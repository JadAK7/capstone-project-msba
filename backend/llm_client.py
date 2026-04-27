"""
llm_client.py
Resilient chat-completion gateway with tenacity retry, per-type circuit
breakers, and multi-provider support.

Environment variables
─────────────────────
  OPENAI_CHAT_MODEL   Model name for chat completions.
                      Default: gpt-4o-mini
                      Other OpenAI examples: gpt-4o, gpt-4-turbo, o1-mini

  LLM_BASE_URL        Base URL for the chat-completion endpoint.
                      Leave unset to use the official OpenAI API.
                      OpenAI-compatible providers — set this + LLM_API_KEY:
                        Groq:       https://api.groq.com/openai/v1
                        Together:   https://api.together.xyz/v1
                        Ollama:     http://localhost:11434/v1
                        OpenRouter: https://openrouter.ai/api/v1
                        Azure:      https://<resource>.openai.azure.com/
                      These providers speak the OpenAI wire protocol so no
                      other code changes are needed.

  LLM_API_KEY         API key for the provider chosen via LLM_BASE_URL.
                      Falls back to OPENAI_API_KEY when not set.

  LLM_PROVIDER        Set to "anthropic" to use the Anthropic SDK instead of
                      the OpenAI SDK. Requires `pip install anthropic`.
                      Supported Anthropic models:
                        claude-3-5-haiku-20241022   (fast, cheap — closest to gpt-4o-mini)
                        claude-3-5-sonnet-20241022  (balanced)
                        claude-3-opus-20240229       (most capable)
                      When LLM_PROVIDER=anthropic, set LLM_API_KEY (or
                      ANTHROPIC_API_KEY) to your Anthropic key and set
                      OPENAI_CHAT_MODEL to one of the claude-* names above.

Switching providers — what changes where
─────────────────────────────────────────
  OpenAI (default)          — no env vars needed beyond OPENAI_API_KEY
  OpenAI-compatible         — set LLM_BASE_URL + LLM_API_KEY
  Anthropic                 — set LLM_PROVIDER=anthropic + LLM_API_KEY +
                              OPENAI_CHAT_MODEL=claude-3-5-haiku-20241022
  Google Gemini             — NOT supported via env var; requires replacing
                              the _call() closure with the google-generativeai
                              SDK (different message schema and response format)

  Embeddings are independent — see embeddings.py / OPENAI_EMBEDDING_MODEL.
  Changing the embedding provider requires a full re-index.

Circuit-breaker call types
──────────────────────────
  "embed"    – text-embedding-3-small/large (embeddings.py)
  "rewrite"  – query rewriter (query_rewriter.py)
  "rerank"   – LLM reranker (reranker.py)
  "generate" – answer generation / grounding (grounding.py, chatbot.py)
  "verify"   – standalone verifier fallback (verifier.py)
"""

import os
import time
import logging
import threading
from typing import Optional, List, Dict, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider / model configuration  (resolved once at import time)
# ---------------------------------------------------------------------------

CHAT_MODEL: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
_LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
_LLM_BASE_URL: Optional[str] = os.environ.get("LLM_BASE_URL", "").strip() or None
_LLM_API_KEY: Optional[str] = (
    os.environ.get("LLM_API_KEY", "").strip()
    or os.environ.get("OPENAI_API_KEY", "").strip()
    or None
)

logger.info(
    "LLM config — provider: %s  model: %s  base_url: %s",
    _LLM_PROVIDER,
    CHAT_MODEL,
    _LLM_BASE_URL or "(OpenAI default)",
)


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

_openai_client = None
_anthropic_client = None
_client_lock = threading.Lock()


def _get_openai_client():
    """Return (or lazily create) the OpenAI-protocol client.

    Works for native OpenAI and any OpenAI-compatible provider (Groq,
    Together, Ollama, OpenRouter, Azure …) — just set LLM_BASE_URL.
    """
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                from openai import OpenAI

                kwargs: Dict[str, Any] = {
                    "api_key": _LLM_API_KEY,
                    "timeout": 30.0,
                }
                if _LLM_BASE_URL:
                    kwargs["base_url"] = _LLM_BASE_URL
                _openai_client = OpenAI(**kwargs)
    return _openai_client


def _get_anthropic_client():
    """Return (or lazily create) the Anthropic client.

    Requires: pip install anthropic
    Set LLM_PROVIDER=anthropic and LLM_API_KEY=<your-anthropic-key>.
    """
    global _anthropic_client
    if _anthropic_client is None:
        with _client_lock:
            if _anthropic_client is None:
                try:
                    import anthropic
                except ImportError as exc:
                    raise ImportError(
                        "LLM_PROVIDER=anthropic requires the 'anthropic' package. "
                        "Install it with: pip install anthropic"
                    ) from exc

                api_key = (
                    _LLM_API_KEY
                    or os.environ.get("ANTHROPIC_API_KEY", "")
                )
                _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ---------------------------------------------------------------------------
# Retry predicate — only retry on 429 and 5xx
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    err = str(exc)
    if "429" in err:
        return True
    for code in ("500", "502", "503", "504"):
        if code in err:
            return True
    return False


# ---------------------------------------------------------------------------
# Circuit breaker (per call type)
# ---------------------------------------------------------------------------

FAILURE_THRESHOLD = 5
COOLDOWN_SECONDS = 60.0


class _CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = FAILURE_THRESHOLD,
        cooldown_seconds: float = COOLDOWN_SECONDS,
        name: str = "default",
    ):
        self._state = self.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
        self._name = name

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._last_failure_time >= self._cooldown:
                    self._state = self.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                prev = self._state
                self._state = self.OPEN
                if prev != self.OPEN:
                    logger.warning(
                        "Circuit breaker OPEN for call type '%s' after %d "
                        "consecutive failures. Will retry in %.0fs.",
                        self._name,
                        self._failure_count,
                        self._cooldown,
                    )

    def allow_request(self) -> bool:
        return self.state in (self.CLOSED, self.HALF_OPEN)


_breakers: Dict[str, _CircuitBreaker] = {
    ct: _CircuitBreaker(name=ct)
    for ct in ("embed", "rewrite", "rerank", "generate", "verify")
}

# Backward-compat alias
_breaker = _breakers["generate"]


# ---------------------------------------------------------------------------
# Provider call implementations
# ---------------------------------------------------------------------------

def _call_openai(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    """Single OpenAI (or OpenAI-compatible) chat call wrapped in tenacity."""
    client = _get_openai_client()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _inner():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=25.0,
        )

    resp = _inner()
    return resp.choices[0].message.content.strip()


def _call_anthropic(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Single Anthropic chat call.

    Anthropic separates the system prompt from the conversation messages.
    This adapter extracts the first system-role message (if present) and
    passes it to the `system=` kwarg, then forwards the rest as `messages`.
    top_p is intentionally omitted — Anthropic recommends not setting both
    temperature and top_p simultaneously.
    """
    client = _get_anthropic_client()

    # Split system message (Anthropic requires it as a separate param)
    system_text = ""
    conversation: List[Dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "system" and not system_text:
            system_text = msg.get("content", "")
        else:
            conversation.append({"role": msg["role"], "content": msg.get("content", "")})

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": conversation,
    }
    if system_text:
        kwargs["system"] = system_text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _inner():
        return client.messages.create(**kwargs)

    resp = _inner()
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LLMUnavailableError(Exception):
    """Raised when the LLM is unavailable (circuit open or retries exhausted)."""
    pass


def chat_completion(
    messages: List[Dict[str, Any]],
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 800,
    top_p: float = 1.0,
    call_type: str = "generate",
) -> str:
    """Resilient chat completion — provider, model, and retry are transparent.

    Args:
        messages:    Chat messages in OpenAI format ([{role, content}, ...]).
                     For Anthropic, the system message is extracted automatically.
        model:       Model name. Defaults to CHAT_MODEL (from OPENAI_CHAT_MODEL env).
        temperature: Sampling temperature.
        max_tokens:  Max response tokens.
        top_p:       Nucleus sampling (ignored for Anthropic).
        call_type:   Circuit-breaker key ('rewrite','rerank','generate','verify').

    Returns:
        The assistant's response as a plain string.

    Raises:
        LLMUnavailableError: Circuit open or all retries exhausted.
    """
    breaker = _breakers.get(call_type, _breakers["generate"])

    if not breaker.allow_request():
        raise LLMUnavailableError(
            f"LLM circuit breaker is open for call type '{call_type}' "
            f"(provider={_LLM_PROVIDER}, model={model}). "
            f"Requests resume after {COOLDOWN_SECONDS:.0f}s cooldown."
        )

    try:
        if _LLM_PROVIDER == "anthropic":
            result = _call_anthropic(messages, model, temperature, max_tokens)
        else:
            result = _call_openai(messages, model, temperature, max_tokens, top_p)

        breaker.record_success()
        return result

    except Exception as e:
        err_msg = str(e)
        # 4xx client errors are not service outages — don't trip the breaker
        if any(code in err_msg for code in ("400", "401", "403", "404", "422")):
            logger.error(
                "LLM client error (not retrying) — provider=%s type=%s: %s",
                _LLM_PROVIDER, call_type, e,
            )
            breaker.record_success()
            raise

        breaker.record_failure()
        logger.error(
            "LLM call failed after retries — provider=%s type=%s: %s",
            _LLM_PROVIDER, call_type, e,
        )
        raise LLMUnavailableError(
            f"LLM unavailable (provider={_LLM_PROVIDER}, type={call_type}): {e}"
        )


def is_llm_available() -> bool:
    """Check if the primary (generate) circuit breaker allows requests."""
    return _breakers["generate"].allow_request()


def is_circuit_open(call_type: str = "generate") -> bool:
    """Check if a specific circuit breaker is currently open."""
    return _breakers.get(call_type, _breakers["generate"]).state == _CircuitBreaker.OPEN


def circuit_breaker_states() -> Dict[str, str]:
    """Return the state of every circuit breaker (for diagnostics)."""
    return {name: b.state for name, b in _breakers.items()}


def llm_info() -> Dict[str, str]:
    """Return current provider / model configuration (for health endpoints)."""
    return {
        "provider": _LLM_PROVIDER,
        "chat_model": CHAT_MODEL,
        "base_url": _LLM_BASE_URL or "https://api.openai.com/v1",
    }
