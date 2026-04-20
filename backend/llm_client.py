"""
llm_client.py
Resilient OpenAI chat completion wrapper with timeout, retry, and circuit breaker.

All modules that call gpt-4o-mini should use `chat_completion()` from this module
instead of calling `client.chat.completions.create()` directly.
"""

import os
import time
import logging
import threading
from typing import Optional, List, Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None
_lock = threading.Lock()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    timeout=30.0,  # Global HTTP timeout (seconds)
                )
    return _client


# ---------------------------------------------------------------------------
# Circuit breaker — fail fast when OpenAI is consistently down
# ---------------------------------------------------------------------------

class _CircuitBreaker:
    """Simple circuit breaker for OpenAI API calls.

    States:
      CLOSED  — normal operation, calls go through
      OPEN    — too many recent failures, calls fail immediately
      HALF    — after cooldown, allow one probe call to test recovery

    Thresholds:
      failure_threshold: consecutive failures before opening the circuit
      cooldown_seconds: how long to wait in OPEN before trying HALF-OPEN
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self._state = self.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

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
                self._state = self.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} consecutive failures. "
                    f"Will retry in {self._cooldown}s."
                )

    def allow_request(self) -> bool:
        s = self.state
        return s in (self.CLOSED, self.HALF_OPEN)


_breaker = _CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LLMUnavailableError(Exception):
    """Raised when the LLM is unavailable (circuit open or retries exhausted)."""
    pass


def chat_completion(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 800,
    top_p: float = 1.0,
    max_retries: int = 3,
    timeout: float = 25.0,
) -> str:
    """Resilient chat completion with retry, backoff, and circuit breaker.

    Args:
        messages: OpenAI chat messages list.
        model: Model name.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        top_p: Nucleus sampling parameter.
        max_retries: Number of retry attempts on transient errors.
        timeout: Per-request timeout in seconds (passed to OpenAI client).

    Returns:
        The assistant's message content string.

    Raises:
        LLMUnavailableError: If circuit is open or all retries exhausted.
    """
    if not _breaker.allow_request():
        raise LLMUnavailableError(
            "OpenAI circuit breaker is open — too many recent failures. "
            "Requests will resume automatically after cooldown."
        )

    client = _get_client()

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=timeout,
            )
            _breaker.record_success()
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err_msg = str(e)
            # Don't retry on client errors (4xx) — they won't fix themselves
            if any(code in err_msg for code in ("400", "401", "403", "404", "422")):
                logger.error(f"LLM client error (not retrying): {e}")
                _breaker.record_success()  # Client errors aren't service outages
                raise

            _breaker.record_failure()

            if attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1), 8)  # 2s, 4s, 8s
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait}s: {e}"
                )
                time.sleep(wait)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                raise LLMUnavailableError(
                    f"OpenAI API unavailable after {max_retries} retries: {e}"
                )


def is_llm_available() -> bool:
    """Check if the LLM circuit breaker allows requests."""
    return _breaker.allow_request()


def is_circuit_open() -> bool:
    """Check if the circuit breaker is currently open (too many failures)."""
    return _breaker.state == _CircuitBreaker.OPEN
