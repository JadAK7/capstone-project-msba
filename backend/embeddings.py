"""
embeddings.py
Centralized embedding generation via OpenAI API, with per-type circuit breaker.

When the 'embed' circuit opens after 5 consecutive failures, callers receive
LLMUnavailableError.  The chatbot handles this by returning a user-facing
error message and skipping cache lookup and retrieval.
"""
from __future__ import annotations

import os
import time
import logging
from typing import List

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model selection via environment variable
#
#   OPENAI_EMBEDDING_MODEL=text-embedding-3-small   (default, 1536 dims)
#   OPENAI_EMBEDDING_MODEL=text-embedding-3-large   (3072 dims, higher quality)
#
# Changing the model requires a full re-index (python scripts/build_index.py)
# because existing vectors in the database have a different dimension.
# The database layer detects the mismatch on startup and auto-migrates.
# ---------------------------------------------------------------------------

_MODEL_DIMS: dict = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _resolve_embedding_config() -> tuple:
    raw = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    model = raw.lower()
    if model not in _MODEL_DIMS:
        logger.warning(
            "Unknown OPENAI_EMBEDDING_MODEL '%s'; supported values: %s. "
            "Defaulting to text-embedding-3-small.",
            raw,
            list(_MODEL_DIMS),
        )
        model = "text-embedding-3-small"
    dim = _MODEL_DIMS[model]
    logger.info("Embedding model: %s  (%d dimensions)", model, dim)
    return model, dim


EMBEDDING_MODEL, EMBEDDING_DIM = _resolve_embedding_config()

_MAX_INPUT_TOKENS = 8192
_MAX_ITEMS_PER_REQUEST = 2048
_MAX_TOKENS_PER_REQUEST = 250_000

_CHARS_PER_TOKEN = 3
_MAX_INPUT_CHARS = _MAX_INPUT_TOKENS * _CHARS_PER_TOKEN  # 24 576 chars

# Try to use tiktoken for accurate token counts (handles non-Latin scripts).
# Falls back to a byte-based estimator that is conservative for Arabic, where
# cl100k_base produces roughly 1 token per UTF-8 byte.
try:
    import tiktoken as _tiktoken
    _ENCODING = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses tiktoken when available. Falls back to a heuristic that is safe for
    bilingual (Arabic + English) batches: ASCII-only text uses ~3 chars/token,
    non-ASCII counts UTF-8 bytes (≈ tokens for cl100k_base on Arabic).
    """
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text, disallowed_special=()))
        except Exception:
            pass
    if text.isascii():
        return len(text) // _CHARS_PER_TOKEN + 1
    return len(text.encode("utf-8")) + 1


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _sanitize(text: str) -> str:
    text = (text or "").strip()
    if not text:
        text = "(empty)"
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]
    return text


def _is_retryable_embed(exc: Exception) -> bool:
    err = str(exc)
    if "429" in err:
        return True
    for code in ("500", "502", "503", "504"):
        if code in err:
            return True
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception(_is_retryable_embed),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_api(client: OpenAI, batch: List[str]) -> List[List[float]]:
    """Single tenacity-wrapped embedding API call."""
    resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
    return [item.embedding for item in resp.data]


def _call_with_retry(client: OpenAI, batch: List[str]) -> List[List[float]]:
    """Call embedding API; 4xx errors are not retried."""
    try:
        return _call_api(client, batch)
    except Exception as e:
        err_msg = str(e)
        if "400" in err_msg or "invalid" in err_msg.lower():
            logger.error("Embedding API validation error (batch of %d): %s", len(batch), e)
            raise
        raise


def embed_text(text: str) -> List[float]:
    """Generate an embedding for a single text string.

    Raises LLMUnavailableError if the embedding circuit breaker is open.
    """
    from .llm_client import _breakers, LLMUnavailableError

    breaker = _breakers["embed"]
    if not breaker.allow_request():
        raise LLMUnavailableError(
            "Embedding circuit breaker is open — too many recent failures."
        )

    client = _get_client()
    try:
        result = _call_with_retry(client, [_sanitize(text)])[0]
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        from .llm_client import LLMUnavailableError as _LLMErr
        raise _LLMErr(f"Embedding failed: {e}") from e


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    if not texts:
        return []

    from .llm_client import _breakers, LLMUnavailableError

    breaker = _breakers["embed"]
    if not breaker.allow_request():
        raise LLMUnavailableError(
            "Embedding circuit breaker is open — too many recent failures."
        )

    client = _get_client()
    all_embeddings: List[List[float]] = []

    batch: List[str] = []
    batch_tokens = 0

    try:
        for raw_text in texts:
            text = _sanitize(raw_text)
            est_tokens = _estimate_tokens(text)

            if batch and (
                batch_tokens + est_tokens > _MAX_TOKENS_PER_REQUEST
                or len(batch) >= _MAX_ITEMS_PER_REQUEST
            ):
                all_embeddings.extend(_call_with_retry(client, batch))
                batch = []
                batch_tokens = 0

            batch.append(text)
            batch_tokens += est_tokens

        if batch:
            all_embeddings.extend(_call_with_retry(client, batch))

        breaker.record_success()
        return all_embeddings

    except Exception as e:
        breaker.record_failure()
        raise
