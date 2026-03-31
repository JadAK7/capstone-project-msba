"""
embeddings.py
Centralized embedding generation via OpenAI API.
"""
from __future__ import annotations

import os
import time
import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# OpenAI limits for this model
_MAX_INPUT_TOKENS = 8192
_MAX_ITEMS_PER_REQUEST = 2048
_MAX_TOKENS_PER_REQUEST = 250_000  # actual limit is 300K; leave headroom

# Use 3 chars per token for truncation (conservative — guarantees we stay
# under the token limit even for dense content like URLs or code).
_CHARS_PER_TOKEN = 3
_MAX_INPUT_CHARS = _MAX_INPUT_TOKENS * _CHARS_PER_TOKEN  # 24 576 chars

_MAX_RETRIES = 3

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _sanitize(text: str) -> str:
    """Ensure a text input is safe for the embeddings API.

    - Replaces empty/whitespace-only strings with a placeholder
    - Truncates to stay under the per-input token limit
    """
    text = (text or "").strip()
    if not text:
        text = "(empty)"
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]
    return text


def _call_with_retry(client, batch: List[str]) -> List[List[float]]:
    """Call the embeddings API with retry on transient errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
            return [item.embedding for item in resp.data]
        except Exception as e:
            err_msg = str(e)
            # Don't retry on validation errors (4xx) — they won't fix themselves
            if "400" in err_msg or "invalid" in err_msg.lower():
                logger.error(f"Embedding API validation error (batch of {len(batch)}): {e}")
                raise
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Embedding API error (attempt {attempt+1}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def embed_text(text: str) -> List[float]:
    """Generate an embedding for a single text string."""
    client = _get_client()
    return _call_with_retry(client, [_sanitize(text)])[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts.

    Handles all OpenAI limits:
    - Per-input: 8 192 tokens  (truncated via _sanitize)
    - Per-request items: 2 048 (batch splits)
    - Per-request tokens: 300K (batch splits at 250K)
    """
    if not texts:
        return []

    client = _get_client()
    all_embeddings: List[List[float]] = []

    batch: List[str] = []
    batch_tokens = 0

    for raw_text in texts:
        text = _sanitize(raw_text)
        est_tokens = len(text) // _CHARS_PER_TOKEN + 1

        if batch and (batch_tokens + est_tokens > _MAX_TOKENS_PER_REQUEST
                      or len(batch) >= _MAX_ITEMS_PER_REQUEST):
            all_embeddings.extend(_call_with_retry(client, batch))
            batch = []
            batch_tokens = 0

        batch.append(text)
        batch_tokens += est_tokens

    if batch:
        all_embeddings.extend(_call_with_retry(client, batch))

    return all_embeddings
