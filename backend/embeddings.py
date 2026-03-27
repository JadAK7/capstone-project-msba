"""
embeddings.py
Centralized embedding generation via OpenAI API.
"""
from __future__ import annotations

import os
import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def embed_text(text: str) -> List[float]:
    """Generate an embedding for a single text string."""
    client = _get_client()
    resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts.

    The OpenAI embeddings API supports batching natively (up to ~2048 inputs).
    For very large batches we chunk into groups of 2000.
    """
    if not texts:
        return []

    client = _get_client()
    all_embeddings: List[List[float]] = []
    batch_size = 2000

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        # Response data is in the same order as input
        all_embeddings.extend([item.embedding for item in resp.data])

    return all_embeddings
