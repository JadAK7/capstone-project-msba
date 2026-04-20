"""
cache.py
In-memory response cache with TTL expiration, LRU eviction, and semantic matching.

Caches full chatbot responses keyed on (search_query, language) to avoid
redundant OpenAI embedding calls and LLM completions
for repeated or similar queries. Designed for the AUB Libraries Assistant
where the query volume is moderate and in-memory caching is sufficient.

Semantic cache layer: when an exact key miss occurs, the cache checks cosine
similarity between the query embedding and all stored entry embeddings.
If any stored entry is above the similarity threshold (default 0.95),
the cached response is returned. This catches semantically identical queries
with slightly different phrasings that the query rewriter didn't normalize.
"""

import atexit
import os
import pickle
import tempfile
import threading
import time
import logging
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class ResponseCache:
    """TTL-based in-memory cache with LRU eviction and semantic similarity matching.

    Keys are (search_query, language) tuples. Values are (answer, debug) tuples.
    Expired entries are lazily evicted on access. When max_size is exceeded,
    the least-recently-used entry is evicted.

    Semantic matching: when an exact key lookup misses, the cache compares the
    query embedding against all stored embeddings. If cosine similarity exceeds
    the threshold, the cached response is returned.

    Args:
        max_size: Maximum number of cached entries (default 256).
        ttl_seconds: Time-to-live for each entry in seconds (default 3600 = 1 hour).
        semantic_threshold: Minimum cosine similarity for semantic cache hit (default 0.95).
    """

    # Default disk path for cache persistence
    _DEFAULT_CACHE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", ".response_cache.pkl",
    )

    # Snapshot every N new puts or every M seconds, whichever comes first
    _SNAPSHOT_INTERVAL_PUTS = 100
    _SNAPSHOT_INTERVAL_SECS = 300  # 5 minutes

    def __init__(
        self,
        max_size: int = 1024,
        ttl_seconds: int = 3600,
        semantic_threshold: float = 0.95,
        persist_path: Optional[str] = None,
    ):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._semantic_threshold = semantic_threshold
        self._persist_path = persist_path or self._DEFAULT_CACHE_PATH
        # OrderedDict for LRU: most-recently-used items are moved to the end
        # Value: (timestamp, value, embedding_or_None)
        self._store: OrderedDict[Tuple[str, str], Tuple[float, Any, Optional[np.ndarray]]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0

        # Snapshot tracking for crash recovery
        self._puts_since_snapshot = 0
        self._last_snapshot_time = time.time()
        self._snapshot_lock = threading.Lock()

        # Try to load persisted cache from disk
        self._load_from_disk()

        # Register atexit as a backup save mechanism
        atexit.register(self.save_to_disk)

    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self._ttl

    def get(self, key: Tuple[str, str]) -> Optional[Any]:
        """Look up a cached response by exact key match.

        Args:
            key: (search_query, language) tuple.

        Returns:
            The cached value (answer, debug) if found and not expired, else None.
        """
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        timestamp, value, _embedding = entry
        if self._is_expired(timestamp):
            del self._store[key]
            self._misses += 1
            return None

        # Cache hit -- move to end (most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def semantic_get(
        self,
        query_embedding: List[float],
        language: str,
    ) -> Optional[Any]:
        """Look up a cached response by semantic similarity.

        Compares the query embedding against all stored embeddings for the
        same language. Returns the value of the most similar entry if it
        exceeds the semantic threshold.

        This is called AFTER an exact-key miss, so it only adds latency
        on cache misses (where we'd spend 2-3 seconds on the full pipeline
        anyway). The numpy dot product over ~256 entries takes <1ms.

        Args:
            query_embedding: The embedding vector of the query.
            language: Language code to match (only compares within same language).

        Returns:
            The cached value (answer, debug) if a semantic match is found, else None.
        """
        if not query_embedding or not self._store:
            return None

        query_vec = np.array(query_embedding, dtype=np.float32)
        best_sim = 0.0
        best_key = None

        # Collect expired keys for cleanup
        expired_keys = []

        for key, (timestamp, _value, embedding) in self._store.items():
            # Only match within same language
            if key[1] != language:
                continue
            if self._is_expired(timestamp):
                expired_keys.append(key)
                continue
            if embedding is None:
                continue

            sim = _cosine_similarity(query_vec, embedding)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        # Clean up expired entries
        for k in expired_keys:
            del self._store[k]

        if best_key is not None and best_sim >= self._semantic_threshold:
            self._store.move_to_end(best_key)
            self._hits += 1
            self._semantic_hits += 1
            _, value, _ = self._store[best_key]
            logger.info(
                f"Semantic cache hit: similarity={best_sim:.4f} "
                f"matched='{best_key[0][:60]}...'"
            )
            return value

        return None

    def put(
        self,
        key: Tuple[str, str],
        value: Any,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Store a response in the cache.

        Args:
            key: (search_query, language) tuple.
            value: The (answer, debug) tuple to cache.
            embedding: Optional query embedding for semantic matching.
        """
        emb_array = np.array(embedding, dtype=np.float32) if embedding is not None else None

        # If key already exists, update it and move to end
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = (time.time(), value, emb_array)
            return

        # Evict oldest (least recently used) if at capacity
        while len(self._store) >= self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug(f"Cache evicted LRU entry: {evicted_key[0][:50]}...")

        self._store[key] = (time.time(), value, emb_array)

        # Periodic snapshot for crash recovery
        self._puts_since_snapshot += 1
        elapsed = time.time() - self._last_snapshot_time
        if (self._puts_since_snapshot >= self._SNAPSHOT_INTERVAL_PUTS
                or elapsed >= self._SNAPSHOT_INTERVAL_SECS):
            self._snapshot_async()

    def _snapshot_async(self) -> None:
        """Save a snapshot to disk in a background thread."""
        if not self._snapshot_lock.acquire(blocking=False):
            return  # Another snapshot is already in progress
        self._puts_since_snapshot = 0
        self._last_snapshot_time = time.time()
        # Release lock after thread finishes
        def _do_snapshot():
            try:
                self.save_to_disk()
            finally:
                self._snapshot_lock.release()
        threading.Thread(target=_do_snapshot, daemon=True).start()

    def invalidate_all(self) -> None:
        """Clear the entire cache and remove the disk snapshot.

        Called when admin modifies content (FAQ CRUD, custom notes, reindex, etc.)
        to ensure stale data is never served.
        """
        count = len(self._store)
        self._store.clear()
        if count > 0:
            logger.info(f"Cache invalidated: {count} entries cleared")
        # Remove disk snapshot so stale data isn't loaded on restart
        try:
            if os.path.exists(self._persist_path):
                os.remove(self._persist_path)
        except OSError:
            pass

    def clear(self) -> None:
        """Invalidate all cached entries.

        Called when admin triggers a reindex so stale data is not served.
        """
        count = len(self._store)
        self._store.clear()
        if count > 0:
            logger.info(f"Cache cleared: {count} entries invalidated")

    def stats(self) -> dict:
        """Return cache statistics for observability.

        Returns:
            Dict with size, max_size, ttl_seconds, hits, misses, hit_rate,
            and semantic_hits count.
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 1),
            "semantic_hits": self._semantic_hits,
            "semantic_threshold": self._semantic_threshold,
        }

    def save_to_disk(self) -> None:
        """Persist cache to disk so it survives process restarts and crashes.

        Uses atomic write (temp file + rename) to avoid corruption if the
        process dies mid-write. Called periodically (every 100 puts or 5 min),
        on shutdown, and via atexit as a backup.
        """
        try:
            # Filter out expired entries before saving
            now = time.time()
            to_save = OrderedDict()
            for key, (timestamp, value, embedding) in self._store.items():
                if now - timestamp <= self._ttl:
                    to_save[key] = (timestamp, value, embedding)

            if not to_save:
                # Nothing to save — remove stale file if it exists
                if os.path.exists(self._persist_path):
                    os.remove(self._persist_path)
                return

            persist_dir = os.path.dirname(self._persist_path)
            os.makedirs(persist_dir, exist_ok=True)

            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(dir=persist_dir, suffix=".pkl.tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, self._persist_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            logger.info(f"Cache saved to disk: {len(to_save)} entries -> {self._persist_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk (non-critical): {e}")

    def _load_from_disk(self) -> None:
        """Load persisted cache from disk on startup.

        Silently skips if file doesn't exist or is corrupt.
        Expired entries are filtered out during loading.
        """
        if not os.path.exists(self._persist_path):
            return

        try:
            with open(self._persist_path, "rb") as f:
                loaded = pickle.load(f)

            if not isinstance(loaded, OrderedDict):
                logger.warning("Cache file has unexpected format, ignoring")
                return

            now = time.time()
            restored = 0
            for key, (timestamp, value, embedding) in loaded.items():
                # Skip expired entries
                if now - timestamp > self._ttl:
                    continue
                if len(self._store) >= self._max_size:
                    break
                self._store[key] = (timestamp, value, embedding)
                restored += 1

            if restored > 0:
                logger.info(f"Cache restored from disk: {restored} entries from {self._persist_path}")

        except Exception as e:
            logger.warning(f"Failed to load cache from disk (starting fresh): {e}")
