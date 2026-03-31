"""
cache.py
In-memory response cache with TTL expiration and LRU eviction.

Caches full chatbot responses keyed on (search_query, language) to avoid
redundant OpenAI embedding calls and LLM completions
for repeated or similar queries. Designed for the AUB Libraries Assistant
where the query volume is moderate and in-memory caching is sufficient.
"""

import time
import logging
from collections import OrderedDict
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ResponseCache:
    """TTL-based in-memory cache with LRU eviction.

    Keys are (search_query, language) tuples. Values are (answer, debug) tuples.
    Expired entries are lazily evicted on access. When max_size is exceeded,
    the least-recently-used entry is evicted.

    Args:
        max_size: Maximum number of cached entries (default 256).
        ttl_seconds: Time-to-live for each entry in seconds (default 3600 = 1 hour).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._max_size = max_size
        self._ttl = ttl_seconds
        # OrderedDict for LRU: most-recently-used items are moved to the end
        self._store: OrderedDict[Tuple[str, str], Tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Tuple[str, str]) -> Optional[Any]:
        """Look up a cached response.

        Args:
            key: (search_query, language) tuple.

        Returns:
            The cached value (answer, debug) if found and not expired, else None.
        """
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        timestamp, value = entry
        if time.time() - timestamp > self._ttl:
            # Entry expired -- remove it
            del self._store[key]
            self._misses += 1
            return None

        # Cache hit -- move to end (most recently used)
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: Tuple[str, str], value: Any) -> None:
        """Store a response in the cache.

        Args:
            key: (search_query, language) tuple.
            value: The (answer, debug) tuple to cache.
        """
        # If key already exists, update it and move to end
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = (time.time(), value)
            return

        # Evict oldest (least recently used) if at capacity
        while len(self._store) >= self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug(f"Cache evicted LRU entry: {evicted_key[0][:50]}...")

        self._store[key] = (time.time(), value)

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
            Dict with size, max_size, ttl_seconds, hits, misses, hit_rate.
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
        }
