"""
stage_timer.py
Lightweight per-stage latency instrumentation for the chat pipeline.

Usage:
    timer = StageTimer()
    with timer.time("query_rewriting"):
        result = rewrite_query(...)
    debug["stage_latencies"] = timer.as_dict()

Multiple calls to time() with the same stage name are accumulated (summed),
which handles stages that span multiple non-contiguous code blocks (e.g.
cache_lookup happens in three places — the total is their combined time).
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict

logger = logging.getLogger(__name__)


class StageTimer:
    """Collects per-stage wall-clock latencies (ms) across a pipeline run."""

    def __init__(self):
        self._latencies: Dict[str, float] = {}

    @contextmanager
    def time(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._latencies[stage] = self._latencies.get(stage, 0.0) + elapsed_ms
            logger.debug("Stage '%s': %.1fms", stage, elapsed_ms)

    def as_dict(self) -> Dict[str, float]:
        """Return a copy of accumulated latencies rounded to 1 decimal place."""
        return {k: round(v, 1) for k, v in self._latencies.items()}
