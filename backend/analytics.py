"""
analytics.py
Chat logging and analytics computation for the admin dashboard.
"""

import json
import os
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Default log file path (project root)
_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_logs.json")


class ChatLogger:
    """Appends chat interaction logs to a JSON-lines file."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or _LOG_FILE

    def log(
        self,
        query: str,
        language: str,
        intent_source: str,
        response_length: int,
        faq_top_score: float,
        db_top_score: float,
        library_top_score: float,
    ) -> None:
        """Append a log entry. Non-blocking best-effort write."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "language": language,
            "intent_source": intent_source,
            "response_length": response_length,
            "faq_top_score": round(faq_top_score, 4),
            "db_top_score": round(db_top_score, 4),
            "library_top_score": round(library_top_score, 4),
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write chat log: {e}")


class AnalyticsComputer:
    """Reads chat logs and computes analytics on demand."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or _LOG_FILE

    def _read_logs(self) -> list:
        """Read all log entries from disk."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Failed to read chat logs: {e}")
        return entries

    def summary(self) -> dict:
        """Compute summary analytics."""
        entries = self._read_logs()
        total = len(entries)

        if total == 0:
            return {
                "total_conversations": 0,
                "today": 0,
                "this_week": 0,
                "this_month": 0,
                "avg_faq_score": 0.0,
                "avg_db_score": 0.0,
                "avg_library_score": 0.0,
                "language_distribution": {"en": 0, "ar": 0},
                "intent_distribution": {},
            }

        now = datetime.utcnow()
        today_str = now.strftime("%Y-%m-%d")
        week_ago = (now - timedelta(days=7)).isoformat() + "Z"
        month_ago = (now - timedelta(days=30)).isoformat() + "Z"

        today_count = 0
        week_count = 0
        month_count = 0
        lang_counter = Counter()
        intent_counter = Counter()
        faq_scores = []
        db_scores = []
        library_scores = []

        for e in entries:
            ts = e.get("timestamp", "")

            if ts.startswith(today_str):
                today_count += 1
            if ts >= week_ago:
                week_count += 1
            if ts >= month_ago:
                month_count += 1

            lang_counter[e.get("language", "en")] += 1
            intent_counter[e.get("intent_source", "unknown")] += 1

            faq_scores.append(e.get("faq_top_score", 0.0))
            db_scores.append(e.get("db_top_score", 0.0))
            library_scores.append(e.get("library_top_score", 0.0))

        def _avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        # Convert counters to percentage dicts
        lang_dist = {}
        for lang, count in lang_counter.items():
            lang_dist[lang] = round(count / total * 100, 1)

        intent_dist = {}
        for intent, count in intent_counter.items():
            intent_dist[intent] = round(count / total * 100, 1)

        return {
            "total_conversations": total,
            "today": today_count,
            "this_week": week_count,
            "this_month": month_count,
            "avg_faq_score": _avg(faq_scores),
            "avg_db_score": _avg(db_scores),
            "avg_library_score": _avg(library_scores),
            "language_distribution": lang_dist,
            "intent_distribution": intent_dist,
        }

    def trends(self, days: int = 30) -> list:
        """Return daily conversation counts for the last N days."""
        entries = self._read_logs()
        now = datetime.utcnow()
        cutoff = (now - timedelta(days=days)).strftime("%Y-%m-%d")

        daily = Counter()
        for e in entries:
            ts = e.get("timestamp", "")
            date_str = ts[:10]  # "YYYY-MM-DD"
            if date_str >= cutoff:
                daily[date_str] += 1

        # Build full date range so every day appears even with 0 count
        result = []
        for i in range(days, -1, -1):
            day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            result.append({"date": day, "count": daily.get(day, 0)})
        return result

    def top_queries(self, limit: int = 20) -> list:
        """Return the most frequent queries."""
        entries = self._read_logs()
        query_counter = Counter()
        last_seen = {}

        for e in entries:
            q = e.get("query", "").strip().lower()
            if q:
                query_counter[q] += 1
                last_seen[q] = e.get("timestamp", "")

        top = query_counter.most_common(limit)
        result = []
        for query, count in top:
            result.append({
                "query": query,
                "count": count,
                "last_asked": last_seen.get(query, ""),
            })
        return result
