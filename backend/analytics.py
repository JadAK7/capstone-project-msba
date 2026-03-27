"""
analytics.py
Chat logging and analytics computation for the admin dashboard.
"""

import json
import os
import logging
from collections import Counter, defaultdict
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
        cache_hit: bool = False,
        response_time_ms: float = 0.0,
        query_word_count: int = 0,
        keyword_intent_fired: bool = False,
        sources_above_threshold: int = 0,
        top_faq_question: str = "",
        top_db_name: str = "",
        generated_answer: str = "",
        retrieved_chunks: Optional[list] = None,
    ) -> None:
        """Append a log entry. Non-blocking best-effort write."""
        now = datetime.utcnow()
        entry = {
            "timestamp": now.isoformat() + "Z",
            "query": query,
            "language": language,
            "intent_source": intent_source,
            "response_length": response_length,
            "faq_top_score": round(faq_top_score, 4),
            "db_top_score": round(db_top_score, 4),
            "library_top_score": round(library_top_score, 4),
            "cache_hit": cache_hit,
            "response_time_ms": round(response_time_ms, 1),
            "query_word_count": query_word_count,
            "keyword_intent_fired": keyword_intent_fired,
            "sources_above_threshold": sources_above_threshold,
            "top_faq_question": top_faq_question,
            "top_db_name": top_db_name,
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "generated_answer": generated_answer,
            "retrieved_chunks": retrieved_chunks or [],
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
            date_str = ts[:10]
            if date_str >= cutoff:
                daily[date_str] += 1

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

    def unanswered_queries(self, limit: int = 50) -> dict:
        """Return queries the bot could not answer (intent_source = 'none (unclear)')."""
        entries = self._read_logs()
        total_queries = len(entries)

        unanswered = [e for e in entries if e.get("intent_source") == "none (unclear)"]
        total_unanswered = len(unanswered)

        query_data = {}
        for e in unanswered:
            q = e.get("query", "").strip()
            q_lower = q.lower()
            if not q_lower:
                continue

            if q_lower not in query_data:
                query_data[q_lower] = {
                    "query": q,
                    "count": 0,
                    "last_asked": "",
                    "language": e.get("language", "en"),
                    "faq_top_score": 0.0,
                    "db_top_score": 0.0,
                    "library_top_score": 0.0,
                }

            query_data[q_lower]["count"] += 1
            ts = e.get("timestamp", "")
            if ts > query_data[q_lower]["last_asked"]:
                query_data[q_lower]["last_asked"] = ts
                query_data[q_lower]["faq_top_score"] = e.get("faq_top_score", 0.0)
                query_data[q_lower]["db_top_score"] = e.get("db_top_score", 0.0)
                query_data[q_lower]["library_top_score"] = e.get("library_top_score", 0.0)
                query_data[q_lower]["language"] = e.get("language", "en")

        sorted_queries = sorted(
            query_data.values(),
            key=lambda x: (-x["count"], x["last_asked"]),
        )[:limit]

        return {
            "total_unanswered": total_unanswered,
            "total_queries": total_queries,
            "queries": sorted_queries,
        }

    # ------------------------------------------------------------------
    # Extended analytics for the charts dashboard
    # ------------------------------------------------------------------

    def compute_extended_summary(self) -> dict:
        """Compute extended summary stats for the charts dashboard."""
        entries = self._read_logs()
        if not entries:
            return {
                "total_entries": 0,
                "estimated_sessions": 0,
                "avg_queries_per_session": 0,
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "cache_hit_rate": 0,
                "unanswered_rate": 0,
                "avg_query_word_count": 0,
            }

        # Session estimation (>5 min gap = new session)
        timestamps = []
        for e in entries:
            ts = e.get("timestamp", "")
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts.rstrip("Z")))
                except ValueError:
                    continue
        timestamps.sort()

        sessions = 1
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i - 1]).total_seconds() > 300:
                sessions += 1

        response_times = [
            e.get("response_time_ms", 0)
            for e in entries
            if e.get("response_time_ms", 0) > 0
        ]
        avg_rt = sum(response_times) / len(response_times) if response_times else 0
        p95_rt = 0
        if response_times:
            sorted_rt = sorted(response_times)
            p95_rt = sorted_rt[min(int(len(sorted_rt) * 0.95), len(sorted_rt) - 1)]

        cache_hits = sum(1 for e in entries if e.get("cache_hit"))
        unanswered = sum(
            1 for e in entries if e.get("intent_source") == "none (unclear)"
        )

        word_counts = [e.get("query_word_count", 0) for e in entries]
        avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0

        return {
            "total_entries": len(entries),
            "estimated_sessions": sessions,
            "avg_queries_per_session": round(len(entries) / sessions, 1)
            if sessions
            else 0,
            "avg_response_time_ms": round(avg_rt, 1),
            "p95_response_time_ms": round(p95_rt, 1),
            "cache_hit_rate": round(cache_hits / len(entries) * 100, 1)
            if entries
            else 0,
            "unanswered_rate": round(unanswered / len(entries) * 100, 1)
            if entries
            else 0,
            "avg_query_word_count": round(avg_wc, 1),
        }

    def compute_charts(self) -> dict:
        """Generate all analytics charts as base64 PNGs."""
        from .chart_generator import ChartGenerator

        entries = self._read_logs()
        generator = ChartGenerator(entries)
        return generator.generate_all()
