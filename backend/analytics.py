"""
analytics.py
Chat logging and analytics computation for the admin dashboard.
All data is stored in PostgreSQL (chat_conversations table) — no file-based logging.
Data persists across container restarts.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from .database import get_connection

logger = logging.getLogger(__name__)


class ChatLogger:
    """Logs chat interactions to the chat_conversations table in PostgreSQL."""

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
        # Hallucination debugging fields
        draft_answer: str = "",
        verified_answer: str = "",
        removed_claims: Optional[list] = None,
        context_sent_to_llm: str = "",
        verification_passed: bool = True,
        # Safety / guard fields
        guard_injection_detected: bool = False,
        guard_out_of_scope: bool = False,
        guard_refusal_reason: str = "",
        guard_matched_patterns: Optional[list] = None,
    ) -> Optional[int]:
        """Insert a log entry into chat_conversations. Returns conversation_id."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO chat_conversations (
                            query, answer, language, chosen_source,
                            faq_top_score, db_top_score, library_top_score,
                            response_time_ms, retrieved_chunks,
                            cache_hit, query_word_count, keyword_intent_fired,
                            sources_above_threshold, top_faq_question, top_db_name,
                            draft_answer, verified_answer, removed_claims,
                            context_sent_to_llm, verification_passed,
                            guard_injection_detected, guard_out_of_scope,
                            guard_refusal_reason, guard_matched_patterns
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            %s, %s,
                            %s, %s
                        ) RETURNING id""",
                        (
                            query,
                            generated_answer,
                            language,
                            intent_source,
                            round(faq_top_score, 4),
                            round(db_top_score, 4),
                            round(library_top_score, 4),
                            round(response_time_ms, 1),
                            json.dumps(retrieved_chunks or []),
                            cache_hit,
                            query_word_count,
                            keyword_intent_fired,
                            sources_above_threshold,
                            top_faq_question,
                            top_db_name,
                            draft_answer,
                            verified_answer,
                            json.dumps(removed_claims or []),
                            (context_sent_to_llm[:3000] if context_sent_to_llm else ""),
                            verification_passed,
                            guard_injection_detected,
                            guard_out_of_scope,
                            guard_refusal_reason,
                            json.dumps(guard_matched_patterns or []),
                        ),
                    )
                    row = cur.fetchone()
                    conversation_id = row[0] if row else None
                conn.commit()
                return conversation_id
        except Exception as e:
            logger.error(f"Failed to log chat to database: {e}")
            return None


class AnalyticsComputer:
    """Reads chat_conversations from PostgreSQL and computes analytics on demand."""

    def _read_logs(self, days: Optional[int] = None) -> list:
        """Read log entries from chat_conversations table.

        Args:
            days: If set, only return entries from the last N days.
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    if days:
                        cur.execute(
                            """SELECT query, answer, language, chosen_source,
                                      faq_top_score, db_top_score, library_top_score,
                                      response_time_ms, cache_hit, query_word_count,
                                      keyword_intent_fired, sources_above_threshold,
                                      top_faq_question, top_db_name,
                                      verification_passed,
                                      guard_injection_detected, guard_out_of_scope,
                                      created_at
                               FROM chat_conversations
                               WHERE created_at >= NOW() - INTERVAL '%s days'
                               ORDER BY created_at ASC""",
                            (days,),
                        )
                    else:
                        cur.execute(
                            """SELECT query, answer, language, chosen_source,
                                      faq_top_score, db_top_score, library_top_score,
                                      response_time_ms, cache_hit, query_word_count,
                                      keyword_intent_fired, sources_above_threshold,
                                      top_faq_question, top_db_name,
                                      verification_passed,
                                      guard_injection_detected, guard_out_of_scope,
                                      created_at
                               FROM chat_conversations
                               ORDER BY created_at ASC"""
                        )
                    rows = cur.fetchall()

            entries = []
            for row in rows:
                ts = row[17]  # created_at
                entries.append({
                    "query": row[0],
                    "generated_answer": row[1],
                    "language": row[2],
                    "intent_source": row[3],
                    "faq_top_score": float(row[4] or 0),
                    "db_top_score": float(row[5] or 0),
                    "library_top_score": float(row[6] or 0),
                    "response_time_ms": float(row[7] or 0),
                    "cache_hit": bool(row[8]),
                    "query_word_count": int(row[9] or 0),
                    "keyword_intent_fired": bool(row[10]),
                    "sources_above_threshold": int(row[11] or 0),
                    "top_faq_question": row[12] or "",
                    "top_db_name": row[13] or "",
                    "verification_passed": bool(row[14]) if row[14] is not None else True,
                    "guard_injection_detected": bool(row[15]),
                    "guard_out_of_scope": bool(row[16]),
                    "timestamp": ts.isoformat() + "Z" if ts else "",
                    "response_length": len(row[1]) if row[1] else 0,
                })
            return entries

        except Exception as e:
            logger.error(f"Failed to read chat logs from database: {e}")
            return []

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
        from datetime import timedelta
        week_ago = (now - timedelta(days=7)).isoformat() + "Z"
        month_ago = (now - timedelta(days=30)).isoformat() + "Z"

        today_count = 0
        week_count = 0
        month_count = 0
        lang_counter = {}
        intent_counter = {}
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

            lang = e.get("language", "en")
            lang_counter[lang] = lang_counter.get(lang, 0) + 1

            intent = e.get("intent_source", "unknown")
            intent_counter[intent] = intent_counter.get(intent, 0) + 1

            faq_scores.append(e.get("faq_top_score", 0.0))
            db_scores.append(e.get("db_top_score", 0.0))
            library_scores.append(e.get("library_top_score", 0.0))

        def _avg(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        lang_dist = {lang: round(count / total * 100, 1) for lang, count in lang_counter.items()}
        intent_dist = {intent: round(count / total * 100, 1) for intent, count in intent_counter.items()}

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
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT DATE(created_at) AS day, COUNT(*) AS count
                           FROM chat_conversations
                           WHERE created_at >= NOW() - INTERVAL '%s days'
                           GROUP BY DATE(created_at)
                           ORDER BY day""",
                        (days,),
                    )
                    rows = cur.fetchall()

            daily = {str(row[0]): row[1] for row in rows}
        except Exception as e:
            logger.error(f"Failed to compute trends: {e}")
            daily = {}

        now = datetime.utcnow()
        from datetime import timedelta
        result = []
        for i in range(days, -1, -1):
            day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            result.append({"date": day, "count": daily.get(day, 0)})
        return result

    def top_queries(self, limit: int = 20) -> list:
        """Return the most frequent queries."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT LOWER(TRIM(query)) AS q,
                                  COUNT(*) AS count,
                                  MAX(created_at) AS last_asked
                           FROM chat_conversations
                           WHERE query IS NOT NULL AND TRIM(query) != ''
                           GROUP BY LOWER(TRIM(query))
                           ORDER BY count DESC
                           LIMIT %s""",
                        (limit,),
                    )
                    rows = cur.fetchall()

            return [
                {
                    "query": row[0],
                    "count": row[1],
                    "last_asked": row[2].isoformat() + "Z" if row[2] else "",
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to compute top queries: {e}")
            return []

    def unanswered_queries(self, limit: int = 50) -> dict:
        """Return queries the bot could not answer (chosen_source = 'none (unclear)')."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Total counts
                    cur.execute("SELECT COUNT(*) FROM chat_conversations")
                    total_queries = cur.fetchone()[0]

                    cur.execute(
                        "SELECT COUNT(*) FROM chat_conversations WHERE chosen_source = 'none (unclear)'"
                    )
                    total_unanswered = cur.fetchone()[0]

                    # Grouped unanswered queries
                    cur.execute(
                        """SELECT LOWER(TRIM(query)) AS q,
                                  COUNT(*) AS count,
                                  MAX(created_at) AS last_asked,
                                  MAX(language) AS language,
                                  MAX(faq_top_score) AS faq_top_score,
                                  MAX(db_top_score) AS db_top_score,
                                  MAX(library_top_score) AS library_top_score
                           FROM chat_conversations
                           WHERE chosen_source = 'none (unclear)'
                             AND query IS NOT NULL AND TRIM(query) != ''
                           GROUP BY LOWER(TRIM(query))
                           ORDER BY count DESC, last_asked DESC
                           LIMIT %s""",
                        (limit,),
                    )
                    rows = cur.fetchall()

            queries = [
                {
                    "query": row[0],
                    "count": row[1],
                    "last_asked": row[2].isoformat() + "Z" if row[2] else "",
                    "language": row[3] or "en",
                    "faq_top_score": float(row[4] or 0),
                    "db_top_score": float(row[5] or 0),
                    "library_top_score": float(row[6] or 0),
                }
                for row in rows
            ]

            return {
                "total_unanswered": total_unanswered,
                "total_queries": total_queries,
                "queries": queries,
            }

        except Exception as e:
            logger.error(f"Failed to compute unanswered queries: {e}")
            return {"total_unanswered": 0, "total_queries": 0, "queries": []}

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
