"""
chatbot.py
Core chatbot logic for the FastAPI backend.
Supports Arabic and English (bilingual).
Uses PostgreSQL + pgvector for vector similarity search.
"""

import os
import re
import numpy as np
from openai import OpenAI
from typing import List, Tuple, Optional
import logging

from .cache import ResponseCache
from .database import get_connection
from .embeddings import embed_text

logger = logging.getLogger(__name__)


class Config:
    EMBEDDING_MODEL = "text-embedding-3-small"

    FAQ_COLLECTION = "faq"
    DB_COLLECTION = "databases"
    LIBRARY_COLLECTION = "library_pages"

    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    LIBRARY_MIN_CONFIDENCE = 0.35
    BOTH_DELTA = 0.06

    # Cross-lingual penalty: Arabic queries against English-indexed data
    # typically score 10-20% lower in cosine similarity. This offset is
    # subtracted from thresholds when the query language differs from the
    # storage language (English) to avoid incorrectly classifying relevant
    # matches as "unclear".
    CROSS_LINGUAL_THRESHOLD_OFFSET = 0.10

    # Conversation history
    MAX_HISTORY_TURNS = 5  # Number of recent turns (user+assistant pairs) to include in LLM context

    # Supported languages
    LANG_EN = "en"
    LANG_AR = "ar"
    DEFAULT_LANG = LANG_EN


class LanguageDetector:
    """Detects whether user input is Arabic or English.

    Uses Unicode character analysis: if the text contains Arabic script
    characters (\u0600-\u06FF, \u0750-\u077F, \uFB50-\uFDFF, \uFE70-\uFEFF),
    it is classified as Arabic. This is reliable for the AUB use case where
    queries are either predominantly Arabic or predominantly English.
    """

    # Unicode ranges for Arabic script
    _ARABIC_PATTERN = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
    )

    @classmethod
    def detect(cls, text: str) -> str:
        """Detect language of input text.

        Returns Config.LANG_AR if the text contains Arabic characters,
        Config.LANG_EN otherwise. For mixed content, Arabic takes precedence
        since an Arabic-speaking user may include English technical terms
        (e.g. database names like 'IEEE' or 'Scopus').
        """
        if not text or not text.strip():
            return Config.DEFAULT_LANG

        arabic_chars = len(cls._ARABIC_PATTERN.findall(text))
        # Even a small amount of Arabic script indicates an Arabic-language query
        if arabic_chars > 0:
            return Config.LANG_AR
        return Config.LANG_EN


class EmbeddingUtils:
    """Utilities for text cleaning."""

    @staticmethod
    def clean_html(s: str) -> str:
        s = re.sub(r"<br\s*/?>", "\n", str(s), flags=re.IGNORECASE)
        s = re.sub(r"<[^>]+>", "", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()

    @staticmethod
    def clean_library_content(s: str) -> str:
        s = re.sub(
            r"^.*?HOME\s*>\s*LIBRARIES[^a-z]*(?=[A-Z][a-z])",
            "", s, count=1, flags=re.DOTALL
        )
        s = re.sub(
            r"\s+SERVICES\s+DIRECTIONS & ACCESSIBILITY\s+FOR ALUMNI.*$",
            "", s, flags=re.DOTALL
        )
        s = re.sub(r"\s+", " ", s).strip()
        return s


def _pgvector_distance_to_similarity(distances: List[float]) -> List[float]:
    """Convert pgvector cosine distances to similarity scores.

    pgvector's <=> operator returns cosine distance (1 - similarity).
    """
    return [1.0 - d for d in distances]


class IntentDetector:
    """Detects user intent from query. Supports English and Arabic keywords."""

    # English database keywords
    DB_KEYWORDS_EN = re.compile(
        r"\b(database|db|which database|where to search|where can i find|"
        r"source for|best database|recommend|ieee|scopus|pubmed|jstor|"
        r"proquest|web of science|find.*articles?|search.*papers?|"
        r"research.*source)\b",
        re.IGNORECASE
    )

    # Arabic database keywords
    DB_KEYWORDS_AR = re.compile(
        r"("
        r"\u0642\u0627\u0639\u062F\u0629\s*\u0628\u064A\u0627\u0646\u0627\u062A|"
        r"\u0642\u0648\u0627\u0639\u062F\s*\u0628\u064A\u0627\u0646\u0627\u062A|"
        r"\u0642\u0627\u0639\u062F\u0629\s*\u0645\u0639\u0644\u0648\u0645\u0627\u062A|"
        r"\u0623\u064A\u0646\s*\u0623\u0628\u062D\u062B|"
        r"\u0623\u064A\u0646\s*\u0623\u062C\u062F|"
        r"\u0645\u0635\u062F\u0631|"
        r"\u0645\u0635\u0627\u062F\u0631|"
        r"\u0623\u0641\u0636\u0644\s*\u0642\u0627\u0639\u062F\u0629|"
        r"\u0623\u0648\u0635\u064A|"
        r"\u062A\u0648\u0635\u064A\u0629|"
        r"\u0623\u0646\u0635\u062D|"
        r"\u0627\u0628\u062D\u062B|"
        r"\u0628\u062D\u062B|"
        r"\u0645\u0642\u0627\u0644\u0627\u062A|"
        r"\u0645\u0642\u0627\u0644\u0629|"
        r"\u0623\u0648\u0631\u0627\u0642\s*\u0628\u062D\u062B\u064A\u0629|"
        r"\u0648\u0631\u0642\u0629\s*\u0628\u062D\u062B\u064A\u0629|"
        r"\u0645\u062C\u0644\u0627\u062A\s*\u0639\u0644\u0645\u064A\u0629|"
        r"\u0645\u062C\u0644\u0629\s*\u0639\u0644\u0645\u064A\u0629|"
        r"\u062F\u0648\u0631\u064A\u0627\u062A|"
        r"\u0631\u0633\u0627\u0644\u0629|"
        r"\u0631\u0633\u0627\u0626\u0644|"
        r"\u0623\u0637\u0631\u0648\u062D\u0629|"
        r"\u0645\u0624\u062A\u0645\u0631|"
        r"\u0645\u0646\u0634\u0648\u0631\u0627\u062A"
        r")",
        re.IGNORECASE
    )

    # English research topic keywords
    RESEARCH_TOPICS_EN = re.compile(
        r"\b(articles?|papers?|journals?|conference|proceedings|"
        r"publications?|research|standards?|thesis|dissertation)\b",
        re.IGNORECASE
    )

    # Arabic research topic keywords
    RESEARCH_TOPICS_AR = re.compile(
        r"("
        r"\u0645\u0642\u0627\u0644|"
        r"\u0628\u062D\u062B|"
        r"\u0623\u0628\u062D\u0627\u062B|"
        r"\u062F\u0631\u0627\u0633\u0629|"
        r"\u062F\u0631\u0627\u0633\u0627\u062A|"
        r"\u0645\u0631\u0627\u062C\u0639|"
        r"\u0645\u0631\u062C\u0639|"
        r"\u0639\u0644\u0645\u064A|"
        r"\u0623\u0643\u0627\u062F\u064A\u0645\u064A"
        r")",
        re.IGNORECASE
    )

    @classmethod
    def is_database_intent(cls, query: str) -> bool:
        """Check if query has database intent. Works for both Arabic and English."""
        has_db_keywords_en = bool(cls.DB_KEYWORDS_EN.search(query))
        has_research_en = bool(cls.RESEARCH_TOPICS_EN.search(query))
        has_db_keywords_ar = bool(cls.DB_KEYWORDS_AR.search(query))
        has_research_ar = bool(cls.RESEARCH_TOPICS_AR.search(query))
        return (
            has_db_keywords_en or has_research_en
            or has_db_keywords_ar or has_research_ar
        )


# ---------------------------------------------------------------------------
# Bilingual prompt and response templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    Config.LANG_EN: (
        "You are a helpful AUB library assistant. Answer the student's question "
        "using the provided context from the library website. "
        "If conversation history is present, use it to understand follow-up "
        "questions (e.g. if the student asks 'how' after asking about borrowing, "
        "they mean 'how to borrow'). "
        "Be concise and directly answer what was asked. "
        "Use markdown formatting for readability."
    ),
    Config.LANG_AR: (
        "\u0623\u0646\u062A \u0645\u0633\u0627\u0639\u062F \u0645\u0643\u062A\u0628\u0629 "
        "\u0627\u0644\u062C\u0627\u0645\u0639\u0629 \u0627\u0644\u0623\u0645\u0631\u064A\u0643\u064A\u0629 "
        "\u0641\u064A \u0628\u064A\u0631\u0648\u062A. "
        "\u0623\u062C\u0628 \u0639\u0644\u0649 \u0633\u0624\u0627\u0644 \u0627\u0644\u0637\u0627\u0644\u0628 "
        "\u0628\u0627\u0633\u062A\u062E\u062F\u0627\u0645 \u0627\u0644\u0633\u064A\u0627\u0642 \u0627\u0644\u0645\u0642\u062F\u0645 "
        "\u0645\u0646 \u0645\u0648\u0642\u0639 \u0627\u0644\u0645\u0643\u062A\u0628\u0629. "
        "\u0625\u0630\u0627 \u0643\u0627\u0646 \u0647\u0646\u0627\u0643 \u0633\u062C\u0644 \u0645\u062D\u0627\u062F\u062B\u0629\u060C "
        "\u0627\u0633\u062A\u062E\u062F\u0645\u0647 \u0644\u0641\u0647\u0645 \u0623\u0633\u0626\u0644\u0629 "
        "\u0627\u0644\u0645\u062A\u0627\u0628\u0639\u0629. "
        "\u0643\u0646 \u0645\u0648\u062C\u0632\u0627\u064B \u0648\u0623\u062C\u0628 \u0645\u0628\u0627\u0634\u0631\u0629 "
        "\u0639\u0644\u0649 \u0645\u0627 \u062A\u0645 \u0633\u0624\u0627\u0644\u0647. "
        "\u0627\u0633\u062A\u062E\u062F\u0645 \u062A\u0646\u0633\u064A\u0642 Markdown \u0644\u0633\u0647\u0648\u0644\u0629 "
        "\u0627\u0644\u0642\u0631\u0627\u0621\u0629. "
        "\u0623\u062C\u0628 \u0628\u0627\u0644\u0644\u063A\u0629 \u0627\u0644\u0639\u0631\u0628\u064A\u0629."
    ),
}

_RESPONSE_TEMPLATES = {
    Config.LANG_EN: {
        "faq_header": "**FAQ Answer**\n\n",
        "db_header": "**Recommended Databases**\n",
        "confidence_label": "confidence",
        "sources_label": "**Sources:**",
        "unclear": (
            "**I'm not quite sure how to help.** You can:\n\n"
            "- Ask about library services (hours, borrowing, access, etc.)\n"
            "- Request database recommendations for your research topic\n\n"
            "**Example:** *'Which databases should I use for engineering articles?'*"
        ),
    },
    Config.LANG_AR: {
        "faq_header": "**\u0625\u062C\u0627\u0628\u0629 \u0645\u0646 \u0627\u0644\u0623\u0633\u0626\u0644\u0629 \u0627\u0644\u0634\u0627\u0626\u0639\u0629**\n\n",
        "db_header": "**\u0642\u0648\u0627\u0639\u062F \u0628\u064A\u0627\u0646\u0627\u062A \u0645\u0648\u0635\u0649 \u0628\u0647\u0627**\n",
        "confidence_label": "\u062B\u0642\u0629",
        "sources_label": "**\u0627\u0644\u0645\u0635\u0627\u062F\u0631:**",
        "unclear": (
            "**\u0644\u0633\u062A \u0645\u062A\u0623\u0643\u062F\u0627\u064B \u0643\u064A\u0641 \u064A\u0645\u0643\u0646\u0646\u064A \u0645\u0633\u0627\u0639\u062F\u062A\u0643.** \u064A\u0645\u0643\u0646\u0643:\n\n"
            "- \u0627\u0644\u0633\u0624\u0627\u0644 \u0639\u0646 \u062E\u062F\u0645\u0627\u062A \u0627\u0644\u0645\u0643\u062A\u0628\u0629 (\u0633\u0627\u0639\u0627\u062A \u0627\u0644\u0639\u0645\u0644\u060C \u0627\u0644\u0625\u0639\u0627\u0631\u0629\u060C \u0627\u0644\u0648\u0635\u0648\u0644\u060C \u0625\u0644\u062E)\n"
            "- \u0637\u0644\u0628 \u062A\u0648\u0635\u064A\u0627\u062A \u0628\u0642\u0648\u0627\u0639\u062F \u0628\u064A\u0627\u0646\u0627\u062A \u0644\u0645\u0648\u0636\u0648\u0639 \u0628\u062D\u062B\u0643\n\n"
            "**\u0645\u062B\u0627\u0644:** *'\u0645\u0627 \u0647\u064A \u0623\u0641\u0636\u0644 \u0642\u0648\u0627\u0639\u062F \u0627\u0644\u0628\u064A\u0627\u0646\u0627\u062A \u0644\u0645\u0642\u0627\u0644\u0627\u062A \u0627\u0644\u0647\u0646\u062F\u0633\u0629\u061F'*"
        ),
    },
}


class LibraryChatbot:
    """Main chatbot class using PostgreSQL + pgvector for retrieval.

    Supports bilingual (Arabic/English) operation. Language is resolved
    in this priority order:
        1. Explicit `language` parameter passed by the caller (from UI toggle)
        2. Auto-detection from the query text via LanguageDetector
    """

    # Minimum cosine similarity for feedback to be considered relevant
    FEEDBACK_MIN_SIMILARITY = 0.85

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self._cache = ResponseCache(max_size=256, ttl_seconds=3600)

        # Verify tables exist and have data
        self.library_available = False
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM library_pages")
                    if cur.fetchone()[0] > 0:
                        self.library_available = True
        except Exception:
            logger.warning("Library pages table not found or empty.")

    def _query_table(
        self,
        table: str,
        query: str,
        n_results: int,
        metadata_cols: List[str],
    ) -> dict:
        """Query a pgvector table for nearest neighbors.

        Returns a dict with the following structure:
            {
                "ids": [[id1, id2, ...]],
                "distances": [[d1, d2, ...]],
                "metadatas": [[{col: val, ...}, ...]],
            }
        """
        query_embedding = embed_text(query)
        query_vec = np.array(query_embedding)

        cols = ", ".join(metadata_cols)
        sql = (
            f"SELECT id, embedding <=> %s AS distance, {cols} "
            f"FROM {table} "
            f"ORDER BY embedding <=> %s "
            f"LIMIT %s"
        )

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query_vec, query_vec, n_results))
                rows = cur.fetchall()

        ids = []
        distances = []
        metadatas = []
        for row in rows:
            ids.append(row[0])
            distances.append(float(row[1]))
            meta = {col: row[2 + i] for i, col in enumerate(metadata_cols)}
            metadatas.append(meta)

        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metadatas],
        }

    def get_collection_count(self, table: str) -> int:
        """Return the number of rows in a table."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                return cur.fetchone()[0]

    def clear_cache(self) -> None:
        """Invalidate all cached responses (e.g. after reindex)."""
        self._cache.clear()

    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self._cache.stats()

    def _lookup_feedback_correction(self, query: str) -> Optional[str]:
        """Check if a similar query has been corrected by an admin.

        Searches the chat_feedback table for negative feedback entries with
        corrected answers, using pgvector similarity on the query embedding.
        Returns the corrected answer if a close match is found (>= 0.85
        cosine similarity), otherwise None.
        """
        try:
            query_embedding = embed_text(query)
            query_vec = np.array(query_embedding)

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT f.corrected_answer, c.query,
                                  1 - (f.embedding <=> %s) AS similarity
                           FROM chat_feedback f
                           JOIN chat_conversations c ON c.id = f.conversation_id
                           WHERE f.rating = -1
                             AND f.corrected_answer IS NOT NULL
                             AND f.embedding IS NOT NULL
                           ORDER BY f.embedding <=> %s
                           LIMIT 1""",
                        (query_vec, query_vec),
                    )
                    row = cur.fetchone()

            if row and row[2] >= self.FEEDBACK_MIN_SIMILARITY:
                logger.info(
                    f"Feedback correction found (similarity={row[2]:.3f}) "
                    f"for query '{query[:60]}' from original '{row[1][:60]}'"
                )
                return row[0]  # corrected_answer
        except Exception as e:
            logger.warning(f"Feedback lookup failed (non-critical): {e}")

        return None

    @staticmethod
    def _build_history_messages(history: Optional[List[dict]] = None) -> List[dict]:
        """Sanitize and truncate conversation history for LLM context."""
        if not history:
            return []

        valid_roles = {"user", "assistant"}
        cleaned = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role in valid_roles and isinstance(content, str) and content.strip():
                cleaned.append({"role": role, "content": content})

        # Keep only the last MAX_HISTORY_TURNS turns (each turn = 2 messages)
        max_entries = Config.MAX_HISTORY_TURNS * 2
        if len(cleaned) > max_entries:
            cleaned = cleaned[-max_entries:]

        return cleaned

    def _resolve_language(self, query: str, language: Optional[str] = None) -> str:
        """Determine response language. Always auto-detects from query text."""
        return LanguageDetector.detect(query)

    @staticmethod
    def _build_search_query(query: str, history: Optional[List[dict]] = None) -> str:
        """Build a context-enriched query for retrieval.

        Short follow-up messages like "how", "why", "tell me more" have no
        semantic content on their own, so we prepend the last user message
        to give the embedding meaningful context.
        """
        if not history or len(query.split()) > 4:
            return query

        last_user_msg = None
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("role") == "user":
                last_user_msg = entry.get("content", "")
                break

        if last_user_msg and last_user_msg.strip():
            return f"{last_user_msg} {query}"
        return query

    def answer(
        self,
        query: str,
        language: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, dict]:
        """Generate answer and return results with debug info."""
        lang = self._resolve_language(query, language)

        search_query = self._build_search_query(query, history)

        # --- Cache lookup ---
        cache_key = (search_query, lang)
        cached = self._cache.get(cache_key)
        if cached is not None:
            answer, debug = cached
            debug = dict(debug)
            debug["cache_hit"] = True
            return answer, debug

        # --- Cache miss: full retrieval + generation pipeline ---

        cross_offset = (
            Config.CROSS_LINGUAL_THRESHOLD_OFFSET
            if lang == Config.LANG_AR
            else 0.0
        )

        # Query all tables
        faq_results = self._query_table("faq", search_query, 5, ["question", "answer"])
        db_results = self._query_table("databases", search_query, 5, ["name", "description"])

        library_results = None
        if self.library_available:
            library_results = self._query_table("library_pages", search_query, 3, ["url", "title", "content"])

        faq_scores = _pgvector_distance_to_similarity(faq_results["distances"][0])
        db_scores = _pgvector_distance_to_similarity(db_results["distances"][0])

        library_scores = []
        if library_results is not None:
            library_scores = _pgvector_distance_to_similarity(library_results["distances"][0])

        best_faq_score = faq_scores[0] if faq_scores else 0.0
        best_db_score = db_scores[0] if db_scores else 0.0
        best_library_score = library_scores[0] if library_scores else 0.0

        is_db_intent = IntentDetector.is_database_intent(query)

        show_faq = best_faq_score >= (Config.FAQ_MIN_CONFIDENCE - cross_offset)
        show_db = is_db_intent or best_db_score >= (Config.DB_MIN_CONFIDENCE - cross_offset)
        show_library = bool(library_scores) and best_library_score >= (Config.LIBRARY_MIN_CONFIDENCE - cross_offset)

        # Build retrieved_chunks: the actual text that was retrieved from each source
        retrieved_chunks = []
        for i, meta in enumerate(faq_results["metadatas"][0]):
            retrieved_chunks.append({
                "source": "faq",
                "score": faq_scores[i] if i < len(faq_scores) else 0.0,
                "text": f"Q: {meta.get('question', '')}\nA: {meta.get('answer', '')}",
            })
        for i, meta in enumerate(db_results["metadatas"][0]):
            retrieved_chunks.append({
                "source": "database",
                "score": db_scores[i] if i < len(db_scores) else 0.0,
                "text": f"{meta.get('name', '')}. {EmbeddingUtils.clean_html(meta.get('description', ''))}",
            })
        if library_results:
            for i, meta in enumerate(library_results["metadatas"][0]):
                retrieved_chunks.append({
                    "source": "library_page",
                    "score": library_scores[i] if i < len(library_scores) else 0.0,
                    "text": f"{meta.get('title', '')}\n{EmbeddingUtils.clean_library_content(meta.get('content', ''))}",
                })

        debug = {
            "faq_scores": [[id_, float(s)] for id_, s in zip(faq_results["ids"][0], faq_scores)],
            "db_scores": [[id_, float(s)] for id_, s in zip(db_results["ids"][0], db_scores)],
            "library_scores": [[id_, float(s)] for id_, s in zip(library_results["ids"][0], library_scores)] if library_results else [],
            "is_db_intent": is_db_intent,
            "show_faq": show_faq,
            "show_db": show_db,
            "show_library": show_library,
            "library_available": self.library_available,
            "chosen_source": None,
            "detected_language": lang,
            "cache_hit": False,
            "faq_metadatas": faq_results["metadatas"][0],
            "db_metadatas": db_results["metadatas"][0],
            "library_metadatas": library_results["metadatas"][0] if library_results else [],
            "retrieved_chunks": retrieved_chunks,
            "query": query,
            "search_query": search_query,
        }

        # --- Check for admin feedback corrections ---
        # If an admin has corrected a similar past query, use their answer
        # as additional context for the LLM to generate a better response.
        feedback_correction = self._lookup_feedback_correction(search_query)
        if feedback_correction:
            debug["chosen_source"] = "admin_feedback_correction"
            debug["feedback_correction"] = feedback_correction
            # Use LLM to blend the corrected answer with the query context
            formatted = self._format_feedback_answer(
                query=search_query,
                corrected_answer=feedback_correction,
                lang=lang,
                history=history,
            )
            result = (formatted, debug)
            self._cache.put(cache_key, result)
            return result

        if show_db and is_db_intent:
            debug["chosen_source"] = "database (keyword intent)"
            result = (self._format_db_recommendations(db_results, db_scores, lang, k=5), debug)
            self._cache.put(cache_key, result)
            return result

        if show_library:
            debug["chosen_source"] = "library pages (scraped)"
            result = (self._format_library_answer(search_query, library_results, library_scores, lang, k=3, history=history), debug)
            self._cache.put(cache_key, result)
            return result

        if show_faq:
            debug["chosen_source"] = "FAQ"
            result = (self._format_faq_answer(faq_results, lang, search_query, history=history), debug)
            self._cache.put(cache_key, result)
            return result

        if show_db:
            debug["chosen_source"] = "database (semantic)"
            result = (self._format_db_recommendations(db_results, db_scores, lang, k=5), debug)
            self._cache.put(cache_key, result)
            return result

        debug["chosen_source"] = "none (unclear)"
        result = (self._format_unclear(lang), debug)
        self._cache.put(cache_key, result)
        return result

    def _format_faq_answer(self, results: dict, lang: str, query: str = "", history: Optional[List[dict]] = None) -> str:
        """Format FAQ answer using the LLM for a natural, readable response."""
        raw_answer = results["metadatas"][0][0]["answer"]
        question = results["metadatas"][0][0].get("question", query)
        templates = _RESPONSE_TEMPLATES[lang]

        system_prompt = _SYSTEM_PROMPTS[lang]
        history_msgs = self._build_history_messages(history)

        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history_msgs)
            messages.append({
                "role": "user",
                "content": f"Context:\nFAQ Question: {question}\nFAQ Answer: {raw_answer}\n\nQuestion: {query}",
            })

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM formatting of FAQ answer failed: {e}")
            answer = raw_answer
            if lang == Config.LANG_AR:
                answer = self._translate_to_arabic(answer)

        return f"{templates['faq_header']}{answer}"

    def _format_db_recommendations(
        self, results: dict, scores: List[float], lang: str, k: int = 5
    ) -> str:
        """Format database recommendations in the appropriate language."""
        templates = _RESPONSE_TEMPLATES[lang]
        lines = [templates["db_header"]]

        for i in range(min(k, len(results["metadatas"][0]))):
            meta = results["metadatas"][0][i]
            score = scores[i]
            name = meta["name"]
            desc = EmbeddingUtils.clean_html(meta["description"])
            if len(desc) > 200:
                desc = desc[:197] + "..."

            if lang == Config.LANG_AR:
                desc = self._translate_to_arabic(desc)

            conf_label = templates["confidence_label"]
            lines.append(f"**{name}** ({conf_label}: {score:.2f})")
            lines.append(f"{desc}\n")

        return "\n".join(lines)

    def _format_library_answer(
        self,
        query: str,
        results: dict,
        scores: List[float],
        lang: str,
        k: int = 3,
        history: Optional[List[dict]] = None,
    ) -> str:
        """Use the LLM to synthesize a clean answer from retrieved library pages."""
        context_parts = []
        sources = []
        for i in range(min(k, len(results["metadatas"][0]))):
            meta = results["metadatas"][0][i]
            content = EmbeddingUtils.clean_library_content(meta["content"])
            context_parts.append(f"Page: {meta['title']}\n{content}")
            sources.append(f"[{meta['title']}]({meta['url']})")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = _SYSTEM_PROMPTS[lang]
        history_msgs = self._build_history_messages(history)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_msgs)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        })

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )

        answer = resp.choices[0].message.content
        source_links = " | ".join(sources)
        templates = _RESPONSE_TEMPLATES[lang]
        return f"{answer}\n\n{templates['sources_label']} {source_links}"

    def _format_feedback_answer(
        self,
        query: str,
        corrected_answer: str,
        lang: str,
        history: Optional[List[dict]] = None,
    ) -> str:
        """Format an answer using an admin-corrected answer as the primary source.

        The LLM adapts the corrected answer to the current query wording and
        language while staying faithful to the admin's correction.
        """
        system_prompt = _SYSTEM_PROMPTS[lang]
        history_msgs = self._build_history_messages(history)

        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history_msgs)
            messages.append({
                "role": "user",
                "content": (
                    f"Context:\n"
                    f"A library administrator has provided this verified answer "
                    f"for a similar question:\n{corrected_answer}\n\n"
                    f"Question: {query}\n\n"
                    f"Use the verified answer above as the primary source. "
                    f"Adapt it to the question if needed, but do not add information "
                    f"that is not in the verified answer."
                ),
            })

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Feedback answer formatting failed: {e}")
            # Fallback: return the corrected answer directly
            if lang == Config.LANG_AR:
                return self._translate_to_arabic(corrected_answer)
            return corrected_answer

    def _format_unclear(self, lang: str) -> str:
        """Format unclear-intent message in the appropriate language."""
        return _RESPONSE_TEMPLATES[lang]["unclear"]

    def _translate_to_arabic(self, text: str) -> str:
        """Translate an English text snippet to Arabic using the LLM."""
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "\u062A\u0631\u062C\u0645 \u0627\u0644\u0646\u0635 "
                            "\u0627\u0644\u062A\u0627\u0644\u064A \u0625\u0644\u0649 "
                            "\u0627\u0644\u0644\u063A\u0629 \u0627\u0644\u0639\u0631\u0628\u064A\u0629. "
                            "\u062D\u0627\u0641\u0638 \u0639\u0644\u0649 "
                            "\u0623\u0633\u0645\u0627\u0621 \u0642\u0648\u0627\u0639\u062F "
                            "\u0627\u0644\u0628\u064A\u0627\u0646\u0627\u062A "
                            "\u0648\u0627\u0644\u0645\u0635\u0637\u0644\u062D\u0627\u062A "
                            "\u0627\u0644\u062A\u0642\u0646\u064A\u0629 "
                            "\u0628\u0627\u0644\u0625\u0646\u062C\u0644\u064A\u0632\u064A\u0629. "
                            "\u0623\u0639\u062F \u0627\u0644\u062A\u0631\u062C\u0645\u0629 "
                            "\u0641\u0642\u0637 \u0628\u062F\u0648\u0646 \u0623\u064A "
                            "\u0634\u0631\u062D \u0625\u0636\u0627\u0641\u064A."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Translation to Arabic failed: {e}")
            return text
