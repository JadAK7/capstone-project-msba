"""
chat.py
Library Chatbot.
Uses PostgreSQL + pgvector for vector storage and similarity search.
Supports Arabic and English (bilingual).
"""

import os
import sys
import re
import numpy as np
from openai import OpenAI
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load environment variables BEFORE importing backend modules so that
# OPENAI_EMBEDDING_MODEL (and other env vars) are visible at import time.
load_dotenv()

from backend.cache import ResponseCache
from backend.database import init_db, get_connection
from backend.embeddings import embed_text, EMBEDDING_MODEL as _EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI()

# Configuration
class Config:
    EMBEDDING_MODEL = _EMBEDDING_MODEL  # resolved from OPENAI_EMBEDDING_MODEL env var
    CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

    FAQ_COLLECTION = "faq"
    DB_COLLECTION = "databases"
    LIBRARY_COLLECTION = "library_pages"

    # Thresholds
    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    LIBRARY_MIN_CONFIDENCE = 0.35
    BOTH_DELTA = 0.06

    # Cross-lingual penalty: Arabic queries against English-indexed data
    # typically score 10-20% lower in cosine similarity.
    CROSS_LINGUAL_THRESHOLD_OFFSET = 0.10

    # Conversation history
    MAX_HISTORY_TURNS = 5  # Number of recent turns (user+assistant pairs) to include in LLM context

    # Supported languages
    LANG_EN = "en"
    LANG_AR = "ar"
    DEFAULT_LANG = LANG_EN


class IntentType(Enum):
    FAQ = "faq"
    DATABASE = "database"
    BOTH = "both"
    UNCLEAR = "unclear"

@dataclass
class SearchResult:
    idx: int
    score: float
    text: str


class LanguageDetector:
    """Detects whether user input is Arabic or English."""

    _ARABIC_PATTERN = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
    )

    @classmethod
    def detect(cls, text: str) -> str:
        if not text or not text.strip():
            return Config.DEFAULT_LANG
        arabic_chars = len(cls._ARABIC_PATTERN.findall(text))
        if arabic_chars > 0:
            return Config.LANG_AR
        return Config.LANG_EN


class EmbeddingUtils:
    """Utilities for text cleaning."""

    @staticmethod
    def clean_html(s: str) -> str:
        """Remove HTML tags and clean text."""
        s = re.sub(r"<br\s*/?>", "\n", str(s), flags=re.IGNORECASE)
        s = re.sub(r"<[^>]+>", "", s)
        s = re.sub(r"\s+\n", "\n", s)
        return s.strip()

    @staticmethod
    def clean_library_content(s: str) -> str:
        """Strip AUB navigation boilerplate from scraped page content."""
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
    """Convert pgvector cosine distances to similarity scores."""
    return [1.0 - d for d in distances]


class IntentDetector:
    """Detects user intent from query. Supports English and Arabic."""

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
    def detect(cls, query: str) -> IntentType:
        """Detect primary intent from query text (English and Arabic)."""
        has_db_en = bool(cls.DB_KEYWORDS_EN.search(query))
        has_research_en = bool(cls.RESEARCH_TOPICS_EN.search(query))
        has_db_ar = bool(cls.DB_KEYWORDS_AR.search(query))
        has_research_ar = bool(cls.RESEARCH_TOPICS_AR.search(query))

        if has_db_en or has_research_en or has_db_ar or has_research_ar:
            return IntentType.DATABASE

        return IntentType.FAQ


# Bilingual templates
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


class LibraryChatbot:
    """Main chatbot class using PostgreSQL + pgvector for retrieval. Bilingual support."""

    def __init__(self):
        self._cache = ResponseCache(max_size=256, ttl_seconds=3600)

        # Check library pages availability
        self.library_available = False
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM library_pages")
                    count = cur.fetchone()[0]
                    if count > 0:
                        self.library_available = True
        except Exception:
            logger.warning("Library pages table not found or empty.")

        # Log loaded counts
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM faq")
                    faq_count = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM databases")
                    db_count = cur.fetchone()[0]
                    if self.library_available:
                        cur.execute("SELECT COUNT(*) FROM library_pages")
                        lib_count = cur.fetchone()[0]
                        logger.info(f"Loaded {faq_count} FAQs, {db_count} databases, and {lib_count} library pages")
                    else:
                        logger.info(f"Loaded {faq_count} FAQs and {db_count} databases")
        except Exception:
            pass

    def _query_table(self, table: str, query: str, n_results: int, metadata_cols: List[str]) -> dict:
        """Query a pgvector table for nearest neighbors."""
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

    @staticmethod
    def _build_search_query(query: str, history: Optional[List[dict]] = None) -> str:
        """Build a context-enriched query for retrieval."""
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

    def answer(self, query: str, history: Optional[List[dict]] = None) -> str:
        """Generate answer for user query. Auto-detects language."""
        lang = LanguageDetector.detect(query)

        search_query = self._build_search_query(query, history)

        # --- Cache lookup ---
        cache_key = (search_query, lang)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

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

        # Convert distances to similarity scores
        faq_scores = _pgvector_distance_to_similarity(faq_results["distances"][0])
        db_scores = _pgvector_distance_to_similarity(db_results["distances"][0])

        library_scores = []
        if library_results is not None:
            library_scores = _pgvector_distance_to_similarity(library_results["distances"][0])

        best_faq_score = faq_scores[0] if faq_scores else 0.0
        best_db_score = db_scores[0] if db_scores else 0.0

        # Detect intent
        intent = IntentDetector.detect(query)

        # Determine response strategy (with cross-lingual offset for Arabic)
        show_faq = best_faq_score >= (Config.FAQ_MIN_CONFIDENCE - cross_offset)
        show_db = (
            intent == IntentType.DATABASE or
            best_db_score >= (Config.DB_MIN_CONFIDENCE - cross_offset)
        )
        show_library = False
        if library_scores:
            show_library = library_scores[0] >= (Config.LIBRARY_MIN_CONFIDENCE - cross_offset)

        # 1. Database intent -> always use database recommendations
        if show_db and intent == IntentType.DATABASE:
            result = self._format_db_recommendations(db_results, db_scores, lang, k=5)
            self._cache.put(cache_key, result)
            return result

        # 2. Scraped library pages -> primary source for non-DB questions
        if show_library:
            result = self._format_library_answer(search_query, library_results, library_scores, lang, k=3, history=history)
            self._cache.put(cache_key, result)
            return result

        # 3. FAQ -> backup if scraped data didn't match
        if show_faq:
            result = self._format_faq_answer(faq_results, lang, search_query, history=history)
            self._cache.put(cache_key, result)
            return result

        # 4. Database recommendations by semantic score (no keyword intent)
        if show_db:
            result = self._format_db_recommendations(db_results, db_scores, lang, k=5)
            self._cache.put(cache_key, result)
            return result

        # 5. Nothing matched
        result = self._format_unclear(lang)
        self._cache.put(cache_key, result)
        return result

    def _format_faq_answer(self, results: dict, lang: str, query: str = "", history: Optional[List[dict]] = None) -> str:
        """Format FAQ answer using the LLM for a natural, readable response."""
        raw_answer = results["metadatas"][0][0]["answer"]
        question = results["metadatas"][0][0].get("question", query)

        system_prompt = _SYSTEM_PROMPTS[lang]
        history_msgs = self._build_history_messages(history)

        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history_msgs)
            messages.append({
                "role": "user",
                "content": f"Context:\nFAQ Question: {question}\nFAQ Answer: {raw_answer}\n\nQuestion: {query}",
            })

            resp = client.chat.completions.create(
                model=Config.CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM formatting of FAQ answer failed: {e}")
            answer = raw_answer
            if lang == Config.LANG_AR:
                answer = self._translate_to_arabic(answer)
            return answer

    def _format_db_recommendations(
        self, results: dict, scores: List[float], lang: str, k: int = 5
    ) -> str:
        """Format database recommendations."""
        if lang == Config.LANG_AR:
            lines = ["\u0642\u0648\u0627\u0639\u062F \u0628\u064A\u0627\u0646\u0627\u062A \u0645\u0648\u0635\u0649 \u0628\u0647\u0627:\n"]
        else:
            lines = ["Recommended databases:\n"]

        conf_label = "\u062B\u0642\u0629" if lang == Config.LANG_AR else "confidence"

        for i in range(min(k, len(results["metadatas"][0]))):
            meta = results["metadatas"][0][i]
            score = scores[i]
            name = meta["name"]
            desc = EmbeddingUtils.clean_html(meta["description"])

            if len(desc) > 200:
                desc = desc[:197] + "..."

            if lang == Config.LANG_AR:
                desc = self._translate_to_arabic(desc)

            lines.append(f"  - {name} ({conf_label}: {score:.2f})")
            lines.append(f"    {desc}\n")

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
            sources.append(f"{meta['title']} -- {meta['url']}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = _SYSTEM_PROMPTS[lang]
        history_msgs = self._build_history_messages(history)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_msgs)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        })

        resp = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )

        answer = resp.choices[0].message.content
        if lang == Config.LANG_AR:
            source_header = "\u0627\u0644\u0645\u0635\u0627\u062F\u0631:"
        else:
            source_header = "Sources:"
        source_list = "\n".join(f"  - {s}" for s in sources)
        return f"{answer}\n\n{source_header}\n{source_list}"

    def _format_unclear(self, lang: str) -> str:
        """Format unclear intent message."""
        if lang == Config.LANG_AR:
            return (
                "\u0644\u0633\u062A \u0645\u062A\u0623\u0643\u062F\u0627\u064B "
                "\u0643\u064A\u0641 \u064A\u0645\u0643\u0646\u0646\u064A "
                "\u0645\u0633\u0627\u0639\u062F\u062A\u0643. \u064A\u0645\u0643\u0646\u0643:\n"
                "  - \u0627\u0644\u0633\u0624\u0627\u0644 \u0639\u0646 "
                "\u062E\u062F\u0645\u0627\u062A \u0627\u0644\u0645\u0643\u062A\u0628\u0629 "
                "(\u0633\u0627\u0639\u0627\u062A \u0627\u0644\u0639\u0645\u0644\u060C "
                "\u0627\u0644\u0625\u0639\u0627\u0631\u0629\u060C "
                "\u0627\u0644\u0648\u0635\u0648\u0644\u060C \u0625\u0644\u062E)\n"
                "  - \u0637\u0644\u0628 \u062A\u0648\u0635\u064A\u0627\u062A "
                "\u0628\u0642\u0648\u0627\u0639\u062F \u0628\u064A\u0627\u0646\u0627\u062A "
                "\u0644\u0645\u0648\u0636\u0648\u0639 \u0628\u062D\u062B\u0643\n\n"
                "\u0645\u062B\u0627\u0644: '\u0645\u0627 \u0647\u064A "
                "\u0623\u0641\u0636\u0644 \u0642\u0648\u0627\u0639\u062F "
                "\u0627\u0644\u0628\u064A\u0627\u0646\u0627\u062A "
                "\u0644\u0645\u0642\u0627\u0644\u0627\u062A "
                "\u0627\u0644\u0647\u0646\u062F\u0633\u0629\u061F'"
            )
        return (
            "I'm not quite sure how to help. You can:\n"
            "  - Ask about library services (hours, borrowing, access, etc.)\n"
            "  - Request database recommendations for your research topic\n"
            "\nExample: 'Which databases should I use for engineering articles?'"
        )

    def _translate_to_arabic(self, text: str) -> str:
        """Translate English text to Arabic using the LLM."""
        try:
            resp = client.chat.completions.create(
                model=Config.CHAT_MODEL,
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


def main():
    """Main chat loop."""
    print("=" * 60)
    print("Library Chatbot / \u0645\u0633\u0627\u0639\u062F \u0627\u0644\u0645\u0643\u062A\u0628\u0629")
    print("=" * 60)
    print("Ask me about library services or get database recommendations!")
    print("\u0627\u0633\u0623\u0644\u0646\u064A \u0639\u0646 \u062E\u062F\u0645\u0627\u062A "
          "\u0627\u0644\u0645\u0643\u062A\u0628\u0629 \u0623\u0648 "
          "\u0627\u062D\u0635\u0644 \u0639\u0644\u0649 \u062A\u0648\u0635\u064A\u0627\u062A "
          "\u0628\u0642\u0648\u0627\u0639\u062F \u0627\u0644\u0628\u064A\u0627\u0646\u0627\u062A!")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    try:
        init_db()
        bot = LibraryChatbot()
    except Exception as e:
        print(f"Error loading chatbot: {e}")
        return

    history = []  # Conversation history for follow-up context

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "bye",
                                  "\u062E\u0631\u0648\u062C",
                                  "\u0645\u0639 \u0627\u0644\u0633\u0644\u0627\u0645\u0629"]:
                print("\nThanks for using the Library Chatbot. Goodbye!")
                print("\u0634\u0643\u0631\u0627\u064B \u0644\u0627\u0633\u062A\u062E\u062F\u0627\u0645\u0643 "
                      "\u0645\u0633\u0627\u0639\u062F \u0627\u0644\u0645\u0643\u062A\u0628\u0629. "
                      "\u0645\u0639 \u0627\u0644\u0633\u0644\u0627\u0645\u0629!")
                break

            answer = bot.answer(query, history=history)
            print(f"\nBot:\n{answer}")

            # Update conversation history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
            # Keep only the last MAX_HISTORY_TURNS turns
            max_entries = Config.MAX_HISTORY_TURNS * 2
            if len(history) > max_entries:
                history = history[-max_entries:]

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\nSorry, I encountered an error. Please try rephrasing your question.")

if __name__ == "__main__":
    main()
