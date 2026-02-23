"""
chatbot.py
Core chatbot logic for the FastAPI backend.
Supports Arabic and English (bilingual).
"""

import os
import re
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    EMBEDDING_MODEL = "text-embedding-3-small"

    FAQ_COLLECTION = "faq"
    DB_COLLECTION = "databases"
    LIBRARY_COLLECTION = "library_pages"

    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    LIBRARY_MIN_CONFIDENCE = 0.35
    BOTH_DELTA = 0.06

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


def _chroma_distance_to_similarity(distances: List[float]) -> List[float]:
    """Convert ChromaDB cosine distances to similarity scores."""
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
    # Covers: database, data, search, find, source, best database,
    # recommend, articles, research papers, where to find, scientific journals
    DB_KEYWORDS_AR = re.compile(
        r"("
        r"\u0642\u0627\u0639\u062F\u0629\s*\u0628\u064A\u0627\u0646\u0627\u062A|"  # قاعدة بيانات
        r"\u0642\u0648\u0627\u0639\u062F\s*\u0628\u064A\u0627\u0646\u0627\u062A|"  # قواعد بيانات
        r"\u0642\u0627\u0639\u062F\u0629\s*\u0645\u0639\u0644\u0648\u0645\u0627\u062A|"  # قاعدة معلومات
        r"\u0623\u064A\u0646\s*\u0623\u0628\u062D\u062B|"  # أين أبحث
        r"\u0623\u064A\u0646\s*\u0623\u062C\u062F|"  # أين أجد
        r"\u0645\u0635\u062F\u0631|"  # مصدر
        r"\u0645\u0635\u0627\u062F\u0631|"  # مصادر
        r"\u0623\u0641\u0636\u0644\s*\u0642\u0627\u0639\u062F\u0629|"  # أفضل قاعدة
        r"\u0623\u0648\u0635\u064A|"  # أوصي (recommend)
        r"\u062A\u0648\u0635\u064A\u0629|"  # توصية (recommendation)
        r"\u0623\u0646\u0635\u062D|"  # أنصح (advise/recommend)
        r"\u0627\u0628\u062D\u062B|"  # ابحث (search)
        r"\u0628\u062D\u062B|"  # بحث (search/research)
        r"\u0645\u0642\u0627\u0644\u0627\u062A|"  # مقالات (articles)
        r"\u0645\u0642\u0627\u0644\u0629|"  # مقالة (article)
        r"\u0623\u0648\u0631\u0627\u0642\s*\u0628\u062D\u062B\u064A\u0629|"  # أوراق بحثية (research papers)
        r"\u0648\u0631\u0642\u0629\s*\u0628\u062D\u062B\u064A\u0629|"  # ورقة بحثية (research paper)
        r"\u0645\u062C\u0644\u0627\u062A\s*\u0639\u0644\u0645\u064A\u0629|"  # مجلات علمية (scientific journals)
        r"\u0645\u062C\u0644\u0629\s*\u0639\u0644\u0645\u064A\u0629|"  # مجلة علمية (scientific journal)
        r"\u062F\u0648\u0631\u064A\u0627\u062A|"  # دوريات (periodicals/journals)
        r"\u0631\u0633\u0627\u0644\u0629|"  # رسالة (thesis)
        r"\u0631\u0633\u0627\u0626\u0644|"  # رسائل (theses)
        r"\u0623\u0637\u0631\u0648\u062D\u0629|"  # أطروحة (dissertation)
        r"\u0645\u0624\u062A\u0645\u0631|"  # مؤتمر (conference)
        r"\u0645\u0646\u0634\u0648\u0631\u0627\u062A"  # منشورات (publications)
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
        r"\u0645\u0642\u0627\u0644|"  # مقال (article)
        r"\u0628\u062D\u062B|"  # بحث (research)
        r"\u0623\u0628\u062D\u0627\u062B|"  # أبحاث (researches)
        r"\u062F\u0631\u0627\u0633\u0629|"  # دراسة (study)
        r"\u062F\u0631\u0627\u0633\u0627\u062A|"  # دراسات (studies)
        r"\u0645\u0631\u0627\u062C\u0639|"  # مراجع (references)
        r"\u0645\u0631\u062C\u0639|"  # مرجع (reference)
        r"\u0639\u0644\u0645\u064A|"  # علمي (scientific)
        r"\u0623\u0643\u0627\u062F\u064A\u0645\u064A"  # أكاديمي (academic)
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
        "using ONLY the provided context from the library website. "
        "Be concise and directly answer what was asked. "
        "If the context doesn't contain the answer, say so. "
        "Use markdown formatting for readability."
    ),
    Config.LANG_AR: (
        "\u0623\u0646\u062A \u0645\u0633\u0627\u0639\u062F \u0645\u0643\u062A\u0628\u0629 "
        "\u0627\u0644\u062C\u0627\u0645\u0639\u0629 \u0627\u0644\u0623\u0645\u0631\u064A\u0643\u064A\u0629 "
        "\u0641\u064A \u0628\u064A\u0631\u0648\u062A. "
        "\u0623\u062C\u0628 \u0639\u0644\u0649 \u0633\u0624\u0627\u0644 \u0627\u0644\u0637\u0627\u0644\u0628 "
        "\u0628\u0627\u0633\u062A\u062E\u062F\u0627\u0645 \u0627\u0644\u0633\u064A\u0627\u0642 \u0627\u0644\u0645\u0642\u062F\u0645 "
        "\u0641\u0642\u0637 \u0645\u0646 \u0645\u0648\u0642\u0639 \u0627\u0644\u0645\u0643\u062A\u0628\u0629. "
        "\u0643\u0646 \u0645\u0648\u062C\u0632\u0627\u064B \u0648\u0623\u062C\u0628 \u0645\u0628\u0627\u0634\u0631\u0629 "
        "\u0639\u0644\u0649 \u0645\u0627 \u062A\u0645 \u0633\u0624\u0627\u0644\u0647. "
        "\u0625\u0630\u0627 \u0644\u0645 \u064A\u062D\u062A\u0648\u0650 \u0627\u0644\u0633\u064A\u0627\u0642 "
        "\u0639\u0644\u0649 \u0627\u0644\u0625\u062C\u0627\u0628\u0629\u060C \u0642\u0644 \u0630\u0644\u0643. "
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
    """Main chatbot class using ChromaDB for retrieval.

    Supports bilingual (Arabic/English) operation. Language is resolved
    in this priority order:
        1. Explicit `language` parameter passed by the caller (from UI toggle)
        2. Auto-detection from the query text via LanguageDetector
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

        embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=Config.EMBEDDING_MODEL,
        )
        chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DIR)

        self.faq_collection = chroma_client.get_collection(
            name=Config.FAQ_COLLECTION,
            embedding_function=embedding_fn,
        )
        self.db_collection = chroma_client.get_collection(
            name=Config.DB_COLLECTION,
            embedding_function=embedding_fn,
        )

        try:
            self.library_collection = chroma_client.get_collection(
                name=Config.LIBRARY_COLLECTION,
                embedding_function=embedding_fn,
            )
            if self.library_collection.count() == 0:
                self.library_collection = None
        except (ValueError, Exception):
            logger.warning("Library pages collection not found.")
            self.library_collection = None

    def _resolve_language(self, query: str, language: Optional[str] = None) -> str:
        """Determine response language.

        Priority:
            1. Explicit language param from the frontend toggle ('en' or 'ar')
            2. Auto-detect from the query text
        """
        if language and language in (Config.LANG_EN, Config.LANG_AR):
            return language
        return LanguageDetector.detect(query)

    def answer(
        self,
        query: str,
        language: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """Generate answer and return results with debug info.

        Args:
            query: The user's question (Arabic or English).
            language: Optional explicit language override ('en' or 'ar').
                      If None, language is auto-detected from the query.

        Returns:
            Tuple of (answer_string, debug_dict).
        """
        lang = self._resolve_language(query, language)

        # Query all collections -- OpenAI embeddings handle Arabic natively
        faq_results = self.faq_collection.query(query_texts=[query], n_results=5)
        db_results = self.db_collection.query(query_texts=[query], n_results=5)

        library_results = None
        if self.library_collection is not None:
            library_results = self.library_collection.query(query_texts=[query], n_results=3)

        faq_scores = _chroma_distance_to_similarity(faq_results["distances"][0])
        db_scores = _chroma_distance_to_similarity(db_results["distances"][0])

        library_scores = []
        if library_results is not None:
            library_scores = _chroma_distance_to_similarity(library_results["distances"][0])

        best_faq_score = faq_scores[0] if faq_scores else 0.0
        best_db_score = db_scores[0] if db_scores else 0.0
        best_library_score = library_scores[0] if library_scores else 0.0

        is_db_intent = IntentDetector.is_database_intent(query)

        show_faq = best_faq_score >= Config.FAQ_MIN_CONFIDENCE
        show_db = is_db_intent or best_db_score >= Config.DB_MIN_CONFIDENCE
        show_library = bool(library_scores) and best_library_score >= Config.LIBRARY_MIN_CONFIDENCE

        debug = {
            "faq_scores": [[id_, float(s)] for id_, s in zip(faq_results["ids"][0], faq_scores)],
            "db_scores": [[id_, float(s)] for id_, s in zip(db_results["ids"][0], db_scores)],
            "library_scores": [[id_, float(s)] for id_, s in zip(library_results["ids"][0], library_scores)] if library_results else [],
            "is_db_intent": is_db_intent,
            "show_faq": show_faq,
            "show_db": show_db,
            "show_library": show_library,
            "library_available": self.library_collection is not None,
            "chosen_source": None,
            "detected_language": lang,
            "faq_metadatas": faq_results["metadatas"][0],
            "db_metadatas": db_results["metadatas"][0],
            "library_metadatas": library_results["metadatas"][0] if library_results else [],
        }

        if show_db and is_db_intent:
            debug["chosen_source"] = "database (keyword intent)"
            return self._format_db_recommendations(db_results, db_scores, lang, k=5), debug

        if show_library:
            debug["chosen_source"] = "library pages (scraped)"
            return self._format_library_answer(query, library_results, library_scores, lang, k=3), debug

        if show_faq:
            debug["chosen_source"] = "FAQ"
            return self._format_faq_answer(faq_results, lang), debug

        if show_db:
            debug["chosen_source"] = "database (semantic)"
            return self._format_db_recommendations(db_results, db_scores, lang, k=5), debug

        debug["chosen_source"] = "none (unclear)"
        return self._format_unclear(lang), debug

    def _format_faq_answer(self, results: dict, lang: str) -> str:
        """Format FAQ answer in the appropriate language.

        The FAQ data is in English. When the language is Arabic, we use the
        LLM to translate the answer so the student receives a natural response.
        """
        answer = results["metadatas"][0][0]["answer"]
        templates = _RESPONSE_TEMPLATES[lang]

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

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )

        answer = resp.choices[0].message.content
        source_links = " | ".join(sources)
        templates = _RESPONSE_TEMPLATES[lang]
        return f"{answer}\n\n{templates['sources_label']} {source_links}"

    def _format_unclear(self, lang: str) -> str:
        """Format unclear-intent message in the appropriate language."""
        return _RESPONSE_TEMPLATES[lang]["unclear"]

    def _translate_to_arabic(self, text: str) -> str:
        """Translate an English text snippet to Arabic using the LLM.

        Used for FAQ answers and database descriptions that are stored in
        English but need to be presented in Arabic. The translation is kept
        concise and preserves any proper nouns or technical terms.
        """
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
            # Fall back to the original English text rather than failing
            return text
