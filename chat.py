"""
chat.py
Library Chatbot with FAQ answers and database recommendations.
Uses ChromaDB for vector storage and similarity search.
Supports Arabic and English (bilingual).
"""

import os
import re
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI()

# Configuration
class Config:
    CHROMA_DIR = "./chroma_db"
    EMBEDDING_MODEL = "text-embedding-3-small"

    FAQ_COLLECTION = "faq"
    DB_COLLECTION = "databases"
    LIBRARY_COLLECTION = "library_pages"

    # Thresholds
    FAQ_HIGH_CONFIDENCE = 0.70
    FAQ_MIN_CONFIDENCE = 0.60
    DB_MIN_CONFIDENCE = 0.45
    LIBRARY_MIN_CONFIDENCE = 0.35
    BOTH_DELTA = 0.06

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


def _chroma_distance_to_similarity(distances: List[float]) -> List[float]:
    """Convert ChromaDB cosine distances to similarity scores."""
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
        "using ONLY the provided context from the library website. "
        "Be concise and directly answer what was asked. "
        "If the context doesn't contain the answer, say so."
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
        "\u0623\u062C\u0628 \u0628\u0627\u0644\u0644\u063A\u0629 \u0627\u0644\u0639\u0631\u0628\u064A\u0629."
    ),
}


class LibraryChatbot:
    """Main chatbot class using ChromaDB for retrieval. Bilingual support."""

    def __init__(self):
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
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

        # Load library pages collection if available
        try:
            self.library_collection = chroma_client.get_collection(
                name=Config.LIBRARY_COLLECTION,
                embedding_function=embedding_fn,
            )
            if self.library_collection.count() == 0:
                self.library_collection = None
            else:
                logger.info(f"Loaded {self.faq_collection.count()} FAQs, "
                           f"{self.db_collection.count()} databases, and "
                           f"{self.library_collection.count()} library pages")
        except (ValueError, Exception):
            logger.warning("Library pages collection not found. Run scrape_aub_library.py to create it.")
            self.library_collection = None
            logger.info(f"Loaded {self.faq_collection.count()} FAQs and "
                       f"{self.db_collection.count()} databases")

    def answer(self, query: str) -> str:
        """Generate answer for user query. Auto-detects language."""
        lang = LanguageDetector.detect(query)

        # Query all collections (ChromaDB handles embedding)
        faq_results = self.faq_collection.query(query_texts=[query], n_results=5)
        db_results = self.db_collection.query(query_texts=[query], n_results=5)

        library_results = None
        if self.library_collection is not None:
            library_results = self.library_collection.query(query_texts=[query], n_results=3)

        # Convert distances to similarity scores
        faq_scores = _chroma_distance_to_similarity(faq_results["distances"][0])
        db_scores = _chroma_distance_to_similarity(db_results["distances"][0])

        library_scores = []
        if library_results is not None:
            library_scores = _chroma_distance_to_similarity(library_results["distances"][0])

        best_faq_score = faq_scores[0] if faq_scores else 0.0
        best_db_score = db_scores[0] if db_scores else 0.0

        # Detect intent
        intent = IntentDetector.detect(query)

        # Determine response strategy
        show_faq = best_faq_score >= Config.FAQ_MIN_CONFIDENCE
        show_db = (
            intent == IntentType.DATABASE or
            best_db_score >= Config.DB_MIN_CONFIDENCE
        )
        show_library = False
        if library_scores:
            show_library = library_scores[0] >= Config.LIBRARY_MIN_CONFIDENCE

        # 1. Database intent -> always use database recommendations
        if show_db and intent == IntentType.DATABASE:
            return self._format_db_recommendations(db_results, db_scores, lang, k=5)

        # 2. Scraped library pages -> primary source for non-DB questions
        if show_library:
            return self._format_library_answer(query, library_results, library_scores, lang, k=3)

        # 3. FAQ -> backup if scraped data didn't match
        if show_faq:
            return self._format_faq_answer(faq_results, lang)

        # 4. Database recommendations by semantic score (no keyword intent)
        if show_db:
            return self._format_db_recommendations(db_results, db_scores, lang, k=5)

        # 5. Nothing matched
        return self._format_unclear(lang)

    def _format_faq_answer(self, results: dict, lang: str) -> str:
        """Format FAQ answer."""
        answer = results["metadatas"][0][0]["answer"]
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

        resp = client.chat.completions.create(
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
        bot = LibraryChatbot()
    except Exception as e:
        print(f"Error loading chatbot: {e}")
        return

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

            answer = bot.answer(query)
            print(f"\nBot:\n{answer}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\nSorry, I encountered an error. Please try rephrasing your question.")

if __name__ == "__main__":
    main()
