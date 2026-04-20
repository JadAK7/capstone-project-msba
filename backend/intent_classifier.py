"""
intent_classifier.py
Unified intent classification for user queries.

Merges intent detection logic previously split between retriever.py (table/page_type
routing) and chatbot.py (IntentDetector for database intent). All keyword patterns
— English and Arabic — live here in a single canonical registry.
"""

import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword patterns — English
# ---------------------------------------------------------------------------

_HOURS_PATTERN = re.compile(
    r"\b(hours?|open(ing)?|clos(e|ing)|schedule|timing|when\s+(is|does|are)|"
    r"what\s+time)\b",
    re.IGNORECASE,
)

_FAQ_PATTERN = re.compile(
    r"\b(how\s+(do|can|to)|what\s+is|where\s+(is|can|do)|can\s+i|"
    r"is\s+(it|there)|do\s+(you|i|they)|policy|policies|rules?|"
    r"allow(ed)?|permit(ted)?|borrow|return|renew|fine|overdue|"
    r"print|scan|copy|wifi|internet|card|membership|access)\b",
    re.IGNORECASE,
)

# Merged database pattern: combines retriever.py's _DB_PATTERN with
# chatbot.py's IntentDetector.DB_KEYWORDS_EN (adds broader coverage like
# "which database", "where to search", "best database", "recommend", etc.)
_DB_PATTERN_EN = re.compile(
    r"\b(database|db|which\s+database|where\s+to\s+search|where\s+can\s+i\s+find|"
    r"source\s+for|best\s+database|recommend|"
    r"ieee|scopus|pubmed|jstor|proquest|web\s+of\s+science|ebsco|"
    r"e-?resource|electronic\s+resource|online\s+database|"
    r"journal|article|paper|research\s+(source|database|tool)|"
    r"find\s+articles?|search\s+for\s+papers?|"
    r"publications?|conference|proceedings|standards?|thesis|dissertation)\b",
    re.IGNORECASE,
)

# Arabic database keywords (from chatbot.py IntentDetector.DB_KEYWORDS_AR)
_DB_PATTERN_AR = re.compile(
    r"("
    r"\u0642\u0627\u0639\u062F\u0629\s*\u0628\u064A\u0627\u0646\u0627\u062A|"  # قاعدة بيانات
    r"\u0642\u0648\u0627\u0639\u062F\s*\u0628\u064A\u0627\u0646\u0627\u062A|"  # قواعد بيانات
    r"\u0642\u0627\u0639\u062F\u0629\s*\u0645\u0639\u0644\u0648\u0645\u0627\u062A|"  # قاعدة معلومات
    r"\u0623\u064A\u0646\s*\u0623\u0628\u062D\u062B|"  # أين أبحث
    r"\u0623\u064A\u0646\s*\u0623\u062C\u062F|"  # أين أجد
    r"\u0645\u0635\u062F\u0631|"  # مصدر
    r"\u0645\u0635\u0627\u062F\u0631|"  # مصادر
    r"\u0623\u0641\u0636\u0644\s*\u0642\u0627\u0639\u062F\u0629|"  # أفضل قاعدة
    r"\u0623\u0648\u0635\u064A|"  # أوصي
    r"\u062A\u0648\u0635\u064A\u0629|"  # توصية
    r"\u0623\u0646\u0635\u062D|"  # أنصح
    r"\u0627\u0628\u062D\u062B|"  # ابحث
    r"\u0628\u062D\u062B|"  # بحث
    r"\u0645\u0642\u0627\u0644\u0627\u062A|"  # مقالات
    r"\u0645\u0642\u0627\u0644\u0629|"  # مقالة
    r"\u0623\u0648\u0631\u0627\u0642\s*\u0628\u062D\u062B\u064A\u0629|"  # أوراق بحثية
    r"\u0648\u0631\u0642\u0629\s*\u0628\u062D\u062B\u064A\u0629|"  # ورقة بحثية
    r"\u0645\u062C\u0644\u0627\u062A\s*\u0639\u0644\u0645\u064A\u0629|"  # مجلات علمية
    r"\u0645\u062C\u0644\u0629\s*\u0639\u0644\u0645\u064A\u0629|"  # مجلة علمية
    r"\u062F\u0648\u0631\u064A\u0627\u062A|"  # دوريات
    r"\u0631\u0633\u0627\u0644\u0629|"  # رسالة
    r"\u0631\u0633\u0627\u0626\u0644|"  # رسائل
    r"\u0623\u0637\u0631\u0648\u062D\u0629|"  # أطروحة
    r"\u0645\u0624\u062A\u0645\u0631|"  # مؤتمر
    r"\u0645\u0646\u0634\u0648\u0631\u0627\u062A|"  # منشورات
    r"\u0645\u0642\u0627\u0644|"  # مقال
    r"\u0623\u0628\u062D\u0627\u062B|"  # أبحاث
    r"\u062F\u0631\u0627\u0633\u0629|"  # دراسة
    r"\u062F\u0631\u0627\u0633\u0627\u062A|"  # دراسات
    r"\u0645\u0631\u0627\u062C\u0639|"  # مراجع
    r"\u0645\u0631\u062C\u0639|"  # مرجع
    r"\u0639\u0644\u0645\u064A|"  # علمي
    r"\u0623\u0643\u0627\u062F\u064A\u0645\u064A"  # أكاديمي
    r")",
    re.IGNORECASE,
)

_CONTACT_PATTERN = re.compile(
    r"\b(contact|email|phone|call|reach|talk\s+to|help\s+desk|"
    r"librarian|staff|ask\s+a\s+librarian|directions?|location|"
    r"where\s+is|floor|map)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_intent(query: str) -> dict:
    """Classify query intent for table routing and page_type filtering.

    Returns dict with:
        tables: list of tables to prioritize (None = all)
        page_types: list of page_type values to filter on (for document_chunks)
        intent: string label ("database", "hours", "contact", "faq", "general")
        is_database_intent: bool — True if any database keyword matched (EN or AR)
    """
    q = query.lower()

    # Check database intent across both English and Arabic patterns.
    # This merges the old retriever._DB_PATTERN + chatbot.IntentDetector logic.
    has_db_en = bool(_DB_PATTERN_EN.search(q))
    has_db_ar = bool(_DB_PATTERN_AR.search(query))  # Arabic: match original case
    is_db = has_db_en or has_db_ar

    if is_db:
        return {
            "tables": ["databases", "faq", "document_chunks"],
            "page_types": ["database_page", "service"],
            "intent": "database",
            "is_database_intent": True,
        }

    # Hours/schedule intent
    if _HOURS_PATTERN.search(q):
        return {
            "tables": ["faq", "document_chunks", "library_pages"],
            "page_types": None,
            "intent": "hours",
            "is_database_intent": False,
        }

    # Contact/location intent
    if _CONTACT_PATTERN.search(q):
        return {
            "tables": ["faq", "document_chunks", "library_pages"],
            "page_types": None,
            "intent": "contact",
            "is_database_intent": False,
        }

    # General FAQ-style questions
    if _FAQ_PATTERN.search(q):
        return {
            "tables": ["faq", "document_chunks", "library_pages"],
            "page_types": None,
            "intent": "faq",
            "is_database_intent": False,
        }

    # Default: search everything (including custom_notes)
    return {
        "tables": None,
        "page_types": None,
        "intent": "general",
        "is_database_intent": False,
    }
