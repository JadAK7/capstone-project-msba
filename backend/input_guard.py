"""
input_guard.py
Pre-processing safety layer: prompt injection detection and domain scope filtering.

Runs BEFORE any retrieval or LLM call. Cheap regex-based checks that short-circuit
malicious or off-topic queries so they never reach the model.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """Result of running input guards on a query."""
    allowed: bool = True
    injection_detected: bool = False
    out_of_scope: bool = False
    refusal_reason: str = ""
    matched_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt injection / jailbreak detection
# ---------------------------------------------------------------------------

# English injection patterns (case-insensitive)
_INJECTION_PATTERNS_EN = [
    # Direct instruction override
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"ignore\s+(your\s+)?(system\s+)?prompt",
    r"ignore\s+(your\s+)?rules",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions|rules|prompt)",
    r"forget\s+(all\s+)?(previous|prior|above|your)\s+(instructions|rules|training|context|prompt)",
    r"forget\s+everything",
    r"override\s+(the\s+)?(system\s+)?prompt",
    r"bypass\s+(the\s+)?(system|safety|content)\s*(prompt|filter|rules)",
    # Role-play injection
    r"you\s+are\s+now\b",
    r"act\s+as\s+(if\s+you\s+are\s+|a\s+|an\s+)?(?!a\s+library|an?\s+aub|the\s+library)",
    r"pretend\s+(you\s+are|to\s+be)\b",
    r"simulate\s+(being|a)\b",
    r"role\s*play\s+as\b",
    r"switch\s+to\s+(a\s+)?new\s+(role|mode|persona)",
    r"new\s+instruction[s]?\s*:",
    r"system\s*:\s*you\s+are",
    # Prompt leaking
    r"(show|reveal|repeat|print|output|display|tell)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)",
    r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)",
    # Forced compliance
    r"just\s+answer\s+(the\s+question\s+)?anyway",
    r"answer\s+without\s+(using\s+)?(context|sources|references|restrictions)",
    r"use\s+your\s+(own\s+)?(general\s+)?knowledge",
    r"don'?t\s+(use|rely\s+on)\s+(the\s+)?(context|sources|retrieved)",
    r"stop\s+being\s+(a\s+)?library\s+assistant",
    r"you\s+can\s+answer\s+anything",
    r"you\s+have\s+no\s+restrictions",
    r"enable\s+(developer|debug|god|admin)\s+mode",
    r"(developer|debug|god|admin|sudo)\s+mode",
    r"do\s+anything\s+now",
    r"DAN\b",
    r"jailbreak",
]

# Arabic injection patterns
_INJECTION_PATTERNS_AR = [
    r"\u062A\u062C\u0627\u0647\u0644\s+(\u062C\u0645\u064A\u0639\s+)?\u0627\u0644\u062A\u0639\u0644\u064A\u0645\u0627\u062A",  # تجاهل (جميع) التعليمات
    r"\u062A\u062C\u0627\u0647\u0644\s+(\u062C\u0645\u064A\u0639\s+)?\u0627\u0644\u0642\u0648\u0627\u0639\u062F",  # تجاهل (جميع) القواعد
    r"\u0627\u0646\u0633\u064E?\s+(\u062C\u0645\u064A\u0639\s+)?\u0627\u0644\u062A\u0639\u0644\u064A\u0645\u0627\u062A",  # انسَ (جميع) التعليمات
    r"\u0623\u0646\u062A\s+\u0627\u0644\u0622\u0646\b",  # أنت الآن
    r"\u062A\u0635\u0631\u0641\s+\u0643\u0640?\b",  # تصرف ك
    r"\u062A\u0638\u0627\u0647\u0631\s+\u0623\u0646\u0643\b",  # تظاهر أنك
    r"\u0623\u0638\u0647\u0631\s+(\u0644\u064A\s+)?\u0627\u0644\u062A\u0639\u0644\u064A\u0645\u0627\u062A",  # أظهر (لي) التعليمات
    r"\u0623\u062C\u0628\s+\u0628\u062F\u0648\u0646\s+(\u0627\u0633\u062A\u062E\u062F\u0627\u0645\s+)?\u0627\u0644\u0633\u064A\u0627\u0642",  # أجب بدون (استخدام) السياق
    r"\u0627\u0633\u062A\u062E\u062F\u0645\s+\u0645\u0639\u0631\u0641\u062A\u0643\s+\u0627\u0644\u0639\u0627\u0645\u0629",  # استخدم معرفتك العامة
]

_COMPILED_INJECTION_EN = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS_EN]
_COMPILED_INJECTION_AR = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS_AR]


def detect_injection(query: str) -> GuardResult:
    """Check if query contains prompt injection or jailbreak patterns.

    Returns GuardResult with injection_detected=True and matched_patterns
    if any injection pattern is found.
    """
    if not query or not query.strip():
        return GuardResult()

    matched = []

    for pattern in _COMPILED_INJECTION_EN:
        if pattern.search(query):
            matched.append(pattern.pattern)

    for pattern in _COMPILED_INJECTION_AR:
        if pattern.search(query):
            matched.append(pattern.pattern)

    if matched:
        logger.warning(
            f"Prompt injection detected in query: '{query[:100]}' "
            f"| matched {len(matched)} pattern(s)"
        )
        return GuardResult(
            allowed=False,
            injection_detected=True,
            refusal_reason="injection_detected",
            matched_patterns=matched,
        )

    return GuardResult()


# ---------------------------------------------------------------------------
# Domain scope filtering
# ---------------------------------------------------------------------------

# Library-domain keywords (English) — if ANY of these appear, query is in-scope
_LIBRARY_SCOPE_EN = re.compile(
    r"\b("
    r"librar[yies]+|book[s]?|borrow|return|renew|loan|overdue|fine[s]?|"
    r"catalog|catalogue|reserve|reservation|hold|"
    r"database[s]?|journal[s]?|article[s]?|paper[s]?|research|"
    r"scopus|pubmed|jstor|proquest|ieee|web\s+of\s+science|ebsco|"
    r"hours|opening|closing|schedule|"
    r"print|scan|copy|photocopy|"
    r"study\s+room|reading\s+room|group\s+study|quiet\s+room|"
    r"interlibrary|ill|"
    r"thesis|dissertation|"
    r"e-?book|e-?resource|electronic|online\s+access|"
    r"wifi|internet|computer|laptop|"
    r"archive[s]?|special\s+collection|"
    r"citation|reference|endnote|zotero|mendeley|"
    r"floor|map|directions?|location|"
    r"policy|policies|rules?|regulations?|"
    r"membership|card|id\s+card|access|"
    r"workshop|training|tutorial|"
    r"staff|librarian|help\s+desk|ask\s+a\s+librarian|"
    r"aub|american\s+university\s+of?\s+beirut|jafet|"
    r"saab\s+medical|engineering|science\s+and\s+agriculture"
    r")\b",
    re.IGNORECASE,
)

# Library-domain keywords (Arabic)
_LIBRARY_SCOPE_AR = re.compile(
    r"("
    r"\u0645\u0643\u062A\u0628\u0629|\u0643\u062A\u0627\u0628|\u0643\u062A\u0628|"  # مكتبة|كتاب|كتب
    r"\u0627\u0633\u062A\u0639\u0627\u0631\u0629|\u0625\u0639\u0627\u0631\u0629|"  # استعارة|إعارة
    r"\u062A\u062C\u062F\u064A\u062F|\u0625\u0631\u062C\u0627\u0639|"  # تجديد|إرجاع
    r"\u0642\u0627\u0639\u062F\u0629\s*\u0628\u064A\u0627\u0646\u0627\u062A|"  # قاعدة بيانات
    r"\u0642\u0648\u0627\u0639\u062F\s*\u0628\u064A\u0627\u0646\u0627\u062A|"  # قواعد بيانات
    r"\u0645\u0642\u0627\u0644|\u0628\u062D\u062B|\u0623\u0628\u062D\u0627\u062B|"  # مقال|بحث|أبحاث
    r"\u0645\u062C\u0644\u0629|\u0645\u062C\u0644\u0627\u062A|"  # مجلة|مجلات
    r"\u0633\u0627\u0639\u0627\u062A\s*\u0627\u0644\u0639\u0645\u0644|"  # ساعات العمل
    r"\u0637\u0628\u0627\u0639\u0629|\u0645\u0633\u062D|\u0646\u0633\u062E|"  # طباعة|مسح|نسخ
    r"\u063A\u0631\u0641\u0629\s*\u062F\u0631\u0627\u0633\u0629|"  # غرفة دراسة
    r"\u0631\u0633\u0627\u0644\u0629|\u0623\u0637\u0631\u0648\u062D\u0629|"  # رسالة|أطروحة
    r"\u0648\u0635\u0648\u0644|\u0628\u0637\u0627\u0642\u0629|"  # وصول|بطاقة
    r"\u0633\u064A\u0627\u0633\u0629|\u0642\u0648\u0627\u0646\u064A\u0646|"  # سياسة|قوانين
    r"\u0645\u0648\u0638\u0641|\u0623\u0645\u064A\u0646\s*\u0645\u0643\u062A\u0628\u0629|"  # موظف|أمين مكتبة
    r"\u062C\u0627\u0641\u064A\u062A"  # جافيت
    r")",
    re.IGNORECASE,
)

# Obvious out-of-scope patterns (things clearly unrelated to a university library)
_OUT_OF_SCOPE_EN = re.compile(
    r"\b("
    r"recipe[s]?|cook|bake|ingredient[s]?|"
    r"weather|temperature|forecast|"
    r"stock\s*(market|price)|crypto|bitcoin|"
    r"write\s+(me\s+)?(a\s+)?(code|program|script|essay|poem|story|song)|"
    r"code\s+(?:in|for|a)\b|"
    r"translate\s+(?:this|the\s+following)\b|"
    r"solve\s+(this\s+)?(math|equation|integral|derivative)|"
    r"what\s+is\s+\d+\s*[\+\-\*\/x]\s*\d+|"
    r"calculate|"
    r"play\s+a\s+game|tell\s+(me\s+)?a\s+joke|"
    r"who\s+is\s+the\s+president|"
    r"capital\s+of\b|"
    r"meaning\s+of\s+life"
    r")\b",
    re.IGNORECASE,
)


def check_domain_scope(query: str) -> GuardResult:
    """Check whether the query is within the library domain.

    Strategy:
    1. If query matches library-scope keywords -> allowed (in scope).
    2. If query matches obvious out-of-scope patterns AND does NOT match
       any library keyword -> blocked.
    3. Ambiguous queries are allowed (benefit of the doubt) — retrieval
       will handle them with low scores leading to abstention.
    """
    if not query or not query.strip():
        return GuardResult()

    has_library_en = bool(_LIBRARY_SCOPE_EN.search(query))
    has_library_ar = bool(_LIBRARY_SCOPE_AR.search(query))

    if has_library_en or has_library_ar:
        return GuardResult()  # In scope

    has_out_of_scope = bool(_OUT_OF_SCOPE_EN.search(query))

    if has_out_of_scope:
        logger.info(f"Out-of-scope query detected: '{query[:100]}'")
        return GuardResult(
            allowed=False,
            out_of_scope=True,
            refusal_reason="out_of_scope",
            matched_patterns=["out_of_scope_keyword"],
        )

    # Ambiguous: let retrieval handle it
    return GuardResult()


# ---------------------------------------------------------------------------
# Combined guard (public API)
# ---------------------------------------------------------------------------

def run_input_guards(query: str) -> GuardResult:
    """Run all input guards: injection detection first, then domain scope.

    Returns the first failing GuardResult, or a passing result if all clear.
    """
    # 1. Injection detection (highest priority)
    injection_result = detect_injection(query)
    if not injection_result.allowed:
        return injection_result

    # 2. Domain scope check
    scope_result = check_domain_scope(query)
    if not scope_result.allowed:
        return scope_result

    return GuardResult()  # All clear


# ---------------------------------------------------------------------------
# Refusal messages
# ---------------------------------------------------------------------------

REFUSAL_MESSAGES = {
    "injection_detected": {
        "en": "I cannot follow that request. I can only answer questions about AUB library services and resources.",
        "ar": "لا يمكنني تنفيذ هذا الطلب. يمكنني فقط الإجابة على الأسئلة المتعلقة بخدمات وموارد مكتبة الجامعة الأمريكية في بيروت.",
    },
    "out_of_scope": {
        "en": "I can only assist with library-related questions, such as library services, hours, databases, borrowing policies, and research resources.",
        "ar": "يمكنني فقط المساعدة في الأسئلة المتعلقة بالمكتبة، مثل خدمات المكتبة وساعات العمل وقواعد البيانات وسياسات الإعارة وموارد البحث.",
    },
    "no_context": {
        "en": "I could not find this information in the available sources. Please contact the library directly or visit the AUB Libraries website.",
        "ar": "لم أتمكن من إيجاد هذه المعلومات في المصادر المتاحة. يرجى التواصل مع المكتبة مباشرة أو زيارة موقع مكتبات الجامعة الأمريكية في بيروت.",
    },
}


def get_refusal_message(reason: str, lang: str = "en") -> str:
    """Get the appropriate refusal message for a given reason and language."""
    messages = REFUSAL_MESSAGES.get(reason, REFUSAL_MESSAGES["no_context"])
    return messages.get(lang, messages["en"])
