"""
query_rewriter.py
LLM-based query rewriting with intent classification folded into the same call.

Pipeline (per request):
  1. Detect Arabic Unicode + mixed-script (hard override — always slow path).
  2. Fast-path: well-formed English, >5 words, no follow-up → skip LLM call.
     Intent falls back to keyword classifier in this case.
  3. Slow path: single LLM call that simultaneously:
       a. Resolves follow-ups from conversation history
       b. Translates Arabic → English for retrieval
       c. Expands short/vague queries
       d. Classifies intent (hours|database|borrowing|contact|general)

The ORIGINAL query is always preserved for final answer generation.
Only the REWRITTEN query is used for embedding / retrieval.
"""

import re
import json
import logging
from typing import List, Optional, Tuple

from .llm_client import chat_completion, LLMUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------

# Arabic Unicode blocks: main Arabic (؀-ۿ), Supplement (ݐ-ݿ),
# Presentation Forms-A (ﭐ-﷿), Presentation Forms-B (ﹰ-﻿).
_ARABIC_PATTERN = re.compile(
    "[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"
)

# Arabic block character class (reused inside compound patterns)
_ARA = "[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"

# Mixed-script: query contains BOTH Arabic Unicode AND Latin letters.
# E.g. "shu هي ساعات" (inline mix of scripts).
_MIXED_SCRIPT_RE = re.compile(
    _ARA + r".*[a-zA-Z]" + "|" + r"[a-zA-Z].*" + _ARA,
    re.DOTALL,
)

# Arabizi digit-substitution: 2=ء, 3=ع, 5=خ, 7=ح adjacent to Latin letters.
# E.g. "3alayk", "7abibi", "ma3ak".
_ARABIZI_DIGIT_RE = re.compile(
    r"(?<![0-9])[2357](?=[a-zA-Z])|(?<=[a-zA-Z])[2357](?![0-9])",
    re.IGNORECASE,
)

# Interrogative scaffolding that dilutes embedding similarity against our
# declarative corpus (FAQ answers, database descriptions, scraped page text).
# Questions starting with these tokens must be normalized into a retrieval-
# optimized form before embedding — otherwise the interrogative structure
# ("which ... should I use", "how do I ...") drops the top score below the
# partial-answer threshold and the bot abstains on queries it could answer.
_INTERROGATIVE_RE = re.compile(
    r"^\s*(which|what|how|where|when|why|who|whom|whose|"
    r"should|shall|can|could|would|do|does|did|is|are|was|were|am)\b",
    re.IGNORECASE,
)


def _is_arabic(text: str) -> bool:
    """True if text contains any Arabic Unicode character."""
    return bool(_ARABIC_PATTERN.search(text))


def _has_arabic_context(text: str) -> bool:
    """True if text needs the slow-path (Arabic Unicode, mixed-script, or Arabizi).

    Hard override for the fast-path skip: any of these conditions means the
    query may require translation or expansion before retrieval.
    """
    if _is_arabic(text):
        return True
    if _MIXED_SCRIPT_RE.search(text):
        return True
    if _ARABIZI_DIGIT_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Valid intents returned by the LLM
# ---------------------------------------------------------------------------

_VALID_LLM_INTENTS = frozenset(
    {"hours", "database", "borrowing", "contact", "general"}
)

# Map LLM intents → the intent labels used by intent_classifier.py
# (borrowing is a sub-type of faq intent)
_INTENT_REMAP = {
    "borrowing": "faq",
}


# ---------------------------------------------------------------------------
# Rewrite prompt — asks for JSON output including intent
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a query rewriter for a university library search system.
Your job is to rewrite the user's query into a clear, self-contained English search query \
optimised for semantic search over a library knowledge base, and to classify its intent.

Rules:
1. If the query is a follow-up (e.g. "what about that?", "how?", "tell me more"), \
resolve it using the conversation history into a complete, standalone question.
2. If the query is in Arabic, translate it to English while preserving the intent.
3. If the query is short or vague (e.g. "hours", "borrowing"), expand it into a \
specific library question (e.g. "What are the library opening hours?", \
"What is the borrowing policy and loan period?").
4. Keep the rewritten query concise (1-2 sentences max).
5. Do NOT answer the question — only rewrite it for search.
6. Preserve specific names, database names, or technical terms exactly.

Return ONLY a valid JSON object (no extra text):
{
  "rewritten_query": "the rewritten English query",
  "language": "ar" or "en",
  "intent": one of "hours" | "database" | "borrowing" | "contact" | "general"
}

Intent definitions:
  hours    — asking about opening/closing times, schedules
  database — asking about research databases, e-resources, journals, finding articles
  borrowing — asking about borrowing, returning, renewing, fines, overdue
  contact  — asking for contact info, email, phone, directions, librarian
  general  — anything else about library services or policies"""


def rewrite_query(
    query: str,
    history: Optional[List[dict]] = None,
    lang: str = "en",
) -> Tuple[str, dict]:
    """Rewrite a user query for optimal retrieval and classify its intent.

    Args:
        query:   Raw user query.
        history: Conversation history (list of {role, content} dicts).
        lang:    Detected language of the query.

    Returns:
        Tuple of (rewritten_query, debug_info).
        rewritten_query is always English and self-contained.
        debug_info keys: original_query, is_arabic, is_short, is_followup,
                         rewrite_skipped, rewritten_query, llm_intent (optional).
    """
    original = query.strip()
    has_arabic = _has_arabic_context(original)
    is_short = len(original.split()) <= 3
    is_followup = _is_followup(original, history)
    is_interrogative = bool(_INTERROGATIVE_RE.match(original))

    debug: dict = {
        "original_query": original,
        "is_arabic": has_arabic,
        "is_short": is_short,
        "is_followup": is_followup,
        "is_interrogative": is_interrogative,
        "rewrite_skipped": False,
    }

    # Fast path: well-formed English query that needs no rewriting.
    # Hard overrides (always slow path):
    #   - Arabic Unicode, mixed-script, or Arabizi
    #   - Interrogative form — embedding scoring against our declarative
    #     corpus drops below the abstain threshold otherwise.
    if (
        not has_arabic
        and not is_short
        and not is_followup
        and not is_interrogative
        and len(original.split()) >= 6
    ):
        debug["rewrite_skipped"] = True
        debug["rewritten_query"] = original
        # intent is None here; callers fall back to keyword classifier
        return original, debug

    # Slow path: call the LLM
    messages = [{"role": "system", "content": _REWRITE_PROMPT}]

    if history:
        recent = history[-6:] if len(history) > 6 else history
        for entry in recent:
            if isinstance(entry, dict) and entry.get("role") in ("user", "assistant"):
                content = entry.get("content", "")
                if content.strip():
                    if entry["role"] == "assistant" and len(content) > 200:
                        content = content[:200] + "..."
                    messages.append({"role": entry["role"], "content": content})

    messages.append({"role": "user", "content": original})

    try:
        raw = chat_completion(
            messages=messages,
            max_tokens=150,
            call_type="rewrite",
        )

        # Parse JSON response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                rewritten = (parsed.get("rewritten_query") or "").strip()
                llm_intent = parsed.get("intent", "")

                # Sanity-check rewritten query
                if not rewritten or len(rewritten) > 300:
                    rewritten = _fallback_rewrite(original, history, has_arabic)
                    debug["rewrite_fallback"] = True
                else:
                    # Validate and remap intent
                    if llm_intent in _VALID_LLM_INTENTS:
                        debug["llm_intent"] = _INTENT_REMAP.get(llm_intent, llm_intent)
                    else:
                        logger.debug(
                            "Rewriter returned unrecognized intent '%s', "
                            "keyword classifier will be used",
                            llm_intent,
                        )

                debug["rewritten_query"] = rewritten
                logger.info(
                    "Query rewrite: '%s' -> '%s' (intent=%s)",
                    original[:60],
                    rewritten[:60],
                    debug.get("llm_intent", "keyword-fallback"),
                )
                return rewritten, debug

            except (json.JSONDecodeError, KeyError) as parse_err:
                logger.warning(
                    "Query rewrite JSON parse failed: %s. Raw: '%s'",
                    parse_err,
                    raw[:200],
                )

        # JSON parse failed — if raw text looks like a plain query, use it
        if raw and 5 < len(raw) < 300 and "\n" not in raw:
            debug["rewritten_query"] = raw.strip()
            debug["rewrite_fallback"] = True
            return raw.strip(), debug

        rewritten = _fallback_rewrite(original, history, has_arabic)
        debug["rewritten_query"] = rewritten
        debug["rewrite_fallback"] = True
        return rewritten, debug

    except Exception as e:
        logger.warning("Query rewrite LLM call failed: %s", e)
        rewritten = _fallback_rewrite(original, history, has_arabic)
        debug["rewritten_query"] = rewritten
        debug["rewrite_error"] = str(e)
        return rewritten, debug


def _is_followup(query: str, history: Optional[List[dict]]) -> bool:
    """Detect if query is a follow-up that needs history context."""
    if not history:
        return False

    q = query.lower().strip()
    if len(q.split()) <= 3:
        return True

    followup_patterns = [
        r"^(what|how|why|when|where|which)\s+(about|else)\b",
        r"^(tell me more|elaborate|explain|expand)",
        r"^(and|but|also|what if)\b",
        r"^(yes|no|ok|sure)\b",
        r"^(the same|that one|this one|it)\b",
        r"^(وماذا|كيف|لماذا|أيضاً)",
    ]

    for pattern in followup_patterns:
        if re.match(pattern, q, re.IGNORECASE):
            return True

    return False


def _fallback_rewrite(
    query: str,
    history: Optional[List[dict]],
    is_arabic: bool,
) -> str:
    """Non-LLM fallback: basic heuristic rewriting."""
    if history:
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("role") == "user":
                prev = entry.get("content", "").strip()
                if prev and prev != query.strip():
                    return f"{prev} {query}"
                break
    return query
