"""
query_rewriter.py
LLM-based query rewriting for improved retrieval.

Replaces the naive heuristic (_build_search_query: prepend last msg if <4 words)
with a proper rewriting pipeline:
  1. Resolve follow-ups using conversation history
  2. Translate Arabic queries to English for embedding (indices are English)
  3. Expand vague/short queries into retrieval-optimized form

The ORIGINAL query is always preserved for final answer generation.
Only the REWRITTEN query is used for embedding / retrieval.
"""

import os
import re
import logging
from typing import List, Optional, Tuple

from .llm_client import chat_completion, LLMUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arabic detection (shared with chatbot.py — lightweight duplicate to avoid
# circular imports)
# ---------------------------------------------------------------------------

_ARABIC_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _is_arabic(text: str) -> bool:
    return bool(_ARABIC_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Core rewriting
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a query rewriter for a university library search system.
Your job is to rewrite the user's query into a clear, self-contained English search query \
optimized for semantic search over a library knowledge base.

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
7. Output ONLY the rewritten query, nothing else."""


def rewrite_query(
    query: str,
    history: Optional[List[dict]] = None,
    lang: str = "en",
) -> Tuple[str, dict]:
    """Rewrite a user query for optimal retrieval.

    Args:
        query: Raw user query.
        history: Conversation history (list of {role, content} dicts).
        lang: Detected language of the query.

    Returns:
        Tuple of (rewritten_query, debug_info).
        rewritten_query is always English and self-contained.
        debug_info contains: original_query, is_arabic, is_short, rewritten_query.
    """
    original = query.strip()
    is_arabic = _is_arabic(original)
    is_short = len(original.split()) <= 3
    is_followup = _is_followup(original, history)

    debug = {
        "original_query": original,
        "is_arabic": is_arabic,
        "is_short": is_short,
        "is_followup": is_followup,
        "rewrite_skipped": False,
    }

    # Fast path: if query is already a well-formed English question (>5 words,
    # not a follow-up, not Arabic), skip the LLM call to save latency/cost.
    if not is_arabic and not is_short and not is_followup and len(original.split()) >= 6:
        debug["rewrite_skipped"] = True
        debug["rewritten_query"] = original
        return original, debug

    # Build messages for the rewriter
    messages = [{"role": "system", "content": _REWRITE_PROMPT}]

    # Include recent history for follow-up resolution
    if history:
        # Only last 3 exchanges to keep token count low
        recent = history[-6:] if len(history) > 6 else history
        for entry in recent:
            if isinstance(entry, dict) and entry.get("role") in ("user", "assistant"):
                content = entry.get("content", "")
                if content.strip():
                    # Truncate long assistant messages
                    if entry["role"] == "assistant" and len(content) > 200:
                        content = content[:200] + "..."
                    messages.append({"role": entry["role"], "content": content})

    messages.append({"role": "user", "content": original})

    try:
        rewritten = chat_completion(
            messages=messages,
            max_tokens=100,
        )

        # Sanity check: if rewriter returned something too long or empty, fallback
        if not rewritten or len(rewritten) > 300:
            rewritten = _fallback_rewrite(original, history, is_arabic)
            debug["rewrite_fallback"] = True

        debug["rewritten_query"] = rewritten
        logger.info(f"Query rewrite: '{original[:60]}' -> '{rewritten[:60]}'")
        return rewritten, debug

    except Exception as e:
        logger.warning(f"Query rewrite LLM call failed: {e}")
        rewritten = _fallback_rewrite(original, history, is_arabic)
        debug["rewritten_query"] = rewritten
        debug["rewrite_error"] = str(e)
        return rewritten, debug


def _is_followup(query: str, history: Optional[List[dict]]) -> bool:
    """Detect if query is a follow-up that needs history context."""
    if not history:
        return False

    q = query.lower().strip()
    # Very short queries with history are likely follow-ups
    if len(q.split()) <= 3:
        return True

    followup_patterns = [
        r"^(what|how|why|when|where|which)\s+(about|else)\b",
        r"^(tell me more|elaborate|explain|expand)",
        r"^(and|but|also|what if)\b",
        r"^(yes|no|ok|sure)\b",
        r"^(the same|that one|this one|it)\b",
        # Arabic follow-up patterns
        r"^(\u0648\u0645\u0627\u0630\u0627|\u0643\u064A\u0641|\u0644\u0645\u0627\u0630\u0627|\u0623\u064A\u0636\u0627\u064B)",  # وماذا|كيف|لماذا|أيضاً
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
    """Non-LLM fallback: basic heuristic rewriting.

    - For follow-ups: prepend last user message
    - For Arabic without LLM: return as-is (embedding will be cross-lingual)
    - For short queries: keep as-is
    """
    if history:
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("role") == "user":
                prev = entry.get("content", "").strip()
                if prev and prev != query.strip():
                    return f"{prev} {query}"
                break

    return query
