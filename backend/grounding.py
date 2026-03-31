"""
grounding.py — Pre-generation grounding checks and claim-level verification.

This module implements the high-precision anti-hallucination pipeline:

1. Answerability classifier (FULL / PARTIAL / NONE) — replaces binary YES/NO
2. Evidence planner — extracts supportable claims BEFORE generation
3. Claim-level post-generation audit — splits answer into atomic claims
   and verifies each individually
4. Query risk classifier — detects high-risk query types that need
   stricter evidence thresholds

The key insight: hallucination happens when the model generates claims
that have no direct textual support.  The fix is to make "no support → no claim"
structurally impossible, not just instructed.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


# ============================================================================
# 1. Query risk classifier
# ============================================================================

# High-risk queries need stronger evidence because wrong answers cause
# real harm (student shows up at wrong time, sends to wrong email, etc.)
_HIGH_RISK_PATTERNS = [
    # Schedule/hours — wrong times waste trips
    re.compile(r"\b(hours?|open(ing)?|clos(e|ing)|schedule|timing|when)\b", re.I),
    # Contact info — wrong email/phone wastes time
    re.compile(r"\b(email|phone|contact|call|reach|address|location|direction)\b", re.I),
    # Policies with consequences — wrong info causes penalties
    re.compile(r"\b(fee|fine|penalty|deadline|overdue|renew|late|cost|price)\b", re.I),
    # Specific procedures — wrong steps cause failures
    re.compile(r"\b(how\s+(?:to|do|can)|step|process|procedure|register|apply|request)\b", re.I),
    # Dates — wrong dates cause missed deadlines
    re.compile(r"\b(date|deadline|due|expire|until|before|after)\b", re.I),
]


def classify_query_risk(query: str) -> str:
    """Classify query as 'high' or 'standard' risk.

    High-risk queries (hours, contacts, fees, dates, procedures) get
    stricter evidence requirements because wrong answers cause real harm.
    """
    hits = sum(1 for pat in _HIGH_RISK_PATTERNS if pat.search(query))
    return "high" if hits >= 1 else "standard"


# ============================================================================
# 2. Answerability classifier (replaces binary YES/NO)
# ============================================================================

def classify_answerability(
    query: str, context: str, risk_level: str = "standard"
) -> Dict:
    """Classify whether context can answer the query.

    Returns dict with:
      - level: "FULL" | "PARTIAL" | "NONE"
      - supported_parts: list of question aspects the context covers
      - missing_parts: list of question aspects the context does NOT cover
      - reason: brief explanation

    The three levels drive different generation strategies:
      FULL    → generate normally (context explicitly supports the answer)
      PARTIAL → answer only supported parts, explicitly list gaps
      NONE    → abstain entirely
    """
    try:
        client = _get_client()

        # For high-risk queries, the bar for FULL is higher
        risk_note = ""
        if risk_level == "high":
            risk_note = (
                "\nIMPORTANT: This is a high-risk query (hours/contacts/fees/dates/procedures). "
                "Classify as FULL only if the context contains the EXACT specific information "
                "requested (exact times, exact emails, exact amounts). "
                "If the context only has general or outdated information, classify as PARTIAL or NONE."
            )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify whether retrieved context passages can answer a user question.\n\n"
                        "Respond with ONLY a JSON object:\n"
                        "{\n"
                        '  "level": "FULL" | "PARTIAL" | "NONE",\n'
                        '  "supported_parts": ["list of question aspects the context explicitly covers"],\n'
                        '  "missing_parts": ["list of question aspects the context does NOT cover"],\n'
                        '  "reason": "one sentence explanation"\n'
                        "}\n\n"
                        "Definitions:\n"
                        "- FULL: The context contains specific, explicit information that directly "
                        "answers the COMPLETE question. Every key aspect of the question is addressed.\n"
                        "- PARTIAL: The context addresses SOME parts of the question but not all. "
                        "Or the context is about the right topic but missing key specifics "
                        "(e.g., mentions a service but not its hours/cost/procedure).\n"
                        "- NONE: The context does not contain information relevant to answering "
                        "the question, or only tangentially relates to the topic.\n\n"
                        "Be strict: 'related topic' ≠ 'answers the question'. "
                        "If the question asks 'What are the hours?' and context only says "
                        "'the library is open to students', that is PARTIAL not FULL — "
                        "it confirms the library exists but doesn't give hours."
                        f"{risk_note}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context[:5000]}",
                },
            ],
            temperature=0.0,
            max_tokens=250,
        )

        raw = resp.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            level = result.get("level", "NONE").upper()
            if level not in ("FULL", "PARTIAL", "NONE"):
                level = "NONE"
            return {
                "level": level,
                "supported_parts": result.get("supported_parts", []),
                "missing_parts": result.get("missing_parts", []),
                "reason": result.get("reason", ""),
            }

    except Exception as e:
        logger.warning(f"Answerability classification failed: {e}")

    # Fail-safe: if classifier fails, treat as PARTIAL (cautious)
    return {
        "level": "PARTIAL",
        "supported_parts": [],
        "missing_parts": ["classification failed"],
        "reason": "Classifier error — defaulting to cautious mode",
    }


# ============================================================================
# 3. Evidence planner — extract supportable claims BEFORE generation
# ============================================================================

def plan_evidence(query: str, context: str) -> Dict:
    """Extract claims the model CAN make, with evidence, BEFORE generation.

    Returns dict with:
      - claims: list of {claim, evidence_snippet, source_tag}
      - unsupported_aspects: parts of the question that have no evidence

    This is the core anti-hallucination mechanism: the model cannot make
    a claim unless it first identifies the supporting text.
    """
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence extraction assistant. Given a question and context "
                        "passages, extract ONLY the claims that can be directly made based on "
                        "the context.\n\n"
                        "For each supportable claim, provide:\n"
                        "- claim: the factual statement\n"
                        "- evidence: the EXACT quote from the context that supports it (copy-paste)\n"
                        "- source: the [Source: ...] tag from the passage\n\n"
                        "Rules:\n"
                        "- Only include claims where you can copy-paste an exact supporting quote\n"
                        "- Do NOT infer, synthesize, or combine facts from different passages\n"
                        "- Do NOT add any information not explicitly in the evidence quote\n"
                        "- For numbers, dates, times, names, emails: the EXACT value must appear in the quote\n"
                        "- If the question asks for something not in the context, list it under unsupported_aspects\n\n"
                        "Return ONLY a JSON object:\n"
                        "{\n"
                        '  "claims": [\n'
                        '    {"claim": "...", "evidence": "exact quote", "source": "[Source: ...]"}\n'
                        "  ],\n"
                        '  "unsupported_aspects": ["list of question parts with no evidence"]\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context[:6000]}",
                },
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        raw = resp.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "claims": result.get("claims", []),
                "unsupported_aspects": result.get("unsupported_aspects", []),
            }

    except Exception as e:
        logger.warning(f"Evidence planning failed: {e}")

    return {"claims": [], "unsupported_aspects": ["evidence planning failed"]}


# ============================================================================
# 4. Claim-level post-generation audit
# ============================================================================

def audit_claims(
    answer: str, context: str, lang: str = "en"
) -> Tuple[str, List[str]]:
    """Split answer into atomic claims and verify each one individually.

    Unlike the previous verifier which checked the whole answer at once,
    this audits EACH claim separately so subtle hallucinations in otherwise
    correct answers are caught.

    Returns (cleaned_answer, removed_claims).
    """
    if not answer or not context:
        return answer, []

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You audit a chatbot answer for unsupported claims.\n\n"
                        "Process:\n"
                        "1. Split the answer into ATOMIC claims (each individual fact).\n"
                        "2. For EACH claim, search the context for the specific supporting text.\n"
                        "3. Mark each claim as SUPPORTED or UNSUPPORTED.\n"
                        "4. Rewrite the answer keeping ONLY supported claims.\n\n"
                        "STRICT rules for marking UNSUPPORTED:\n"
                        "- Any number, time, date, email, phone, URL, price, or name that does "
                        "not appear verbatim in the context → UNSUPPORTED\n"
                        "- Any service, policy, or capability not explicitly described → UNSUPPORTED\n"
                        "- Any conclusion from combining two separate context passages → UNSUPPORTED\n"
                        "- Any generalization (typically, usually, most, often) → UNSUPPORTED\n"
                        "- Any claim about what a service does NOT offer (unless context explicitly "
                        "says it's unavailable) → UNSUPPORTED\n\n"
                        "Return ONLY a JSON object:\n"
                        "{\n"
                        '  "audit": [\n'
                        '    {"claim": "...", "verdict": "SUPPORTED|UNSUPPORTED", '
                        '"evidence": "exact quote or null"}\n'
                        "  ],\n"
                        '  "cleaned_answer": "answer with only supported claims, properly formatted",\n'
                        '  "removed_claims": ["list of removed unsupported claims"]\n'
                        "}\n\n"
                        "If removing claims leaves the answer empty, set cleaned_answer to:\n"
                        '"I could not verify this information from the available sources. '
                        'Please contact the library directly."\n\n'
                        "Preserve markdown formatting and citations in cleaned_answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context[:6000]}\n\n"
                        f"Answer to audit:\n{answer}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=1200,
        )

        raw = resp.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            cleaned = result.get("cleaned_answer", answer)
            removed = result.get("removed_claims", [])

            # If the audit stripped everything meaningful, use abstention
            if cleaned and len(cleaned.strip()) < 30:
                if lang == "ar":
                    cleaned = (
                        "لم أتمكن من التحقق من هذه المعلومات من المصادر المتاحة. "
                        "يرجى التواصل مع المكتبة مباشرة."
                    )
                else:
                    cleaned = (
                        "I could not verify this information from the available sources. "
                        "Please contact the library directly."
                    )

            if removed:
                logger.info(
                    f"Claim audit removed {len(removed)} claims: "
                    f"{[c[:60] for c in removed]}"
                )

            return cleaned, removed

    except Exception as e:
        logger.warning(f"Claim audit failed: {e}")

    # Fail-safe: return answer with disclaimer
    return answer + "\n\n*Please verify this information with the library directly.*", \
           ["AUDIT_ERROR: claim audit failed"]


# ============================================================================
# 5. Support-aware reranking prompt
# ============================================================================

SUPPORT_RERANK_PROMPT = (
    "You are an evidence judge for a university library chatbot. "
    "Given a user question and numbered text passages, score each passage on "
    "how well it provides DIRECT EVIDENCE to answer the question.\n\n"
    "This is NOT about topic relevance — it is about answer support.\n"
    "A passage about 'library hours' is relevant to a question about hours, "
    "but only SUPPORTS the answer if it contains the actual hour values.\n\n"
    "Scoring guidelines:\n"
    "- 0.9-1.0: Contains the EXACT answer (specific facts, numbers, times, names)\n"
    "- 0.7-0.8: Contains key information needed to answer (specific but incomplete)\n"
    "- 0.4-0.6: Related topic but missing the specific details asked about\n"
    "- 0.1-0.3: Tangentially related, no useful evidence\n"
    "- 0.0: Irrelevant\n\n"
    "Important:\n"
    "- A passage that mentions a topic without giving specifics scores 0.4-0.5, not 0.7+\n"
    "- A passage that gives exact numbers/times/names scores 0.7+\n"
    "- Prefer passages with quotable facts over passages with descriptions\n\n"
    "Return ONLY a JSON array: [{\"index\": 0, \"score\": 0.8}, ...]. "
    "Include ALL passages. No explanation."
)
