"""
grounding.py — Pre-generation grounding checks and claim-level verification.

This module implements the high-precision anti-hallucination pipeline:

1. Query risk classifier — detects high-risk query types that need
   stricter evidence thresholds
2. Answerability classifier (FULL / PARTIAL / NONE) — replaces binary YES/NO
3. Evidence planner — extracts supportable claims BEFORE generation
4. Generate-with-inline-verification — generates an answer constrained by
   the evidence plan and verifies each claim inline in a single LLM call
   (replaces the previous separate generation + post-generation audit)

The key insight: hallucination happens when the model generates claims
that have no direct textual support.  The fix is to make "no support → no claim"
structurally impossible, not just instructed.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from .llm_client import chat_completion, LLMUnavailableError

logger = logging.getLogger(__name__)


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
        # For high-risk queries, the bar for FULL is higher
        risk_note = ""
        if risk_level == "high":
            risk_note = (
                "\nIMPORTANT: This is a high-risk query (hours/contacts/fees/dates/procedures). "
                "Classify as FULL only if the context contains the EXACT specific information "
                "requested (exact times, exact emails, exact amounts). "
                "If the context only has general or outdated information, classify as PARTIAL or NONE."
            )

        raw = chat_completion(
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
            max_tokens=250,
            call_type="generate",
        )
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
        raw = chat_completion(
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
            max_tokens=1000,
            call_type="generate",
        )
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
# 4. Generate-with-inline-verification (replaces separate generation + audit)
# ============================================================================

def _parse_chunk_index(chunk_id: str) -> Optional[int]:
    """Parse 'chunk_0' → 0, 'chunk_12' → 12; return None on failure."""
    try:
        return int(chunk_id.split("_")[-1])
    except (ValueError, IndexError, AttributeError):
        return None


def _validate_chunk_ids(claims: List[dict], n_chunks: int) -> List[str]:
    """Return a list of invalid chunk_id strings (referencing out-of-range chunks)."""
    invalid = []
    for claim in claims:
        cid = claim.get("chunk_id", "")
        if not cid:
            continue
        idx = _parse_chunk_index(cid)
        if idx is None or not (0 <= idx < n_chunks):
            invalid.append(cid)
    return invalid


def _generate_answer_raw(
    query: str,
    context: str,
    evidence_plan: Dict,
    system_prompt: str,
    lang: str,
    partial_context: bool,
    history_msgs: Optional[List[dict]],
) -> str:
    """Generation-only call (no inline verification).

    Used as the first stage of the two-stage fallback when the structured
    single-call JSON parse fails.
    """
    planned_claims = evidence_plan.get("claims", [])
    unsupported_aspects = evidence_plan.get("unsupported_aspects", [])

    evidence_section = "=== PRE-VERIFIED EVIDENCE (use ONLY these facts) ===\n"
    for i, claim in enumerate(planned_claims):
        evidence_section += (
            f"{i+1}. Claim: {claim.get('claim', '')}\n"
            f"   Evidence: \"{claim.get('evidence', '')}\"\n"
            f"   Source: {claim.get('source', 'unknown')}\n"
        )
    if unsupported_aspects:
        evidence_section += (
            "\n=== UNSUPPORTED ASPECTS (say you don't have info on these) ===\n"
            + "\n".join(f"- {asp}" for asp in unsupported_aspects)
        )

    partial_note = ""
    if partial_context:
        partial_note = (
            "\n\n⚠️ PARTIAL CONTEXT: Answer only the supported parts above. "
            "Explicitly acknowledge missing information."
        )

    messages = [{"role": "system", "content": system_prompt + partial_note}]
    if history_msgs:
        messages.extend(history_msgs)
    messages.append({
        "role": "user",
        "content": (
            f"{evidence_section}\n\n"
            f"Context:\n{context[:6000]}\n\n"
            f"Question: {query}\n\n"
            "Write a response using ONLY the pre-verified evidence above. "
            "Cite sources (Source: ...) for every claim. "
            "For unsupported aspects, explicitly say you don't have that information."
        ),
    })

    return chat_completion(
        messages=messages,
        max_tokens=1200,
        top_p=0.85,
        call_type="generate",
    )


def generate_and_verify(
    query: str,
    context: str,
    evidence_plan: Dict,
    system_prompt: str,
    lang: str = "en",
    partial_context: bool = False,
    history_msgs: Optional[List[dict]] = None,
) -> Tuple[str, str, List[str]]:
    """Generate an answer and verify each claim inline in a single LLM call.

    The LLM returns structured JSON with:
      - answer:   final markdown answer
      - claims:   [{claim, chunk_id, confidence}] — chunk_id references [Chunk-N]
      - verified: bool — LLM self-reports if all claims are grounded
      - flags:    [] or list of issues found during self-check
      - removed_claims: claims excluded by the LLM

    Post-generation checks (Python, no extra LLM call):
      - Regex output safety check (always runs)
      - Local chunk_id range validation

    Conditional fallback (one extra LLM call):
      - If verified=false OR flags non-empty OR invalid chunk references
        → call verify_answer() from verifier.py on the generated answer

    Structured-output parse failure fallback:
      - If the LLM returns unparseable JSON → fall back to two-stage behavior:
        generate (plain text) then verify_answer() separately.

    Returns (final_answer, draft_answer, removed_claims).
    """
    from .verifier import check_output_safety, verify_answer

    planned_claims = evidence_plan.get("claims", [])
    unsupported_aspects = evidence_plan.get("unsupported_aspects", [])

    # Number context passages so the LLM can reference them by ID
    context_parts = context.split("\n\n---\n\n")
    n_chunks = len(context_parts)
    numbered_context = "\n\n---\n\n".join(
        f"[Chunk-{i}]\n{part}" for i, part in enumerate(context_parts)
    )

    evidence_section = "=== PRE-VERIFIED EVIDENCE (use ONLY these facts) ===\n"
    for i, claim in enumerate(planned_claims):
        evidence_section += (
            f"{i+1}. Claim: {claim.get('claim', '')}\n"
            f"   Evidence: \"{claim.get('evidence', '')}\"\n"
            f"   Source: {claim.get('source', 'unknown')}\n"
        )

    if unsupported_aspects:
        evidence_section += (
            "\n=== UNSUPPORTED ASPECTS (you MUST say you don't have info) ===\n"
            + "\n".join(f"- {asp}" for asp in unsupported_aspects)
        )

    partial_note = ""
    if partial_context:
        partial_note = (
            "\n\n⚠️ PARTIAL CONTEXT: Only some aspects of the question are supported. "
            "You MUST:\n"
            "- State ONLY the supported facts listed above\n"
            "- Explicitly say what information is missing\n"
            "- Do NOT fill in gaps with general knowledge\n"
        )

    verification_instructions = (
        "\n\n=== INLINE VERIFICATION (MANDATORY) ===\n"
        "After writing your answer, return ONLY a JSON object (no text outside):\n"
        "{\n"
        '  "answer": "your complete markdown-formatted answer with citations",\n'
        '  "claims": [\n'
        '    {"claim": "atomic fact from your answer", "chunk_id": "chunk_0", "confidence": 0.95}\n'
        "  ],\n"
        '  "verified": true,\n'
        '  "flags": [],\n'
        '  "removed_claims": ["claims you excluded because they had no chunk support"]\n'
        "}\n\n"
        "STRICT rules:\n"
        f"- Context passages are labelled [Chunk-0] through [Chunk-{n_chunks - 1}]\n"
        "- In each claim, set chunk_id to the label of the passage that supports it "
        "(e.g. 'chunk_0', 'chunk_2')\n"
        "- If a claim has no supporting chunk, EXCLUDE it from the answer and add to removed_claims\n"
        "- Set 'verified': false if you had to exclude any claims\n"
        "- Set 'flags' to a non-empty list describing any issues (unsupported claims, etc.)\n"
        "- Any number, time, date, email, URL, or name MUST appear verbatim in the chunk\n"
        "- Do NOT combine facts from different chunks to create a new claim\n"
        "- Do NOT use generalizations (typically, usually, generally)\n"
        "- If ALL claims are unsupported, set answer to: "
        '"I could not verify this information from the available sources. '
        'Please contact the library directly."'
    )

    messages = [{"role": "system", "content": system_prompt + partial_note}]
    if history_msgs:
        messages.extend(history_msgs)
    messages.append({
        "role": "user",
        "content": (
            f"{evidence_section}\n\n"
            f"Context:\n{numbered_context[:6000]}\n\n"
            f"Question: {query}\n\n"
            "Write a response using ONLY the pre-verified evidence above. "
            "For any unsupported aspect, explicitly state you don't have that information. "
            "Cite the Source tag for every claim."
            f"{verification_instructions}"
        ),
    })

    try:
        raw = chat_completion(
            messages=messages,
            max_tokens=1200,
            top_p=0.85,
            call_type="generate",
        )

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            # ---- Structured-output parse failure: two-stage fallback ----
            logger.warning(
                "generate_and_verify: JSON not found in LLM response; "
                "falling back to two-stage generate→verify"
            )
            draft_raw = raw if (raw and len(raw) > 20) else _generate_answer_raw(
                query, context, evidence_plan, system_prompt, lang,
                partial_context, history_msgs,
            )
            final, removed = verify_answer(query, draft_raw, context, lang)
            return final, draft_raw, removed + ["PARSE_ERROR: structured output fallback used"]

        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("generate_and_verify: JSON decode failed; two-stage fallback")
            draft_raw = _generate_answer_raw(
                query, context, evidence_plan, system_prompt, lang,
                partial_context, history_msgs,
            )
            final, removed = verify_answer(query, draft_raw, context, lang)
            return final, draft_raw, removed + ["PARSE_ERROR: JSON decode fallback used"]

        answer = result.get("answer", "")
        claims = result.get("claims", [])
        verified = result.get("verified", True)
        flags = result.get("flags", [])
        removed = result.get("removed_claims", [])
        draft = answer

        # ---- Regex safety check (always runs, no LLM) ----
        is_safe, violation = check_output_safety(answer)
        if not is_safe:
            logger.warning("generate_and_verify safety violation: %s", violation)
            if lang == "ar":
                refusal = "يمكنني فقط الإجابة على الأسئلة المتعلقة بخدمات وموارد مكتبة الجامعة."
            else:
                refusal = "I can only answer questions about AUB library services and resources."
            return refusal, draft, [f"SAFETY_VIOLATION: {violation}"]

        # ---- Local deterministic chunk_id validation (no LLM) ----
        invalid_refs = _validate_chunk_ids(claims, n_chunks)
        if invalid_refs:
            logger.warning(
                "generate_and_verify: chunk_ids %s are out of range (n_chunks=%d)",
                invalid_refs,
                n_chunks,
            )

        # ---- Conditional fallback: external verifier LLM call ----
        # Triggered when: LLM self-reports failure, flags non-empty, or invalid chunk refs.
        if not verified or flags or invalid_refs:
            reason_parts = []
            if not verified:
                reason_parts.append("verified=false")
            if flags:
                reason_parts.append(f"flags={flags[:3]}")
            if invalid_refs:
                reason_parts.append(f"invalid_chunk_ids={invalid_refs}")
            logger.info(
                "generate_and_verify triggering external verifier: %s",
                ", ".join(reason_parts),
            )
            verified_answer, extra_removed = verify_answer(query, answer, context, lang)
            removed = list(removed) + extra_removed
            if extra_removed:
                logger.info(
                    "External verifier removed %d additional claims", len(extra_removed)
                )
            answer = verified_answer

        # Abstention guard
        if not answer or len(answer.strip()) < 30:
            if lang == "ar":
                answer = (
                    "لم أتمكن من التحقق من هذه المعلومات من المصادر المتاحة. "
                    "يرجى التواصل مع المكتبة مباشرة."
                )
            else:
                answer = (
                    "I could not verify this information from the available sources. "
                    "Please contact the library directly."
                )

        if removed:
            logger.info(
                "Inline verification removed %d claim(s): %s",
                len(removed),
                [c[:60] for c in removed],
            )

        return answer, draft, removed

    except LLMUnavailableError:
        raise
    except Exception as e:
        logger.error("generate_and_verify failed: %s", e)
        parts = [
            f"- {c.get('claim', '')} {c.get('source', '')}"
            for c in planned_claims[:5]
        ]
        fallback = "\n".join(parts) if parts else "An error occurred."
        return (
            fallback + "\n\n*Please verify this information with the library directly.*",
            fallback,
            ["GENERATION_ERROR: generate-and-verify failed"],
        )


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
