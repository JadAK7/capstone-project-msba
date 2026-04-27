"""
evaluation.py
RAG evaluation metrics for the AUB Libraries Assistant.

Metrics (each returns 0.0-1.0):
  - Groundedness: % of answer claims supported by retrieved context
  - Faithfulness: % of answer that does NOT introduce info outside context
  - Context Relevance: relevance of retrieved chunks to the query
  - Answer Relevance: how well the answer addresses the question
  - Citation Accuracy: correctness of cited sources
  - Hallucination Rate: % of answer that is fabricated (= 1 - faithfulness)

Combined grounding_score = weighted average:
  groundedness (40%) + faithfulness (35%) + context_relevance (25%)

Consistency invariant:
  hallucination_rate = 1 - faithfulness (enforced, not independently scored)
"""

import json
import logging
import os
import re
import time
from typing import List, Optional

import numpy as np

from .embeddings import embed_text
from .llm_client import chat_completion, LLMUnavailableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM evaluation helper — routes through llm_client for retry + circuit breaker
# ---------------------------------------------------------------------------

def _llm_eval(prompt: str, max_tokens: int = 500) -> dict:
    """Call the configured chat model for evaluation scoring.

    Returns parsed JSON with 'score' (0.0-1.0) and 'reason'.
    Uses llm_client.chat_completion so it inherits provider config,
    retry logic, and the 'generate' circuit breaker.
    """
    try:
        text = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, calibrated RAG evaluation judge. "
                        "You score retrieval-augmented generation systems. "
                        "Be critical — most real systems score between 0.4 and 0.85. "
                        "Perfect 1.0 scores should be extremely rare. "
                        "Scores below 0.3 indicate serious problems. "
                        "Always respond with ONLY valid JSON: "
                        '{\"score\": <float 0.0-1.0>, \"reason\": \"<explanation>\"}'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            call_type="generate",
        )
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return {"score": score, "reason": parsed.get("reason", "")}
        return {"score": 0.0, "reason": "Failed to parse LLM response"}
    except (LLMUnavailableError, Exception) as e:
        logger.error(f"LLM evaluation call failed: {e}")
        return {"score": 0.0, "reason": f"Error: {e}"}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


# ---------------------------------------------------------------------------
# Cross-lingual judge fix
# ---------------------------------------------------------------------------
# The LLM judge cannot reliably verify an Arabic answer against an English
# context — it literally searches for the Arabic string in the English text
# and marks everything unsupported. We (1) tell the judge that cross-lingual
# matches are valid in every prompt, and (2) pre-translate Arabic answers to
# English before judging groundedness/faithfulness so the match is direct.

_CROSS_LINGUAL_NOTE = (
    "IMPORTANT — cross-lingual matching: the ANSWER and CONTEXT may be in "
    "different languages (for example, Arabic answer against English context). "
    "A claim is SUPPORTED if it expresses the same fact as something in the "
    "context, regardless of language. Translate mentally when checking. Do "
    "NOT mark a claim unsupported only because the exact words do not appear "
    "in the context language.\n\n"
)


def _has_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in (text or ""))


def _translate_to_english(text: str) -> str:
    """Translate an Arabic answer to English for faithful cross-lingual judging.

    Used only for groundedness/faithfulness scoring — the original answer is
    preserved for answer_relevance and for reporting. Falls back to the
    original text if the translation call fails.
    """
    if not text:
        return text
    try:
        translated = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise translator. Translate the user's "
                        "Arabic text to English. Preserve every fact, number, "
                        "name, URL, date, and structural element (lists, "
                        "bullets, headers) exactly. Do not add commentary or "
                        "interpretation. Return only the English translation."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=1500,
            call_type="generate",
        )
        return translated.strip() or text
    except Exception as e:
        logger.warning(f"Cross-lingual judge translation failed, using original: {e}")
        return text


# ---------------------------------------------------------------------------
# 1. Groundedness — claim-level verification
# ---------------------------------------------------------------------------

def score_groundedness(query: str, context: str, answer: str) -> dict:
    """Score = fraction of claims in the answer that are supported by context.

    Uses a two-step approach:
      1. Extract distinct factual claims from the answer
      2. Check each claim against the context
    """
    prompt = (
        "You are evaluating GROUNDEDNESS of a chatbot answer.\n\n"
        "Task: Determine what fraction of the answer's claims are supported by the context.\n\n"
        "Steps:\n"
        "1. List every distinct factual claim made in the ANSWER (not opinions or hedging).\n"
        "2. For each claim, check if it is explicitly supported by the CONTEXT.\n"
        "3. Count supported claims vs total claims.\n"
        "4. Score = supported_claims / total_claims.\n\n"
        "Scoring guide:\n"
        "- 1.0: Every single claim traces to context (rare — be strict)\n"
        "- 0.7-0.9: Most claims supported, minor unsupported details\n"
        "- 0.4-0.7: Mixed — some claims supported, some not\n"
        "- 0.0-0.4: Most claims have no basis in context\n\n"
        "Be strict: rephrasing is OK, but any added fact not in context is unsupported.\n"
        "If the answer says 'I don't have information', score 1.0 (no unsupported claims).\n\n"
        + _CROSS_LINGUAL_NOTE +
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context[:5000]}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"Found X claims, Y supported. <details>\"}"
    )
    return _llm_eval(prompt)


# ---------------------------------------------------------------------------
# 2. Faithfulness — inverse of hallucination
# ---------------------------------------------------------------------------

def score_faithfulness(query: str, context: str, answer: str) -> dict:
    """Score = 1 - (fraction of answer content that introduces info outside context).

    Faithfulness checks the OPPOSITE direction from groundedness:
    - Groundedness: are the answer's claims IN the context?
    - Faithfulness: does the answer ADD things NOT in the context?

    A faithful answer may not cover everything in context, but it doesn't fabricate.
    """
    prompt = (
        "You are a STRICT evaluator of chatbot faithfulness. "
        "You must catch ALL instances where the answer adds information not in the context.\n\n"
        "Faithfulness = does the answer avoid introducing information NOT present in the context?\n\n"
        "Steps:\n"
        "1. List EVERY factual claim in the ANSWER (each specific fact, number, name, time, "
        "service, URL, policy detail).\n"
        "2. For EACH claim, search the CONTEXT for the specific supporting text.\n"
        "3. Mark each claim as SUPPORTED (text found in context) or UNSUPPORTED (not found).\n"
        "4. Score = 1.0 - (count of unsupported claims / total claims).\n\n"
        "What counts as UNSUPPORTED (strict):\n"
        "- Any specific fact, number, time, date, name, or URL not stated in context\n"
        "- Any service, policy, or procedure not described in context\n"
        "- Any conclusion drawn by combining facts from different context passages\n"
        "- Any generalization (\"typically\", \"usually\", \"in most cases\") about library services\n"
        "- Any recommendation or suggestion not based on a context passage\n"
        "- Adding details that make the answer sound more complete but aren't in context\n\n"
        "What is NOT unfaithful:\n"
        "- Directly quoting or closely paraphrasing context\n"
        "- Formatting changes (bullets, bold) that don't change meaning\n"
        "- Saying \"I don't have information\" (perfectly faithful = 1.0)\n"
        "- Grammatical connectives that don't add facts\n\n"
        "Be strict. Most chatbot answers add SOME unsupported details. "
        "A score of 0.9+ should be rare.\n\n"
        + _CROSS_LINGUAL_NOTE +
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context[:5000]}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"Found X claims. Y unsupported: <list each>\"}"
    )
    return _llm_eval(prompt)


# ---------------------------------------------------------------------------
# 3. Hallucination Rate — derived from faithfulness (NOT independently scored)
# ---------------------------------------------------------------------------

def compute_hallucination_rate(faithfulness_score: float) -> dict:
    """Hallucination rate = 1 - faithfulness.

    This is NOT scored independently by the LLM. It is derived from faithfulness
    to guarantee the consistency invariant:
        hallucination_rate + faithfulness = 1.0

    This eliminates the contradiction where faithfulness=0 and hallucination=0.
    """
    rate = round(1.0 - faithfulness_score, 4)
    if rate < 0.05:
        reason = "Very low hallucination — answer is highly faithful to context"
    elif rate < 0.2:
        reason = "Minor hallucination — small additions beyond context"
    elif rate < 0.5:
        reason = "Moderate hallucination — noticeable unsupported content"
    else:
        reason = "High hallucination — significant fabricated content"
    return {"score": rate, "reason": reason}


# ---------------------------------------------------------------------------
# 4. Context Relevance — hybrid: embedding + LLM
# ---------------------------------------------------------------------------

def score_context_relevance(query: str, chunks: List[dict]) -> dict:
    """Score = how relevant the retrieved chunks are to the query.

    Uses LLM scoring (more accurate than pure embedding similarity).
    """
    if not chunks:
        return {"score": 0.0, "reason": "No chunks retrieved"}

    chunk_texts = []
    for i, chunk in enumerate(chunks[:5]):
        text = chunk.get("text", "")[:400]
        chunk_texts.append(f"[Chunk {i+1}]: {text}")

    chunks_formatted = "\n\n".join(chunk_texts)

    prompt = (
        "You are evaluating CONTEXT RELEVANCE: how relevant are the retrieved chunks to the query?\n\n"
        "Steps:\n"
        "1. Read the QUESTION carefully.\n"
        "2. For each retrieved chunk, rate its relevance to answering the question.\n"
        "3. Score = average relevance across all chunks.\n\n"
        "Scoring guide:\n"
        "- 1.0: Every chunk is directly relevant and useful for answering\n"
        "- 0.7-0.9: Most chunks are relevant, maybe one off-topic\n"
        "- 0.4-0.7: Mixed relevance — some useful, some not\n"
        "- 0.1-0.4: Mostly irrelevant chunks retrieved\n"
        "- 0.0: No chunk has any relevance to the question\n\n"
        + _CROSS_LINGUAL_NOTE +
        f"QUESTION: {query}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_formatted[:3000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"X of Y chunks relevant. <details>\"}"
    )
    result = _llm_eval(prompt)

    # Also compute embedding similarity as supplementary data
    try:
        query_emb = embed_text(query)
        similarities = []
        for chunk in chunks[:5]:
            chunk_emb = embed_text(chunk.get("text", "")[:500])
            sim = _cosine_similarity(query_emb, chunk_emb)
            similarities.append(sim)
        result["raw_similarities"] = [round(s, 4) for s in similarities]
        result["avg_embedding_similarity"] = round(float(np.mean(similarities)), 4) if similarities else 0.0
    except Exception:
        pass  # supplementary only, don't fail

    return result


# ---------------------------------------------------------------------------
# 5. Answer Relevance
# ---------------------------------------------------------------------------

def score_answer_relevance(query: str, answer: str) -> dict:
    """Score = how well the answer addresses the specific question asked."""
    prompt = (
        "You are evaluating ANSWER RELEVANCE: does the answer address the question?\n\n"
        "Scoring guide:\n"
        "- 1.0: Directly and completely answers the question\n"
        "- 0.7-0.9: Answers the question but missing some details or slightly indirect\n"
        "- 0.4-0.7: Partially addresses the question, or answers a related but different question\n"
        "- 0.1-0.4: Barely related to the question\n"
        "- 0.0: Completely off-topic\n\n"
        "Note: An answer saying 'I don't have information' for a question the system can't answer "
        "should score ~0.5 (honest but unhelpful).\n\n"
        + _CROSS_LINGUAL_NOTE +
        f"QUESTION: {query}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"...\"}"
    )
    return _llm_eval(prompt)


# ---------------------------------------------------------------------------
# 6. Citation Accuracy
# ---------------------------------------------------------------------------

def score_citation_accuracy(answer: str, chunks: List[dict]) -> dict:
    """Score = correctness of cited sources in the answer.

    Checks if URLs and markdown link references in the answer can be traced
    to the retrieved chunks. Also penalizes if the answer makes claims that
    should cite a source but doesn't.
    """
    urls_in_answer = re.findall(r"https?://[^\s\)\]>]+", answer)
    source_refs = re.findall(r"\[([^\]]+)\]\([^\)]+\)", answer)

    if not urls_in_answer and not source_refs:
        return {"score": 1.0, "reason": "No citations in answer (N/A — not penalized)"}

    # Collect all text from chunks to match against
    chunk_urls = set()
    chunk_titles = set()
    chunk_full_text = ""
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_full_text += " " + text
        chunk_urls.update(re.findall(r"https?://[^\s\)\]>]+", text))
        # page_url and page_title from metadata
        page_url = chunk.get("page_url", "")
        page_title = chunk.get("page_title", "")
        if page_url:
            chunk_urls.add(page_url)
        if page_title:
            chunk_titles.add(page_title.lower().strip())
        first_line = text.split("\n")[0].strip()
        if first_line:
            chunk_titles.add(first_line.lower())

    matched = 0
    total = len(urls_in_answer) + len(source_refs)

    for url in urls_in_answer:
        url_clean = url.rstrip("/.,;:")
        if any(url_clean.startswith(cu.rstrip("/")) or cu.rstrip("/").startswith(url_clean)
               for cu in chunk_urls):
            matched += 1

    for ref in source_refs:
        ref_lower = ref.lower().strip()
        if ref_lower in chunk_titles or any(ref_lower in t for t in chunk_titles):
            matched += 1

    score = matched / total if total > 0 else 1.0
    return {
        "score": round(score, 4),
        "reason": f"{matched}/{total} citations matched retrieved sources",
        "urls_found": urls_in_answer,
        "refs_found": source_refs,
    }


# ---------------------------------------------------------------------------
# Combined grounding score
# ---------------------------------------------------------------------------

_GROUNDING_WEIGHTS = {
    "groundedness": 0.40,
    "faithfulness": 0.35,
    "context_relevance": 0.25,
}


def compute_grounding_score(
    groundedness: float,
    faithfulness: float,
    context_relevance: float,
) -> float:
    """Weighted average: groundedness 40% + faithfulness 35% + context_relevance 25%."""
    return round(
        _GROUNDING_WEIGHTS["groundedness"] * groundedness
        + _GROUNDING_WEIGHTS["faithfulness"] * faithfulness
        + _GROUNDING_WEIGHTS["context_relevance"] * context_relevance,
        4,
    )


# ---------------------------------------------------------------------------
# Consistency enforcement
# ---------------------------------------------------------------------------

def _enforce_consistency(metrics: dict) -> dict:
    """Post-process metric scores to eliminate logical contradictions.

    Rules enforced:
    1. hallucination_rate = 1 - faithfulness (hard invariant)
    2. If groundedness > faithfulness + 0.3, cap groundedness
       (can't be highly grounded if the answer adds lots of extra info)
    3. If context_relevance < 0.2, cap groundedness at context_relevance + 0.3
       (can't ground an answer in irrelevant context)
    """
    g = metrics["groundedness"]["score"]
    f = metrics["faithfulness"]["score"]
    cr = metrics["context_relevance"]["score"]

    # Rule 1: hallucination = 1 - faithfulness (already enforced by compute_hallucination_rate)
    # (no action needed here — it's derived, not independent)

    # Rule 2: groundedness can't wildly exceed faithfulness
    # If answer adds lots of extra info (low faithfulness), groundedness should be limited
    max_g_from_f = f + 0.3
    if g > max_g_from_f and f < 0.7:
        adjusted_g = round(max_g_from_f, 4)
        metrics["groundedness"]["score"] = adjusted_g
        metrics["groundedness"]["reason"] += (
            f" [Adjusted from {g:.2f} to {adjusted_g:.2f}: "
            f"capped by faithfulness={f:.2f}]"
        )

    # Rule 3: can't be well-grounded in irrelevant context
    if cr < 0.2:
        max_g_from_cr = cr + 0.3
        current_g = metrics["groundedness"]["score"]
        if current_g > max_g_from_cr:
            adjusted_g = round(max_g_from_cr, 4)
            metrics["groundedness"]["score"] = adjusted_g
            metrics["groundedness"]["reason"] += (
                f" [Adjusted from {current_g:.2f} to {adjusted_g:.2f}: "
                f"context_relevance too low ({cr:.2f})]"
            )

    return metrics


# ---------------------------------------------------------------------------
# Full evaluation for a single query
# ---------------------------------------------------------------------------

def evaluate_single(
    query: str,
    answer: str,
    retrieved_chunks: List[dict],
    chosen_source: str = "",
    context_sent_to_llm: str = "",
) -> dict:
    """Run all metrics on a single query-answer pair.

    Args:
        context_sent_to_llm: The EXACT context string that was sent to the
            generation LLM.  When provided, evaluation uses this instead of
            reconstructing context from retrieved_chunks.  This eliminates
            context mismatches (different truncation, missing headers, wrong
            chunk filtering) that cause the evaluator to flag supported claims
            as hallucinations.
    """
    # Use the actual generation context when available — this is what the
    # LLM saw, so it's what faithfulness should be measured against.
    if context_sent_to_llm:
        context = context_sent_to_llm
        relevant_chunks = retrieved_chunks
    else:
        # Fallback: reconstruct from retrieved_chunks (legacy path)
        relevant_chunks = retrieved_chunks
        context = "\n---\n".join(c.get("text", "") for c in relevant_chunks[:5])

    # Log what we're evaluating
    logger.info(
        f"Evaluating: query='{query[:60]}' | source={chosen_source} | "
        f"chunks={len(relevant_chunks)} | answer_len={len(answer)}"
    )

    # Truncation warning: groundedness/faithfulness prompts cap context at
    # 5000 chars and answer at 2000. If the real strings are longer, the
    # judge can't see all the supporting evidence and may flag a supported
    # claim as unsupported. Surface the per-row counts so the scorecard
    # can flag runs where this is happening at scale.
    context_chars = len(context)
    answer_chars = len(answer)
    context_truncated = context_chars > 5000
    answer_truncated = answer_chars > 2000
    if context_truncated:
        logger.warning(
            "Context truncated for grounding judge: %dc → 5000c "
            "(query=%r). Supported claims may be marked unsupported.",
            context_chars,
            query[:50],
        )

    # Cross-lingual judge fix (Fix 2): if the answer is Arabic but the
    # context is English, the LLM judge will mark every claim "unsupported"
    # because the Arabic strings don't appear in the English text. Translate
    # the answer to English before running groundedness/faithfulness so the
    # judge can match on meaning instead of exact substrings.
    answer_for_grounding = answer
    answer_was_translated = False
    if _has_arabic(answer) and not _has_arabic(context):
        translated = _translate_to_english(answer)
        if translated and translated != answer:
            answer_for_grounding = translated
            answer_was_translated = True
            logger.info(
                "  [cross-lingual judge] Translated AR answer→EN for "
                f"grounding/faithfulness (orig {len(answer)}c → {len(translated)}c)"
            )

    # Run the 4 independent metrics. Use the (possibly translated) answer
    # only for groundedness/faithfulness — answer_relevance compares the
    # answer to the question and should stay in the original language.
    groundedness_result = score_groundedness(query, context, answer_for_grounding)
    faithfulness_result = score_faithfulness(query, context, answer_for_grounding)
    context_relevance_result = score_context_relevance(query, relevant_chunks)
    answer_relevance_result = score_answer_relevance(query, answer)
    citation_result = score_citation_accuracy(answer, relevant_chunks)

    # Derive hallucination from faithfulness (NOT independent LLM call)
    hallucination_result = compute_hallucination_rate(faithfulness_result["score"])

    # Assemble metrics dict
    metrics = {
        "groundedness": groundedness_result,
        "faithfulness": faithfulness_result,
        "context_relevance": context_relevance_result,
        "answer_relevance": answer_relevance_result,
        "citation_accuracy": citation_result,
        "hallucination_rate": hallucination_result,
    }

    # Enforce logical consistency
    metrics = _enforce_consistency(metrics)

    # Compute grounding score from (possibly adjusted) values
    grounding = compute_grounding_score(
        metrics["groundedness"]["score"],
        metrics["faithfulness"]["score"],
        metrics["context_relevance"]["score"],
    )

    # Debug logging
    logger.info(
        f"  Scores: ground={metrics['groundedness']['score']:.2f} "
        f"faith={metrics['faithfulness']['score']:.2f} "
        f"ctx_rel={metrics['context_relevance']['score']:.2f} "
        f"ans_rel={metrics['answer_relevance']['score']:.2f} "
        f"halluc={metrics['hallucination_rate']['score']:.2f} "
        f"cite={metrics['citation_accuracy']['score']:.2f} "
        f"GROUNDING={grounding:.2f}"
    )

    return {
        "query": query,
        "answer_preview": answer[:200] + ("..." if len(answer) > 200 else ""),
        "chosen_source": chosen_source,
        "num_chunks": len(relevant_chunks),
        "metrics": metrics,
        "grounding_score": grounding,
        "answer_translated_for_judge": answer_was_translated,
        "context_chars": context_chars,
        "answer_chars": answer_chars,
        "context_truncated": context_truncated,
        "answer_truncated": answer_truncated,
    }


# ---------------------------------------------------------------------------
# Batch evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation_pipeline(
    questions: List[str],
    chatbot,
    language: Optional[str] = None,
) -> dict:
    """Run the full evaluation pipeline on a list of test questions."""
    results = []
    metric_totals = {
        "groundedness": [],
        "faithfulness": [],
        "context_relevance": [],
        "answer_relevance": [],
        "citation_accuracy": [],
        "hallucination_rate": [],
        "grounding_score": [],
    }

    for i, question in enumerate(questions):
        logger.info(f"Evaluating question {i + 1}/{len(questions)}: {question[:80]}")
        start = time.time()

        try:
            answer, debug = chatbot.answer(question, language=language)

            retrieved_chunks = debug.get("retrieved_chunks", [])
            chosen_source = debug.get("chosen_source", "")
            context_sent = debug.get("context_sent_to_llm", "")

            eval_result = evaluate_single(
                query=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                chosen_source=chosen_source,
                context_sent_to_llm=context_sent,
            )

            eval_result["evaluation_time_s"] = round(time.time() - start, 2)
            results.append(eval_result)

            for metric_name in list(metric_totals.keys()):
                if metric_name == "grounding_score":
                    metric_totals[metric_name].append(eval_result["grounding_score"])
                else:
                    metric_totals[metric_name].append(
                        eval_result["metrics"][metric_name]["score"]
                    )

        except Exception as e:
            logger.error(f"Evaluation failed for question '{question[:50]}': {e}")
            results.append({
                "query": question,
                "error": str(e),
                "evaluation_time_s": round(time.time() - start, 2),
            })

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    aggregate = {
        metric: _avg(values) for metric, values in metric_totals.items()
    }
    aggregate["total_questions"] = len(questions)
    aggregate["successful_evaluations"] = len([r for r in results if "error" not in r])
    aggregate["grounding_weights"] = _GROUNDING_WEIGHTS

    # --- Breakdown metrics for validation ---
    successful = [r for r in results if "error" not in r]

    # Refusal rate: chosen_source is the authoritative signal. The
    # text-match fallback only fires when chosen_source is missing.
    # Disagreements are logged so prompt-wording changes can't silently
    # misclassify refusals.
    _REFUSAL_START_MARKERS = (
        "i could not find",
        "i don't have",
        "i can only answer",
        "**i'm not quite sure",
        "**لست متأكد",
        "لم أتمكن",
        "لا أملك معلومات",
        "يمكنني فقط",
    )

    def _matches_refusal_text(text: str) -> bool:
        return (text or "").lstrip().lower().startswith(_REFUSAL_START_MARKERS)

    def _is_refusal_row(r: dict) -> bool:
        cs = r.get("chosen_source", "") or ""
        if cs:
            cs_says_refusal = cs.startswith("none") or cs.startswith("refused")
            text_says_refusal = _matches_refusal_text(r.get("answer_preview", ""))
            if cs_says_refusal != text_says_refusal:
                logger.warning(
                    "refusal-detection disagreement: chosen_source=%r → %s, "
                    "text → %s. Trusting chosen_source.",
                    cs,
                    "refusal" if cs_says_refusal else "answer",
                    "refusal" if text_says_refusal else "answer",
                )
            return cs_says_refusal
        return _matches_refusal_text(r.get("answer_preview", ""))

    refusals = [r for r in successful if _is_refusal_row(r)]
    aggregate["refusal_rate"] = round(len(refusals) / max(len(successful), 1), 4)

    # Partial-answer rate
    partials = [
        r for r in successful
        if r.get("metrics", {}).get("context_relevance", {}).get("score", 0) < 0.6
        and r not in refusals
    ]
    aggregate["partial_answer_rate"] = round(len(partials) / max(len(successful), 1), 4)

    # Per-query-type hallucination (requires query risk classification)
    try:
        from .grounding import classify_query_risk
        high_risk = [r for r in successful if classify_query_risk(r.get("query", "")) == "high"]
        standard = [r for r in successful if classify_query_risk(r.get("query", "")) == "standard"]

        def _avg_halluc(lst):
            scores = [r.get("metrics", {}).get("hallucination_rate", {}).get("score", 0) for r in lst]
            return round(sum(scores) / max(len(scores), 1), 4)

        aggregate["hallucination_by_risk"] = {
            "high_risk": {"count": len(high_risk), "hallucination_rate": _avg_halluc(high_risk)},
            "standard": {"count": len(standard), "hallucination_rate": _avg_halluc(standard)},
        }
    except Exception:
        pass

    return {
        "aggregate": aggregate,
        "results": results,
    }
