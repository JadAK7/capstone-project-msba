"""
evaluation.py
RAG evaluation metrics for the AUB Libraries Assistant.

Metrics (each returns 0.0–1.0):
  - Groundedness: Is the answer supported by the retrieved context?
  - Faithfulness: Does the answer avoid adding information not in the context?
  - Context Relevance: Do the retrieved chunks match the query?
  - Answer Relevance: Does the answer address the question?
  - Citation Accuracy: Are sources/links in the answer accurate?
  - Hallucination Rate: How much of the answer is fabricated? (0 = no hallucination)

Combined grounding_score = weighted average of groundedness, faithfulness, context_relevance.
"""

import json
import logging
import os
import re
import time
from typing import List, Optional

import numpy as np
from openai import OpenAI

from .embeddings import embed_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-based evaluation helpers
# ---------------------------------------------------------------------------

_eval_client: Optional[OpenAI] = None


def _get_eval_client() -> OpenAI:
    global _eval_client
    if _eval_client is None:
        _eval_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _eval_client


def _llm_score(prompt: str, max_tokens: int = 150) -> dict:
    """Call GPT-4o-mini to score an evaluation dimension.

    The prompt must instruct the model to return JSON with a "score" (0.0–1.0)
    and a short "reason".
    """
    client = _get_eval_client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG evaluation judge. "
                        "Always respond with valid JSON: {\"score\": <float 0.0-1.0>, \"reason\": \"<brief explanation>\"}. "
                        "Nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON even if wrapped in markdown code fences
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"score": 0.0, "reason": "Failed to parse LLM response"}
    except Exception as e:
        logger.error(f"LLM evaluation call failed: {e}")
        return {"score": 0.0, "reason": f"Error: {e}"}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def score_groundedness(query: str, context: str, answer: str) -> dict:
    """Is every claim in the answer supported by the retrieved context?"""
    prompt = (
        "Evaluate whether the ANSWER is grounded in (supported by) the CONTEXT.\n"
        "Score 1.0 if every claim in the answer can be traced to the context.\n"
        "Score 0.0 if the answer has no basis in the context.\n\n"
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context[:3000]}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"...\"}"
    )
    return _llm_score(prompt)


def score_faithfulness(query: str, context: str, answer: str) -> dict:
    """Does the answer avoid adding information that is NOT in the context?"""
    prompt = (
        "Evaluate the FAITHFULNESS of the ANSWER to the CONTEXT.\n"
        "Score 1.0 if the answer contains ONLY information from the context "
        "(no fabricated details, no external knowledge added).\n"
        "Score 0.0 if the answer is mostly fabricated or adds significant "
        "information not present in the context.\n"
        "Note: Formatting, rephrasing, or summarizing context is acceptable.\n\n"
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context[:3000]}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"...\"}"
    )
    return _llm_score(prompt)


def score_context_relevance(query: str, chunks: List[dict]) -> dict:
    """Do the retrieved chunks actually match the user's query?

    Uses embedding similarity between query and each chunk, averaged.
    Falls back to LLM if embeddings fail.
    """
    if not chunks:
        return {"score": 0.0, "reason": "No chunks retrieved"}

    try:
        query_emb = embed_text(query)
        similarities = []
        for chunk in chunks[:5]:  # Limit to top-5 for cost
            chunk_emb = embed_text(chunk.get("text", "")[:500])
            sim = _cosine_similarity(query_emb, chunk_emb)
            similarities.append(sim)

        avg_sim = float(np.mean(similarities)) if similarities else 0.0
        # Normalize: cosine sim for relevant text is typically 0.3–0.8
        # Map to 0–1 range with 0.3 as floor and 0.8 as ceiling
        normalized = max(0.0, min(1.0, (avg_sim - 0.2) / 0.6))
        return {
            "score": round(normalized, 4),
            "reason": f"Avg cosine similarity: {avg_sim:.4f} across {len(similarities)} chunks",
            "raw_similarities": [round(s, 4) for s in similarities],
        }
    except Exception as e:
        logger.warning(f"Embedding-based context relevance failed, using LLM: {e}")
        context_text = "\n---\n".join(c.get("text", "")[:300] for c in chunks[:5])
        prompt = (
            "Rate how relevant the retrieved CONTEXT chunks are to the QUESTION.\n"
            "Score 1.0 if all chunks are highly relevant.\n"
            "Score 0.0 if none are relevant.\n\n"
            f"QUESTION: {query}\n\n"
            f"CONTEXT CHUNKS:\n{context_text[:3000]}\n\n"
            "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"...\"}"
        )
        return _llm_score(prompt)


def score_answer_relevance(query: str, answer: str) -> dict:
    """Does the answer actually address what was asked?"""
    prompt = (
        "Evaluate whether the ANSWER directly addresses the QUESTION.\n"
        "Score 1.0 if the answer fully and directly answers the question.\n"
        "Score 0.0 if the answer is completely off-topic or unhelpful.\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"...\"}"
    )
    return _llm_score(prompt)


def score_citation_accuracy(answer: str, chunks: List[dict]) -> dict:
    """Are citations/sources in the answer accurate?

    Checks if URLs or source references in the answer match retrieved chunks.
    Returns 1.0 if no citations are present (not applicable).
    """
    # Extract URLs and source references from answer
    urls_in_answer = re.findall(r"https?://[^\s\)]+", answer)
    source_refs = re.findall(r"\[([^\]]+)\]\([^\)]+\)", answer)

    if not urls_in_answer and not source_refs:
        return {"score": 1.0, "reason": "No citations in answer (N/A)"}

    # Collect URLs and titles from chunks
    chunk_urls = set()
    chunk_titles = set()
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_urls.update(re.findall(r"https?://[^\s\)]+", text))
        # Library page chunks have title as first line
        first_line = text.split("\n")[0].strip()
        if first_line:
            chunk_titles.add(first_line.lower())

    matched = 0
    total = len(urls_in_answer) + len(source_refs)

    for url in urls_in_answer:
        # Check if URL (or its prefix) appears in any chunk
        if any(url.rstrip("/").startswith(cu.rstrip("/")) or cu.rstrip("/").startswith(url.rstrip("/"))
               for cu in chunk_urls):
            matched += 1

    for ref in source_refs:
        if ref.lower() in chunk_titles or any(ref.lower() in t for t in chunk_titles):
            matched += 1

    score = matched / total if total > 0 else 1.0
    return {
        "score": round(score, 4),
        "reason": f"{matched}/{total} citations matched retrieved sources",
        "urls_found": urls_in_answer,
        "refs_found": source_refs,
    }


def score_hallucination_rate(query: str, context: str, answer: str) -> dict:
    """What fraction of the answer is hallucinated (not in context)?

    Returns 0.0 for no hallucination (best), 1.0 for fully hallucinated (worst).
    Note: this is inverted from the other scores — lower is better.
    """
    prompt = (
        "Evaluate the HALLUCINATION RATE of the ANSWER given the CONTEXT.\n"
        "Score 0.0 if NOTHING in the answer is hallucinated (everything is supported by context).\n"
        "Score 1.0 if the ENTIRE answer is hallucinated (nothing matches context).\n"
        "Common hallucinations: made-up URLs, invented facts, wrong numbers, "
        "fabricated database names, non-existent services.\n\n"
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context[:3000]}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "Return JSON: {\"score\": <0.0-1.0>, \"reason\": \"...\"}"
    )
    return _llm_score(prompt)


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
    """Weighted average of groundedness, faithfulness, and context_relevance."""
    return round(
        _GROUNDING_WEIGHTS["groundedness"] * groundedness
        + _GROUNDING_WEIGHTS["faithfulness"] * faithfulness
        + _GROUNDING_WEIGHTS["context_relevance"] * context_relevance,
        4,
    )


# ---------------------------------------------------------------------------
# Full evaluation for a single query
# ---------------------------------------------------------------------------


def evaluate_single(
    query: str,
    answer: str,
    retrieved_chunks: List[dict],
    chosen_source: str = "",
) -> dict:
    """Run all metrics on a single query-answer pair.

    Args:
        query: The user question.
        answer: The generated answer.
        retrieved_chunks: List of dicts with 'source', 'score', 'text' keys.
        chosen_source: Which source was chosen (e.g. "FAQ", "database").

    Returns:
        Dict with per-metric scores and the combined grounding_score.
    """
    # Build context string from the chunks that were actually used
    # Filter to chunks from the chosen source if possible
    source_map = {
        "FAQ": "faq",
        "database (keyword intent)": "database",
        "database (semantic)": "database",
        "library pages (scraped)": "library_page",
    }
    source_filter = source_map.get(chosen_source, "")

    if source_filter:
        relevant_chunks = [c for c in retrieved_chunks if c.get("source") == source_filter]
    else:
        relevant_chunks = retrieved_chunks

    # If no relevant chunks after filtering, use all chunks
    if not relevant_chunks:
        relevant_chunks = retrieved_chunks

    context = "\n---\n".join(c.get("text", "") for c in relevant_chunks[:5])

    # Run all metrics
    groundedness_result = score_groundedness(query, context, answer)
    faithfulness_result = score_faithfulness(query, context, answer)
    context_relevance_result = score_context_relevance(query, relevant_chunks)
    answer_relevance_result = score_answer_relevance(query, answer)
    citation_result = score_citation_accuracy(answer, relevant_chunks)
    hallucination_result = score_hallucination_rate(query, context, answer)

    grounding = compute_grounding_score(
        groundedness_result["score"],
        faithfulness_result["score"],
        context_relevance_result["score"],
    )

    return {
        "query": query,
        "answer_preview": answer[:200] + ("..." if len(answer) > 200 else ""),
        "chosen_source": chosen_source,
        "num_chunks": len(relevant_chunks),
        "metrics": {
            "groundedness": groundedness_result,
            "faithfulness": faithfulness_result,
            "context_relevance": context_relevance_result,
            "answer_relevance": answer_relevance_result,
            "citation_accuracy": citation_result,
            "hallucination_rate": hallucination_result,
        },
        "grounding_score": grounding,
    }


# ---------------------------------------------------------------------------
# Batch evaluation pipeline
# ---------------------------------------------------------------------------


def run_evaluation_pipeline(
    questions: List[str],
    chatbot,
    language: Optional[str] = None,
) -> dict:
    """Run the full evaluation pipeline on a list of test questions.

    Args:
        questions: List of test question strings.
        chatbot: An initialized LibraryChatbot instance.
        language: Optional language override.

    Returns:
        Dict with per-question results and aggregate scores.
    """
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

            eval_result = evaluate_single(
                query=question,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                chosen_source=chosen_source,
            )

            eval_result["evaluation_time_s"] = round(time.time() - start, 2)
            results.append(eval_result)

            # Accumulate for averages
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

    # Compute aggregate scores
    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    aggregate = {
        metric: _avg(values) for metric, values in metric_totals.items()
    }
    aggregate["total_questions"] = len(questions)
    aggregate["successful_evaluations"] = len([r for r in results if "error" not in r])
    aggregate["grounding_weights"] = _GROUNDING_WEIGHTS

    return {
        "aggregate": aggregate,
        "results": results,
    }
