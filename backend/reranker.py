"""
reranker.py
LLM-based reranking of retrieved chunks.
Takes top candidates from hybrid retrieval, scores relevance, returns top results.

Includes deduplication to remove near-duplicate chunks before scoring.
"""

import json
import re
import logging
import os
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _get_chunk_text(result: dict) -> str:
    """Extract the display text from a retrieval result."""
    meta = result.get("metadata", {})

    # document_chunks table
    if "chunk_text" in meta:
        return meta["chunk_text"]

    # FAQ table
    if "question" in meta and "answer" in meta:
        return f"Q: {meta['question']}\nA: {meta['answer']}"

    # Databases table
    if "name" in meta and "description" in meta:
        return f"{meta['name']}. {meta['description']}"

    # Library pages table
    if "title" in meta and "content" in meta:
        return f"{meta['title']}\n{meta['content'][:500]}"

    return str(meta)


def _deduplicate_candidates(
    candidates: List[dict],
    similarity_threshold: float = 0.85,
) -> List[dict]:
    """Remove near-duplicate chunks based on text overlap.

    Uses Jaccard similarity on word sets to detect duplicates.
    Keeps the candidate with the higher RRF score.
    """
    if not candidates:
        return []

    def _word_set(result: dict) -> set:
        text = _get_chunk_text(result).lower()
        return set(text.split())

    kept = []
    kept_word_sets = []

    for cand in candidates:
        cand_words = _word_set(cand)
        is_dup = False
        for existing_words in kept_word_sets:
            if not cand_words or not existing_words:
                continue
            intersection = len(cand_words & existing_words)
            union = len(cand_words | existing_words)
            if union > 0 and intersection / union >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
            kept_word_sets.append(cand_words)

    logger.info(f"Deduplication: {len(candidates)} -> {len(kept)} candidates")
    return kept


def rerank(
    query: str,
    candidates: List[dict],
    top_k: int = 8,
    min_score: float = 0.5,
) -> List[dict]:
    """Rerank retrieval candidates using LLM-based relevance scoring.

    Args:
        query: The user's question.
        candidates: List of retrieval results (from retriever.hybrid_retrieve).
        top_k: Max number of chunks to keep after reranking.
        min_score: Minimum relevance score (0-1) to include.

    Returns:
        Reranked list of candidates, each with an added "rerank_score" field.
    """
    if not candidates:
        return []

    # Deduplicate before reranking to avoid wasting LLM tokens on near-duplicates
    candidates = _deduplicate_candidates(candidates[:20])

    # Limit input to avoid token limits — take top 15 candidates max
    candidates = candidates[:15]

    # Build the scoring prompt
    chunk_descriptions = []
    for i, cand in enumerate(candidates):
        text = _get_chunk_text(cand)
        meta = cand.get("metadata", {})
        # Add source context to help the reranker
        source_info = ""
        page_title = meta.get("page_title", meta.get("title", ""))
        section_title = meta.get("section_title", "")
        if page_title:
            source_info = f"(Page: {page_title}"
            if section_title:
                source_info += f" > {section_title}"
            source_info += ") "
        if len(text) > 1800:
            text = text[:1800] + "..."
        chunk_descriptions.append(f"[{i}] {source_info}{text}")

    chunks_text = "\n---\n".join(chunk_descriptions)

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
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
                        "- A passage that gives exact numbers/times/names/procedures scores 0.7+\n"
                        "- Prefer passages with quotable facts over passages with descriptions\n\n"
                        "Return ONLY a JSON array: [{\"index\": 0, \"score\": 0.8}, ...]. "
                        "Include ALL passages. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nPassages:\n{chunks_text}",
                },
            ],
            temperature=0.0,
            max_tokens=600,
        )

        raw = resp.choices[0].message.content.strip()
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            logger.warning("Reranker failed to parse LLM response, using RRF scores")
            return _fallback_rerank(candidates, top_k)

        # Apply scores to candidates
        score_map = {item["index"]: item["score"] for item in scores if "index" in item and "score" in item}

        for i, cand in enumerate(candidates):
            cand["rerank_score"] = score_map.get(i, 0.0)

        # Sort by rerank score, filter by min_score
        reranked = [c for c in candidates if c["rerank_score"] >= min_score]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"LLM reranking failed: {e}, falling back to RRF scores")
        return _fallback_rerank(candidates, top_k)


def _fallback_rerank(candidates: List[dict], top_k: int) -> List[dict]:
    """Fallback: use the existing RRF scores as rerank scores."""
    for cand in candidates:
        cand["rerank_score"] = cand.get("rrf_score", cand.get("vector_score", 0.0))
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
