"""
reranker.py
LLM-based reranking of retrieved chunks with source-priority-aware scoring.

v2 improvements:
  - Source-priority boost: graduated boost based on source trust tier
  - FAQ score dampening: compensates for FAQ entries' inflated similarity scores
  - Source type metadata preserved and used in scoring
  - Better logging of source selection reasoning

Includes deduplication to remove near-duplicate chunks before scoring.
"""

import json
import re
import logging
import os
from typing import List, Optional

from .llm_client import chat_completion, LLMUnavailableError
from .source_config import (
    SOURCE_CONFIG, get_source_type, get_source_trust,
    FACULTY_TEXT, SCRAPED_WEBSITE, FACULTY_FAQ,
)

logger = logging.getLogger(__name__)


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

    # Custom notes table
    if "label" in meta and "content" in meta:
        return f"{meta['label']}. {meta['content']}"

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


def _apply_source_priority_boost(candidates: List[dict]) -> None:
    """Apply source-priority boost to rerank scores.

    This is the core source-aware ranking logic:
      1. Each candidate gets a boost proportional to its source's trust tier
      2. FAQ entries get a dampening factor to compensate for inflated similarity
      3. The boost is small enough that a clearly better match from a lower-priority
         source still wins, but large enough to break ties in favor of higher-trust sources

    Modifies candidates in-place (adds/updates rerank_score).
    """
    cfg = SOURCE_CONFIG

    for cand in candidates:
        source_type = cand.get("source_type", get_source_type(cand.get("source_table", "")))
        trust = get_source_trust(source_type)
        raw_score = cand.get("rerank_score", 0.0)
        vector_score = cand.get("vector_score", 0.0)

        # Faculty text vector floor: custom notes are admin-curated answers.
        # The LLM reranker penalizes notes that don't contain quotable facts
        # (e.g. "refer to this link"), but when the note is topically relevant
        # (high vector similarity), it IS the intended answer. Use vector
        # score as a floor so the reranker can't bury it.
        if source_type == FACULTY_TEXT and vector_score >= cfg.faculty_text_vector_floor_threshold:
            vector_floor = vector_score * cfg.faculty_text_vector_floor_multiplier
            if vector_floor > raw_score:
                logger.info(
                    f"Faculty text vector floor: raw={raw_score:.3f} → floor={vector_floor:.3f} "
                    f"(vector={vector_score:.3f})"
                )
                raw_score = vector_floor

        # FAQ dampening: short clean FAQ entries get inflated evidence scores
        # because they're concentrated and easy to match. Apply a small reduction.
        if source_type == FACULTY_FAQ:
            raw_score *= cfg.faq_score_dampening

        # Source priority boost: proportional to trust tier.
        # This is the ONLY place trust boosts are applied (not in RRF retrieval).
        # This is a soft tiebreaker — when two chunks have similar evidence scores,
        # the higher-trust source wins.
        boost = cfg.rerank_source_boost_weight * trust

        # Freshness boost for scraped content on time-sensitive queries
        if source_type == SCRAPED_WEBSITE and cand.get("freshness_sensitive"):
            boost += cfg.freshness_boost * cfg.rerank_source_boost_weight

        final_score = raw_score + boost

        cand["rerank_score"] = min(1.0, final_score)
        cand["raw_rerank_score"] = raw_score  # Preserve pre-boost score for debugging
        cand["source_boost"] = boost


def rerank(
    query: str,
    candidates: List[dict],
    top_k: int = 8,
    min_score: float = 0.5,
) -> List[dict]:
    """Rerank retrieval candidates using LLM-based relevance scoring + source priority.

    Pipeline:
      1. Deduplicate near-identical chunks
      2. LLM scores each chunk on evidence support (0-1)
      3. Apply source-priority boost (graduated by trust tier)
      4. Apply FAQ score dampening
      5. Sort by final score, filter by min_score

    Each result gets:
      - rerank_score: final score (evidence + source boost)
      - raw_rerank_score: LLM score before source adjustment
      - source_boost: the boost applied for source priority
    """
    cfg = SOURCE_CONFIG

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
        source_type = cand.get("source_type", get_source_type(cand.get("source_table", "")))
        if page_title:
            source_info = f"(Page: {page_title}"
            if section_title:
                source_info += f" > {section_title}"
            source_info += f") [source: {source_type}] "
        else:
            source_info = f"[source: {source_type}] "
        if len(text) > 1800:
            text = text[:1800] + "..."
        chunk_descriptions.append(f"[{i}] {source_info}{text}")

    chunks_text = "\n---\n".join(chunk_descriptions)

    try:
        raw = chat_completion(
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
                        "- Prefer passages with quotable facts over passages with descriptions\n"
                        "- Score based purely on evidence quality — source priority is handled separately\n\n"
                        "Return ONLY a JSON array: [{\"index\": 0, \"score\": 0.8}, ...]. "
                        "Include ALL passages. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nPassages:\n{chunks_text}",
                },
            ],
            max_tokens=600,
            call_type="rerank",
        )
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            logger.warning("Reranker failed to parse LLM response, using RRF scores")
            return _fallback_rerank(candidates, top_k)

        # Apply LLM scores to candidates
        score_map = {item["index"]: item["score"] for item in scores if "index" in item and "score" in item}

        for i, cand in enumerate(candidates):
            cand["rerank_score"] = score_map.get(i, 0.0)

        # Apply source-priority boost (graduated by trust tier, with FAQ dampening)
        _apply_source_priority_boost(candidates)

        # Sort by final score, filter by min_score
        reranked = [c for c in candidates if c["rerank_score"] >= cfg.rerank_min_score]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Log source selection reasoning
        if reranked:
            top = reranked[0]
            logger.info(
                f"Reranker top result: source_type={top.get('source_type')} "
                f"raw_score={top.get('raw_rerank_score', 0):.3f} "
                f"boost={top.get('source_boost', 0):.3f} "
                f"final={top.get('rerank_score', 0):.3f} "
                f"| {len(reranked)} candidates above threshold"
            )
            # Log source distribution
            source_dist = {}
            for c in reranked:
                st = c.get("source_type", "unknown")
                source_dist[st] = source_dist.get(st, 0) + 1
            logger.info(f"Reranker source distribution: {source_dist}")

        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"LLM reranking failed: {e}, falling back to RRF scores")
        return _fallback_rerank(candidates, top_k)


def _fallback_rerank(candidates: List[dict], top_k: int) -> List[dict]:
    """Fallback: use the existing RRF scores as rerank scores."""
    for cand in candidates:
        cand["rerank_score"] = cand.get("rrf_score", cand.get("vector_score", 0.0))
    # Still apply source priority boost even in fallback
    _apply_source_priority_boost(candidates)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
