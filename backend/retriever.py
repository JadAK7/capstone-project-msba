"""
retriever.py
Hybrid retrieval: combines vector (semantic) search with keyword (BM25-like) search.
Retrieves candidates from all sources, merges and deduplicates results.

v4 improvements (source-aware retrieval):
  - Source priority boost in RRF scoring (gentle advantage for higher-trust sources)
  - Per-source candidate limits (configurable via source_config)
  - Minimum source diversity guarantee (prevents one source from dominating pool)
  - Source type metadata attached to every result
  - Intent-based table pre-filtering (carried forward from v3)
  - page_type filtering for document_chunks (carried forward from v3)
  - phraseto_tsquery + plainto_tsquery combined keyword search
  - Basic synonym expansion for common library terms
"""

import re
import logging
from collections import defaultdict
from typing import List, Optional

import numpy as np

from .database import get_connection
from .embeddings import embed_text
from .source_config import (
    SOURCE_CONFIG, TABLE_TO_SOURCE_TYPE,
    get_source_type, get_source_trust_for_table, is_freshness_sensitive,
    FACULTY_TEXT, SCRAPED_WEBSITE, FACULTY_FAQ,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synonym expansion for keyword search
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "hours": ["opening hours", "schedule", "timing", "open", "close"],
    "borrow": ["loan", "checkout", "check out", "lending"],
    "return": ["return", "give back"],
    "renew": ["renewal", "extend", "extension"],
    "fine": ["penalty", "overdue fee", "late fee"],
    "wifi": ["internet", "wireless", "network"],
    "print": ["printing", "printer", "photocopy", "scan"],
    "reserve": ["reservation", "booking", "book a room"],
    "room": ["study room", "group room", "reading room", "quiet room"],
    "database": ["e-resource", "electronic resource", "online database"],
    "thesis": ["dissertation", "capstone", "research paper"],
    "interlibrary": ["ill", "interlibrary loan", "document delivery"],
    "card": ["id card", "library card", "membership"],
}


def _expand_synonyms(query: str) -> str:
    """Expand query with synonyms for better keyword recall."""
    query_lower = query.lower()
    expansions = []

    for keyword, synonyms in _SYNONYMS.items():
        if keyword in query_lower:
            for syn in synonyms:
                if syn not in query_lower:
                    expansions.append(syn)
                    break  # One expansion per keyword to avoid noise

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


# ---------------------------------------------------------------------------
# Intent-based table + page_type filtering
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

_DB_PATTERN = re.compile(
    r"\b(database|db|recommend|ieee|scopus|pubmed|jstor|proquest|"
    r"web\s+of\s+science|ebsco|e-?resource|journal|article|paper|"
    r"research\s+(source|database|tool)|find\s+articles?|"
    r"search\s+for\s+papers?)\b",
    re.IGNORECASE,
)

_CONTACT_PATTERN = re.compile(
    r"\b(contact|email|phone|call|reach|talk\s+to|help\s+desk|"
    r"librarian|staff|ask\s+a\s+librarian|directions?|location|"
    r"where\s+is|floor|map)\b",
    re.IGNORECASE,
)


def _classify_query_intent(query: str) -> dict:
    """Classify query intent for table/page_type filtering.

    Returns dict with:
        tables: list of tables to prioritize (or None for all)
        page_types: list of page_type values to filter on (for document_chunks)
        intent: string label for logging
    """
    q = query.lower()

    # Database intent: only search databases table (+ FAQ for "how to access" type)
    if _DB_PATTERN.search(q):
        return {
            "tables": ["databases", "faq", "document_chunks"],
            "page_types": ["database_page", "service"],
            "intent": "database",
        }

    # Hours/schedule intent: search all page types — hours info can appear on
    # pages classified as "general", "service", or "hours_contact"
    if _HOURS_PATTERN.search(q):
        return {
            "tables": ["faq", "document_chunks", "library_pages"],
            "page_types": None,
            "intent": "hours",
        }

    # Contact/location intent: also don't restrict page types
    if _CONTACT_PATTERN.search(q):
        return {
            "tables": ["faq", "document_chunks", "library_pages"],
            "page_types": None,
            "intent": "contact",
        }

    # General FAQ-style questions
    if _FAQ_PATTERN.search(q):
        return {
            "tables": ["faq", "document_chunks", "library_pages"],
            "page_types": None,  # search all page types
            "intent": "faq",
        }

    # Default: search everything (including custom_notes)
    return {
        "tables": None,
        "page_types": None,
        "intent": "general",
    }


# ---------------------------------------------------------------------------
# BM25-like keyword scoring (phrase + plain tsquery combined)
# ---------------------------------------------------------------------------

def _keyword_search(
    table: str,
    query: str,
    text_column: str,
    metadata_cols: List[str],
    n_results: int = 20,
    page_type_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Keyword-based search using PostgreSQL full-text search.

    Uses both phraseto_tsquery (exact phrase matching) and plainto_tsquery
    (individual word matching), with phrase matches boosted.
    Falls back to ILIKE if full-text search returns no results.
    """
    cols = ", ".join(metadata_cols)

    # Build page_type filter clause for document_chunks
    type_clause = ""
    type_params = []
    if page_type_filter and table == "document_chunks":
        placeholders = ", ".join(["%s"] * len(page_type_filter))
        type_clause = f"AND page_type IN ({placeholders})"
        type_params = list(page_type_filter)

    # Expand query with synonyms for better recall
    expanded_query = _expand_synonyms(query)

    # Combined: phrase match (boosted 2x) + plain match
    fts_sql = (
        f"SELECT id, "
        f"(ts_rank_cd(to_tsvector('english', {text_column}), phraseto_tsquery('english', %s)) * 2.0 "
        f"+ ts_rank_cd(to_tsvector('english', {text_column}), plainto_tsquery('english', %s))) AS rank, "
        f"{cols} "
        f"FROM {table} "
        f"WHERE (to_tsvector('english', {text_column}) @@ plainto_tsquery('english', %s) "
        f"   OR to_tsvector('english', {text_column}) @@ phraseto_tsquery('english', %s)) "
        f"{type_clause} "
        f"ORDER BY rank DESC "
        f"LIMIT %s"
    )

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                params = [query, expanded_query, expanded_query, query] + type_params + [n_results]
                cur.execute(fts_sql, params)
                rows = cur.fetchall()

                if rows:
                    results = []
                    for row in rows:
                        meta = {col: row[2 + i] for i, col in enumerate(metadata_cols)}
                        results.append({
                            "id": row[0],
                            "keyword_score": float(row[1]),
                            "metadata": meta,
                        })
                    return results

                # Fallback: ILIKE for simple substring matching
                words = [w for w in query.split() if len(w) > 2]
                if not words:
                    return []

                conditions = " OR ".join([f"{text_column} ILIKE %s" for _ in words])
                type_where = ""
                fallback_type_params = []
                if page_type_filter and table == "document_chunks":
                    placeholders = ", ".join(["%s"] * len(page_type_filter))
                    type_where = f"AND page_type IN ({placeholders})"
                    fallback_type_params = list(page_type_filter)

                ilike_sql = (
                    f"SELECT id, {cols} "
                    f"FROM {table} "
                    f"WHERE ({conditions}) {type_where} "
                    f"LIMIT %s"
                )
                params = [f"%{w}%" for w in words] + fallback_type_params + [n_results]
                cur.execute(ilike_sql, params)
                rows = cur.fetchall()

                results = []
                for row in rows:
                    meta = {col: row[1 + i] for i, col in enumerate(metadata_cols)}
                    text_lower = " ".join(str(v) for v in meta.values()).lower()
                    overlap = sum(1 for w in words if w.lower() in text_lower)
                    score = overlap / max(len(words), 1)
                    results.append({
                        "id": row[0],
                        "keyword_score": score,
                        "metadata": meta,
                    })
                return results

    except Exception as e:
        logger.warning(f"Keyword search on {table} failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Vector (semantic) search
# ---------------------------------------------------------------------------

def _vector_search(
    table: str,
    query: str,
    metadata_cols: List[str],
    n_results: int = 20,
    page_type_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Semantic search using pgvector cosine distance.

    Optionally filters by page_type for the document_chunks table.
    """
    try:
        query_embedding = embed_text(query)
        query_vec = np.array(query_embedding)
        cols = ", ".join(metadata_cols)

        # Build page_type filter
        type_clause = ""
        type_params = []
        if page_type_filter and table == "document_chunks":
            placeholders = ", ".join(["%s"] * len(page_type_filter))
            type_clause = f"WHERE page_type IN ({placeholders})"
            type_params = list(page_type_filter)

        if type_clause:
            sql = (
                f"SELECT id, 1 - (embedding <=> %s) AS similarity, {cols} "
                f"FROM {table} "
                f"{type_clause} "
                f"ORDER BY embedding <=> %s "
                f"LIMIT %s"
            )
            params = [query_vec] + type_params + [query_vec, n_results]
        else:
            sql = (
                f"SELECT id, 1 - (embedding <=> %s) AS similarity, {cols} "
                f"FROM {table} "
                f"ORDER BY embedding <=> %s "
                f"LIMIT %s"
            )
            params = [query_vec, query_vec, n_results]

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = []
        for row in rows:
            meta = {col: row[2 + i] for i, col in enumerate(metadata_cols)}
            results.append({
                "id": row[0],
                "vector_score": float(row[1]),
                "metadata": meta,
            })
        return results

    except Exception as e:
        logger.warning(f"Vector search on {table} failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Hybrid merge with Reciprocal Rank Fusion (RRF) + source priority boost
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    vector_results: List[dict],
    keyword_results: List[dict],
    k: int = 60,
    vector_weight: float = 0.65,
    keyword_weight: float = 0.35,
) -> List[dict]:
    """Merge vector and keyword results using weighted Reciprocal Rank Fusion."""
    scores = defaultdict(lambda: {
        "rrf_score": 0.0, "vector_score": 0.0, "keyword_score": 0.0, "metadata": {},
    })

    for rank, result in enumerate(vector_results):
        rid = result["id"]
        scores[rid]["rrf_score"] += vector_weight * (1.0 / (k + rank + 1))
        scores[rid]["vector_score"] = result.get("vector_score", 0.0)
        scores[rid]["metadata"] = result["metadata"]

    for rank, result in enumerate(keyword_results):
        rid = result["id"]
        scores[rid]["rrf_score"] += keyword_weight * (1.0 / (k + rank + 1))
        scores[rid]["keyword_score"] = result.get("keyword_score", 0.0)
        if not scores[rid]["metadata"]:
            scores[rid]["metadata"] = result["metadata"]

    merged = [{"id": rid, **data} for rid, data in scores.items()]
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Table-specific retrieval configurations
# ---------------------------------------------------------------------------

_TABLE_CONFIGS = {
    "faq": {
        "metadata_cols": ["question", "answer"],
        "text_column": "document",
        "source_label": "faq",
    },
    "databases": {
        "metadata_cols": ["name", "description"],
        "text_column": "document",
        "source_label": "database",
    },
    "library_pages": {
        "metadata_cols": ["url", "title", "content"],
        "text_column": "document",
        "source_label": "library_page",
    },
    "document_chunks": {
        "metadata_cols": ["chunk_text", "page_url", "page_title", "section_title", "page_type"],
        "text_column": "chunk_text",
        "source_label": "chunk",
    },
    "custom_notes": {
        "metadata_cols": ["label", "content"],
        "text_column": "document",
        "source_label": "custom_note",
    },
}


# ---------------------------------------------------------------------------
# Source diversity: ensure minimum representation from each source
# ---------------------------------------------------------------------------

def _ensure_source_diversity(
    all_results: List[dict],
    n_final: int,
    min_per_source: int,
) -> List[dict]:
    """Ensure minimum candidates from each source type in the final pool.

    Strategy:
      1. Group results by source_type
      2. Reserve min_per_source slots for each source (if available)
      3. Fill remaining slots from the global ranked list
    This prevents a single source (e.g. FAQ) from crowding out all others.
    """
    if min_per_source <= 0:
        return all_results[:n_final]

    # Group by source type
    by_source: dict = defaultdict(list)
    for r in all_results:
        by_source[r["source_type"]].append(r)

    reserved = []
    reserved_ids = set()

    # Reserve top candidates from each source type
    for source_type, candidates in by_source.items():
        for cand in candidates[:min_per_source]:
            if cand["id"] not in reserved_ids:
                reserved.append(cand)
                reserved_ids.add(cand["id"])

    # Fill remaining from global ranking (skip already-reserved)
    remaining_slots = n_final - len(reserved)
    for r in all_results:
        if remaining_slots <= 0:
            break
        if r["id"] not in reserved_ids:
            reserved.append(r)
            reserved_ids.add(r["id"])
            remaining_slots -= 1

    # Sort final pool by boosted RRF score
    reserved.sort(key=lambda x: x["rrf_score"], reverse=True)
    return reserved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    tables: Optional[List[str]] = None,
    n_vector: int = 20,
    n_keyword: int = 15,
    n_final: int = 30,
    page_type_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Perform source-aware hybrid retrieval across specified tables.

    Each result includes:
      - id, rrf_score, vector_score, keyword_score
      - metadata (table-specific fields)
      - source_table (DB table name)
      - source_label (human-readable)
      - source_type (logical source: faculty_text, scraped_website, faculty_faq, databases)
      - source_trust (trust score from config)

    Source priority is applied as a gentle boost to RRF scores, so higher-trust
    sources get a small advantage without overriding strong relevance signals.
    """
    cfg = SOURCE_CONFIG

    if tables is None:
        tables = ["faq", "databases", "library_pages", "document_chunks", "custom_notes"]

    # Verify which tables actually exist and have data
    valid_tables = []
    for table in tables:
        if table not in _TABLE_CONFIGS:
            continue
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    if count > 0:
                        valid_tables.append(table)
                    else:
                        logger.info(f"Skipping table '{table}': 0 rows")
        except Exception as e:
            logger.warning(f"Skipping table '{table}': {e}")
            continue

    # Check if query is time-sensitive (for freshness boost on scraped content)
    freshness_sensitive = is_freshness_sensitive(query)

    all_results = []

    for table in valid_tables:
        config = _TABLE_CONFIGS[table]
        source_type = get_source_type(table)
        source_trust = get_source_trust_for_table(table)

        # Per-source candidate limits from config
        table_n_vector = cfg.per_source_n_vector.get(source_type, n_vector)
        table_n_keyword = cfg.per_source_n_keyword.get(source_type, n_keyword)

        # page_type filter only applies to document_chunks
        pt_filter = page_type_filter if table == "document_chunks" else None

        vector_results = _vector_search(
            table, query, config["metadata_cols"], table_n_vector,
            page_type_filter=pt_filter,
        )
        keyword_results = _keyword_search(
            table, query, config["text_column"], config["metadata_cols"],
            table_n_keyword, page_type_filter=pt_filter,
        )

        merged = _reciprocal_rank_fusion(vector_results, keyword_results)

        # Apply source priority boost to RRF scores.
        # This is a small additive boost proportional to source trust,
        # giving higher-priority sources a gentle advantage.
        source_boost = cfg.rrf_source_boost_weight * source_trust

        # Extra boost for scraped content on time-sensitive queries
        if freshness_sensitive and source_type == SCRAPED_WEBSITE:
            source_boost += cfg.freshness_boost * cfg.rrf_source_boost_weight

        for result in merged:
            result["source_table"] = table
            result["source_label"] = config["source_label"]
            result["source_type"] = source_type
            result["source_trust"] = source_trust
            result["rrf_score"] += source_boost

        all_results.extend(merged)

    # Sort by boosted RRF score across all tables
    all_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Ensure source diversity: don't let one source crowd out the pool
    final_results = _ensure_source_diversity(
        all_results,
        n_final=cfg.n_final_candidates,
        min_per_source=cfg.min_candidates_per_source,
    )

    # Log source distribution for debugging
    source_counts = defaultdict(int)
    for r in final_results:
        source_counts[r["source_type"]] += 1
    logger.info(
        f"Hybrid retrieval: {len(all_results)} total candidates from {valid_tables}, "
        f"returning {len(final_results)} | source distribution: {dict(source_counts)}"
        + (f" | page_type_filter={page_type_filter}" if page_type_filter else "")
        + (" | freshness_boost=ON" if freshness_sensitive else "")
    )

    return final_results


def classify_query_intent(query: str) -> dict:
    """Public API for query intent classification. Used by chatbot.py."""
    return _classify_query_intent(query)


def vector_only_retrieve(
    query: str,
    table: str,
    metadata_cols: List[str],
    n_results: int = 5,
) -> List[dict]:
    """Pure vector search for a single table (backward compatibility)."""
    return _vector_search(table, query, metadata_cols, n_results)
