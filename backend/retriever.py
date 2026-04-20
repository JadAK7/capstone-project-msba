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
from .intent_classifier import classify_intent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synonym expansion for keyword search
# ---------------------------------------------------------------------------

_SYNONYMS_EN = {
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

_SYNONYMS_AR = {
    "\u0633\u0627\u0639\u0627\u062a": ["\u0645\u0648\u0627\u0639\u064a\u062f", "\u0623\u0648\u0642\u0627\u062a", "\u062f\u0648\u0627\u0645"],  # ساعات → مواعيد، أوقات، دوام
    "\u0627\u0633\u062a\u0639\u0627\u0631\u0629": ["\u0625\u0639\u0627\u0631\u0629", "\u0627\u0633\u062a\u0644\u0627\u0641"],  # استعارة → إعارة، استلاف
    "\u0625\u0631\u062c\u0627\u0639": ["\u0631\u062f", "\u0625\u0639\u0627\u062f\u0629"],  # إرجاع → رد، إعادة
    "\u062a\u062c\u062f\u064a\u062f": ["\u062a\u0645\u062f\u064a\u062f"],  # تجديد → تمديد
    "\u063a\u0631\u0627\u0645\u0629": ["\u063a\u0631\u0627\u0645\u0627\u062a", "\u0631\u0633\u0648\u0645 \u062a\u0623\u062e\u064a\u0631"],  # غرامة → غرامات، رسوم تأخير
    "\u0648\u0627\u064a \u0641\u0627\u064a": ["\u0627\u0646\u062a\u0631\u0646\u062a", "\u0625\u0646\u062a\u0631\u0646\u062a", "\u0634\u0628\u0643\u0629"],  # واي فاي → انترنت، إنترنت، شبكة
    "\u0637\u0628\u0627\u0639\u0629": ["\u0637\u0628\u0639", "\u0645\u0633\u062d", "\u062a\u0635\u0648\u064a\u0631"],  # طباعة → طبع، مسح، تصوير
    "\u062d\u062c\u0632": ["\u062d\u062c\u0648\u0632\u0627\u062a", "\u062d\u062c\u0632 \u063a\u0631\u0641\u0629"],  # حجز → حجوزات، حجز غرفة
    "\u063a\u0631\u0641\u0629": ["\u063a\u0631\u0641\u0629 \u062f\u0631\u0627\u0633\u0629", "\u0642\u0627\u0639\u0629", "\u063a\u0631\u0641\u0629 \u0645\u0637\u0627\u0644\u0639\u0629"],  # غرفة → غرفة دراسة، قاعة، غرفة مطالعة
    "\u0642\u0627\u0639\u062f\u0629 \u0628\u064a\u0627\u0646\u0627\u062a": ["\u0642\u0648\u0627\u0639\u062f \u0628\u064a\u0627\u0646\u0627\u062a", "\u0645\u0635\u0627\u062f\u0631 \u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a\u0629"],  # قاعدة بيانات → قواعد بيانات، مصادر إلكترونية
    "\u0631\u0633\u0627\u0644\u0629": ["\u0623\u0637\u0631\u0648\u062d\u0629", "\u0628\u062d\u062b", "\u0631\u0633\u0627\u0626\u0644"],  # رسالة → أطروحة، بحث، رسائل
    "\u0628\u0637\u0627\u0642\u0629": ["\u0628\u0637\u0627\u0642\u0629 \u0645\u0643\u062a\u0628\u0629", "\u0647\u0648\u064a\u0629", "\u0639\u0636\u0648\u064a\u0629"],  # بطاقة → بطاقة مكتبة، هوية، عضوية
}

# Regex to detect Arabic characters in the query
_ARABIC_CHARS_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def _expand_synonyms(query: str) -> str:
    """Expand query with synonyms for better keyword recall.

    Supports both English and Arabic. For Arabic queries, expansions are
    drawn from the Arabic synonym table. For English, from the English one.
    If the query is mixed, both tables are checked.
    """
    query_lower = query.lower()
    expansions = []

    has_arabic = bool(_ARABIC_CHARS_RE.search(query))

    # English synonyms
    for keyword, synonyms in _SYNONYMS_EN.items():
        if keyword in query_lower:
            for syn in synonyms:
                if syn not in query_lower:
                    expansions.append(syn)
                    break

    # Arabic synonyms
    if has_arabic:
        for keyword, synonyms in _SYNONYMS_AR.items():
            if keyword in query:
                for syn in synonyms:
                    if syn not in query:
                        expansions.append(syn)
                        break

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query



# Intent classification is now handled by intent_classifier.py


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

    # Use 'simple' text search config for Arabic (no stemming, but proper tokenization).
    # The 'english' config doesn't handle Arabic script at all.
    has_arabic = bool(_ARABIC_CHARS_RE.search(query))
    fts_config = "simple" if has_arabic else "english"

    # Combined: phrase match (boosted 2x) + plain match
    fts_sql = (
        f"SELECT id, "
        f"(ts_rank_cd(to_tsvector('{fts_config}', {text_column}), phraseto_tsquery('{fts_config}', %s)) * 2.0 "
        f"+ ts_rank_cd(to_tsvector('{fts_config}', {text_column}), plainto_tsquery('{fts_config}', %s))) AS rank, "
        f"{cols} "
        f"FROM {table} "
        f"WHERE (to_tsvector('{fts_config}', {text_column}) @@ plainto_tsquery('{fts_config}', %s) "
        f"   OR to_tsvector('{fts_config}', {text_column}) @@ phraseto_tsquery('{fts_config}', %s)) "
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
    query_vec: Optional[np.ndarray] = None,
) -> List[dict]:
    """Semantic search using pgvector cosine distance.

    Optionally filters by page_type for the document_chunks table.
    If query_vec is provided, skips embedding generation (avoids redundant API calls).
    """
    try:
        if query_vec is None:
            query_embedding = embed_text(query)
            query_vec = np.array(query_embedding)
        else:
            query_vec = np.array(query_vec)
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
    query_embedding: Optional[List[float]] = None,
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

    # Embed query ONCE and reuse across all table searches
    if query_embedding is None:
        query_embedding = embed_text(query)
    query_vec = np.array(query_embedding)

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
            query_vec=query_vec,
        )
        keyword_results = _keyword_search(
            table, query, config["text_column"], config["metadata_cols"],
            table_n_keyword, page_type_filter=pt_filter,
        )

        merged = _reciprocal_rank_fusion(vector_results, keyword_results)

        # Source trust boosts are applied once, at reranking only (reranker.py).
        # RRF scores here are pure retrieval-quality signals.
        for result in merged:
            result["source_table"] = table
            result["source_label"] = config["source_label"]
            result["source_type"] = source_type
            result["source_trust"] = source_trust
            result["freshness_sensitive"] = freshness_sensitive

        all_results.extend(merged)

    # Sort by RRF score across all tables
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
    """Public API for query intent classification. Delegates to intent_classifier."""
    return classify_intent(query)


def vector_only_retrieve(
    query: str,
    table: str,
    metadata_cols: List[str],
    n_results: int = 5,
) -> List[dict]:
    """Pure vector search for a single table (backward compatibility)."""
    return _vector_search(table, query, metadata_cols, n_results)
