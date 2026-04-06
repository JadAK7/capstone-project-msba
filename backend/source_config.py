"""
source_config.py
Centralized configuration for multi-source RAG retrieval with source priority.

Source taxonomy:
  - faculty_text   (custom_notes table)  — Highest trust: curated by library faculty
  - scraped_website (document_chunks + library_pages) — Second: live website content
  - faculty_faq    (faq table)           — Lowest: short FAQ pairs, useful as fallback
  - databases      (databases table)     — Separate: routed via intent detection, not ranked

All tuning knobs for source-aware retrieval live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Source type enum — maps DB tables to logical source types
# ---------------------------------------------------------------------------

# Canonical source type names
FACULTY_TEXT = "faculty_text"
SCRAPED_WEBSITE = "scraped_website"
FACULTY_FAQ = "faculty_faq"
DATABASES = "databases"

# Map each DB table to its logical source type
TABLE_TO_SOURCE_TYPE: Dict[str, str] = {
    "custom_notes": FACULTY_TEXT,
    "document_chunks": SCRAPED_WEBSITE,
    "library_pages": SCRAPED_WEBSITE,
    "faq": FACULTY_FAQ,
    "databases": DATABASES,
}

# Reverse: which tables belong to each source type
SOURCE_TYPE_TO_TABLES: Dict[str, List[str]] = {
    FACULTY_TEXT: ["custom_notes"],
    SCRAPED_WEBSITE: ["document_chunks", "library_pages"],
    FACULTY_FAQ: ["faq"],
    DATABASES: ["databases"],
}


# ---------------------------------------------------------------------------
# Source priority configuration
# ---------------------------------------------------------------------------

@dataclass
class SourcePriorityConfig:
    """All tuning knobs for source-aware retrieval and ranking.

    Adjust these values to change how aggressively the system prefers
    higher-priority sources over lower-priority ones.
    """

    # --- Source trust tiers (higher = more trusted) ---
    # Used to compute source priority boosts in retrieval and reranking.
    # Scale: 0.0 to 1.0.  The *difference* between tiers matters more
    # than the absolute values.
    source_trust: Dict[str, float] = field(default_factory=lambda: {
        FACULTY_TEXT: 1.0,       # Highest trust
        SCRAPED_WEBSITE: 0.7,    # Second
        FACULTY_FAQ: 0.4,        # Lowest (still useful as fallback)
        DATABASES: 0.8,          # Separate track, but decent trust
    })

    # --- RRF source boost ---
    # After RRF merge, each result's score is boosted by:
    #   rrf_score += rrf_source_boost_weight * source_trust[source_type]
    # This gives higher-priority sources a gentle advantage in the
    # candidate pool without overriding strong relevance signals.
    rrf_source_boost_weight: float = 0.005

    # --- Per-source candidate limits ---
    # How many candidates to retrieve per source from vector/keyword search.
    # Higher values for lower-trust sources ensure they still appear in the
    # pool when they're genuinely the best match.
    per_source_n_vector: Dict[str, int] = field(default_factory=lambda: {
        FACULTY_TEXT: 20,
        SCRAPED_WEBSITE: 25,  # More candidates because chunks are numerous
        FACULTY_FAQ: 15,
        DATABASES: 15,
    })
    per_source_n_keyword: Dict[str, int] = field(default_factory=lambda: {
        FACULTY_TEXT: 15,
        SCRAPED_WEBSITE: 20,
        FACULTY_FAQ: 12,
        DATABASES: 12,
    })

    # --- Reranker source boost ---
    # After LLM reranking, the rerank_score is adjusted:
    #   final_score = rerank_score + rerank_source_boost_weight * source_trust[source_type]
    # faculty_text gets: +0.15 * 1.0 = +0.15
    # scraped gets:      +0.15 * 0.7 = +0.105
    # faq gets:          +0.15 * 0.4 = +0.06
    # Net advantage of faculty_text over scraped: ~0.045
    rerank_source_boost_weight: float = 0.15

    # --- Source precedence threshold ---
    # When comparing best scores across sources for final selection:
    # A higher-priority source "wins" if its best score is within this
    # margin of the lower-priority source's best score.
    # Example: faculty_text scores 0.65, scraped scores 0.80 → gap 0.15
    # If precedence_margin = 0.15, faculty_text still wins.
    precedence_margin: float = 0.15

    # --- Minimum source diversity ---
    # Before reranking, ensure at least this many candidates from each
    # source type (if available). Prevents one source from dominating
    # the candidate pool entirely.
    min_candidates_per_source: int = 2

    # --- Global candidate pool size ---
    n_final_candidates: int = 30  # Total candidates after RRF merge
    rerank_top_k: int = 8         # Candidates to keep after reranking
    rerank_min_score: float = 0.45

    # --- Confidence thresholds (for answer generation) ---
    confident_threshold: float = 0.60
    partial_threshold: float = 0.45

    # --- Faculty text (custom_notes) vector floor ---
    # Custom notes are admin-curated answers. The LLM reranker scores "evidence
    # support" — but a note like "refer to this link" has low evidence even
    # though it IS the intended answer. When a custom note is topically relevant
    # (high vector similarity), we use the vector score as a floor so the
    # reranker can't kill it.
    #
    # Logic: if source_type == faculty_text and vector_score >= threshold:
    #   rerank_score = max(rerank_score, vector_score * multiplier)
    #
    # Example: vector=0.74, rerank=0.40, multiplier=1.1 → floor=0.81, final=0.81+boost
    faculty_text_vector_floor_threshold: float = 0.60
    faculty_text_vector_floor_multiplier: float = 1.1

    # --- FAQ-specific controls ---
    # FAQ entries are short and clean, which inflates their similarity scores.
    # This dampening factor is applied to FAQ rerank scores to compensate.
    # 1.0 = no dampening, 0.9 = 10% reduction.
    faq_score_dampening: float = 0.92

    # --- Freshness boost for scraped content ---
    # When the query is about dynamic/time-sensitive info (hours, events),
    # scraped content gets an extra boost since it's from the live website.
    freshness_boost: float = 0.05
    freshness_query_patterns: List[str] = field(default_factory=lambda: [
        "hours", "open", "close", "schedule", "timing", "when",
        "current", "today", "now", "event", "upcoming",
    ])


# Singleton config — import and use throughout the app
SOURCE_CONFIG = SourcePriorityConfig()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_source_type(table_name: str) -> str:
    """Get the logical source type for a given DB table name."""
    return TABLE_TO_SOURCE_TYPE.get(table_name, "unknown")


def get_source_trust(source_type: str) -> float:
    """Get the trust score for a source type."""
    return SOURCE_CONFIG.source_trust.get(source_type, 0.5)


def get_source_trust_for_table(table_name: str) -> float:
    """Get the trust score for a DB table."""
    source_type = get_source_type(table_name)
    return get_source_trust(source_type)


def is_freshness_sensitive(query: str) -> bool:
    """Check if a query is about time-sensitive information."""
    q_lower = query.lower()
    return any(pattern in q_lower for pattern in SOURCE_CONFIG.freshness_query_patterns)
