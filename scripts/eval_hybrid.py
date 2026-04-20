#!/usr/bin/env python3
"""
eval_hybrid.py
Compares hybrid retrieval (vector + keyword + RRF) vs. vector-only retrieval.

For 15 queries containing specific library terms, runs both modes and compares:
  - Whether a key term appears in the top-5 results (term-match recall@5)
  - Number of unique results added by the keyword path
  - Top-1 vector score vs. hybrid RRF score

Backs up: "Hybrid retrieval outperforms pure vector search: adding keyword
search with synonym expansion improved recall for specific library terms"

Usage:
    python scripts/eval_hybrid.py
    python scripts/eval_hybrid.py --output results/hybrid.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.embeddings import embed_text
from backend.retriever import _vector_search, _keyword_search, _reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Test queries — chosen to benefit from keyword matching (specific named terms)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {"query": "ILL interlibrary loan request process",    "key_term": "interlibrary", "table": "faq"},
    {"query": "Jafet library schedule and hours",         "key_term": "jafet",        "table": "document_chunks"},
    {"query": "Saab Medical library location access",     "key_term": "saab",         "table": "document_chunks"},
    {"query": "PubMed biomedical database articles",      "key_term": "pubmed",       "table": "databases"},
    {"query": "IEEE Xplore engineering research papers",  "key_term": "ieee",         "table": "databases"},
    {"query": "JSTOR humanities social sciences archive", "key_term": "jstor",        "table": "databases"},
    {"query": "overdue fine penalty late return book",    "key_term": "overdue",      "table": "faq"},
    {"query": "Endnote citation reference manager",      "key_term": "endnote",      "table": "faq"},
    {"query": "ABI INFORM business management database", "key_term": "abi",          "table": "databases"},
    {"query": "Scopus research citations impact factor",  "key_term": "scopus",       "table": "databases"},
    {"query": "ProQuest dissertations theses database",   "key_term": "proquest",     "table": "databases"},
    {"query": "alumni library card membership benefits",  "key_term": "alumni",       "table": "faq"},
    {"query": "thesis dissertation submission library",   "key_term": "thesis",       "table": "faq"},
    {"query": "wifi wireless internet connection campus", "key_term": "wifi",         "table": "faq"},
    {"query": "photocopy scan printer document service",  "key_term": "scan",         "table": "faq"},
]

_TABLE_METADATA = {
    "faq":            ["question", "answer"],
    "databases":      ["name", "description"],
    "document_chunks": ["chunk_text", "page_url", "page_title", "section_title", "page_type"],
}

_TABLE_TEXT_COL = {
    "faq":            "document",
    "databases":      "document",
    "document_chunks": "chunk_text",
}


def _has_key_term(results: list, term: str, meta_keys: list, top_k: int = 5) -> bool:
    """Check if any of the top_k results contain the key term."""
    term_lower = term.lower()
    for r in results[:top_k]:
        meta = r.get("metadata", {})
        for key in meta_keys:
            if term_lower in str(meta.get(key, "")).lower():
                return True
    return False


def run_hybrid_eval(output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("HYBRID vs. VECTOR-ONLY RETRIEVAL COMPARISON")
    print("=" * 70)
    print(f"Test queries: {len(TEST_QUERIES)} | Metric: key-term recall@5\n")

    results = []
    hybrid_only_hits = 0   # hybrid finds term, vector misses
    vector_only_hits = 0   # vector finds term, hybrid misses
    both_hit = 0
    neither_hit = 0
    total_new_results = 0

    for item in TEST_QUERIES:
        query = item["query"]
        key_term = item["key_term"]
        table = item["table"]
        meta_cols = _TABLE_METADATA[table]
        text_col = _TABLE_TEXT_COL[table]

        # Embed once, reuse for both modes
        query_vec = np.array(embed_text(query))

        # --- Vector-only ---
        t0 = time.time()
        vec_results = _vector_search(
            table=table, query=query, metadata_cols=meta_cols,
            n_results=10, query_vec=query_vec,
        )
        vec_time_ms = (time.time() - t0) * 1000

        # --- Hybrid (vector + keyword merged via RRF 65/35) ---
        t0 = time.time()
        kw_results = _keyword_search(
            table=table, query=query, text_column=text_col,
            metadata_cols=meta_cols, n_results=10,
        )
        merged = _reciprocal_rank_fusion(vec_results, kw_results,
                                         vector_weight=0.65, keyword_weight=0.35)
        hybrid_time_ms = (time.time() - t0) * 1000

        # Metrics
        vec_top_score = vec_results[0]["vector_score"] if vec_results else 0.0
        hybrid_top_rrf = merged[0]["rrf_score"] if merged else 0.0

        vec_ids = {r["id"] for r in vec_results[:10]}
        hybrid_ids = {r["id"] for r in merged[:10]}
        new_in_hybrid = len(hybrid_ids - vec_ids)
        total_new_results += new_in_hybrid

        vec_hit = _has_key_term(vec_results, key_term, meta_cols, top_k=5)
        hyb_hit = _has_key_term(merged, key_term, meta_cols, top_k=5)

        if hyb_hit and not vec_hit:
            winner = "hybrid"
            hybrid_only_hits += 1
        elif vec_hit and not hyb_hit:
            winner = "vector"
            vector_only_hits += 1
        elif hyb_hit and vec_hit:
            winner = "both"
            both_hit += 1
        else:
            winner = "neither"
            neither_hit += 1

        result = {
            "query": query,
            "key_term": key_term,
            "table": table,
            "vector_top_score": round(vec_top_score, 4),
            "hybrid_top_rrf_score": round(hybrid_top_rrf, 4),
            "vector_has_term_in_top5": vec_hit,
            "hybrid_has_term_in_top5": hyb_hit,
            "new_results_added_by_keyword": new_in_hybrid,
            "winner": winner,
            "vector_time_ms": round(vec_time_ms, 1),
            "hybrid_time_ms": round(hybrid_time_ms, 1),
            "keyword_results_count": len(kw_results),
        }
        results.append(result)

        v_mark = "[FOUND]" if vec_hit else "[MISS] "
        h_mark = "[FOUND]" if hyb_hit else "[MISS] "
        print(f"  {query[:48]!r}  key={key_term!r}")
        print(f"    Vector:  score={vec_top_score:.3f} {v_mark}")
        print(f"    Hybrid:  rrf  ={hybrid_top_rrf:.3f} {h_mark}  "
              f"+{new_in_hybrid} new results from keyword path")
        print(f"    Winner: {winner.upper()}")

    # Aggregates
    total = len(results)
    hybrid_recall = (hybrid_only_hits + both_hit) / total
    vector_recall = (vector_only_hits + both_hit) / total
    recall_improvement = hybrid_recall - vector_recall
    avg_new = total_new_results / total

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<42} {'Value':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Hybrid recall@5 (key term in top 5)':<42} {hybrid_recall:>9.1%}")
    print(f"  {'Vector-only recall@5':<42} {vector_recall:>9.1%}")
    print(f"  {'Recall improvement from hybrid':<42} {recall_improvement:>+9.1%}")
    print(f"  {'Hybrid-only wins (hybrid finds, vector misses)':<42} {hybrid_only_hits:>10}")
    print(f"  {'Vector-only wins':<42} {vector_only_hits:>10}")
    print(f"  {'Both found the term':<42} {both_hit:>10}")
    print(f"  {'Neither found the term':<42} {neither_hit:>10}")
    print(f"  {'Avg new results added by keyword path':<42} {avg_new:>9.1f}")
    print()
    print(f"Conclusion: Hybrid retrieval achieves {hybrid_recall:.0%} term-match recall@5 "
          f"vs. {vector_recall:.0%} for vector-only,")
    print(f"a {recall_improvement:+.0%} improvement. The keyword path added an average of "
          f"{avg_new:.1f} new candidates per query.")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_queries": total,
            "rrf_weights": {"vector": 0.65, "keyword": 0.35},
            "recall_k": 5,
        },
        "aggregate": {
            "hybrid_recall_at_5": round(hybrid_recall, 4),
            "vector_recall_at_5": round(vector_recall, 4),
            "recall_improvement": round(recall_improvement, 4),
            "hybrid_only_wins": hybrid_only_hits,
            "vector_only_wins": vector_only_hits,
            "both_found": both_hit,
            "neither_found": neither_hit,
            "avg_new_results_from_keyword": round(avg_new, 2),
        },
        "per_query": results,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"eval_hybrid_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare hybrid vs. vector-only retrieval")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    init_db()
    run_hybrid_eval(output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
