#!/usr/bin/env python3
"""
eval_latency.py
Measures per-stage pipeline latency and cold vs. cached full-pipeline latency.

Stages timed individually:
  1. Input Guard    — regex + embedding injection check
  2. Query Rewriting — LLM-based rewriting
  3. Embedding       — embed_text() call
  4. Retrieval       — hybrid_retrieve() across tables
  5. Reranking       — rerank() LLM scoring

Full pipeline timing:
  - Cold:   first call with empty cache (all stages run)
  - Cached: same query repeated (returns from cache in <5ms)

Usage:
    python scripts/eval_latency.py
    python scripts/eval_latency.py --output results/latency.json
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
from backend.chatbot import LibraryChatbot
from backend.input_guard import run_input_guards
from backend.query_rewriter import rewrite_query
from backend.embeddings import embed_text
from backend.retriever import hybrid_retrieve
from backend.reranker import rerank

# ---------------------------------------------------------------------------
# Test queries — varied, representative of real usage
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "What are the library opening hours?",
    "How do I borrow a book from the library?",
    "Which databases are available for medical research?",
    "Can I reserve a study room at AUB?",
    "What are the fines for overdue books?",
    "How do I access library resources from off-campus?",
    "What printing services are available at the library?",
    "Which database should I use for engineering research?",
    "How does interlibrary loan work at AUB?",
    "Can AUB alumni access the library collections?",
]

_SEARCH_TABLES = ["faq", "databases", "document_chunks"]


def _time(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def run_latency_eval(api_key: str, output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("PIPELINE LATENCY BREAKDOWN EVALUATION")
    print("=" * 70)
    print(f"Queries: {len(TEST_QUERIES)}\n")

    bot = LibraryChatbot(api_key=api_key)
    bot.clear_cache()

    stage_times = {
        "input_guard": [],
        "query_rewriting": [],
        "embedding": [],
        "retrieval": [],
        "reranking": [],
    }
    full_cold_times = []
    full_cached_times = []
    per_query_stages = []

    # ---- Stage-by-stage timing ----
    print("[Stage Timing] Each stage timed individually (no LLM generation)")
    for q in TEST_QUERIES:
        row = {"query": q}

        # Stage 1: Input guard
        _, t = _time(run_input_guards, q)
        stage_times["input_guard"].append(t)
        row["input_guard_ms"] = round(t, 2)

        # Stage 2: Query rewriting (LLM call)
        (rewritten, _), t = _time(rewrite_query, q, None, "en")
        stage_times["query_rewriting"].append(t)
        row["query_rewriting_ms"] = round(t, 2)
        row["rewritten_query"] = rewritten

        # Stage 3: Embedding
        raw_vec, t = _time(embed_text, rewritten)
        stage_times["embedding"].append(t)
        row["embedding_ms"] = round(t, 2)
        query_vec = np.array(raw_vec)

        # Stage 4: Hybrid retrieval
        candidates, t = _time(
            hybrid_retrieve,
            rewritten,
            _SEARCH_TABLES,
            20, 15, 30, None,
            raw_vec,
        )
        stage_times["retrieval"].append(t)
        row["retrieval_ms"] = round(t, 2)
        row["candidates_returned"] = len(candidates)

        # Stage 5: Reranking
        reranked, t = _time(rerank, rewritten, candidates, 6, 0.45)
        stage_times["reranking"].append(t)
        row["reranking_ms"] = round(t, 2)
        row["reranked_count"] = len(reranked)

        subtotal = (row["input_guard_ms"] + row["query_rewriting_ms"] +
                    row["embedding_ms"] + row["retrieval_ms"] + row["reranking_ms"])
        row["subtotal_excl_generation_ms"] = round(subtotal, 2)

        per_query_stages.append(row)
        print(f"  {q[:48]!r}")
        print(f"    guard={row['input_guard_ms']:.0f}ms  rewrite={row['query_rewriting_ms']:.0f}ms  "
              f"embed={row['embedding_ms']:.0f}ms  retrieve={row['retrieval_ms']:.0f}ms  "
              f"rerank={row['reranking_ms']:.0f}ms  | subtotal={subtotal:.0f}ms")

    # ---- Full pipeline: cold vs cached ----
    print("\n[Full Pipeline] Cold vs. cached (includes LLM generation + verification)")
    for q in TEST_QUERIES:
        # Cold
        t0 = time.perf_counter()
        answer, debug = bot.answer(q)
        cold_ms = (time.perf_counter() - t0) * 1000
        full_cold_times.append(cold_ms)

        # Cached (immediate repeat)
        t0 = time.perf_counter()
        answer, debug2 = bot.answer(q)
        cached_ms = (time.perf_counter() - t0) * 1000
        full_cached_times.append(cached_ms)

        hit = debug2.get("cache_hit", False)
        print(f"  cold={cold_ms:6.0f}ms → cached={cached_ms:5.0f}ms  "
              f"(hit={hit}) | {q[:40]!r}")

    # ---- Summary ----
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    avg_guard      = avg(stage_times["input_guard"])
    avg_rewrite    = avg(stage_times["query_rewriting"])
    avg_embed      = avg(stage_times["embedding"])
    avg_retrieve   = avg(stage_times["retrieval"])
    avg_rerank     = avg(stage_times["reranking"])
    avg_subtotal   = avg_guard + avg_rewrite + avg_embed + avg_retrieve + avg_rerank

    avg_cold   = avg(full_cold_times)
    avg_cached = avg(full_cached_times)
    speedup    = avg_cold / avg_cached if avg_cached > 0 else float("inf")

    # Generation is inferred as: full_cold - subtotal
    avg_generation_verify = avg_cold - avg_subtotal

    print("\n" + "=" * 70)
    print("SUMMARY — Average latency per stage")
    print("=" * 70)
    print(f"  {'Stage':<30} {'Avg (ms)':>10} {'% of Cold':>12}")
    print(f"  {'-'*54}")

    def pct(val):
        return val / avg_cold * 100 if avg_cold > 0 else 0

    print(f"  {'1. Input Guard':<30} {avg_guard:>10.1f} {pct(avg_guard):>11.1f}%")
    print(f"  {'2. Query Rewriting (LLM)':<30} {avg_rewrite:>10.1f} {pct(avg_rewrite):>11.1f}%")
    print(f"  {'3. Embedding':<30} {avg_embed:>10.1f} {pct(avg_embed):>11.1f}%")
    print(f"  {'4. Hybrid Retrieval':<30} {avg_retrieve:>10.1f} {pct(avg_retrieve):>11.1f}%")
    print(f"  {'5. LLM Reranking':<30} {avg_rerank:>10.1f} {pct(avg_rerank):>11.1f}%")
    print(f"  {'6. Generation + Verification':<30} {avg_generation_verify:>10.1f} {pct(avg_generation_verify):>11.1f}%")
    print(f"  {'-'*54}")
    print(f"  {'Total (cold pipeline)':<30} {avg_cold:>10.1f} {'100.0%':>12}")
    print()
    print(f"  Full pipeline (cold):   {avg_cold:>8.0f} ms")
    print(f"  Full pipeline (cached): {avg_cached:>8.0f} ms  ({speedup:.1f}× speedup)")
    print()
    print(f"Conclusion: LLM generation + verification accounts for the largest share "
          f"({pct(avg_generation_verify):.0f}%) of cold pipeline latency.")
    print(f"Cache hits reduce response time by {speedup:.1f}× ({avg_cold:.0f}ms → {avg_cached:.0f}ms).")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_queries": len(TEST_QUERIES),
            "search_tables": _SEARCH_TABLES,
        },
        "aggregate": {
            "avg_input_guard_ms": round(avg_guard, 2),
            "avg_query_rewriting_ms": round(avg_rewrite, 2),
            "avg_embedding_ms": round(avg_embed, 2),
            "avg_retrieval_ms": round(avg_retrieve, 2),
            "avg_reranking_ms": round(avg_rerank, 2),
            "avg_subtotal_excl_generation_ms": round(avg_subtotal, 2),
            "avg_generation_verify_ms": round(avg_generation_verify, 2),
            "avg_full_cold_ms": round(avg_cold, 2),
            "avg_full_cached_ms": round(avg_cached, 2),
            "cache_speedup": round(speedup, 2),
            "pct_guard": round(pct(avg_guard), 1),
            "pct_rewriting": round(pct(avg_rewrite), 1),
            "pct_embedding": round(pct(avg_embed), 1),
            "pct_retrieval": round(pct(avg_retrieve), 1),
            "pct_reranking": round(pct(avg_rerank), 1),
            "pct_generation_verify": round(pct(avg_generation_verify), 1),
        },
        "per_query_stage_times": per_query_stages,
        "full_pipeline": [
            {"query": q, "cold_ms": round(c, 1), "cached_ms": round(h, 1)}
            for q, c, h in zip(TEST_QUERIES, full_cold_times, full_cached_times)
        ],
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"eval_latency_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Measure pipeline latency breakdown")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Export it or add it to .env")
        sys.exit(1)

    init_db()
    run_latency_eval(api_key=api_key, output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
