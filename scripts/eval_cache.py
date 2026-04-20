#!/usr/bin/env python3
"""
eval_cache.py
Measures response cache effectiveness: hit rates and latency savings.

Three rounds:
  Round 1 — Cold:       cache is empty, all queries miss → baseline latency
  Round 2 — Exact:      same queries repeated → should all hit exact cache
  Round 3 — Semantic:   paraphrased versions → tests semantic similarity cache (≥0.95)

Backs up: "Caching reduces latency and API cost: repeated/similar queries hit
the in-memory cache, avoiding redundant embedding + LLM calls"

Usage:
    python scripts/eval_cache.py
    python scripts/eval_cache.py --output results/cache.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot

# ---------------------------------------------------------------------------
# 10 query pairs: (original, paraphrase)
# Paraphrases should be close enough to trigger semantic cache (cosine ≥ 0.95)
# ---------------------------------------------------------------------------

QUERY_PAIRS = [
    (
        "What are the library opening hours?",
        "When is the library open?",
    ),
    (
        "How do I borrow a book from the library?",
        "What is the process for borrowing books?",
    ),
    (
        "What are the fines for overdue books?",
        "How much do I pay for returning books late?",
    ),
    (
        "Can I reserve a study room?",
        "How do I book a group study room?",
    ),
    (
        "How do I renew a borrowed book?",
        "Can I extend my book loan period?",
    ),
    (
        "Can I access databases from off-campus?",
        "Is remote access to library resources possible?",
    ),
    (
        "What printing services are available at the library?",
        "Where can I print documents in the library?",
    ),
    (
        "How do I request an interlibrary loan?",
        "What is the ILL service at the AUB library?",
    ),
    (
        "How many books can I borrow at once?",
        "What is the borrowing limit for students?",
    ),
    (
        "Which databases are available for medical research?",
        "What medical databases does AUB library provide?",
    ),
]


def _run_round(bot: LibraryChatbot, queries: list, label: str) -> list:
    """Send a list of queries and collect latency + cache hit info."""
    print(f"\n[Round: {label}]")
    round_results = []
    for q in queries:
        t0 = time.time()
        answer, debug = bot.answer(q)
        elapsed_ms = (time.time() - t0) * 1000

        cache_hit = debug.get("cache_hit", False)
        semantic_hit = debug.get("semantic_cache_hit", False)

        label_str = "SEMANTIC HIT" if semantic_hit else ("HIT    " if cache_hit else "MISS   ")
        print(f"  [{elapsed_ms:6.0f}ms] {label_str} | {q[:52]!r}")

        round_results.append({
            "query": q,
            "latency_ms": round(elapsed_ms, 1),
            "cache_hit": cache_hit,
            "semantic_cache_hit": semantic_hit,
        })
    return round_results


def run_cache_eval(api_key: str, output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("CACHE EFFECTIVENESS EVALUATION")
    print("=" * 70)
    print(f"Query pairs: {len(QUERY_PAIRS)} | Semantic threshold: 0.95\n")

    bot = LibraryChatbot(api_key=api_key)
    bot.clear_cache()

    originals = [p[0] for p in QUERY_PAIRS]
    paraphrases = [p[1] for p in QUERY_PAIRS]

    # Round 1: cold
    cold_results = _run_round(bot, originals, "COLD (cache empty — baseline)")
    cold_lats = [r["latency_ms"] for r in cold_results]
    cold_hits = sum(1 for r in cold_results if r["cache_hit"])
    print(f"  Avg: {sum(cold_lats)/len(cold_lats):.0f}ms | Hits: {cold_hits}/{len(originals)}")

    # Round 2: exact repeat
    exact_results = _run_round(bot, originals, "EXACT REPEAT (should all hit cache)")
    exact_lats = [r["latency_ms"] for r in exact_results]
    exact_hits = sum(1 for r in exact_results if r["cache_hit"])
    print(f"  Avg: {sum(exact_lats)/len(exact_lats):.0f}ms | Hits: {exact_hits}/{len(originals)}")

    # Round 3: paraphrases (semantic cache)
    semantic_results = _run_round(bot, paraphrases, "PARAPHRASED QUERIES (semantic cache)")
    semantic_lats = [r["latency_ms"] for r in semantic_results]
    semantic_hits = sum(1 for r in semantic_results if r["cache_hit"])
    true_semantic = sum(1 for r in semantic_results if r["semantic_cache_hit"])
    print(f"  Avg: {sum(semantic_lats)/len(semantic_lats):.0f}ms | "
          f"Hits: {semantic_hits}/{len(paraphrases)} | Semantic-only: {true_semantic}/{len(paraphrases)}")

    # Compute summary metrics
    avg_cold = sum(cold_lats) / len(cold_lats)
    avg_exact = sum(exact_lats) / len(exact_lats)
    avg_semantic = sum(semantic_lats) / len(semantic_lats)
    speedup_exact = avg_cold / avg_exact if avg_exact > 0 else float("inf")
    speedup_semantic = avg_cold / avg_semantic if avg_semantic > 0 else float("inf")
    exact_hit_rate = exact_hits / len(originals)
    semantic_hit_rate = semantic_hits / len(paraphrases)
    true_semantic_rate = true_semantic / len(paraphrases)

    # Estimate API call savings per 100 queries
    # Cold: 1 embedding + 2 LLM calls per query
    # Cached: 0 LLM calls, at most 1 embedding for semantic check
    estimated_calls_saved_exact = exact_hits * 3        # embed + rerank + generate
    estimated_calls_saved_semantic = semantic_hits * 2  # rerank + generate

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Mode':<28} {'Avg Latency':>12} {'Hit Rate':>10} {'Speedup':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Cold (baseline)':<28} {avg_cold:>10.0f}ms {'—':>10} {'1.0×':>10}")
    print(f"  {'Exact repeat':<28} {avg_exact:>10.0f}ms {exact_hit_rate:>9.0%} {speedup_exact:>9.1f}×")
    print(f"  {'Paraphrased (semantic)':<28} {avg_semantic:>10.0f}ms {semantic_hit_rate:>9.0%} {speedup_semantic:>9.1f}×")
    print()
    print(f"  Semantic-only cache hits: {true_semantic}/{len(paraphrases)} ({true_semantic_rate:.0%})")
    print(f"  API calls saved (exact round):    {estimated_calls_saved_exact} of {len(originals)*3} possible")
    print(f"  API calls saved (semantic round): {estimated_calls_saved_semantic} of {len(paraphrases)*2} possible")
    print()
    print(f"Conclusion: Exact cache hits are {speedup_exact:.1f}× faster than cold queries.")
    print(f"The semantic cache caught {true_semantic_rate:.0%} of paraphrased queries,")
    print(f"delivering a {speedup_semantic:.1f}× speedup on near-duplicate inputs.")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_query_pairs": len(QUERY_PAIRS),
            "semantic_threshold": 0.95,
            "cache_max_size": 1024,
            "cache_ttl_seconds": 3600,
        },
        "aggregate": {
            "cold_avg_latency_ms": round(avg_cold, 1),
            "exact_avg_latency_ms": round(avg_exact, 1),
            "semantic_avg_latency_ms": round(avg_semantic, 1),
            "exact_hit_rate": round(exact_hit_rate, 4),
            "semantic_hit_rate": round(semantic_hit_rate, 4),
            "true_semantic_hit_rate": round(true_semantic_rate, 4),
            "speedup_exact": round(speedup_exact, 2),
            "speedup_semantic": round(speedup_semantic, 2),
            "api_calls_saved_exact": estimated_calls_saved_exact,
            "api_calls_saved_semantic": estimated_calls_saved_semantic,
        },
        "cold_round": cold_results,
        "exact_round": exact_results,
        "semantic_round": semantic_results,
        "query_pairs": [{"original": o, "paraphrase": p} for o, p in QUERY_PAIRS],
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"eval_cache_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate cache effectiveness")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Export it or add it to .env")
        sys.exit(1)

    init_db()
    run_cache_eval(api_key=api_key, output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
