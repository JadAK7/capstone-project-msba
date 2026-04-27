"""
Estimates per-query OpenAI cost for the full pipeline and each naive baseline,
then produces a cost-quality Pareto report.

This is an ESTIMATE based on:
  (a) The number and type of OpenAI calls each pipeline stage makes
      (read from backend/ source code)
  (b) Per-stage token averages inferred from the answer length and typical
      context sizes observed in the golden-set eval output
  (c) OpenAI list pricing as of 2025-01 (edit PRICING below if stale)

For exact numbers, wire `usage` from the Anthropic/OpenAI client responses
into backend/llm_client.py and re-run. Estimates are good enough to ground a
cost-quality tradeoff discussion in the defense.

Run:
    python scripts/eval_cost.py <eval_run_dir> [out_path]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


# -----------------------------------------------------------------------------
# Pricing (USD per 1M tokens). Update if stale.
# Source: OpenAI list pricing, January 2025.
# -----------------------------------------------------------------------------
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def _embed_cost(tokens: int) -> float:
    p = PRICING["text-embedding-3-small"]
    return tokens * p["input"] / 1_000_000


def _chat_cost(in_tok: int, out_tok: int, model: str = "gpt-4o-mini") -> float:
    p = PRICING[model]
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


# -----------------------------------------------------------------------------
# Per-stage token estimates (per query). These reflect the typical input/output
# size of each stage in backend/chatbot.py as of the current pipeline. See the
# "estimation_basis" block in the output JSON for what each number represents.
# -----------------------------------------------------------------------------
STAGE_TOKENS = {
    "input_guard_embed": {"in": 25, "out": 0, "kind": "embed"},
    "query_rewriter": {"in": 300, "out": 60, "kind": "chat"},
    "retrieval_embed": {"in": 30, "out": 0, "kind": "embed"},
    # LLM reranker: one call that scores ~15 candidate chunks in batch
    "reranker": {"in": 1800, "out": 120, "kind": "chat"},
    "generator": {"in": 1500, "out": 300, "kind": "chat"},
    "verifier": {"in": 900, "out": 80, "kind": "chat"},
    # Lightweight alternative used in B5
    "generator_simple": {"in": 900, "out": 200, "kind": "chat"},
    # Full-text BM25 is free (no API call)
}

# Which stages each baseline invokes.
BASELINE_STAGES = {
    "B0": [  # Full pipeline
        "input_guard_embed",
        "query_rewriter",
        "retrieval_embed",
        "reranker",
        "generator",
        "verifier",
    ],
    "B1": ["generator"],                                     # LLM-only
    "B2": ["generator_simple"],                              # BM25-only + raw top (no embed, no reranker)
    "B3": ["retrieval_embed", "generator_simple"],           # Vector-only + raw top
    "B4": ["retrieval_embed"],                               # FAQ nearest-match verbatim
    "B5": ["retrieval_embed", "generator_simple"],           # Retrieve + generic summarize
}


def stage_cost(stage: str) -> float:
    s = STAGE_TOKENS[stage]
    if s["kind"] == "embed":
        return _embed_cost(s["in"])
    return _chat_cost(s["in"], s["out"])


def baseline_cost(b_id: str) -> dict:
    stages = BASELINE_STAGES[b_id]
    by_stage = {s: stage_cost(s) for s in stages}
    return {
        "stages_invoked": stages,
        "per_stage_usd": {k: round(v, 6) for k, v in by_stage.items()},
        "total_per_query_usd": round(sum(by_stage.values()), 6),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "cost_estimate.json"

    baselines = json.load(open(run_dir / "baselines.json"))

    labels = {b: baselines["summary"][b]["label"] for b in baselines["summary"]}
    quality = {
        b: {
            "grounding_score": baselines["summary"][b]["overall"]["grounding_score"],
            "answer_relevance": baselines["summary"][b]["overall"]["answer_relevance"],
            "faithfulness": baselines["summary"][b]["overall"]["faithfulness"],
            "groundedness": baselines["summary"][b]["overall"]["groundedness"],
            "mean_elapsed_ms": baselines["summary"][b]["overall"]["mean_elapsed_ms"],
            "n": baselines["summary"][b]["overall"]["n"],
        }
        for b in baselines["summary"]
    }

    cost = {b: baseline_cost(b) for b in BASELINE_STAGES}

    # Cost/quality ratios + Pareto ranking
    pareto = []
    for b in cost:
        c = cost[b]["total_per_query_usd"]
        q = quality[b]["grounding_score"]
        pareto.append(
            {
                "baseline": b,
                "label": labels[b],
                "cost_usd_per_query": c,
                "grounding_score": q,
                "usd_per_grounding_point": round(c / q, 6) if q > 0 else None,
                "quality_per_usd": round(q / c, 2) if c > 0 else None,
                "mean_elapsed_ms": quality[b]["mean_elapsed_ms"],
            }
        )
    # Sort by cost ascending
    pareto.sort(key=lambda x: x["cost_usd_per_query"])

    # Pareto-optimal = no other point has lower cost AND higher quality
    optimal = set()
    for i, p in enumerate(pareto):
        dominated = any(
            q["cost_usd_per_query"] <= p["cost_usd_per_query"]
            and q["grounding_score"] >= p["grounding_score"]
            and (q["cost_usd_per_query"] < p["cost_usd_per_query"]
                 or q["grounding_score"] > p["grounding_score"])
            for q in pareto
        )
        if not dominated:
            optimal.add(p["baseline"])

    # Annual cost projection at representative query volumes
    b0 = cost["B0"]["total_per_query_usd"]
    volumes = [1_000, 10_000, 100_000, 1_000_000]
    annual = {f"{v:,} queries": round(b0 * v, 2) for v in volumes}

    report = {
        "pricing_usd_per_1m_tokens": PRICING,
        "stage_token_assumptions": STAGE_TOKENS,
        "baseline_stages": BASELINE_STAGES,
        "estimation_basis": (
            "Token counts are per-stage averages inferred from backend/chatbot.py "
            "and typical context sizes in the golden-set eval output. For exact "
            "numbers, add token usage logging to backend/llm_client.py and rerun."
        ),
        "per_baseline_cost": cost,
        "per_baseline_quality": quality,
        "cost_vs_quality_pareto": pareto,
        "pareto_optimal_baselines": sorted(optimal),
        "annual_cost_full_pipeline_usd": annual,
    }
    out.write_text(json.dumps(report, indent=2))

    print(f"\n=== PER-QUERY COST ESTIMATES ===\n")
    print(f"{'id':<4} {'label':<28} {'cost $':>10} {'gs':>7} {'$/gs-pt':>10} {'latency_ms':>12} {'pareto':>7}")
    for p in pareto:
        star = "  *" if p["baseline"] in optimal else "   "
        print(
            f"{p['baseline']:<4} {p['label'][:28]:<28} "
            f"{p['cost_usd_per_query']:>10.5f} "
            f"{p['grounding_score']:>7.3f} "
            f"{(p['usd_per_grounding_point'] or 0):>10.5f} "
            f"{p['mean_elapsed_ms']:>12.0f} "
            f"{star:>7}"
        )
    print("\n  * = Pareto-optimal (no other baseline has lower cost AND higher quality)")

    print(f"\n=== ANNUAL COST PROJECTION (Full pipeline, @ ${cost['B0']['total_per_query_usd']:.5f}/query) ===")
    for v, c in annual.items():
        print(f"  {v:>18} → ${c:>10,.2f}/year")

    b0_cost = cost["B0"]["total_per_query_usd"]
    b3_cost = cost["B3"]["total_per_query_usd"]
    b0_q = quality["B0"]["grounding_score"]
    b3_q = quality["B3"]["grounding_score"]
    print(f"\n=== KEY TRADEOFFS ===")
    print(
        f"  Full (B0) vs Vector-only (B3): {b0_cost/b3_cost:.1f}× more expensive "
        f"for {100*(b0_q-b3_q)/b3_q:+.1f}% grounding delta"
    )
    b1_cost = cost["B1"]["total_per_query_usd"]
    b1_q = quality["B1"]["grounding_score"]
    print(
        f"  Full (B0) vs LLM-only (B1):    {b0_cost/b1_cost:.1f}× more expensive "
        f"for {100*(b0_q-b1_q)/b1_q:+.1f}% grounding delta"
    )

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
