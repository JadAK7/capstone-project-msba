"""
Compares two full eval runs to quantify run-to-run variance from LLM-as-judge
stochasticity. Reads golden.json from two eval_run_* directories, reports
per-metric mean absolute delta and threshold-flip rates.

Usage:
    python scripts/eval_run_variance.py <run_a_dir> <run_b_dir> [out_path]
"""
import json
import statistics
import sys
from pathlib import Path

METRIC_KEYS = [
    "answer_relevance",
    "groundedness",
    "faithfulness",
    "context_relevance",
    "citation_accuracy",
]


def get_metric(row, key):
    m = row.get("metrics") or {}
    v = m.get(key)
    if isinstance(v, dict):
        v = v.get("score")
    return v


def analyze(run_a_dir: Path, run_b_dir: Path) -> dict:
    a = json.load(open(run_a_dir / "golden.json"))
    b = json.load(open(run_b_dir / "golden.json"))

    headline = {}
    for k, va in a["table1_overall"]["metrics"].items():
        vb = b["table1_overall"]["metrics"][k]
        headline[k] = {"run_a": va, "run_b": vb, "abs_delta": abs(va - vb)}

    ra = {r["id"]: r for r in a["raw_results"]}
    rb = {r["id"]: r for r in b["raw_results"]}
    common = sorted(set(ra) & set(rb))

    per_metric = {}
    for mk in METRIC_KEYS:
        deltas, flips_50, n = [], 0, 0
        for qid in common:
            va = get_metric(ra[qid], mk)
            vb = get_metric(rb[qid], mk)
            if va is None or vb is None:
                continue
            n += 1
            deltas.append(abs(va - vb))
            if (va >= 0.5) != (vb >= 0.5):
                flips_50 += 1
        per_metric[mk] = {
            "n": n,
            "mean_abs_delta": statistics.mean(deltas) if deltas else None,
            "max_abs_delta": max(deltas) if deltas else None,
            "threshold_flips_at_0.5": flips_50,
        }

    gs_deltas = []
    gs_flips = 0
    for qid in common:
        va, vb = ra[qid].get("grounding_score"), rb[qid].get("grounding_score")
        if va is None or vb is None:
            continue
        gs_deltas.append(abs(va - vb))
        if (va >= 0.6) != (vb >= 0.6):
            gs_flips += 1
    composite = {
        "n": len(gs_deltas),
        "mean_abs_delta": statistics.mean(gs_deltas),
        "median_abs_delta": statistics.median(gs_deltas),
        "max_abs_delta": max(gs_deltas),
        "threshold_flips_at_0.6": gs_flips,
    }

    behavior = {
        "abstention_flips": sum(
            1 for q in common if bool(ra[q].get("abstained")) != bool(rb[q].get("abstained"))
        ),
        "chosen_source_flips": sum(
            1 for q in common if ra[q].get("chosen_source") != rb[q].get("chosen_source")
        ),
        "n_common": len(common),
    }

    return {
        "run_a": str(run_a_dir),
        "run_b": str(run_b_dir),
        "headline": headline,
        "per_metric_variance": per_metric,
        "grounding_composite_variance": composite,
        "behavior_flips": behavior,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    run_a = Path(sys.argv[1])
    run_b = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("run_variance.json")
    result = analyze(run_a, run_b)
    out.write_text(json.dumps(result, indent=2))

    print(f"\nRun-to-run variance: {run_a.name}  vs  {run_b.name}")
    print(f"(same pipeline, same golden set, same day — pure LLM-as-judge noise)\n")
    print("Headline metric | Run A | Run B | |Δ|")
    for k, v in result["headline"].items():
        print(f"  {k:28s}  {v['run_a']:.4f}  {v['run_b']:.4f}  {v['abs_delta']:.4f}")
    c = result["grounding_composite_variance"]
    print(f"\nPer-question grounding_score_composite (n={c['n']}):")
    print(f"  mean abs delta:   {c['mean_abs_delta']:.4f}")
    print(f"  median abs delta: {c['median_abs_delta']:.4f}")
    print(f"  max abs delta:    {c['max_abs_delta']:.4f}")
    print(f"  threshold flips @0.6: {c['threshold_flips_at_0.6']}/{c['n']}")
    bh = result["behavior_flips"]
    print(f"\nBehavior flips across {bh['n_common']} shared questions:")
    print(f"  abstention flips:     {bh['abstention_flips']}")
    print(f"  chosen_source flips:  {bh['chosen_source_flips']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
