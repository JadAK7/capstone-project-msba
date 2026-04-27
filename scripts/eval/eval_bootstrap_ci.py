"""
Non-parametric bootstrap 95% CIs for the headline eval metrics.

Inputs:
    <run_dir>/golden.json      -- main golden-set results (raw_results array)
    <run_dir>/baselines.json   -- baseline comparison results (per_question array)

Outputs a JSON report with:
  (1) CI for each headline metric on the golden set (grounding, faithfulness, etc.)
  (2) Paired bootstrap CI for (Full pipeline - each baseline) per metric
      — paired resampling preserves per-question correlation, tightening CIs
      vs. independent resampling
  (3) A proportion of bootstrap resamples in which the full pipeline beats the
      baseline (a bootstrap analog of a one-sided p-value)

Run:
    python scripts/eval_bootstrap_ci.py <eval_run_dir> [out_path] [--n-boot N]

    out_path defaults to <eval_run_dir>/bootstrap_ci.json
    n_boot defaults to 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


METRIC_KEYS = [
    "answer_relevance",
    "groundedness",
    "faithfulness",
    "context_relevance",
    "citation_accuracy",
]


def _get_metric(row: dict, key: str):
    """Extract a scalar metric from raw_results rows (handles both the
    {score, reason} dict shape and the flat scalar shape)."""
    m = row.get("metrics") or {}
    v = m.get(key)
    if isinstance(v, dict):
        v = v.get("score")
    return v


def _boot_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Percentile bootstrap 95% CI for the mean."""
    if len(values) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(len(values)),
    }


def _paired_boot(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
):
    """Paired bootstrap: resample question indices, compute (mean_a - mean_b)
    on each resample. Returns CI for the delta and proportion where a > b."""
    assert len(a) == len(b)
    if len(a) == 0:
        return {"delta": None, "ci_low": None, "ci_high": None, "p_a_gt_b": None, "n": 0}
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "delta": float(a.mean() - b.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_a_gt_b": float((deltas > 0).mean()),
        "n": int(len(a)),
    }


def headline_cis(golden: dict, n_boot: int, rng: np.random.Generator) -> dict:
    rows = golden["raw_results"]
    out: dict = {"n_rows": len(rows), "metrics": {}}
    for mk in METRIC_KEYS:
        vals = np.array([v for r in rows if (v := _get_metric(r, mk)) is not None])
        out["metrics"][mk] = _boot_ci(vals, n_boot, rng)

    # Hallucination rate: defined in golden.json as mean of "hallucination" field per row
    # where that score is >= 0.5. Mirror that definition here on the raw metric.
    halluc = np.array(
        [v for r in rows if (v := _get_metric(r, "hallucination")) is not None]
    )
    if halluc.size:
        out["metrics"]["hallucination_rate"] = _boot_ci(
            (halluc >= 0.5).astype(float), n_boot, rng
        )

    gs = np.array(
        [r["grounding_score"] for r in rows if r.get("grounding_score") is not None]
    )
    out["metrics"]["grounding_score_composite"] = _boot_ci(gs, n_boot, rng)

    kc = np.array(
        [r["keyword_coverage"] for r in rows if r.get("keyword_coverage") is not None]
    )
    out["metrics"]["keyword_coverage_avg"] = _boot_ci(kc, n_boot, rng)

    abst = np.array([1.0 if r.get("abstained") else 0.0 for r in rows])
    out["metrics"]["abstention_rate"] = _boot_ci(abst, n_boot, rng)

    return out


def baseline_deltas(
    baselines: dict, n_boot: int, rng: np.random.Generator
) -> dict:
    pq = baselines["per_question"]
    baseline_ids = list(baselines["summary"].keys())
    labels = {b: baselines["summary"][b]["label"] for b in baseline_ids}
    metrics_here = ["answer_relevance", "groundedness", "faithfulness", "grounding_score"]

    # Build aligned arrays per baseline per metric
    arr = {b: {m: [] for m in metrics_here} for b in baseline_ids}
    for q in pq:
        bq = q["baselines"]
        for b in baseline_ids:
            scores = bq.get(b) or {}
            for m in metrics_here:
                arr[b][m].append(scores.get(m))

    # Prune to rows where ALL baselines have scores for ALL metrics (paired)
    valid_idx = []
    for i in range(len(pq)):
        ok = all(
            arr[b][m][i] is not None for b in baseline_ids for m in metrics_here
        )
        if ok:
            valid_idx.append(i)

    aligned = {
        b: {
            m: np.array([arr[b][m][i] for i in valid_idx], dtype=float)
            for m in metrics_here
        }
        for b in baseline_ids
    }

    full_id = "B0"  # Full pipeline
    out: dict = {
        "labels": labels,
        "n_questions": len(valid_idx),
        "deltas_vs_full": {},
        "absolute_means": {
            b: {
                m: {"mean": float(aligned[b][m].mean()), "n": int(len(aligned[b][m]))}
                for m in metrics_here
            }
            for b in baseline_ids
        },
    }
    for b in baseline_ids:
        if b == full_id:
            continue
        per_metric = {}
        for m in metrics_here:
            per_metric[m] = _paired_boot(
                aligned[full_id][m], aligned[b][m], n_boot, rng
            )
        out["deltas_vs_full"][b] = {"label": labels[b], "metrics": per_metric}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="Directory containing golden.json + baselines.json")
    p.add_argument("out", type=Path, nargs="?", default=None,
                   help="Output JSON path (default: <run_dir>/bootstrap_ci.json)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Number of bootstrap resamples (default: 1000)")
    args = p.parse_args()

    run_dir = args.run_dir
    out = args.out or run_dir / "bootstrap_ci.json"
    n_boot = args.n_boot

    rng = np.random.default_rng(seed=42)

    golden = json.load(open(run_dir / "golden.json"))
    baselines = json.load(open(run_dir / "baselines.json"))

    result = {
        "run_dir": str(run_dir),
        "n_bootstrap": n_boot,
        "random_seed": 42,
        "headline": headline_cis(golden, n_boot, rng),
        "baselines_paired": baseline_deltas(baselines, n_boot, rng),
    }
    out.write_text(json.dumps(result, indent=2))

    print(f"\n=== HEADLINE METRICS (95% bootstrap CI, n_boot={n_boot}) ===")
    print(f"{'metric':<30} {'mean':>8} {'CI_low':>8} {'CI_high':>8}  n")
    for k, v in result["headline"]["metrics"].items():
        if v["mean"] is None:
            continue
        print(
            f"  {k:<28} {v['mean']:>8.4f} {v['ci_low']:>8.4f} {v['ci_high']:>8.4f}  {v['n']}"
        )

    print(f"\n=== BASELINE DELTAS vs Full pipeline (paired bootstrap) ===")
    bp = result["baselines_paired"]
    print(f"n={bp['n_questions']} paired questions\n")
    for b, info in bp["deltas_vs_full"].items():
        print(f"  [{b}] {info['label']}")
        for m, d in info["metrics"].items():
            mark = " *" if d["ci_low"] > 0 or d["ci_high"] < 0 else "  "
            print(
                f"    {m:<22} Δ={d['delta']:>+.4f}  "
                f"CI=[{d['ci_low']:>+.4f}, {d['ci_high']:>+.4f}]  "
                f"P(full>base)={d['p_a_gt_b']:.2f}{mark}"
            )
        print()
    print("(* = CI does not include 0, i.e. difference is significant at α=0.05)")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
