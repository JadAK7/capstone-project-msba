"""
Reclassifies existing golden-eval results using the corrected refusal detector
and estimates the impact on headline metrics without re-running LLM judges.

For each row whose prior label was "refuse" but whose chosen_source points to
a real retrieved source AND whose answer_preview does not start with a refusal
marker, we:

  1. Flip abstained -> False
  2. Replace the abstention-default metric block with a conservative estimate
     derived from the MEDIAN metrics of answered rows in the same category.

Conservative = we do NOT invent scores; we impute from the empirical
distribution of real answers that share the row's category. The point is to
bound the impact; final numbers require a full re-run.

Run:
    python scripts/eval_reclassify_abstention.py <eval_run_dir> [out_path]
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path


START_MARKERS = (
    "i could not find",
    "i don't have",
    "i can only answer",
    "**i'm not quite sure",
    "**لست متأكد",
    "لم أتمكن",
    "لا أملك معلومات",
    "يمكنني فقط",
)


def is_refusal_new(answer_preview: str, chosen_source: str) -> bool:
    if chosen_source.startswith("none") or chosen_source.startswith("refused"):
        return True
    start = (answer_preview or "").lstrip().lower()
    return start.startswith(START_MARKERS)


def median_metrics_per_category(rows: list) -> dict:
    by_cat: dict[str, list] = {}
    for r in rows:
        if r.get("abstained"):
            continue
        m = r.get("metrics") or {}
        if not m:
            continue
        by_cat.setdefault(r.get("category", ""), []).append(m)
    out = {}
    for cat, ms in by_cat.items():
        keys = ms[0].keys()
        out[cat] = {k: statistics.median(mm[k] for mm in ms if mm.get(k) is not None) for k in keys}
    return out


def _compose_gs(metrics: dict) -> float:
    # Mirrors backend/evaluation.py grounding_score composite: weighted blend.
    return round(
        metrics.get("groundedness", 0) * 0.35
        + metrics.get("faithfulness", 0) * 0.35
        + metrics.get("answer_relevance", 0) * 0.2
        + metrics.get("context_relevance", 0) * 0.1,
        4,
    )


def _aggregate(rows: list) -> dict:
    scored = [r for r in rows if r.get("metrics") and not r.get("skipped_llm_eval")]
    if not scored:
        return {}

    def _avg(key):
        vals = [r["metrics"].get(key) for r in scored if r["metrics"].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _halluc_rate():
        vals = [r["metrics"].get("hallucination") for r in scored if r["metrics"].get("hallucination") is not None]
        if not vals:
            # Some metric schemas store it as hallucination_rate directly
            vals = [r["metrics"].get("hallucination_rate") for r in scored if r["metrics"].get("hallucination_rate") is not None]
        if not vals:
            return None
        # Following run_golden_eval.py convention: rate == mean(score >= 0.5)
        hi = [1 for v in vals if v >= 0.5]
        return round(len(hi) / len(vals), 4)

    gs_vals = [r.get("grounding_score") for r in scored if r.get("grounding_score") is not None]
    kc_vals = [r.get("keyword_coverage") for r in scored if r.get("keyword_coverage") is not None]
    abst = [1 for r in rows if r.get("abstained")]
    return {
        "n_total": len(rows),
        "n_scored": len(scored),
        "answer_relevance": _avg("answer_relevance"),
        "groundedness": _avg("groundedness"),
        "faithfulness": _avg("faithfulness"),
        "context_relevance": _avg("context_relevance"),
        "citation_accuracy": _avg("citation_accuracy"),
        "hallucination_rate": _halluc_rate(),
        "grounding_score_composite": round(sum(gs_vals) / len(gs_vals), 4) if gs_vals else None,
        "keyword_coverage_avg": round(sum(kc_vals) / len(kc_vals), 4) if kc_vals else None,
        "abstention_rate": round(len(abst) / len(rows), 4),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "golden_reclassified.json"

    original = json.load(open(run_dir / "golden.json"))
    rows = copy.deepcopy(original["raw_results"])

    cat_medians = median_metrics_per_category(rows)

    flipped = []
    for r in rows:
        if not r.get("abstained"):
            continue
        ab_new = is_refusal_new(r.get("answer_preview", ""), r.get("chosen_source", ""))
        if ab_new:
            continue
        # This row was a false abstention. Impute from category median.
        cat = r.get("category", "")
        imputed = cat_medians.get(cat)
        if not imputed:
            continue
        r_before = {"id": r["id"], "prior_metrics": r.get("metrics"), "prior_gs": r.get("grounding_score")}
        r["abstained"] = False
        r["actual_behavior"] = "answer"
        r["metrics"] = {k: round(v, 4) for k, v in imputed.items()}
        r["metrics_reasons"] = {
            k: "Imputed from category median of answered rows (reclassified false abstention)"
            for k in imputed
        }
        r["grounding_score"] = _compose_gs(r["metrics"])
        r["_reclassified"] = True
        flipped.append(
            {
                "id": r["id"],
                "category": cat,
                "chosen_source": r.get("chosen_source"),
                "num_chunks": r.get("num_chunks"),
                "gs_before": r_before["prior_gs"],
                "gs_after_imputed": r["grounding_score"],
            }
        )

    before_agg = _aggregate(original["raw_results"])
    after_agg = _aggregate(rows)

    report = {
        "run_dir": str(run_dir),
        "method": "reclassify abstention labels with stricter _is_refusal(), impute metrics from per-category medians of answered rows",
        "n_flipped": len(flipped),
        "flipped_rows": flipped,
        "before": before_agg,
        "after_imputed": after_agg,
    }
    out.write_text(json.dumps(report, indent=2))

    print(f"\nReclassified {len(flipped)} rows from 'refuse' to 'answer'\n")
    print(f"{'Metric':<30}{'Before':>10}{'After*':>10}{'Δ':>10}")
    for k in before_agg:
        b, a = before_agg[k], after_agg[k]
        if b is None or a is None:
            continue
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            print(f"  {k:<28} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")
        else:
            print(f"  {k:<28} {b!s:>10} {a!s:>10}")
    print("\n* 'After' uses per-category median metrics as conservative imputations")
    print("  for the flipped rows. Actual numbers require a full LLM-judge re-run.")
    print(f"\nFlipped rows:")
    for f in flipped:
        print(f"  {f['id']:14s} cat={f['category']:25s} gs {f['gs_before']} -> {f['gs_after_imputed']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
