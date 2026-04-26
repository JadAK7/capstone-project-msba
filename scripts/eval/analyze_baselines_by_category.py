#!/usr/bin/env python3
"""
analyze_baselines_by_category.py
Post-hoc per-category breakdown of any eval_baselines JSON output.

Works against baselines.json files that pre-date the category field on
per-question records, by looking up each question's category in
data/golden_set.json via its id.

Reports:
  - Per-category table for each baseline (answer_relevance, grounding_score)
  - Architecture margin B0 − best-other-baseline, per category, per metric
  - Verdict per category: which approach wins, on the fair metric

Usage:
    python scripts/analyze_baselines_by_category.py <path/to/baselines.json>
    python scripts/analyze_baselines_by_category.py eval_run_20260423_120009/baselines.json
"""

import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN = os.path.join(PROJECT_ROOT, "data", "golden_set.json")


def _avg(values):
    vs = [v for v in values if v is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/analyze_baselines_by_category.py <baselines.json>")
        sys.exit(1)
    path = sys.argv[1]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    with open(GOLDEN, encoding="utf-8") as f:
        gold = json.load(f)

    # Look up category by question id
    id_to_cat = {q["id"]: q.get("category", "") for q in gold["questions"]}

    per_q = data["per_question"]
    # Attach category to each question record (if not already present)
    for q in per_q:
        if not q.get("category"):
            q["category"] = id_to_cat.get(q["id"], "")

    # Collect baseline keys actually present
    baseline_keys = sorted({k for q in per_q for k in q["baselines"].keys()})

    # Build per-category aggregates
    categories = sorted({q["category"] for q in per_q if q["category"]})

    # Header
    print("\n" + "=" * 100)
    print(f"  PER-CATEGORY BASELINE ANALYSIS  ({path})")
    print("=" * 100)
    print(f"  {len(per_q)} questions across {len(categories)} categories: {categories}")

    def _collect(scope_filter):
        """Returns dict[baseline_key] = {'rel': [..], 'gs': [..]}"""
        acc = {k: {"rel": [], "gs": []} for k in baseline_keys}
        for q in per_q:
            if not scope_filter(q):
                continue
            for k in baseline_keys:
                b = q["baselines"].get(k)
                if not b:
                    continue
                if b.get("answer_relevance") is not None:
                    acc[k]["rel"].append(b["answer_relevance"])
                if b.get("grounding_score") is not None:
                    acc[k]["gs"].append(b["grounding_score"])
        return acc

    def _print_scope(title, filt):
        acc = _collect(filt)
        n = sum(1 for q in per_q if filt(q))
        if n == 0:
            return
        print(f"\n  {title}  (n={n})")
        print(f"  {'-'*88}")
        print(f"  {'key':<4}  {'rel*':>7}  {'gs':>6}   (rel* = answer_relevance, the fair metric)")
        for k in baseline_keys:
            r = _avg(acc[k]["rel"])
            g = _avg(acc[k]["gs"])
            if r is None and g is None:
                continue
            r_s = f"{r:.3f}" if r is not None else "  n/a"
            g_s = f"{g:.3f}" if g is not None else "  n/a"
            print(f"  {k:<4}  {r_s:>7}  {g_s:>6}")

        # Margin: B0 vs best other
        if "B0" in acc:
            for metric in ("rel", "gs"):
                vals = {k: _avg(acc[k][metric]) for k in baseline_keys if _avg(acc[k][metric]) is not None}
                if "B0" not in vals or len(vals) < 2:
                    continue
                b0 = vals["B0"]
                others = {k: v for k, v in vals.items() if k != "B0"}
                best = max(others, key=others.get)
                margin = round(b0 - others[best], 4)
                name = "answer_relevance" if metric == "rel" else "grounding_score"
                verdict = ""
                if metric == "rel":
                    if margin > 0.03:    verdict = "  ← B0 wins (architecture earns complexity)"
                    elif margin < -0.03: verdict = "  ← baseline wins (architecture costs more than it adds here)"
                    else:                verdict = "  ← tie"
                print(f"    B0 {name:<16} = {b0:.3f}   best other ({best}) = {others[best]:.3f}   "
                      f"margin = {margin:+.3f}{verdict}")

    # Overall
    _print_scope("OVERALL", lambda q: True)

    # Language
    _print_scope("ENGLISH",  lambda q: q.get("language") == "en")
    _print_scope("ARABIC",   lambda q: q.get("language") == "ar")

    # Per category
    print(f"\n  {'='*88}")
    print(f"  BY CATEGORY")
    print(f"  {'='*88}")
    for cat in categories:
        _print_scope(cat, lambda q, c=cat: q.get("category") == c)

    print(f"\n  {'='*88}")
    print("  Read:")
    print("  - Use answer_relevance (rel*) as the fair headline. grounding_score is")
    print("    biased toward baselines that return chunk text verbatim.")
    print("  - If a category shows 'baseline wins on rel*', the architecture is adding")
    print("    cost without adding value on that question type. That is a real finding,")
    print("    not a defect in measurement.")


if __name__ == "__main__":
    main()
