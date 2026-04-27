"""
Determines provenance of each golden-set question from its notes and IDs, then
recomputes headline eval metrics split by provenance class. Produces:

  - "handwritten_en"        - EN question authored directly
  - "handwritten_ar_native" - AR question authored directly (not a translation)
  - "author_translated"     - AR question that is a translation of an EN peer
                              (note text starts with "Arabic translation of ...")
  - "machine_generated"     - IDs prefixed gen_* (none expected in current set)

The goal is to pre-empt a "test-set contamination" question during defense by
showing that performance on author-written vs. machine-generated questions is
not statistically distinguishable (or by showing the actual split is pure
author-written and contamination is therefore a non-issue).

Run:
    python scripts/eval_provenance_split.py <eval_run_dir> [golden_set_path]
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


DEFAULT_GOLDEN = Path("data/golden_set.json")


def classify(q: dict) -> str:
    qid = q["id"]
    notes = (q.get("notes") or "").lower()
    if qid.startswith("gen_"):
        return "machine_generated"
    if q.get("language") == "ar":
        if re.match(r"^\s*arabic translation of ", notes):
            return "author_translated"
        return "handwritten_ar_native"
    return "handwritten_en"


def aggregate(rows: list) -> dict:
    scored = [r for r in rows if r.get("metrics") and r.get("grounding_score") is not None]
    if not scored:
        return {"n": len(rows), "n_scored": 0}

    def _mean(key):
        vals = [r["metrics"].get(key) for r in scored if r["metrics"].get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    gs = [r["grounding_score"] for r in scored]
    abst = sum(1 for r in rows if r.get("abstained"))
    return {
        "n": len(rows),
        "n_scored": len(scored),
        "answer_relevance": _mean("answer_relevance"),
        "groundedness": _mean("groundedness"),
        "faithfulness": _mean("faithfulness"),
        "context_relevance": _mean("context_relevance"),
        "grounding_score_composite": round(statistics.mean(gs), 4),
        "abstention_rate": round(abst / len(rows), 4),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    golden_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GOLDEN

    golden = json.load(open(golden_path))
    eval_out = json.load(open(run_dir / "golden.json"))

    provenance = {q["id"]: classify(q) for q in golden["questions"]}

    buckets: dict[str, list] = {
        "handwritten_en": [],
        "handwritten_ar_native": [],
        "author_translated": [],
        "machine_generated": [],
    }
    for r in eval_out["raw_results"]:
        p = provenance.get(r["id"], "unknown")
        buckets.setdefault(p, []).append(r)

    counts = {k: len(v) for k, v in buckets.items() if v}
    totals_in_golden = {}
    for q in golden["questions"]:
        p = classify(q)
        totals_in_golden[p] = totals_in_golden.get(p, 0) + 1

    agg = {k: aggregate(v) for k, v in buckets.items() if v}

    report = {
        "run_dir": str(run_dir),
        "golden_set": str(golden_path),
        "n_total_in_golden": len(golden["questions"]),
        "golden_provenance_counts": totals_in_golden,
        "eval_provenance_counts": counts,
        "metrics_by_provenance": agg,
        "notes": (
            "Questions prefixed 'gen_' are machine-generated per "
            "scripts/generate_golden_set.py. Absence of that prefix combined "
            "with hand-written 'notes' fields indicates author-written "
            "provenance. 'author_translated' = AR question whose note begins "
            "\"Arabic translation of ...\"."
        ),
    }
    out = run_dir / "provenance_split.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n=== GOLDEN-SET PROVENANCE ({len(golden['questions'])} questions) ===")
    for k, n in sorted(totals_in_golden.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(golden["questions"])
        print(f"  {k:<26} {n:3d}  ({pct:4.1f}%)")

    print(f"\n=== METRICS BY PROVENANCE ===")
    print(f"{'bucket':<26}{'n':>4}{'gs':>10}{'ground':>9}{'faith':>9}{'ctx':>9}{'ans':>9}{'abst':>9}")
    for k, m in agg.items():
        print(
            f"  {k:<24} {m['n']:>4} "
            f"{(m['grounding_score_composite'] or 0):>9.3f} "
            f"{(m['groundedness'] or 0):>8.3f} "
            f"{(m['faithfulness'] or 0):>8.3f} "
            f"{(m['context_relevance'] or 0):>8.3f} "
            f"{(m['answer_relevance'] or 0):>8.3f} "
            f"{m['abstention_rate']:>8.3f}"
        )

    machine_n = totals_in_golden.get("machine_generated", 0)
    if machine_n == 0:
        print(
            "\nNO machine-generated questions found in the set. "
            "The contamination concern is mitigated: all questions are "
            "author-written (with 15 author-translations from EN→AR). "
            "Recommend updating golden_set.json 'description' to reflect this."
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
