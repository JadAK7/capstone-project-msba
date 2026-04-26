"""
Error taxonomy on the N lowest-scoring golden-set questions.

Classifies each failure into one of:
  - KB_COVERAGE_GAP        : system correctly refused because no relevant chunks retrieved (num_chunks==0)
  - EVAL_CLASSIFICATION_BUG: system answered but the regex-based refusal detector flagged it (false abstention)
  - RETRIEVAL_BELOW_THRESHOLD: chunks were retrieved but the confidence threshold caused refusal
  - CROSS_LINGUAL_FAITHFULNESS: Arabic answer marked unsupported despite reading faithfully against English context
  - GENERATION_FAITHFULNESS_DRIFT: system answered but judge flagged unsupported claims
  - JUDGE_INCONSISTENT     : answer appears correct and chunks are relevant; judge may have scored too harshly
  - OTHER                  : needs manual review

Run:
    python scripts/eval_error_taxonomy.py <eval_run_dir> [n_bottom=20] [out_path]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REFUSAL_START_MARKERS = [
    "i could not find",
    "i don't have",
    "**i'm not quite sure",
    "**لست متأكد",
    "لم أتمكن",
]


def classify(row: dict) -> tuple[str, str]:
    """Return (category, one-line justification)."""
    ab = row.get("actual_behavior")
    src = row.get("chosen_source", "") or ""
    n_chunks = row.get("num_chunks") or 0
    preview = (row.get("answer_preview") or "").lower()
    starts_refusal = any(preview.startswith(m) for m in REFUSAL_START_MARKERS)
    lang = row.get("language")
    metrics = row.get("metrics") or {}

    def mscore(k):
        v = metrics.get(k)
        return v.get("score") if isinstance(v, dict) else v

    if ab == "refuse":
        if n_chunks == 0 or src.startswith("none"):
            if not starts_refusal and len(preview) > 30:
                return "EVAL_CLASSIFICATION_BUG", (
                    "refuse label but answer_preview has real content; "
                    "likely 'please contact the library' false-positive in _is_refusal()"
                )
            return "KB_COVERAGE_GAP", "num_chunks==0 and system correctly refused"
        if n_chunks > 0 and starts_refusal:
            return "RETRIEVAL_BELOW_THRESHOLD", (
                f"{n_chunks} chunks retrieved but generator refused "
                f"(source={src!r})"
            )
        if n_chunks > 0 and not starts_refusal and len(preview) > 30:
            return "EVAL_CLASSIFICATION_BUG", (
                "refuse label but answer_preview has substantive content; "
                "regex refusal detector matched appended fallback phrase"
            )
        return "OTHER", f"refuse, chunks={n_chunks}, src={src}"

    # answered cases
    gs = row.get("grounding_score") or 0.0
    faith = mscore("faithfulness") or 1.0
    ground = mscore("groundedness") or 1.0
    if lang == "ar" and faith < 0.5 and ground > 0.7:
        return "CROSS_LINGUAL_FAITHFULNESS", (
            f"Arabic answer, groundedness={ground:.2f} but faithfulness={faith:.2f} "
            "— judge may penalize translation-level mismatches"
        )
    if gs < 0.75:
        return "GENERATION_FAITHFULNESS_DRIFT", (
            f"answered, gs={gs:.2f}, faithfulness={faith:.2f}, "
            f"groundedness={ground:.2f}"
        )
    return "JUDGE_INCONSISTENT", f"answer looks correct; judge scored {gs:.2f}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    n_bottom = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else run_dir / "error_taxonomy.json"

    d = json.load(open(run_dir / "golden.json"))
    rows = d["raw_results"]
    rows_sorted = sorted(
        rows, key=lambda r: (r.get("grounding_score") if r.get("grounding_score") is not None else 1.0)
    )
    bottom = rows_sorted[:n_bottom]

    by_cat = defaultdict(list)
    for r in bottom:
        cat, why = classify(r)
        by_cat[cat].append(
            {
                "id": r["id"],
                "question": r["question"],
                "category": r.get("category"),
                "language": r.get("language"),
                "grounding_score": r.get("grounding_score"),
                "chosen_source": r.get("chosen_source"),
                "num_chunks": r.get("num_chunks"),
                "actual_behavior": r.get("actual_behavior"),
                "expected_behavior": r.get("expected_behavior"),
                "answer_preview": (r.get("answer_preview") or "")[:220],
                "justification": why,
            }
        )

    counts = {k: len(v) for k, v in by_cat.items()}
    result = {
        "run_dir": str(run_dir),
        "n_bottom": n_bottom,
        "counts": counts,
        "by_category": by_cat,
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nError taxonomy on bottom {n_bottom} golden questions")
    print(f"Source: {run_dir}/golden.json\n")
    total = sum(counts.values())
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total
        print(f"  {cat:32s} {n:3d}  ({pct:4.1f}%)")
    print(f"\n  {'TOTAL':32s} {total:3d}")
    print(f"\nExamples by category:")
    for cat, items in by_cat.items():
        print(f"\n  [{cat}]  n={len(items)}")
        for it in items[:3]:
            print(f"    - {it['id']:14s} {it['question'][:80]}")
            print(f"      gs={it['grounding_score']}  chunks={it['num_chunks']}  src={it['chosen_source']!r}")
            print(f"      why: {it['justification']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
