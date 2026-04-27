"""
Merges reviewed reference answers from data/reference_answers_draft.json
into data/golden_set.json by adding a "reference_answer" field to each
matching question.

Rows where reference_answer is empty are skipped (no merge). Rows that
still carry needs_user_review=True are merged anyway but flagged in the
printout so you know the draft state is in the set.

Run:
    python scripts/merge_reference_answers.py
    python scripts/merge_reference_answers.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(DATA_DIR / "golden_set.json"))
    parser.add_argument("--refs", default=str(DATA_DIR / "reference_answers_draft.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    golden = json.load(open(args.golden, encoding="utf-8"))
    refs = json.load(open(args.refs, encoding="utf-8"))

    ref_map = {r["id"]: r for r in refs["references"]}
    merged, skipped, flagged = [], [], []

    for q in golden["questions"]:
        ref = ref_map.get(q["id"])
        if not ref:
            continue
        ra = (ref.get("reference_answer") or "").strip()
        if not ra:
            skipped.append(q["id"])
            continue
        q["reference_answer"] = ra
        merged.append(q["id"])
        if ref.get("needs_user_review"):
            flagged.append(q["id"])

    print(f"Merge summary:")
    print(f"  Merged (reference_answer added): {len(merged)}")
    print(f"  Skipped (empty reference):       {len(skipped)}")
    print(f"  Flagged as needs_user_review:    {len(flagged)}")
    if skipped:
        print(f"\n  Skipped IDs:")
        for qid in skipped:
            print(f"    - {qid}")
    if flagged:
        print(f"\n  Flagged IDs (pre-drafted, unreviewed):")
        for qid in flagged:
            print(f"    - {qid}")

    if args.dry_run:
        print(f"\n(dry run) Would write {args.golden}")
        return

    with open(args.golden, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.golden}")


if __name__ == "__main__":
    main()
