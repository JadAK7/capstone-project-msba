"""
Pre-drafts reference (ground-truth) answers for a subset of golden-set
questions by matching them against the source CSVs and extracting the
first 1-2 sentences of the relevant source answer. Writes an editable
JSON file that the user reviews/tightens, then merges back into
data/golden_set.json.

Pipeline:
    1. Select 25 diagnostic/representative questions covering FAQ, DB,
       policy, and Arabic categories.
    2. For each FAQ-direct question: match to library_faq_clean.csv.
    3. For each database question: match to Databases description.csv
       by the first keyword in answer_keywords that names a DB.
    4. For policy/Arabic/other: leave a placeholder ("") for the user.
    5. Write data/reference_answers_draft.json with both the pre-drafted
       text and the source span it came from, so the user can edit.

Run:
    python scripts/draft_reference_answers.py
    python scripts/draft_reference_answers.py --out data/references.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", t)
    return [p.strip() for p in parts if p.strip()]


def _first_sentences(text: str, n: int = 2, max_chars: int = 280) -> str:
    s = _sentences(text)
    out = " ".join(s[:n])
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "..."
    return out


def load_faqs() -> list[dict]:
    rows = []
    with open(DATA_DIR / "library_faq_clean.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                rows.append({"q": q, "a": a})
    return rows


def load_dbs() -> list[dict]:
    rows = []
    with open(DATA_DIR / "Databases description.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = (r.get("name") or "").strip()
            d = (r.get("description") or "").strip()
            if n and d:
                d = re.sub(r"<[^>]+>", " ", d)
                d = re.sub(r"\s+", " ", d).strip()
                rows.append({"name": n, "desc": d})
    return rows


def best_faq_match(q: str, faqs: list[dict]) -> tuple[float, dict | None]:
    q_low = q.lower()
    best = (0.0, None)
    for f in faqs:
        s = SequenceMatcher(None, q_low, f["q"].lower()).ratio()
        if s > best[0]:
            best = (s, f)
    return best


def best_db_match(keywords: list[str], dbs: list[dict]) -> dict | None:
    """Find a DB whose name matches any of the expected keywords."""
    kw_lower = {k.lower() for k in keywords}
    for db in dbs:
        name_parts = {p.lower() for p in re.split(r"[\s/]+", db["name"]) if p}
        if kw_lower & name_parts:
            return db
    # Fall back: substring match on db name
    for db in dbs:
        for k in keywords:
            if len(k) > 3 and k.lower() in db["name"].lower():
                return db
    return None


def select_questions(golden: list[dict]) -> list[dict]:
    """Pick 25 questions: 10 FAQ, 5 DB, 5 policy, 5 Arabic."""
    by_cat: dict[str, list] = {}
    for q in golden:
        if q.get("expected_behavior") != "answer":
            continue
        by_cat.setdefault(q["category"], []).append(q)

    # Use deterministic ordering so the output is reproducible
    pick = []
    pick.extend(by_cat.get("faq_direct", [])[:10])
    pick.extend(by_cat.get("database_recommendation", [])[:5])
    pick.extend(by_cat.get("policy_hours", [])[:5])
    pick.extend(by_cat.get("arabic", [])[:5])
    return pick


def draft_for_question(q: dict, faqs: list[dict], dbs: list[dict]) -> dict:
    cat = q["category"]
    out = {
        "id": q["id"],
        "question": q["question"],
        "category": cat,
        "language": q.get("language", "en"),
        "difficulty": q.get("difficulty"),
        "expected_keywords": q.get("answer_keywords", []),
        "notes": q.get("notes", ""),
        "reference_answer": "",
        "source_span": "",
        "source_match_score": None,
        "needs_user_review": True,
    }

    if cat == "faq_direct":
        score, match = best_faq_match(q["question"], faqs)
        if match and score >= 0.5:
            out["reference_answer"] = _first_sentences(match["a"])
            out["source_span"] = match["a"][:500]
            out["source_match_score"] = round(score, 3)
            out["needs_user_review"] = False  # pre-drafted, just tighten if desired
        else:
            out["reference_answer"] = ""  # user writes
    elif cat == "database_recommendation":
        db = best_db_match(q.get("answer_keywords", []), dbs)
        if db:
            out["reference_answer"] = f"{db['name']}: {_first_sentences(db['desc'], n=1, max_chars=220)}"
            out["source_span"] = f"{db['name']}: {db['desc'][:500]}"
            out["source_match_score"] = 1.0
            out["needs_user_review"] = False
    elif cat == "arabic":
        # Many ar_* questions are translations of EN peers — note field holds
        # the hint. We leave the reference blank for the user but surface the
        # peer ID so they can translate the EN reference once it's written.
        m = re.match(r"^\s*arabic translation of (\w+)", q.get("notes", "").lower())
        if m:
            out["peer_en_id"] = m.group(1).strip(".,;")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden",
        default=str(DATA_DIR / "golden_set.json"),
        help="Path to golden set",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "reference_answers_draft.json"),
        help="Output JSON",
    )
    args = parser.parse_args()

    golden = json.load(open(args.golden, encoding="utf-8"))
    faqs = load_faqs()
    dbs = load_dbs()

    picked = select_questions(golden["questions"])
    drafts = [draft_for_question(q, faqs, dbs) for q in picked]

    summary = {
        "total": len(drafts),
        "prefilled": sum(1 for d in drafts if d["reference_answer"]),
        "needs_user_input": sum(1 for d in drafts if not d["reference_answer"]),
        "by_category": {},
    }
    for d in drafts:
        c = d["category"]
        summary["by_category"].setdefault(c, {"total": 0, "prefilled": 0})
        summary["by_category"][c]["total"] += 1
        if d["reference_answer"]:
            summary["by_category"][c]["prefilled"] += 1

    out_doc = {
        "description": (
            "Reference (ground-truth) answers for Ragas answer_correctness. "
            "FAQ and database rows are pre-drafted from source CSVs; please "
            "tighten to 1-2 sentences if long. Policy, follow-up, and Arabic "
            "rows require you to fill in reference_answer. For Arabic rows "
            "that are translations of EN peers (see peer_en_id), the easiest "
            "path is to write the EN reference first, then translate. When "
            "done, run: python scripts/merge_reference_answers.py"
        ),
        "summary": summary,
        "references": drafts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)

    print(f"\nDrafted {summary['total']} reference answers → {args.out}")
    print(f"  Pre-filled from source: {summary['prefilled']}")
    print(f"  Need your input:        {summary['needs_user_input']}")
    print(f"\nBy category:")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<28} prefilled {s['prefilled']:>2}/{s['total']}")

    print(f"\nNext step: open {args.out}, review the pre-drafted answers,")
    print(f"  write anything marked needs_user_review: true, then run:")
    print(f"    python scripts/merge_reference_answers.py")


if __name__ == "__main__":
    main()
