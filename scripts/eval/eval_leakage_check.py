"""
Leakage check: for each golden-set question, find the most similar FAQ row
in the embedding space and flag near-duplicates.

Why this exists:
    A juror's first instinct on a high score is "did you test on training
    data?". If a golden question is a paraphrase of an FAQ row, the system
    can answer it perfectly via the FAQ retriever and the eval over-credits
    end-to-end performance. This script gives a one-shot answer to that
    challenge: how many golden questions have a near-twin in the FAQ?

Method:
    1. Embed each golden question (text-embedding-3-small via
       backend.embeddings).
    2. For each, query the `faq` table with pgvector cosine distance and
       take the nearest row.
    3. Flag anything with similarity > LEAKAGE_THRESHOLD (default 0.95) —
       at that level the FAQ is effectively the same question.

Output (JSON):
    - n_questions, n_flagged, leakage_rate
    - per-question nearest match (question, similarity, faq_question)
    - flagged_examples: subset where similarity > threshold

Run:
    python scripts/eval_leakage_check.py
    python scripts/eval_leakage_check.py --output results/leakage.json
    python scripts/eval_leakage_check.py --threshold 0.93
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db, get_connection
from backend.embeddings import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DEFAULT_THRESHOLD = 0.95


def nearest_faq(query_embedding) -> dict | None:
    """Return the single closest FAQ row by pgvector cosine distance.

    pgvector's <=> operator is cosine distance ∈ [0, 2]. Cosine
    similarity = 1 - distance, capped at [-1, 1].
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM faq
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (query_embedding, query_embedding),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "faq_id": row[0],
                "faq_question": row[1],
                "faq_answer_preview": (row[2] or "")[:200],
                "similarity": float(row[3]),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=str,
                        default=os.path.join(DATA_DIR, "golden_set.json"),
                        help="Path to golden set JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: leakage_<ts>.json)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold for leakage flag (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)

    questions = golden.get("questions", [])
    logger.info(f"Loaded {len(questions)} golden questions from {args.golden}")

    init_db()

    per_question = []
    flagged = []
    for i, q in enumerate(questions):
        q_text = q["question"]
        try:
            q_emb = embed_text(q_text)
            match = nearest_faq(q_emb)
        except Exception as e:
            logger.error(f"[{i+1}/{len(questions)}] {q['id']} embed/query failed: {e}")
            per_question.append({
                "id": q["id"],
                "question": q_text,
                "category": q.get("category"),
                "language": q.get("language"),
                "error": str(e),
            })
            continue

        sim = match["similarity"] if match else 0.0
        rec = {
            "id": q["id"],
            "question": q_text,
            "category": q.get("category"),
            "language": q.get("language"),
            "nearest_faq_id": match["faq_id"] if match else None,
            "nearest_faq_question": match["faq_question"] if match else None,
            "similarity": round(sim, 4),
            "leakage_flag": sim > args.threshold,
        }
        per_question.append(rec)
        if rec["leakage_flag"]:
            flagged.append(rec)
        logger.info(
            f"[{i+1}/{len(questions)}] {q['id']} sim={sim:.3f} "
            f"{'⚠ LEAKAGE' if rec['leakage_flag'] else 'ok'}"
        )

    close_db()

    n = len(per_question)
    n_flagged = len(flagged)
    leakage_rate = round(n_flagged / max(n, 1), 4)

    # Distribution buckets — useful even if nothing crosses the threshold,
    # because the curve shape tells you whether the set is *close* to leaking.
    buckets = {">0.95": 0, "0.90-0.95": 0, "0.80-0.90": 0, "0.70-0.80": 0, "<0.70": 0}
    for r in per_question:
        s = r.get("similarity", 0)
        if s > 0.95:        buckets[">0.95"] += 1
        elif s > 0.90:      buckets["0.90-0.95"] += 1
        elif s > 0.80:      buckets["0.80-0.90"] += 1
        elif s > 0.70:      buckets["0.70-0.80"] += 1
        else:               buckets["<0.70"] += 1

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "golden_set": args.golden,
            "threshold": args.threshold,
        },
        "summary": {
            "n_questions": n,
            "n_flagged": n_flagged,
            "leakage_rate": leakage_rate,
            "similarity_distribution": buckets,
        },
        "flagged_examples": flagged,
        "per_question": per_question,
    }

    output_path = args.output
    if not output_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_leakage_{ts}.json",
        )
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== GOLDEN-SET LEAKAGE CHECK ===")
    print(f"  Threshold:         sim > {args.threshold}")
    print(f"  Questions checked: {n}")
    print(f"  Flagged:           {n_flagged}  ({leakage_rate:.1%})")
    print(f"\n  Similarity distribution to nearest FAQ:")
    for bucket, count in buckets.items():
        bar = "█" * count if count else ""
        print(f"    {bucket:>12}  n={count:<3}  {bar}")
    if flagged:
        print(f"\n  Flagged questions (test contamination risk):")
        for r in flagged:
            print(f"    [{r['id']}] sim={r['similarity']:.3f}")
            print(f"      golden: {r['question'][:80]}")
            print(f"      faq   : {r['nearest_faq_question'][:80]}")
    else:
        print("\n  No leakage detected at this threshold — golden set is independent.")
    print(f"\n  Wrote {output_path}")


if __name__ == "__main__":
    main()
