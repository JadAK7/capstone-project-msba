#!/usr/bin/env python3
"""
eval_rewriting.py
Measures query rewriting coverage across query categories.

Sends 40 queries through rewrite_query() across 4 categories:
  - arabic:           should always be rewritten (is_arabic=True)
  - followup:         should always be rewritten (is_followup=True)
  - short_vague:      should always be rewritten (is_short=True, <=3 words)
  - well_formed_en:   fast-path skip (>=6 words, not Arabic, not followup)

Backs up: "Query rewriting is triggered for X% of all queries"

Usage:
    python scripts/eval_rewriting.py
    python scripts/eval_rewriting.py --output results/rewriting.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.query_rewriter import rewrite_query

# ---------------------------------------------------------------------------
# Test query sets — 10 per category, 40 total
# ---------------------------------------------------------------------------

QUERIES = {
    "arabic": [
        "ما هي ساعات عمل المكتبة؟",
        "كيف أستعير كتاباً من المكتبة؟",
        "ما هي غرامة تأخير إعادة الكتب؟",
        "هل يمكنني حجز غرفة دراسة؟",
        "كيف أجدد استعارة كتاب؟",
        "هل يمكنني الوصول إلى قواعد البيانات من خارج الحرم؟",
        "ما هي خدمات الطباعة المتاحة في المكتبة؟",
        "كيف تعمل خدمة الإعارة بين المكتبات؟",
        "هل يمكن لخريجي الجامعة استخدام المكتبة؟",
        "كم عدد الكتب التي يمكنني استعارتها في وقت واحد؟",
    ],
    "followup": [
        "what about the fines?",
        "how long exactly?",
        "and for graduate students?",
        "tell me more",
        "what else can I do?",
        "ok but how do I renew?",
        "what if I lose it?",
        "is that free?",
        "yes but where exactly?",
        "and online access?",
    ],
    "short_vague": [
        "hours",
        "borrow",
        "printing",
        "databases",
        "fines",
        "wifi",
        "rooms",
        "ILL",
        "renew",
        "access",
    ],
    "well_formed_en": [
        "What are the opening hours of Jafet library on weekdays?",
        "How many books can I borrow at a time as a graduate student?",
        "Which databases are available for business and finance research?",
        "What is the policy for returning books after the due date?",
        "Can I access library electronic resources from off-campus?",
        "How do I reserve a study room at the AUB library?",
        "What printing and scanning services are available at the library?",
        "How can I request an interlibrary loan for a book not in AUB?",
        "What library services are available for AUB alumni members?",
        "How do I set up remote access to library databases from home?",
    ],
}

# Dummy history for follow-up queries (simulates a real conversation)
_FOLLOWUP_HISTORY = [
    {
        "role": "user",
        "content": "What are the library borrowing policies?",
    },
    {
        "role": "assistant",
        "content": (
            "The library allows students to borrow up to 10 books for 4 weeks. "
            "Renewals can be done online or at the desk."
        ),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_rewriting_eval(output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("QUERY REWRITING COVERAGE EVALUATION")
    print("=" * 70)
    print(f"Total queries: {sum(len(v) for v in QUERIES.values())} "
          f"across {len(QUERIES)} categories\n")

    all_results = []
    category_stats = {}

    for category, queries in QUERIES.items():
        print(f"[{category.upper()}] ({len(queries)} queries)")
        cat_results = []

        for q in queries:
            history = _FOLLOWUP_HISTORY if category == "followup" else None
            lang = "ar" if category == "arabic" else "en"

            t0 = time.time()
            rewritten, debug = rewrite_query(query=q, history=history, lang=lang)
            elapsed_ms = (time.time() - t0) * 1000

            was_rewritten = not debug.get("rewrite_skipped", False)
            result = {
                "category": category,
                "original_query": q,
                "rewritten_query": rewritten,
                "was_rewritten": was_rewritten,
                "rewrite_skipped": debug.get("rewrite_skipped", False),
                "is_arabic": debug.get("is_arabic", False),
                "is_short": debug.get("is_short", False),
                "is_followup": debug.get("is_followup", False),
                "rewrite_fallback": debug.get("rewrite_fallback", False),
                "rewrite_changed": rewritten.strip() != q.strip(),
                "elapsed_ms": round(elapsed_ms, 1),
            }
            cat_results.append(result)
            all_results.append(result)

            status = "REWRITTEN" if was_rewritten else "SKIPPED "
            print(f"  [{status}] {q[:55]!r}")
            if was_rewritten and rewritten.strip() != q.strip():
                print(f"           -> {rewritten[:55]!r}")

        rewritten_count = sum(1 for r in cat_results if r["was_rewritten"])
        category_stats[category] = {
            "total": len(cat_results),
            "rewritten": rewritten_count,
            "skipped": len(cat_results) - rewritten_count,
            "rewrite_rate": round(rewritten_count / len(cat_results), 4),
        }
        print()

    # Overall
    total = len(all_results)
    total_rewritten = sum(1 for r in all_results if r["was_rewritten"])
    overall_rate = total_rewritten / total

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Category':<22} {'Total':>6} {'Rewritten':>10} {'Rate':>8}")
    print(f"  {'-'*50}")
    for cat, stats in category_stats.items():
        print(f"  {cat:<22} {stats['total']:>6} {stats['rewritten']:>10} {stats['rewrite_rate']:>7.1%}")
    print(f"  {'-'*50}")
    print(f"  {'OVERALL':<22} {total:>6} {total_rewritten:>10} {overall_rate:>7.1%}")
    print()
    print(f"Conclusion: Query rewriting is triggered for {overall_rate:.1%} of all queries.")
    print(f"  Arabic: {category_stats['arabic']['rewrite_rate']:.0%} | "
          f"Follow-ups: {category_stats['followup']['rewrite_rate']:.0%} | "
          f"Short/vague: {category_stats['short_vague']['rewrite_rate']:.0%} | "
          f"Well-formed EN: {category_stats['well_formed_en']['rewrite_rate']:.0%}")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_queries": total,
            "categories": list(QUERIES.keys()),
            "queries_per_category": {k: len(v) for k, v in QUERIES.items()},
        },
        "overall": {
            "total": total,
            "rewritten": total_rewritten,
            "skipped": total - total_rewritten,
            "rewrite_rate": round(overall_rate, 4),
        },
        "by_category": category_stats,
        "per_query": all_results,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"eval_rewriting_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate query rewriting coverage")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    init_db()
    run_rewriting_eval(output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
