#!/usr/bin/env python3
"""
eval_arabic_gap.py
Measures the cosine similarity gap between Arabic and English queries.

For each of 20 paired questions (same question in English + Arabic),
embeds both and queries the knowledge base. Computes the actual gap in
top-1 cosine similarity score, measuring the real cross-lingual penalty.

Backs up: "Arabic queries score X% lower in cosine similarity"

Usage:
    python scripts/eval_arabic_gap.py
    python scripts/eval_arabic_gap.py --output results/arabic_gap.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.retriever import _vector_search

# ---------------------------------------------------------------------------
# 20 parallel question pairs — identical intent, different language
# ---------------------------------------------------------------------------

QUESTION_PAIRS = [
    (
        "What are the library opening hours?",
        "ما هي ساعات عمل المكتبة؟",
    ),
    (
        "How do I borrow a book from the library?",
        "كيف أستعير كتاباً من المكتبة؟",
    ),
    (
        "What is the fine for overdue books?",
        "ما هي غرامة تأخير إعادة الكتب؟",
    ),
    (
        "Can I reserve a study room?",
        "هل يمكنني حجز غرفة دراسة؟",
    ),
    (
        "How do I renew a borrowed book?",
        "كيف أجدد استعارة كتاب مستعار؟",
    ),
    (
        "Can I access databases from off-campus?",
        "هل يمكنني الوصول إلى قواعد البيانات من خارج الحرم الجامعي؟",
    ),
    (
        "What printing services are available at the library?",
        "ما هي خدمات الطباعة المتاحة في المكتبة؟",
    ),
    (
        "How does interlibrary loan work?",
        "كيف تعمل خدمة الإعارة بين المكتبات؟",
    ),
    (
        "Can AUB alumni use the library?",
        "هل يمكن لخريجي الجامعة الأمريكية في بيروت استخدام المكتبة؟",
    ),
    (
        "How many books can I borrow at once?",
        "كم عدد الكتب التي يمكنني استعارتها في وقت واحد؟",
    ),
    (
        "Where can I find medical research databases?",
        "أين يمكنني إيجاد قواعد بيانات البحث الطبي؟",
    ),
    (
        "What databases are available for engineering research?",
        "ما هي قواعد البيانات المتاحة للبحث الهندسي؟",
    ),
    (
        "How do I connect to the library WiFi?",
        "كيف أتصل بشبكة واي فاي في المكتبة؟",
    ),
    (
        "What library services are available for graduate students?",
        "ما هي خدمات المكتبة المتاحة لطلاب الدراسات العليا؟",
    ),
    (
        "What is the borrowing period for books?",
        "ما هي مدة استعارة الكتب؟",
    ),
    (
        "Can I request a book that is not in the library collection?",
        "هل يمكنني طلب كتاب غير موجود في مجموعة المكتبة؟",
    ),
    (
        "What time does Jafet library close?",
        "متى تغلق مكتبة جافيت؟",
    ),
    (
        "How do I cite sources using APA format?",
        "كيف أستشهد بالمصادر باستخدام تنسيق APA؟",
    ),
    (
        "What scanning and copying services are available?",
        "ما هي خدمات المسح الضوئي والنسخ المتاحة؟",
    ),
    (
        "How do I get a library card?",
        "كيف أحصل على بطاقة المكتبة؟",
    ),
]

# Tables to search and their metadata columns
_SEARCH_CONFIG = {
    "faq": ["question", "answer"],
    "document_chunks": ["chunk_text", "page_url", "page_title", "section_title", "page_type"],
}


def get_top_similarity(query: str, table: str, meta_cols: list) -> float:
    """Return the top-1 cosine similarity score for a query against a table."""
    results = _vector_search(
        table=table,
        query=query,
        metadata_cols=meta_cols,
        n_results=1,
    )
    return results[0]["vector_score"] if results else 0.0


def run_arabic_gap_eval(output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("ARABIC vs. ENGLISH COSINE SIMILARITY GAP EVALUATION")
    print("=" * 70)
    print(f"Pairs: {len(QUESTION_PAIRS)} | Tables: {', '.join(_SEARCH_CONFIG.keys())}\n")

    results = []
    gaps_pct = []

    for i, (en_q, ar_q) in enumerate(QUESTION_PAIRS):
        # Get best score across both tables for each language
        en_scores = {t: get_top_similarity(en_q, t, cols) for t, cols in _SEARCH_CONFIG.items()}
        ar_scores = {t: get_top_similarity(ar_q, t, cols) for t, cols in _SEARCH_CONFIG.items()}

        en_best = max(en_scores.values())
        ar_best = max(ar_scores.values())

        abs_gap = en_best - ar_best
        rel_gap_pct = (abs_gap / en_best * 100) if en_best > 0 else 0.0
        gaps_pct.append(rel_gap_pct)

        result = {
            "pair_id": i + 1,
            "english_query": en_q,
            "arabic_query": ar_q,
            "english_best_score": round(en_best, 4),
            "arabic_best_score": round(ar_best, 4),
            "absolute_gap": round(abs_gap, 4),
            "relative_gap_pct": round(rel_gap_pct, 2),
            "scores_by_table": {
                "english": {t: round(s, 4) for t, s in en_scores.items()},
                "arabic": {t: round(s, 4) for t, s in ar_scores.items()},
            },
        }
        results.append(result)

        direction = "lower" if abs_gap > 0 else "higher"
        print(f"[{i+1:02d}] EN={en_best:.3f}  AR={ar_best:.3f}  "
              f"Gap={abs_gap:+.3f} ({rel_gap_pct:+.1f}% {direction})")
        print(f"      {en_q[:60]!r}")

    # Aggregate statistics
    gaps = np.array(gaps_pct)
    mean_gap = float(np.mean(gaps))
    median_gap = float(np.median(gaps))
    std_gap = float(np.std(gaps))
    min_gap = float(np.min(gaps))
    max_gap = float(np.max(gaps))

    arabic_lower = sum(1 for r in results if r["absolute_gap"] > 0)
    arabic_higher = sum(1 for r in results if r["absolute_gap"] < 0)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mean gap (EN score − AR score):  {mean_gap:+.1f}%")
    print(f"  Median gap:                      {median_gap:+.1f}%")
    print(f"  Std deviation:                   {std_gap:.1f}%")
    print(f"  Range:                           {min_gap:.1f}% to {max_gap:.1f}%")
    print(f"  Arabic scored lower:             {arabic_lower}/{len(results)} pairs ({arabic_lower/len(results):.0%})")
    print(f"  Arabic scored higher/equal:      {arabic_higher}/{len(results)} pairs")
    print()

    if mean_gap > 0:
        print(f"Conclusion: Arabic queries score an average of {mean_gap:.1f}% lower in cosine")
        print(f"similarity than equivalent English queries, confirming the cross-lingual penalty.")
        print(f"The 0.10 threshold offset in Config compensates for this gap.")
    else:
        print(f"Conclusion: On this dataset, Arabic queries do not consistently score lower.")
        print(f"Mean gap = {mean_gap:.1f}%. The cross-lingual penalty may be offset by the")
        print(f"query rewriter translating Arabic to English before embedding.")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_pairs": len(QUESTION_PAIRS),
            "tables_searched": list(_SEARCH_CONFIG.keys()),
            "note": (
                "Arabic queries are NOT rewritten before embedding in this test. "
                "In production, the query rewriter translates Arabic to English first, "
                "which eliminates the gap. This test measures the raw penalty "
                "to validate why the rewriter and threshold offset exist."
            ),
        },
        "aggregate": {
            "mean_gap_pct": round(mean_gap, 2),
            "median_gap_pct": round(median_gap, 2),
            "std_gap_pct": round(std_gap, 2),
            "min_gap_pct": round(min_gap, 2),
            "max_gap_pct": round(max_gap, 2),
            "arabic_scored_lower_count": arabic_lower,
            "arabic_scored_higher_count": arabic_higher,
            "total_pairs": len(results),
            "pct_pairs_arabic_lower": round(arabic_lower / len(results) * 100, 1),
        },
        "per_pair": results,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_arabic_gap_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic vs. English cosine similarity gap")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    init_db()
    run_arabic_gap_eval(output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
