#!/usr/bin/env python3
"""
eval_bilingual_consistency.py
Measures whether Arabic and English versions of the same question produce
equivalent answers after the query rewriter translates Arabic → English.

For 15 paired questions, runs both through the FULL pipeline (with rewriting
enabled), then measures semantic similarity between the two answers using
OpenAI embeddings.

Backs up: "The query rewriter closes the cross-lingual gap: Arabic and English
queries produce semantically equivalent answers."

High similarity (≥0.85) = the rewriter successfully bridged the language gap.
Low similarity (<0.70) = the Arabic path produced a different / incomplete answer.

Usage:
    python scripts/eval_bilingual_consistency.py
    python scripts/eval_bilingual_consistency.py --output results/bilingual.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot
from backend.embeddings import embed_text

# ---------------------------------------------------------------------------
# 15 question pairs — identical intent, English + Arabic
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
        "What are the fines for overdue books?",
        "ما هي غرامة تأخير إعادة الكتب؟",
    ),
    (
        "How do I renew a borrowed book?",
        "كيف أجدد استعارة كتاب مستعار؟",
    ),
    (
        "Can I access library databases from off-campus?",
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
        "Which databases are available for medical research?",
        "ما هي قواعد البيانات المتاحة للبحث الطبي؟",
    ),
    (
        "What is the borrowing period for books?",
        "ما هي مدة استعارة الكتب؟",
    ),
    (
        "Can I reserve a study room at the library?",
        "هل يمكنني حجز غرفة دراسة في المكتبة؟",
    ),
    (
        "What time does Jafet library close?",
        "متى تغلق مكتبة جافيت؟",
    ),
    (
        "How do I get a library card?",
        "كيف أحصل على بطاقة المكتبة؟",
    ),
    (
        "What scanning and copying services are available?",
        "ما هي خدمات المسح الضوئي والنسخ المتاحة؟",
    ),
]

# Thresholds for classifying answer similarity
_HIGH_CONSISTENCY = 0.85   # answers are semantically equivalent
_LOW_CONSISTENCY  = 0.70   # answers diverge meaningfully


def cosine_similarity(v1: list, v2: list) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def run_bilingual_eval(api_key: str, output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("BILINGUAL ANSWER CONSISTENCY EVALUATION")
    print("=" * 70)
    print(f"Question pairs: {len(QUESTION_PAIRS)} | Consistency threshold: {_HIGH_CONSISTENCY}\n")
    print("Note: Both English and Arabic queries run through the full pipeline.")
    print("      Arabic is translated to English by the query rewriter before retrieval.\n")

    bot = LibraryChatbot(api_key=api_key)
    bot.clear_cache()

    results = []
    similarities = []
    high_consistency = 0
    low_consistency = 0
    blocked_count = 0

    for i, (en_q, ar_q) in enumerate(QUESTION_PAIRS):
        print(f"[{i+1:02d}] {en_q[:55]!r}")

        # English answer
        t0 = time.time()
        en_answer, en_debug = bot.answer(en_q)
        en_time = (time.time() - t0) * 1000

        # Arabic answer
        t0 = time.time()
        ar_answer, ar_debug = bot.answer(ar_q)
        ar_time = (time.time() - t0) * 1000

        # Check if either was blocked
        en_blocked = en_debug.get("guard_blocked", False)
        ar_blocked = ar_debug.get("guard_blocked", False)

        if en_blocked or ar_blocked:
            blocked_count += 1
            print(f"      EN blocked={en_blocked} AR blocked={ar_blocked} — skipping similarity")
            results.append({
                "pair_id": i + 1,
                "english_query": en_q,
                "arabic_query": ar_q,
                "english_answer_preview": en_answer[:120],
                "arabic_answer_preview": ar_answer[:120],
                "answer_similarity": None,
                "consistency_level": "blocked",
                "en_blocked": en_blocked,
                "ar_blocked": ar_blocked,
                "en_rewritten": en_debug.get("rewrite_debug", {}).get("rewritten_query", en_q),
                "ar_rewritten": ar_debug.get("rewrite_debug", {}).get("rewritten_query", ar_q),
                "en_latency_ms": round(en_time, 1),
                "ar_latency_ms": round(ar_time, 1),
            })
            continue

        # Embed both answers and compute cosine similarity
        en_vec = embed_text(en_answer)
        ar_vec = embed_text(ar_answer)
        sim = cosine_similarity(en_vec, ar_vec)
        similarities.append(sim)

        if sim >= _HIGH_CONSISTENCY:
            high_consistency += 1
            level = "HIGH"
            mark = "✓"
        elif sim >= _LOW_CONSISTENCY:
            level = "MED"
            mark = "~"
        else:
            low_consistency += 1
            level = "LOW"
            mark = "✗"

        ar_rewritten = ar_debug.get("rewrite_debug", {}).get("rewritten_query", ar_q)
        print(f"      AR rewritten → {ar_rewritten[:55]!r}")
        print(f"      Similarity: {sim:.3f} [{level} {mark}]")

        results.append({
            "pair_id": i + 1,
            "english_query": en_q,
            "arabic_query": ar_q,
            "arabic_rewritten_to": ar_rewritten,
            "english_answer_preview": en_answer[:120],
            "arabic_answer_preview": ar_answer[:120],
            "answer_similarity": round(sim, 4),
            "consistency_level": level,
            "en_blocked": False,
            "ar_blocked": False,
            "en_latency_ms": round(en_time, 1),
            "ar_latency_ms": round(ar_time, 1),
        })

    # Aggregate stats (exclude blocked pairs)
    valid = [r for r in results if r.get("answer_similarity") is not None]
    if similarities:
        avg_sim = float(np.mean(similarities))
        median_sim = float(np.median(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))
        high_rate = high_consistency / len(valid)
    else:
        avg_sim = median_sim = min_sim = max_sim = high_rate = 0.0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<48} {'Value':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Avg answer similarity (EN vs AR answer)':<48} {avg_sim:>8.3f}")
    print(f"  {'Median answer similarity':<48} {median_sim:>8.3f}")
    print(f"  {'Min / Max similarity':<48} {min_sim:.3f} / {max_sim:.3f}")
    print(f"  {'High consistency pairs (≥{:.0f}%)':<48} {high_consistency:>6}/{len(valid)}".format(_HIGH_CONSISTENCY * 100))
    print(f"  {'Low consistency pairs (<{:.0f}%)':<48} {low_consistency:>6}/{len(valid)}".format(_LOW_CONSISTENCY * 100))
    print(f"  {'Blocked / skipped pairs':<48} {blocked_count:>6}/{len(QUESTION_PAIRS)}")
    print()
    print(f"Conclusion: Arabic and English queries produce answers with an average "
          f"semantic similarity of {avg_sim:.2f}.")
    if avg_sim >= _HIGH_CONSISTENCY:
        print(f"The query rewriter successfully bridges the cross-lingual gap — "
              f"{high_rate:.0%} of pairs achieve high consistency (≥{_HIGH_CONSISTENCY:.0%}).")
    else:
        print(f"Some divergence remains: {low_consistency} pair(s) scored below {_LOW_CONSISTENCY:.0%}, "
              f"indicating the Arabic path may produce incomplete or different answers for those queries.")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "num_pairs": len(QUESTION_PAIRS),
            "high_consistency_threshold": _HIGH_CONSISTENCY,
            "low_consistency_threshold": _LOW_CONSISTENCY,
            "note": (
                "Answer similarity is computed as cosine similarity between "
                "OpenAI embeddings of the English and Arabic answers. "
                "Arabic queries are translated to English by the query rewriter "
                "before retrieval — this test validates that translation produces "
                "equivalent answers."
            ),
        },
        "aggregate": {
            "avg_answer_similarity": round(avg_sim, 4),
            "median_answer_similarity": round(median_sim, 4),
            "min_similarity": round(min_sim, 4),
            "max_similarity": round(max_sim, 4),
            "high_consistency_count": high_consistency,
            "low_consistency_count": low_consistency,
            "high_consistency_rate": round(high_rate, 4),
            "blocked_pairs": blocked_count,
            "valid_pairs": len(valid),
        },
        "per_pair": results,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_bilingual_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate bilingual answer consistency")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    init_db()
    run_bilingual_eval(api_key=api_key, output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
