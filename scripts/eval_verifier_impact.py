#!/usr/bin/env python3
"""
eval_verifier_impact.py
Measures the impact of the post-generation claim verifier.

For 20 queries, compares draft_answer vs verified_answer from the debug output:
  - How often does the verifier modify the answer?
  - How many claims are removed on average?
  - Does removing claims reduce answer length?
  - Are removals correlated with low retrieval confidence?

Also runs 5 adversarially crafted queries designed to provoke hallucinations,
to test whether the verifier catches them.

Backs up: "Post-generation claim verification reduces hallucination by removing
unsupported claims from X% of answers."

Usage:
    python scripts/eval_verifier_impact.py
    python scripts/eval_verifier_impact.py --output results/verifier.json
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
from backend.chatbot import LibraryChatbot

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

# 20 standard queries — measure verifier modification rate
STANDARD_QUERIES = [
    "What are the library opening hours?",
    "How do I borrow a book from the library?",
    "What are the fines for overdue books?",
    "Can I reserve a study room at AUB?",
    "How do I renew a borrowed book?",
    "Can I access library databases from off-campus?",
    "What printing services are available at the library?",
    "How do I request an interlibrary loan?",
    "Can AUB alumni use the library?",
    "How many books can I borrow at once?",
    "Which databases are available for medical research?",
    "What is PubMed and how do I access it?",
    "What is the borrowing period for books?",
    "Does AUB library have access to JSTOR?",
    "How do I connect to library WiFi?",
    "What are the quiet study areas in the library?",
    "How do I cite sources using APA format?",
    "What is the interlibrary loan service at AUB?",
    "Where is the Saab Medical library?",
    "How do I get a library card?",
]

# 5 adversarial queries — designed to provoke hallucination
# These ask for specific details unlikely to be in the KB
ADVERSARIAL_QUERIES = [
    "What are the exact GPS coordinates of Jafet library?",
    "How much does it cost to print in color per page at AUB library?",
    "What is the maximum fine amount before library privileges are suspended?",
    "How many floors does Jafet library have and what is on each floor?",
    "What are the exact hours for the Jafet library self-checkout machine?",
]


def run_verifier_eval(api_key: str, output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("CLAIM VERIFIER IMPACT EVALUATION")
    print("=" * 70)
    print(f"Standard queries: {len(STANDARD_QUERIES)} | Adversarial: {len(ADVERSARIAL_QUERIES)}\n")

    bot = LibraryChatbot(api_key=api_key)
    bot.clear_cache()

    standard_results = []
    adversarial_results = []

    modified_count = 0
    total_claims_removed = 0
    total_char_reduction = 0

    # --- Standard queries ---
    print("[STANDARD QUERIES] — measuring verifier modification rate")
    for q in STANDARD_QUERIES:
        t0 = time.time()
        answer, debug = bot.answer(q)
        elapsed = (time.time() - t0) * 1000

        draft = debug.get("draft_answer", "")
        removed = debug.get("removed_claims", [])
        verified_flag = debug.get("verified", True)
        cache_hit = debug.get("cache_hit", False)

        # Skip cached answers (no draft vs verified distinction)
        if cache_hit:
            print(f"  [CACHED] {q[:55]!r}")
            standard_results.append({
                "query": q,
                "cached": True,
                "modified": None,
                "claims_removed": None,
            })
            continue

        modified = len(removed) > 0 and removed != [""]
        if modified:
            modified_count += 1
        num_removed = len([r for r in removed if r and not r.startswith("ANSWERABILITY")])
        total_claims_removed += num_removed

        char_diff = len(draft) - len(answer) if draft else 0
        total_char_reduction += max(0, char_diff)

        status = f"MODIFIED ({num_removed} claim(s) removed)" if modified else "UNCHANGED"
        print(f"  [{status}] {q[:50]!r}")
        if modified and removed:
            for claim in removed[:2]:
                print(f"    - removed: {str(claim)[:80]!r}")

        standard_results.append({
            "query": q,
            "cached": False,
            "modified": modified,
            "claims_removed": num_removed,
            "removed_claims": removed,
            "draft_length": len(draft),
            "final_length": len(answer),
            "char_reduction": char_diff,
            "verified_flag": verified_flag,
            "latency_ms": round(elapsed, 1),
        })

    non_cached = [r for r in standard_results if not r.get("cached") and r.get("modified") is not None]
    modification_rate = modified_count / len(non_cached) if non_cached else 0.0
    avg_removed = total_claims_removed / len(non_cached) if non_cached else 0.0
    avg_char_reduction = total_char_reduction / len(non_cached) if non_cached else 0.0

    print(f"\n  Modification rate: {modification_rate:.0%} "
          f"({modified_count}/{len(non_cached)} answers had claims removed)")
    print(f"  Avg claims removed per modified answer: {avg_removed:.1f}")

    # --- Adversarial queries ---
    print("\n[ADVERSARIAL QUERIES] — probing hallucination on specific unknowns")
    adv_modified = 0
    adv_abstained = 0
    for q in ADVERSARIAL_QUERIES:
        t0 = time.time()
        answer, debug = bot.answer(q)
        elapsed = (time.time() - t0) * 1000

        draft = debug.get("draft_answer", "")
        removed = debug.get("removed_claims", [])
        cache_hit = debug.get("cache_hit", False)

        modified = len(removed) > 0 and removed != [""]
        abstained = any(p in answer.lower() for p in [
            "could not find", "not find this", "please contact", "cannot find",
            "not available", "لم أتمكن", "يرجى التواصل"
        ])

        if modified:
            adv_modified += 1
        if abstained:
            adv_abstained += 1

        num_removed = len([r for r in removed if r])
        if abstained:
            status = "ABSTAINED ✓"
        elif modified:
            status = f"MODIFIED  ~ ({num_removed} claim(s) removed)"
        else:
            status = "ANSWERED  (check for hallucination)"

        print(f"  [{status}] {q[:55]!r}")
        print(f"    -> {answer[:100]!r}")

        adversarial_results.append({
            "query": q,
            "cached": cache_hit,
            "modified_by_verifier": modified,
            "abstained": abstained,
            "claims_removed": num_removed,
            "removed_claims": removed,
            "answer_preview": answer[:200],
            "draft_preview": draft[:200] if draft else "",
            "latency_ms": round(elapsed, 1),
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<52} {'Value':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Verifier modification rate (standard queries)':<52} {modification_rate:>7.0%}")
    print(f"  {'Avg claims removed per answer (all standard)':<52} {avg_removed:>8.2f}")
    print(f"  {'Avg character reduction per modified answer':<52} {avg_char_reduction:>8.0f}")
    print(f"  {'Adversarial: modified by verifier':<52} {adv_modified:>6}/{len(ADVERSARIAL_QUERIES)}")
    print(f"  {'Adversarial: abstained (safest outcome)':<52} {adv_abstained:>6}/{len(ADVERSARIAL_QUERIES)}")
    print()
    print(f"Conclusion: The claim verifier modifies {modification_rate:.0%} of answers by removing "
          f"unsupported claims (avg {avg_removed:.1f} per answer). "
          f"On adversarial queries probing specific unknown details, "
          f"the system abstained or modified {adv_abstained + adv_modified}/{len(ADVERSARIAL_QUERIES)} times.")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "standard_queries_count": len(STANDARD_QUERIES),
            "adversarial_queries_count": len(ADVERSARIAL_QUERIES),
        },
        "aggregate": {
            "modification_rate": round(modification_rate, 4),
            "avg_claims_removed": round(avg_removed, 2),
            "avg_char_reduction": round(avg_char_reduction, 1),
            "answers_modified": modified_count,
            "non_cached_answers": len(non_cached),
            "adversarial_modified": adv_modified,
            "adversarial_abstained": adv_abstained,
        },
        "standard_results": standard_results,
        "adversarial_results": adversarial_results,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"eval_verifier_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate claim verifier impact")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    init_db()
    run_verifier_eval(api_key=api_key, output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
