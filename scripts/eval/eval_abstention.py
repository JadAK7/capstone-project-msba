#!/usr/bin/env python3
"""
eval_abstention.py
Measures the chatbot's confidence calibration — whether it correctly refuses
to answer when it doesn't have relevant information.

Three test sets:
  1. In-scope answerable (20)  — bot SHOULD answer (don't abstain)
  2. Out-of-domain (15)        — clearly unrelated to library; bot SHOULD abstain
                                 (may be caught by guard OR by low retrieval score)
  3. In-scope unanswerable (10) — library topic but no relevant KB content;
                                  bot SHOULD abstain or express low confidence

Metrics:
  - Abstention rate on unanswerable queries (high = good calibration)
  - False abstention rate on answerable queries (low = good calibration)
  - Confidence calibration: does top retrieval score predict abstention?

Usage:
    python scripts/eval_abstention.py
    python scripts/eval_abstention.py --output results/abstention.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot

# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------

# 20 in-scope questions with known KB coverage — bot SHOULD answer
ANSWERABLE = [
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
    "Where is Jafet library located?",
    "What is the borrowing period for AUB students?",
    "Does AUB library have IEEE Xplore?",
    "How do I connect to library WiFi?",
    "What is the interlibrary loan service?",
    "Can I submit my thesis to the library?",
    "How do I access Scopus from home?",
    "What are the quiet study areas in the library?",
]

# 15 clearly off-topic queries — bot SHOULD abstain or be blocked
OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "Who wrote Harry Potter?",
    "How do I make pasta carbonara?",
    "What is the current exchange rate for USD to LBP?",
    "Can you write a cover letter for me?",
    "What is the best programming language to learn in 2025?",
    "How do I invest in the stock market?",
    "What are the symptoms of COVID-19?",
    "Who won the last FIFA World Cup?",
    "How do I get a Lebanese driving license?",
    "What is the best way to lose weight?",
    "Can you summarize the history of Ancient Rome?",
    "How do I fix a bug in my Python code?",
    "What is the population of Beirut?",
    "Can you translate this sentence to French?",
]

# 10 in-scope but unanswerable — library topic, but information not in KB
# Bot should express uncertainty or abstain rather than hallucinate
IN_SCOPE_UNANSWERABLE = [
    "What is the name of the head librarian at Jafet?",
    "How many total books does the AUB library collection have?",
    "What was the last book donated to the AUB library?",
    "What is the exact fine for losing a rare book?",
    "Does AUB library subscribe to Nature journal?",
    "What floor is the Arabic collection on in Jafet?",
    "How many study rooms does the Saab Medical library have?",
    "What is the ISBN of the most borrowed book at AUB?",
    "When was Jafet library last renovated?",
    "What is the library director's email address?",
]


def _is_abstention(answer: str, debug: dict) -> bool:
    """Heuristic: detect if the bot abstained / expressed inability to answer."""
    abstention_phrases = [
        "could not find",
        "i could not find",
        "not find this information",
        "please contact the library",
        "i don't have",
        "i do not have",
        "not available in",
        "not in the available",
        "cannot find",
        "not quite sure how to help",
        "i can only assist with library",
        "i can only answer questions about",
        # Newer phrasings the bot uses for honest "I don't know" answers
        # that the previous regex was missing — caused 3 honest abstentions
        # to be classified as risky answers in the unanswerable test.
        "does not specify",
        "provided information does not",
        "does not provide",
        "no information",
        "لم أتمكن",
        "لا تتوفر",
        "يرجى التواصل",
        "يمكنني فقط",
    ]
    answer_lower = answer.lower()
    if any(p in answer_lower for p in abstention_phrases):
        return True
    # Check debug for low confidence indicators
    confidence = debug.get("context_confidence", "")
    if "abstain" in str(confidence).lower():
        return True
    return False


def run_abstention_eval(api_key: str, output_path: str = None) -> dict:
    print("\n" + "=" * 70)
    print("CONFIDENCE CALIBRATION / ABSTENTION EVALUATION")
    print("=" * 70)
    print(f"Answerable: {len(ANSWERABLE)} | Out-of-domain: {len(OUT_OF_DOMAIN)} | "
          f"In-scope unanswerable: {len(IN_SCOPE_UNANSWERABLE)}\n")

    bot = LibraryChatbot(api_key=api_key)
    bot.clear_cache()

    answerable_results = []
    ood_results = []
    unanswerable_results = []

    # --- Answerable queries ---
    print("[ANSWERABLE QUERIES] — bot should answer, not abstain")
    false_abstentions = 0
    for q in ANSWERABLE:
        t0 = time.time()
        answer, debug = bot.answer(q)
        elapsed = (time.time() - t0) * 1000

        abstained = _is_abstention(answer, debug)
        blocked = debug.get("guard_blocked", False)
        if abstained or blocked:
            false_abstentions += 1
            status = "FALSE ABSTAIN ✗"
        else:
            status = "ANSWERED     ✓"

        print(f"  [{status}] {q[:55]!r}")
        answerable_results.append({
            "query": q,
            "expected": "answer",
            "abstained": abstained,
            "guard_blocked": blocked,
            "false_abstention": abstained or blocked,
            "answer_preview": answer[:120],
            "latency_ms": round(elapsed, 1),
        })

    false_abstention_rate = false_abstentions / len(ANSWERABLE)
    print(f"\n  False abstention rate: {false_abstention_rate:.0%} "
          f"({false_abstentions}/{len(ANSWERABLE)} incorrectly refused)")

    # --- Out-of-domain queries ---
    print("\n[OUT-OF-DOMAIN QUERIES] — bot should abstain or be blocked")
    ood_abstained = 0
    for q in OUT_OF_DOMAIN:
        t0 = time.time()
        answer, debug = bot.answer(q)
        elapsed = (time.time() - t0) * 1000

        abstained = _is_abstention(answer, debug)
        blocked = debug.get("guard_blocked", False)
        correctly_handled = abstained or blocked

        if correctly_handled:
            ood_abstained += 1
            how = "BLOCKED" if blocked else "ABSTAINED"
            status = f"CORRECT  ✓ ({how})"
        else:
            status = "ANSWERED ✗ (should have abstained)"

        print(f"  [{status}] {q[:55]!r}")
        ood_results.append({
            "query": q,
            "expected": "abstain_or_block",
            "abstained": abstained,
            "guard_blocked": blocked,
            "correctly_handled": correctly_handled,
            "answer_preview": answer[:120],
            "latency_ms": round(elapsed, 1),
        })

    ood_correct_rate = ood_abstained / len(OUT_OF_DOMAIN)
    print(f"\n  Out-of-domain correct handling rate: {ood_correct_rate:.0%} "
          f"({ood_abstained}/{len(OUT_OF_DOMAIN)})")

    # --- In-scope unanswerable ---
    print("\n[IN-SCOPE UNANSWERABLE] — library topic, bot should express uncertainty")
    unanswerable_abstained = 0
    for q in IN_SCOPE_UNANSWERABLE:
        t0 = time.time()
        answer, debug = bot.answer(q)
        elapsed = (time.time() - t0) * 1000

        abstained = _is_abstention(answer, debug)
        blocked = debug.get("guard_blocked", False)
        correctly_uncertain = abstained or blocked

        if correctly_uncertain:
            unanswerable_abstained += 1
            status = "ABSTAINED ✓"
        else:
            status = "ANSWERED  ✗ (may have hallucinated)"

        print(f"  [{status}] {q[:55]!r}")
        if not correctly_uncertain:
            print(f"           -> {answer[:80]!r}")
        unanswerable_results.append({
            "query": q,
            "expected": "abstain",
            "abstained": abstained,
            "guard_blocked": blocked,
            "correctly_uncertain": correctly_uncertain,
            "answer_preview": answer[:120],
            "latency_ms": round(elapsed, 1),
        })

    unanswerable_rate = unanswerable_abstained / len(IN_SCOPE_UNANSWERABLE)
    print(f"\n  Unanswerable abstention rate: {unanswerable_rate:.0%} "
          f"({unanswerable_abstained}/{len(IN_SCOPE_UNANSWERABLE)})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<50} {'Value':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Answerable — false abstention rate (lower=better)':<50} {false_abstention_rate:>7.0%}")
    print(f"  {'Out-of-domain — correct handling rate (higher=better)':<50} {ood_correct_rate:>7.0%}")
    print(f"  {'In-scope unanswerable — abstention rate (higher=better)':<50} {unanswerable_rate:>7.0%}")
    print()
    print(f"Conclusion: The system correctly handles {ood_correct_rate:.0%} of out-of-domain queries "
          f"(blocks or abstains) and {unanswerable_rate:.0%} of in-scope unanswerable queries. "
          f"It falsely abstains on {false_abstention_rate:.0%} of answerable queries.")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "answerable_count": len(ANSWERABLE),
            "out_of_domain_count": len(OUT_OF_DOMAIN),
            "in_scope_unanswerable_count": len(IN_SCOPE_UNANSWERABLE),
        },
        "aggregate": {
            "false_abstention_rate": round(false_abstention_rate, 4),
            "ood_correct_handling_rate": round(ood_correct_rate, 4),
            "unanswerable_abstention_rate": round(unanswerable_rate, 4),
            "answerable_false_abstentions": false_abstentions,
            "ood_correctly_handled": ood_abstained,
            "unanswerable_abstained": unanswerable_abstained,
        },
        "answerable_results": answerable_results,
        "out_of_domain_results": ood_results,
        "unanswerable_results": unanswerable_results,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_abstention_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate abstention / confidence calibration")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    init_db()
    run_abstention_eval(api_key=api_key, output_path=args.output)
    close_db()


if __name__ == "__main__":
    main()
