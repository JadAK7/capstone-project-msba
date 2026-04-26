#!/usr/bin/env python3
"""
eval_intent.py
Evaluate the regex-based intent classifier against a labeled set.

Measures:
  - overall accuracy
  - per-intent precision / recall / F1
  - confusion matrix

Why this test matters:
The intent classifier routes queries to the correct source tables (databases,
scraped pages, FAQ). Misclassification causes bad retrieval and wasted LLM
tokens. A jury will ask why a regex is used instead of an LLM classifier;
this test answers whether the regex is accurate enough to justify the simpler,
zero-latency design.

Usage:
    python scripts/eval_intent.py
    python scripts/eval_intent.py --output eval_intent.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.intent_classifier import classify_intent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


def _f1(p, r):
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str,
                        default=os.path.join(DATA_DIR, "intent_labels.json"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    intents = data.get("intent_labels", ["database", "hours", "contact", "faq", "general"])

    # Run classifier on each question
    predictions = []
    for q in questions:
        pred = classify_intent(q["query"]).get("intent", "general")
        predictions.append({
            "id": q["id"],
            "query": q["query"],
            "expected": q["intent"],
            "predicted": pred,
            "correct": pred == q["intent"],
            "lang": q.get("lang", "en"),
        })

    # Overall metrics
    n = len(predictions)
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = round(correct / n, 4) if n else 0.0

    # Per-intent P / R / F1
    per_intent = {}
    for intent in intents:
        tp = sum(1 for p in predictions if p["expected"] == intent and p["predicted"] == intent)
        fp = sum(1 for p in predictions if p["expected"] != intent and p["predicted"] == intent)
        fn = sum(1 for p in predictions if p["expected"] == intent and p["predicted"] != intent)
        support = tp + fn
        precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
        per_intent[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Confusion matrix (expected → predicted)
    confusion = defaultdict(lambda: Counter())
    for p in predictions:
        confusion[p["expected"]][p["predicted"]] += 1
    confusion = {k: dict(v) for k, v in confusion.items()}

    # Report
    print("\n" + "=" * 72)
    print("  INTENT CLASSIFIER EVALUATION (regex-based)")
    print("=" * 72)
    print(f"  Overall accuracy: {accuracy:.1%}  ({correct}/{n})\n")
    print(f"  {'Intent':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-'*50}")
    for intent in intents:
        m = per_intent[intent]
        print(f"  {intent:<12} {m['precision']:>7.1%} {m['recall']:>7.1%} {m['f1']:>7.1%} {m['support']:>8}")

    print("\n  Confusion matrix (rows=expected, cols=predicted):")
    header = "  " + " " * 12 + "".join(f"{i:>10}" for i in intents)
    print(header)
    for row in intents:
        counts = confusion.get(row, {})
        cells = "".join(f"{counts.get(col, 0):>10}" for col in intents)
        print(f"  {row:<12}{cells}")

    # Per-question errors for manual inspection
    errors = [p for p in predictions if not p["correct"]]
    if errors:
        print(f"\n  Misclassifications ({len(errors)}):")
        for e in errors:
            print(f"    [{e['expected']:<8} → {e['predicted']:<8}] {e['query'][:70]}")

    # Save
    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions": n,
        },
        "overall_accuracy": accuracy,
        "per_intent": per_intent,
        "confusion_matrix": confusion,
        "predictions": predictions,
    }

    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_intent_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
