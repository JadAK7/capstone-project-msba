#!/usr/bin/env python3
"""
eval_feedback_similarity.py
Evaluate the admin feedback similarity threshold (default FEEDBACK_MIN_SIMILARITY=0.85).

When a student asks a question similar to one an admin has previously corrected,
the pipeline reuses the correction. This test measures whether the 0.85 cosine
similarity threshold correctly distinguishes:
  - SAME-intent pairs (correction should transfer)     → above threshold
  - SIMILAR-but-different pairs (should NOT transfer)  → below threshold
  - DIFFERENT-topic pairs (obviously below threshold)  → well below threshold

For each candidate threshold T, reports precision, recall, F1 of the
"treat as same-intent" decision. Plots the ROC-style curve so you can
justify the chosen threshold on real distinguishability evidence.

Usage:
    python scripts/eval_feedback_similarity.py
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

from backend.embeddings import embed_texts

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CHATBOT_DEFAULT_THRESHOLD = 0.85


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str,
                        default=os.path.join(DATA_DIR, "feedback_similarity_set.json"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.set, encoding="utf-8") as f:
        data = json.load(f)
    pairs = data["pairs"]

    # Batch-embed all unique strings once
    texts = []
    for p in pairs:
        texts.append(p["a"]); texts.append(p["b"])
    vectors = embed_texts(texts)
    vectors = [np.array(v, dtype=np.float64) for v in vectors]

    # Compute similarity per pair
    for i, p in enumerate(pairs):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        p["similarity"] = round(_cosine(va, vb), 4)
        p["gold_same"] = (p["relation"] == "same")

    # Summary stats per relation
    rel_stats = {}
    for rel in ("same", "similar", "different"):
        sims = [p["similarity"] for p in pairs if p["relation"] == rel]
        if sims:
            rel_stats[rel] = {
                "n": len(sims),
                "min":  round(min(sims), 4),
                "mean": round(sum(sims) / len(sims), 4),
                "max":  round(max(sims), 4),
            }

    # Threshold sweep: classify "same" if sim >= T
    thresholds = [round(0.50 + 0.02 * i, 2) for i in range(26)]  # 0.50 .. 1.00
    sweep = []
    for T in thresholds:
        tp = sum(1 for p in pairs if p["gold_same"] and p["similarity"] >= T)
        fp = sum(1 for p in pairs if not p["gold_same"] and p["similarity"] >= T)
        fn = sum(1 for p in pairs if p["gold_same"] and p["similarity"] < T)
        tn = sum(1 for p in pairs if not p["gold_same"] and p["similarity"] < T)
        precision, recall, f1 = _prf1(tp, fp, fn)
        sweep.append({"threshold": T, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                      "precision": precision, "recall": recall, "f1": f1})

    best_f1 = max(sweep, key=lambda x: x["f1"])
    chosen = next((s for s in sweep if s["threshold"] == CHATBOT_DEFAULT_THRESHOLD), None)

    # Report
    print("\n" + "=" * 72)
    print("  FEEDBACK SIMILARITY THRESHOLD EVALUATION")
    print("=" * 72)
    print(f"  n pairs = {len(pairs)}  (configured threshold = {CHATBOT_DEFAULT_THRESHOLD})\n")

    print("  Similarity distribution by relation:")
    print(f"  {'relation':<10} {'n':>4} {'min':>8} {'mean':>8} {'max':>8}")
    for rel, s in rel_stats.items():
        print(f"  {rel:<10} {s['n']:>4} {s['min']:>8.3f} {s['mean']:>8.3f} {s['max']:>8.3f}")

    print(f"\n  Threshold sweep (classify 'same' if sim >= T):")
    print(f"  {'T':>5} {'prec':>7} {'rec':>7} {'F1':>7} {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4}")
    for s in sweep:
        mark = ""
        if s["threshold"] == best_f1["threshold"]:
            mark += "  ← best F1"
        if s["threshold"] == CHATBOT_DEFAULT_THRESHOLD:
            mark += "  ← configured"
        print(f"  {s['threshold']:>5.2f} {s['precision']:>7.1%} {s['recall']:>7.1%} "
              f"{s['f1']:>7.1%} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {s['tn']:>4}{mark}")

    print(f"\n  Best F1: {best_f1['f1']:.1%} at T={best_f1['threshold']}")
    if chosen:
        print(f"  At configured T={CHATBOT_DEFAULT_THRESHOLD}: "
              f"P={chosen['precision']:.1%} R={chosen['recall']:.1%} F1={chosen['f1']:.1%}")
        if abs(chosen["f1"] - best_f1["f1"]) < 0.02:
            print(f"  → configured threshold is within 2% of the optimum on this set.")
        else:
            print(f"  → best is T={best_f1['threshold']}; delta F1 = {best_f1['f1'] - chosen['f1']:+.1%}")

    # Save
    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_pairs": len(pairs),
            "configured_threshold": CHATBOT_DEFAULT_THRESHOLD,
        },
        "relation_stats": rel_stats,
        "threshold_sweep": sweep,
        "best_f1": best_f1,
        "pairs": pairs,
    }
    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_feedback_sim_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
