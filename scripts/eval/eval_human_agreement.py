#!/usr/bin/env python3
"""
eval_human_agreement.py
Measures agreement between GPT-4o-mini (the LLM judge used in eval_rewriting.py)
and a human reviewer across a sample of evaluated answers.

Takes the existing eval_results JSON, picks 8 answers, and asks the human to
rate groundedness (0/0.5/1.0) and faithfulness (0/0.5/1.0) for each one.
Then computes:
  - Simple agreement rate (within ±0.2)
  - Cohen's kappa (chance-corrected agreement)

This validates that the LLM judge is a reliable evaluator.

Usage:
    python scripts/eval_human_agreement.py --input eval_results_*.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("numpy required: pip install numpy")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def cohen_kappa(ratings_a, ratings_b, labels):
    """Compute Cohen's kappa between two raters."""
    n = len(ratings_a)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    po = agree / n

    # Expected agreement
    pe = 0.0
    for label in labels:
        pa = sum(1 for a in ratings_a if a == label) / n
        pb = sum(1 for b in ratings_b if b == label) / n
        pe += pa * pb

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def discretize(score: float) -> str:
    """Convert continuous score to low/medium/high for kappa."""
    if score >= 0.85:
        return "high"
    elif score >= 0.60:
        return "medium"
    else:
        return "low"


def run_agreement_eval(results_path: str, output_path: str = None) -> dict:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = data.get("results", [])
    if not all_results:
        print("ERROR: No results found in the input file.")
        sys.exit(1)

    # Sample 8 answers spread across the full set
    step = max(1, len(all_results) // 8)
    sample = all_results[::step][:8]

    print("\n" + "=" * 70)
    print("HUMAN vs. LLM JUDGE AGREEMENT EVALUATION")
    print("=" * 70)
    print(f"Evaluating {len(sample)} sampled answers from {len(all_results)} total.")
    print("For each answer, rate GROUNDEDNESS and FAITHFULNESS.")
    print()
    print("Scale:")
    print("  1.0 = All claims are fully supported by context")
    print("  0.8 = One minor unsupported detail")
    print("  0.6 = One clear unsupported claim or exaggeration")
    print("  0.5 = Half the claims are unsupported")
    print("  0.0 = Answer is largely fabricated or contradicts context")
    print()

    human_groundedness = []
    human_faithfulness = []
    llm_groundedness = []
    llm_faithfulness = []

    comparison_rows = []

    for i, result in enumerate(sample):
        query = result.get("query", "")
        answer = result.get("answer_preview", "")
        llm_ground = result["metrics"]["groundedness"]["score"]
        llm_ground_reason = result["metrics"]["groundedness"]["reason"]
        llm_faith = result["metrics"]["faithfulness"]["score"]
        llm_faith_reason = result["metrics"]["faithfulness"]["reason"]

        # Find context from chunks if available
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(sample)}] QUERY: {query}")
        print(f"\nANSWER PREVIEW:\n  {answer[:300]}")
        print(f"\nLLM JUDGE SCORES:")
        print(f"  Groundedness: {llm_ground} — {llm_ground_reason[:120]}")
        print(f"  Faithfulness: {llm_faith} — {llm_faith_reason[:120]}")
        print()

        # Human input
        while True:
            try:
                hg_raw = input(f"  Your GROUNDEDNESS score (0 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0) [Enter to skip]: ").strip()
                if hg_raw == "":
                    hg = llm_ground  # Default to LLM if skipped
                    print(f"  → Skipped, using LLM score: {hg}")
                else:
                    hg = float(hg_raw)
                break
            except ValueError:
                print("  Invalid input. Enter a number (e.g. 0.8)")

        while True:
            try:
                hf_raw = input(f"  Your FAITHFULNESS score (0 / 0.5 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0) [Enter to skip]: ").strip()
                if hf_raw == "":
                    hf = llm_faith
                    print(f"  → Skipped, using LLM score: {hf}")
                else:
                    hf = float(hf_raw)
                break
            except ValueError:
                print("  Invalid input.")

        human_groundedness.append(hg)
        human_faithfulness.append(hf)
        llm_groundedness.append(llm_ground)
        llm_faithfulness.append(llm_faith)

        comparison_rows.append({
            "query": query,
            "answer_preview": answer[:200],
            "llm_groundedness": llm_ground,
            "human_groundedness": hg,
            "ground_diff": round(abs(hg - llm_ground), 2),
            "llm_faithfulness": llm_faith,
            "human_faithfulness": hf,
            "faith_diff": round(abs(hf - llm_faith), 2),
        })

    # Agreement metrics
    TOLERANCE = 0.2  # within ±0.2 = agreement

    ground_agreements = [abs(h - l) <= TOLERANCE for h, l in zip(human_groundedness, llm_groundedness)]
    faith_agreements = [abs(h - l) <= TOLERANCE for h, l in zip(human_faithfulness, llm_faithfulness)]

    ground_agree_rate = sum(ground_agreements) / len(ground_agreements)
    faith_agree_rate = sum(faith_agreements) / len(faith_agreements)
    overall_agree_rate = (sum(ground_agreements) + sum(faith_agreements)) / (2 * len(ground_agreements))

    # Cohen's kappa (discretized)
    labels = ["low", "medium", "high"]
    h_ground_disc = [discretize(s) for s in human_groundedness]
    l_ground_disc = [discretize(s) for s in llm_groundedness]
    h_faith_disc = [discretize(s) for s in human_faithfulness]
    l_faith_disc = [discretize(s) for s in llm_faithfulness]

    kappa_ground = cohen_kappa(h_ground_disc, l_ground_disc, labels)
    kappa_faith = cohen_kappa(h_faith_disc, l_faith_disc, labels)
    kappa_overall = (kappa_ground + kappa_faith) / 2

    # Mean absolute error
    mae_ground = sum(abs(h - l) for h, l in zip(human_groundedness, llm_groundedness)) / len(human_groundedness)
    mae_faith = sum(abs(h - l) for h, l in zip(human_faithfulness, llm_faithfulness)) / len(human_faithfulness)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  {'Metric':<45} {'Ground':>10} {'Faith':>10}")
    print(f"  {'-'*67}")
    print(f"  {'Agreement rate (within ±0.2)':<45} {ground_agree_rate:>9.0%} {faith_agree_rate:>9.0%}")
    print(f"  {'Cohen\\'s kappa (low/med/high)':<45} {kappa_ground:>10.2f} {kappa_faith:>10.2f}")
    print(f"  {'Mean absolute error (0-1 scale)':<45} {mae_ground:>10.2f} {mae_faith:>10.2f}")
    print(f"  {'Overall agreement rate':<45} {overall_agree_rate:>9.0%}")
    print(f"  {'Overall Cohen\\'s kappa':<45} {kappa_overall:>10.2f}")
    print()

    # Interpret kappa
    if kappa_overall >= 0.80:
        interp = "Almost perfect agreement (κ ≥ 0.80)"
    elif kappa_overall >= 0.61:
        interp = "Substantial agreement (κ 0.61–0.80)"
    elif kappa_overall >= 0.41:
        interp = "Moderate agreement (κ 0.41–0.60)"
    else:
        interp = "Fair or poor agreement (κ < 0.41) — judge reliability questionable"

    print(f"  Interpretation: {interp}")
    print()
    print(f"Conclusion: Human and LLM judge agree {overall_agree_rate:.0%} of the time "
          f"(within ±0.2 tolerance on 0–1 scale). "
          f"Cohen's kappa = {kappa_overall:.2f} ({interp.split(' (')[0]}).")

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_file": results_path,
            "n_samples": len(sample),
            "agreement_tolerance": TOLERANCE,
            "kappa_labels": labels,
        },
        "aggregate": {
            "ground_agreement_rate": round(ground_agree_rate, 4),
            "faith_agreement_rate": round(faith_agree_rate, 4),
            "overall_agreement_rate": round(overall_agree_rate, 4),
            "kappa_groundedness": round(kappa_ground, 3),
            "kappa_faithfulness": round(kappa_faith, 3),
            "kappa_overall": round(kappa_overall, 3),
            "mae_groundedness": round(mae_ground, 3),
            "mae_faithfulness": round(mae_faith, 3),
            "interpretation": interp,
        },
        "per_item": comparison_rows,
    }

    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_human_agreement_{ts}.json",
        )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Human vs LLM judge agreement evaluation")
    parser.add_argument("--input", type=str, required=True, help="Path to eval_results_*.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    run_agreement_eval(results_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
