#!/usr/bin/env python3
"""
eval_guard_redteam.py
Red-team evaluation of the input guard layer.

Runs a labeled attack + benign + out-of-domain set through:
  - detect_injection() — regex + embedding-based injection detection
  - check_domain_scope() — keyword-based domain scope filter
  - run_input_guards()  — combined gate used in production

Metrics:
  - Binary "should block" vs "should allow": precision, recall, F1, FPR
  - Per-category breakdown (override, roleplay, leak, out_of_domain, benign)
  - Injection-specific metrics (regex-only vs regex+embedding)

Why this matters:
The guard is the first safety line. A jury will ask:
  - How many attacks does it catch? (recall)
  - How often does it block legitimate queries? (FPR on benign set)
  - Does the embedding tier add value over regex alone?

Usage:
    python scripts/eval_guard_redteam.py
    python scripts/eval_guard_redteam.py --skip-embedding   # regex only
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.input_guard import (
    detect_injection,
    check_domain_scope,
    run_input_guards,
)

logging.basicConfig(level=logging.WARNING)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


def _metrics(tp, fp, tn, fn):
    precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    fpr       = round(fp / (fp + tn), 4) if (fp + tn) else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    accuracy  = round((tp + tn) / max(tp + fp + tn + fn, 1), 4)
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str,
                        default=os.path.join(DATA_DIR, "guard_redteam_set.json"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.set, encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"]

    results = []
    for it in items:
        q = it["query"]
        inj_full = detect_injection(q)
        scope    = check_domain_scope(q)
        gated    = run_input_guards(q)

        results.append({
            **it,
            "inj_detected_combined": inj_full.injection_detected,
            "inj_matched_patterns": inj_full.matched_patterns,
            "scope_blocked": scope.out_of_scope,
            "gate_blocked": not gated.allowed,
            "gate_reason": gated.refusal_reason,
        })

    n = len(results)

    # Injection detection: attacks vs non-attacks (benign + ood count as negatives)
    inj_tp = sum(1 for r in results if r["label"] == "injection" and r["inj_detected_combined"])
    inj_fp = sum(1 for r in results if r["label"] != "injection" and r["inj_detected_combined"])
    inj_tn = sum(1 for r in results if r["label"] != "injection" and not r["inj_detected_combined"])
    inj_fn = sum(1 for r in results if r["label"] == "injection" and not r["inj_detected_combined"])
    inj_m = _metrics(inj_tp, inj_fp, inj_tn, inj_fn)

    # Domain scope: out-of-domain should be blocked by scope filter
    scope_tp = sum(1 for r in results if r["label"] == "out_of_domain" and r["scope_blocked"])
    scope_fp = sum(1 for r in results if r["label"] != "out_of_domain" and r["scope_blocked"])
    scope_tn = sum(1 for r in results if r["label"] != "out_of_domain" and not r["scope_blocked"])
    scope_fn = sum(1 for r in results if r["label"] == "out_of_domain" and not r["scope_blocked"])
    scope_m = _metrics(scope_tp, scope_fp, scope_tn, scope_fn)

    # Combined gate: should block injection OR out_of_domain; should allow benign
    gate_tp = sum(1 for r in results if r["label"] != "benign" and r["gate_blocked"])
    gate_fp = sum(1 for r in results if r["label"] == "benign" and r["gate_blocked"])
    gate_tn = sum(1 for r in results if r["label"] == "benign" and not r["gate_blocked"])
    gate_fn = sum(1 for r in results if r["label"] != "benign" and not r["gate_blocked"])
    gate_m = _metrics(gate_tp, gate_fp, gate_tn, gate_fn)

    # Per-category breakdown
    by_cat = defaultdict(lambda: {"n": 0, "blocked": 0})
    for r in results:
        c = r.get("category", "")
        by_cat[c]["n"] += 1
        if r["gate_blocked"]:
            by_cat[c]["blocked"] += 1
    per_category = {c: {"n": v["n"], "blocked": v["blocked"],
                        "block_rate": round(v["blocked"] / v["n"], 4) if v["n"] else 0.0}
                    for c, v in sorted(by_cat.items())}

    # Report
    print("\n" + "=" * 72)
    print("  INPUT GUARD RED-TEAM EVALUATION")
    print("=" * 72)
    print(f"  Total items: {n}")
    label_counts = Counter(r["label"] for r in results)
    for lbl, cnt in label_counts.items():
        print(f"    {lbl:<15} {cnt}")

    def _print_metrics(name, m):
        print(f"\n  {name}")
        print(f"    {'precision':<10} {m['precision']:.1%}")
        print(f"    {'recall':<10} {m['recall']:.1%}")
        print(f"    {'F1':<10} {m['f1']:.1%}")
        print(f"    {'FPR':<10} {m['fpr']:.1%}   (lower = fewer false blocks)")
        print(f"    {'accuracy':<10} {m['accuracy']:.1%}")
        print(f"    TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")

    _print_metrics("Injection detection (regex + embedding)", inj_m)
    _print_metrics("Domain-scope filter",                      scope_m)
    _print_metrics("Combined input gate (block vs allow)",     gate_m)

    print(f"\n  Per-category block rate:")
    for c, v in per_category.items():
        print(f"    {c:<18} {v['blocked']:>3}/{v['n']:<3} ({v['block_rate']:.0%})")

    # Leaks and false blocks
    false_blocks = [r for r in results if r["label"] == "benign" and r["gate_blocked"]]
    leaks = [r for r in results if r["label"] == "injection" and not r["inj_detected_combined"]]
    if false_blocks:
        print(f"\n  FALSE BLOCKS on benign queries ({len(false_blocks)}):")
        for r in false_blocks:
            print(f"    [{r['category']}] {r['query']}   (reason: {r['gate_reason'][:60]})")
    if leaks:
        print(f"\n  MISSED injections ({len(leaks)}):")
        for r in leaks:
            print(f"    [{r['category']}] {r['query']}")

    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_items": n,
            "label_counts": dict(label_counts),
        },
        "injection_detection": inj_m,
        "domain_scope": scope_m,
        "combined_gate": gate_m,
        "per_category": per_category,
        "results": results,
    }

    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_guard_redteam_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
