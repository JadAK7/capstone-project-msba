#!/usr/bin/env python3
"""
eval_threshold_sweep.py
Sweep the context-sufficiency thresholds (partial / confident) to show
whether the configured values are at or near the operating point that
balances answer coverage against abstention precision.

Decision rule under test (backend/chatbot.py:755):
    top_score >= confident_threshold → confident answer
    partial <= top_score < confident → partial answer (extra caution)
    top_score < partial              → abstain

For each candidate threshold T in a sweep:
  - abstain = (top_rerank_score < T)
  - for golden questions with expected_behavior='answer': abstaining is a FALSE abstention
  - for golden questions with expected_behavior in {refuse, guard_block}: abstaining is CORRECT
  - plus: among non-abstained questions, how good is the answer (grounding_score)?

The script runs each golden question ONCE through the full pipeline to capture
top_rerank_score, answer, and retrieved chunks. All threshold sweeping is
post-hoc, so the cost is O(golden_set_size) not O(golden × thresholds).

Usage:
    python scripts/eval_threshold_sweep.py
    python scripts/eval_threshold_sweep.py --n 30              # subset
    python scripts/eval_threshold_sweep.py --bypass-admin      # exclude custom_notes
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import patch
from contextlib import ExitStack

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot
from backend.evaluation import evaluate_single

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CFG_CONFIDENT = 0.60
CFG_PARTIAL   = 0.45


def _admin_bypass():
    from backend import retriever as ret
    from backend.chatbot import LibraryChatbot as LC
    original = ret.hybrid_retrieve

    def _no_custom_notes(*args, **kwargs):
        tables = kwargs.get("tables")
        if tables is None and len(args) >= 2:
            tables = args[1]
        if tables:
            filtered = [t for t in tables if t != "custom_notes"]
            if "tables" in kwargs:
                kwargs["tables"] = filtered
            else:
                args = (args[0], filtered) + args[2:]
        return original(*args, **kwargs)

    stack = ExitStack()
    stack.enter_context(patch.object(ret, "hybrid_retrieve", side_effect=_no_custom_notes))
    stack.enter_context(patch.object(LC, "_lookup_feedback", return_value=None))
    return stack


def _top_score_from_debug(debug: dict) -> float:
    # Prefer the explicit field (set on abstain), fall back to retrieved_chunks[0].score
    if "top_rerank_score" in debug:
        return float(debug["top_rerank_score"])
    chunks = debug.get("retrieved_chunks", [])
    if chunks:
        return float(chunks[0].get("score", 0.0))
    return 0.0


def _score_answer(q, answer, debug):
    source = debug.get("chosen_source", "")
    is_refusal = (
        source.startswith("none")
        or "could not find" in answer.lower()
        or "لم أتمكن" in answer
    )
    if is_refusal:
        return {"answered": False, "grounding_score": None, "answer_relevance": None}
    try:
        r = evaluate_single(
            query=q["question"], answer=answer,
            retrieved_chunks=debug.get("retrieved_chunks", []),
            chosen_source=source,
            context_sent_to_llm=debug.get("context_sent_to_llm", ""),
        )
        return {
            "answered": True,
            "grounding_score": r["grounding_score"],
            "answer_relevance": r["metrics"]["answer_relevance"]["score"],
        }
    except Exception as e:
        return {"answered": True, "grounding_score": None, "answer_relevance": None, "error": str(e)}


def run_once(chatbot, q, bypass_admin: bool):
    if bypass_admin:
        with _admin_bypass():
            answer, debug = chatbot.answer(q["question"])
    else:
        answer, debug = chatbot.answer(q["question"])
    top_score = _top_score_from_debug(debug)
    scored = _score_answer(q, answer, debug)
    return {
        "id": q["id"],
        "question": q["question"],
        "expected_behavior": q.get("expected_behavior", "answer"),
        "category": q.get("category", ""),
        "top_score": top_score,
        "actually_abstained": not scored["answered"],
        "grounding_score": scored["grounding_score"],
        "answer_relevance": scored["answer_relevance"],
        "chosen_source": debug.get("chosen_source", ""),
    }


def sweep(per_q, thresholds):
    """For each threshold T, simulate the abstention decision and compute metrics."""
    rows = []
    for T in thresholds:
        # Simulated abstention per question at this threshold
        sim_results = []
        for q in per_q:
            sim_abstain = q["top_score"] < T
            expected_answer = q["expected_behavior"] == "answer"
            expected_refuse = q["expected_behavior"] in ("refuse", "guard_block")
            sim_results.append({
                "expected_answer": expected_answer,
                "expected_refuse": expected_refuse,
                "sim_abstain": sim_abstain,
                "gs": q["grounding_score"],
            })

        n = len(sim_results)
        n_ans = sum(1 for r in sim_results if r["expected_answer"])
        n_ref = sum(1 for r in sim_results if r["expected_refuse"])

        # On answer questions: sim_abstain is BAD (false abstention)
        false_abstention = sum(1 for r in sim_results if r["expected_answer"] and r["sim_abstain"])
        # On refuse questions: sim_abstain is GOOD
        correct_refusal = sum(1 for r in sim_results if r["expected_refuse"] and r["sim_abstain"])
        missed_refusal  = sum(1 for r in sim_results if r["expected_refuse"] and not r["sim_abstain"])

        answer_coverage    = round(1 - (false_abstention / n_ans), 4) if n_ans else 0.0
        abstention_recall  = round(correct_refusal / n_ref, 4) if n_ref else 0.0
        # Precision of abstention: of all things we would abstain on, how many should we?
        all_sim_abstained  = sum(1 for r in sim_results if r["sim_abstain"])
        abstention_precision = round(correct_refusal / all_sim_abstained, 4) if all_sim_abstained else 0.0

        # Grounding score on questions that remain answered (only for expected_answer)
        kept_gs = [r["gs"] for r in sim_results if r["expected_answer"] and not r["sim_abstain"] and r["gs"] is not None]
        avg_gs = round(sum(kept_gs) / len(kept_gs), 4) if kept_gs else None

        rows.append({
            "threshold": T,
            "n": n,
            "answer_coverage": answer_coverage,         # how many answer-questions we still answer
            "abstention_recall": abstention_recall,      # how many refuse-questions we correctly drop
            "abstention_precision": abstention_precision,
            "mean_grounding_score_answered": avg_gs,
            "false_abstention": false_abstention,
            "correct_refusal": correct_refusal,
            "missed_refusal": missed_refusal,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str,
                        default=os.path.join(DATA_DIR, "golden_set.json"))
    parser.add_argument("--n", type=int, default=0,
                        help="Subset size; 0 = use all")
    parser.add_argument("--bypass-admin", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set."); sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if args.n:
        questions = questions[:args.n]

    init_db()
    chatbot = LibraryChatbot(os.environ["OPENAI_API_KEY"])

    print(f"  Running pipeline on {len(questions)} golden questions "
          f"{'[admin bypass ON] ' if args.bypass_admin else ''}...")
    per_q = []
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        try:
            per_q.append(run_once(chatbot, q, args.bypass_admin))
        except Exception as e:
            print(f"    [err] {q.get('id')}: {e}")
        if i % 5 == 0:
            print(f"    {i}/{len(questions)}  ({time.time()-t0:.1f}s)")
    close_db()

    # Sweep
    thresholds = [round(0.20 + 0.025 * i, 3) for i in range(33)]  # 0.20 .. 1.00
    rows = sweep(per_q, thresholds)

    # Report
    print("\n" + "=" * 82)
    print("  CONFIDENCE THRESHOLD SWEEP")
    print("=" * 82)
    n_ans = sum(1 for q in per_q if q["expected_behavior"] == "answer")
    n_ref = sum(1 for q in per_q if q["expected_behavior"] in ("refuse", "guard_block"))
    print(f"  n answer-expected = {n_ans}   n refuse-expected = {n_ref}\n")
    print(f"  {'T':>5} {'coverage':>9} {'abst_rec':>9} {'abst_prec':>10} {'mean_gs':>9}   verdict")

    for r in rows:
        mark = ""
        if abs(r["threshold"] - CFG_PARTIAL) < 1e-6:   mark += " ← configured partial"
        if abs(r["threshold"] - CFG_CONFIDENT) < 1e-6: mark += " ← configured confident"
        gs = f"{r['mean_grounding_score_answered']:.3f}" if r["mean_grounding_score_answered"] is not None else "  n/a"
        print(f"  {r['threshold']:>5.3f} {r['answer_coverage']:>8.1%} "
              f"{r['abstention_recall']:>8.1%} {r['abstention_precision']:>9.1%} "
              f"{gs:>9}{mark}")

    # Summary: sweet-spot = max(answer_coverage * abstention_precision + lambda * mean_gs)
    # We'll report best-F1-like composite: harmonic mean of coverage and abst_recall
    def _composite(r):
        p, rc = r["answer_coverage"], r["abstention_recall"]
        return 2 * p * rc / (p + rc) if (p + rc) else 0.0
    best = max(rows, key=_composite)
    print(f"\n  Best coverage/abstention balance: T={best['threshold']:.3f}  "
          f"coverage={best['answer_coverage']:.1%}  abst_recall={best['abstention_recall']:.1%}")
    print(f"  Configured partial={CFG_PARTIAL}, confident={CFG_CONFIDENT}")

    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions": len(per_q),
            "bypass_admin": args.bypass_admin,
            "configured_partial": CFG_PARTIAL,
            "configured_confident": CFG_CONFIDENT,
        },
        "per_question": per_q,
        "sweep": rows,
        "best_balance": best,
    }
    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_threshold_sweep_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
