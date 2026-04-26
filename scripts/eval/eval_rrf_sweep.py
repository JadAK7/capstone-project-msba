#!/usr/bin/env python3
"""
eval_rrf_sweep.py
Sweep the RRF vector/keyword weight split and measure retrieval quality.

Current hybrid retrieval (backend/retriever.py:301-302) uses:
    vector_weight = 0.65
    keyword_weight = 0.35  (complement)

This test patches _reciprocal_rank_fusion to use different weight splits,
runs the pipeline on a subset of the golden set, and reports grounding_score
at each split. The goal is to justify 0.65/0.35 by showing it is at or near
the quality optimum, not a guess.

Usage:
    python scripts/eval_rrf_sweep.py
    python scripts/eval_rrf_sweep.py --n 15
    python scripts/eval_rrf_sweep.py --weights 0.0 0.35 0.5 0.65 0.8 1.0
    python scripts/eval_rrf_sweep.py --bypass-admin
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
from backend import retriever as ret

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DEFAULT_WEIGHTS = [0.0, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]


def _admin_bypass():
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


def _patched_rrf(vector_weight: float):
    """Returns a patched _reciprocal_rank_fusion that uses the given weight split."""
    original = ret._reciprocal_rank_fusion
    kw_weight = 1.0 - vector_weight

    def _impl(vector_results, keyword_results, k=60, **kwargs):
        return original(vector_results, keyword_results, k=k,
                        vector_weight=vector_weight,
                        keyword_weight=kw_weight)
    return _impl


def _is_abstained(answer, source):
    return (source.startswith("none")
            or "could not find" in answer.lower()
            or "لم أتمكن" in answer)


def _score(q, answer, debug):
    source = debug.get("chosen_source", "")
    if _is_abstained(answer, source):
        return {"abstained": True, "answer_relevance": 0.3, "groundedness": 1.0,
                "grounding_score": 0.43}
    try:
        r = evaluate_single(
            query=q["question"], answer=answer,
            retrieved_chunks=debug.get("retrieved_chunks", []),
            chosen_source=source,
            context_sent_to_llm=debug.get("context_sent_to_llm", ""),
        )
        return {
            "abstained": False,
            "answer_relevance": r["metrics"]["answer_relevance"]["score"],
            "groundedness":     r["metrics"]["groundedness"]["score"],
            "grounding_score":  r["grounding_score"],
        }
    except Exception as e:
        return {"abstained": False, "error": str(e),
                "answer_relevance": 0, "groundedness": 0, "grounding_score": 0}


def run_weight(chatbot, questions, vector_weight: float, bypass_admin: bool):
    patched = _patched_rrf(vector_weight)
    scores = []

    with ExitStack() as stack:
        if bypass_admin:
            stack.enter_context(_admin_bypass())
        stack.enter_context(patch.object(ret, "_reciprocal_rank_fusion", side_effect=patched))

        for q in questions:
            t0 = time.time()
            try:
                answer, debug = chatbot.answer(q["question"])
                s = _score(q, answer, debug)
                s["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
                s["id"] = q["id"]
                s["chosen_source"] = debug.get("chosen_source", "")
            except Exception as e:
                s = {"id": q["id"], "error": str(e)}
            scores.append(s)

    ok = [s for s in scores if "error" not in s]
    def _avg(k):
        vals = [s[k] for s in ok if k in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
    return {
        "vector_weight": vector_weight,
        "keyword_weight": round(1.0 - vector_weight, 4),
        "n_scored": len(ok),
        "answer_relevance": _avg("answer_relevance"),
        "groundedness": _avg("groundedness"),
        "grounding_score": _avg("grounding_score"),
        "abstention_rate": round(sum(1 for s in ok if s.get("abstained")) / max(len(ok), 1), 4),
        "mean_elapsed_ms": _avg("elapsed_ms"),
        "per_question": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str,
                        default=os.path.join(DATA_DIR, "golden_set.json"))
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    parser.add_argument("--bypass-admin", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set."); sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        data = json.load(f)
    # Use answer-expected, English questions for a clean sweep
    candidates = [q for q in data["questions"]
                  if q.get("expected_behavior") == "answer" and q.get("language") == "en"]
    import random; random.seed(42)
    questions = random.sample(candidates, min(args.n, len(candidates)))

    init_db()
    chatbot = LibraryChatbot(os.environ["OPENAI_API_KEY"])

    results = {}
    print(f"\n  Sweeping RRF vector_weight over {args.weights} on {len(questions)} questions "
          f"{'[admin bypass ON] ' if args.bypass_admin else ''}...\n")
    for w in args.weights:
        print(f"  -- vector_weight = {w:.2f} --")
        vr = run_weight(chatbot, questions, w, args.bypass_admin)
        results[f"vw_{w:.2f}"] = vr
        print(f"     rel={vr['answer_relevance']:.3f}  grd={vr['groundedness']:.3f}  "
              f"gs={vr['grounding_score']:.3f}  abst={vr['abstention_rate']:.2%}  "
              f"ms={vr['mean_elapsed_ms']:.0f}")

    close_db()

    # Summary table
    print("\n" + "=" * 72)
    print("  RRF WEIGHT SWEEP SUMMARY")
    print("=" * 72)
    print(f"  {'vec_w':>6} {'kw_w':>6} {'rel':>7} {'grd':>7} {'gs':>7} {'abst':>7}")
    for w in args.weights:
        r = results[f"vw_{w:.2f}"]
        mark = "  ← configured" if abs(w - 0.65) < 1e-6 else ""
        print(f"  {w:>6.2f} {1-w:>6.2f} {r['answer_relevance']:>7.3f} {r['groundedness']:>7.3f} "
              f"{r['grounding_score']:>7.3f} {r['abstention_rate']:>7.2%}{mark}")

    best = max(args.weights, key=lambda w: results[f"vw_{w:.2f}"]["grounding_score"])
    print(f"\n  Best grounding_score at vector_weight = {best:.2f}")
    print(f"  Configured: 0.65")

    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions": len(questions),
            "bypass_admin": args.bypass_admin,
            "weights_tested": args.weights,
        },
        "results": results,
    }
    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_rrf_sweep_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
