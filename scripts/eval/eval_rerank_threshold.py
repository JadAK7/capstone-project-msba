#!/usr/bin/env python3
"""
eval_rerank_threshold.py
Sweep the reranker's min_score filter and measure end-to-end quality.

backend/source_config.py sets:
    rerank_min_score = 0.45

This threshold filters out reranked chunks with score < min_score.
If it's too high, the generator is starved of context and the pipeline
abstains more often than needed. If it's too low, noisy low-relevance
chunks reach the LLM and hurt groundedness. The sweep shows where the
configured value sits on the quality curve.

Usage:
    python scripts/eval_rerank_threshold.py
    python scripts/eval_rerank_threshold.py --n 15
    python scripts/eval_rerank_threshold.py --thresholds 0.2 0.3 0.4 0.45 0.5 0.6 0.7
    python scripts/eval_rerank_threshold.py --bypass-admin
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
from backend.source_config import SOURCE_CONFIG

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DEFAULT_THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65]


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


def run_threshold(chatbot, questions, threshold: float, bypass_admin: bool):
    scores = []
    with ExitStack() as stack:
        if bypass_admin:
            stack.enter_context(_admin_bypass())
        # Temporarily override the config value
        original_min = SOURCE_CONFIG.rerank_min_score
        SOURCE_CONFIG.rerank_min_score = threshold
        try:
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
        finally:
            SOURCE_CONFIG.rerank_min_score = original_min

    ok = [s for s in scores if "error" not in s]
    def _avg(k):
        vals = [s[k] for s in ok if k in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
    return {
        "rerank_min_score": threshold,
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
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--bypass-admin", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set."); sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        data = json.load(f)
    candidates = [q for q in data["questions"]
                  if q.get("expected_behavior") == "answer" and q.get("language") == "en"]
    import random; random.seed(42)
    questions = random.sample(candidates, min(args.n, len(candidates)))

    init_db()
    chatbot = LibraryChatbot(os.environ["OPENAI_API_KEY"])

    configured = SOURCE_CONFIG.rerank_min_score
    results = {}
    print(f"\n  Sweeping rerank_min_score over {args.thresholds} on {len(questions)} questions "
          f"{'[admin bypass ON] ' if args.bypass_admin else ''}...\n")
    for T in args.thresholds:
        print(f"  -- rerank_min_score = {T:.3f} --")
        vr = run_threshold(chatbot, questions, T, args.bypass_admin)
        results[f"T_{T:.3f}"] = vr
        print(f"     rel={vr['answer_relevance']:.3f}  grd={vr['groundedness']:.3f}  "
              f"gs={vr['grounding_score']:.3f}  abst={vr['abstention_rate']:.2%}  "
              f"ms={vr['mean_elapsed_ms']:.0f}")

    close_db()

    # Summary
    print("\n" + "=" * 72)
    print("  RERANK MIN_SCORE SWEEP SUMMARY")
    print("=" * 72)
    print(f"  {'T':>7} {'rel':>7} {'grd':>7} {'gs':>7} {'abst':>7}")
    for T in args.thresholds:
        r = results[f"T_{T:.3f}"]
        mark = "  ← configured" if abs(T - configured) < 1e-6 else ""
        print(f"  {T:>7.3f} {r['answer_relevance']:>7.3f} {r['groundedness']:>7.3f} "
              f"{r['grounding_score']:>7.3f} {r['abstention_rate']:>7.2%}{mark}")

    best = max(args.thresholds, key=lambda T: results[f"T_{T:.3f}"]["grounding_score"])
    print(f"\n  Best grounding_score at T = {best:.3f}")
    print(f"  Configured: {configured}")

    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions": len(questions),
            "bypass_admin": args.bypass_admin,
            "thresholds_tested": args.thresholds,
            "configured_threshold": configured,
        },
        "results": results,
    }
    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_rerank_threshold_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
