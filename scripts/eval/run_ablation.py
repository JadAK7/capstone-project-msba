#!/usr/bin/env python3
"""
run_ablation.py
Ablation study: disable individual pipeline stages and measure the impact.

Two modes
─────────
--mode general   (default)
    Run all variants against a random 20-question English subset of the
    golden set.  Fast but insensitive — stages won't show large deltas
    because the fast-path already skips rewriting for well-formed English.

--mode stress   (recommended for presentations)
    Run each stage's ablation ONLY against the questions specifically
    designed to stress that stage (data/ablation_stress_set.json):
      • Rewriting  → Arabic queries + one-word queries + follow-up fragments
      • BM25       → exact proper nouns, acronyms, rare terms
      • Reranking  → ambiguous/multi-hop queries where many chunks look correct
    These are the conditions where each stage does real work, so the
    deltas are meaningful.

Output table (stress mode):
  ─── REWRITING ABLATION (n=15, Arabic + short + follow-up) ────────
  Full pipeline          73.2%   92.1%   71.8%
  No query rewriting     48.5%   88.0%   59.2%   (-12.6%)  ← meaningful delta

  ─── BM25 ABLATION (n=10, exact phrases + proper nouns) ───────────
  Full pipeline          77.1%   93.5%   74.0%
  Vector-only (no BM25)  61.4%   90.0%   64.2%   (-9.8%)

  ─── RERANKING ABLATION (n=10, ambiguous / multi-hop) ────────────
  Full pipeline          70.8%   91.0%   69.5%
  No LLM reranking       62.3%   88.5%   63.4%   (-6.1%)

Usage:
    python scripts/run_ablation.py --mode stress
    python scripts/run_ablation.py --mode general
    python scripts/run_ablation.py --mode stress --output results/ablation.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot
from backend.evaluation import evaluate_single

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")


# ---------------------------------------------------------------------------
# Pipeline variant patches (unchanged)
# ---------------------------------------------------------------------------

from contextlib import ExitStack


def _admin_bypass_stack():
    """ExitStack of patches that remove admin-curated shortcuts:
      - exclude custom_notes (faculty_text) from retrieval
      - disable admin feedback correction lookup

    Used to measure retrieval-stage contributions without the curated
    faculty_text source swallowing most queries before BM25 / rerank
    can produce a differential signal.
    """
    from backend import retriever as ret
    from backend.chatbot import LibraryChatbot

    original_retrieve = ret.hybrid_retrieve

    def _retrieve_without_custom_notes(*args, **kwargs):
        tables = kwargs.get("tables")
        if tables is None and len(args) >= 2:
            tables = args[1]
        if tables:
            filtered = [t for t in tables if t != "custom_notes"]
            if "tables" in kwargs:
                kwargs["tables"] = filtered
            else:
                args = (args[0], filtered) + args[2:]
        return original_retrieve(*args, **kwargs)

    stack = ExitStack()
    stack.enter_context(patch.object(ret, "hybrid_retrieve", side_effect=_retrieve_without_custom_notes))
    stack.enter_context(patch.object(LibraryChatbot, "_lookup_feedback", return_value=None))
    return stack


def _with_admin_bypass(runner):
    """Wrap a variant runner so it executes under the admin-bypass patches."""
    def wrapped(chatbot, question):
        with _admin_bypass_stack():
            return runner(chatbot, question)
    return wrapped


def _run_full(chatbot, question):
    return chatbot.answer(question)


def _run_no_rewrite(chatbot, question):
    """Force raw query straight to retrieval — no translation, no expansion."""
    from backend import query_rewriter as qr

    def _bypass(query, history=None, lang="en"):
        # Critically: do NOT mark is_arabic=False for Arabic queries.
        # We just skip the LLM rewrite and use the raw text as-is.
        return query, {
            "original_query": query,
            "is_arabic": False,
            "is_short": len(query.split()) <= 3,
            "is_followup": False,
            "rewrite_skipped": True,
            "rewritten_query": query,
        }

    with patch.object(qr, "rewrite_query", side_effect=_bypass):
        return chatbot.answer(question)


def _run_vector_only(chatbot, question):
    """Disable BM25: keyword search always returns empty."""
    from backend import retriever as ret

    def _no_keywords(*args, **kwargs):
        return []

    with patch.object(ret, "_keyword_search", side_effect=_no_keywords):
        return chatbot.answer(question)


def _run_no_rerank(chatbot, question):
    """Skip LLM reranking: use raw RRF scores directly."""
    from backend import reranker as rr

    def _use_fallback(query, candidates, top_k=8, min_score=0.5):
        return rr._fallback_rerank(candidates, top_k)

    with patch.object(rr, "rerank", side_effect=_use_fallback):
        return chatbot.answer(question)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _avg(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def _is_abstained(answer: str, chosen_source: str) -> bool:
    return (
        chosen_source.startswith("none")
        or "could not find" in answer.lower()
        or "لم أتمكن" in answer
    )


def score_one(q_text, answer, debug) -> dict:
    retrieved = debug.get("retrieved_chunks", [])
    context = debug.get("context_sent_to_llm", "")
    source = debug.get("chosen_source", "")

    if _is_abstained(answer, source):
        return {"answer_relevance": 0.3, "groundedness": 1.0, "faithfulness": 1.0,
                "grounding_score": 0.43, "abstained": True, "chosen_source": source}

    r = evaluate_single(query=q_text, answer=answer, retrieved_chunks=retrieved,
                        chosen_source=source, context_sent_to_llm=context)
    return {
        "answer_relevance": r["metrics"]["answer_relevance"]["score"],
        "groundedness":     r["metrics"]["groundedness"]["score"],
        "faithfulness":     r["metrics"]["faithfulness"]["score"],
        "grounding_score":  r["grounding_score"],
        "abstained":        False,
        "chosen_source":    source,
    }


def run_variant(label, runner, questions, chatbot) -> dict:
    scores = []
    for q in questions:
        q_text = q["question"]
        t0 = time.time()
        try:
            answer, debug = runner(chatbot, q_text)
            s = score_one(q_text, answer, debug)
            s["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            s["question"] = q_text
            s["id"] = q.get("id", "")
        except Exception as e:
            logger.warning("Error on '%s': %s", q_text[:60], e)
            s = {"question": q_text, "id": q.get("id", ""), "error": str(e)}
        scores.append(s)

    ok = [s for s in scores if "error" not in s]
    return {
        "label": label,
        "n_tested": len(questions),
        "n_scored": len(ok),
        "answer_relevance": _avg([s["answer_relevance"] for s in ok]),
        "groundedness":     _avg([s["groundedness"]     for s in ok]),
        "grounding_score":  _avg([s["grounding_score"]  for s in ok]),
        "abstention_rate":  round(len([s for s in ok if s.get("abstained")]) / max(len(ok), 1), 4),
        "mean_elapsed_ms":  _avg([s.get("elapsed_ms", 0) for s in ok]),
        "per_question":     scores,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

W = 72

def _section(title):
    print(f"\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")
    print(f"  {'System Version':<30}  {'Ans Rel':>8}  {'Grounded':>9}  {'Grounding':>10}")
    print(f"  {'·'*62}")


def _row(vr, baseline_rel=None, baseline_grnd=None):
    rel  = vr["answer_relevance"]
    grnd = vr["grounding_score"]
    g    = vr["groundedness"]
    delta = ""
    if baseline_grnd is not None:
        d = grnd - baseline_grnd
        delta = f"  ({d:+.1%})"
        if d < -0.05:
            delta += " ← stage contributes"
        elif d < 0:
            delta += " (minor)"
        else:
            delta += " (no impact)"
    print(f"  {vr['label']:<30}  {rel:>8.1%}  {g:>9.1%}  {grnd:>10.1%}{delta}")


# ---------------------------------------------------------------------------
# STRESS MODE: per-stage targeted ablation
# ---------------------------------------------------------------------------

def run_stress(chatbot, stress_questions, output, bypass_admin: bool = False) -> dict:
    by_stage = {}
    for q in stress_questions:
        stage = q.get("stress_stage", "unknown")
        by_stage.setdefault(stage, []).append(q)

    def _wrap(fn):
        return _with_admin_bypass(fn) if bypass_admin else fn

    if bypass_admin:
        print(f"\n  [admin bypass ON] custom_notes excluded and feedback lookup disabled.")

    all_results = {}

    # ── Rewriting ablation ──────────────────────────────────────────
    rw_qs = by_stage.get("rewriting", [])
    if rw_qs:
        print(f"\n{'='*W}")
        print(f"  REWRITING ABLATION  (n={len(rw_qs)}, Arabic + short queries + follow-ups)")
        print(f"  These are the queries where the rewriter DOES real work:")
        print(f"  Arabic → needs translation; short queries → need expansion.")
        _section("Rewriting ablation results")
        full = run_variant("Full pipeline", _wrap(_run_full), rw_qs, chatbot)
        nowr = run_variant("No query rewriting", _wrap(_run_no_rewrite), rw_qs, chatbot)
        _row(full)
        _row(nowr, full["answer_relevance"], full["grounding_score"])
        all_results["rewriting"] = {"full": full, "no_rewrite": nowr,
                                    "n": len(rw_qs), "stage_delta_grounding": round(nowr["grounding_score"] - full["grounding_score"], 4)}

    # ── BM25 ablation ───────────────────────────────────────────────
    bm_qs = by_stage.get("bm25", [])
    if bm_qs:
        print(f"\n{'='*W}")
        print(f"  BM25 ABLATION  (n={len(bm_qs)}, proper nouns + acronyms + exact phrases)")
        print(f"  These are the queries where keyword matching beats semantic search:")
        print(f"  'ScholarWorks', 'Jafet', 'EEE', 'ILL', 'Al Manhal' — rare exact terms.")
        _section("BM25 ablation results")
        full = run_variant("Full pipeline (hybrid)", _wrap(_run_full), bm_qs, chatbot)
        vect = run_variant("Vector-only (no BM25)", _wrap(_run_vector_only), bm_qs, chatbot)
        _row(full)
        _row(vect, full["answer_relevance"], full["grounding_score"])
        all_results["bm25"] = {"full": full, "vector_only": vect,
                               "n": len(bm_qs), "stage_delta_grounding": round(vect["grounding_score"] - full["grounding_score"], 4)}

    # ── Reranking ablation ──────────────────────────────────────────
    rr_qs = by_stage.get("reranking", [])
    if rr_qs:
        print(f"\n{'='*W}")
        print(f"  RERANKING ABLATION  (n={len(rr_qs)}, ambiguous + multi-hop + overlapping topics)")
        print(f"  These are queries where retrieval returns many plausible-sounding chunks.")
        print(f"  Without reranking, the wrong chunk surfaces first and the answer is wrong or hedged.")
        _section("Reranking ablation results")
        full   = run_variant("Full pipeline", _wrap(_run_full), rr_qs, chatbot)
        norrk  = run_variant("No LLM reranking (RRF only)", _wrap(_run_no_rerank), rr_qs, chatbot)
        _row(full)
        _row(norrk, full["answer_relevance"], full["grounding_score"])
        all_results["reranking"] = {"full": full, "no_rerank": norrk,
                                    "n": len(rr_qs), "stage_delta_grounding": round(norrk["grounding_score"] - full["grounding_score"], 4)}

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  ABLATION SUMMARY — STAGE CONTRIBUTIONS (stress-test mode)")
    print(f"{'='*W}")
    print(f"  {'Stage':<22}  {'n':>4}  {'Full':>8}  {'Ablated':>8}  {'Delta':>8}  {'Verdict'}")
    print(f"  {'─'*66}")

    stage_info = [
        ("rewriting", "Query rewriting",  "no_rewrite",  "no_rewrite"),
        ("bm25",      "BM25 keyword",     "vector_only", "vector_only"),
        ("reranking", "LLM reranking",    "no_rerank",   "no_rerank"),
    ]
    for stage, name, ablated_key, _ in stage_info:
        if stage not in all_results:
            continue
        data = all_results[stage]
        full_g  = data["full"]["grounding_score"]
        abl_key = next(k for k in data if k not in ("full", "n", "stage_delta_grounding"))
        abl_g   = data[abl_key]["grounding_score"]
        delta   = abl_g - full_g
        verdict = "contributes" if delta < -0.03 else "minor" if delta < 0 else "no measurable impact"
        print(f"  {name:<22}  {data['n']:>4}  {full_g:>8.1%}  {abl_g:>8.1%}  {delta:>+7.1%}  {verdict}")

    print(f"{'='*W}\n")
    print("  Negative delta = stage helps. The bigger the drop, the more it contributes.")
    print("  Rewriting delta is most important: it directly validates the Arabic claim.\n")

    return all_results


# ---------------------------------------------------------------------------
# GENERAL MODE: original random-subset approach
# ---------------------------------------------------------------------------

def run_general(chatbot, golden_questions, n, variants_to_run, output, bypass_admin: bool = False) -> dict:
    import random
    random.seed(42)
    GENERAL_CATEGORIES = {"faq_direct", "policy_hours", "database_recommendation", "follow_up_ambiguous"}
    candidates = [q for q in golden_questions
                  if q.get("category") in GENERAL_CATEGORIES
                  and q.get("language") == "en"
                  and q.get("expected_behavior") == "answer"]
    questions = random.sample(candidates, min(n, len(candidates)))

    VARIANTS = {
        "full":        ("Full pipeline",            _run_full),
        "no_rewrite":  ("No query rewriting",       _run_no_rewrite),
        "vector_only": ("Vector-only (no BM25)",    _run_vector_only),
        "no_rerank":   ("No LLM reranking",         _run_no_rerank),
    }

    results = {}
    baseline_rel = baseline_grnd = None

    print(f"\n{'='*W}")
    print(f"  ABLATION STUDY — GENERAL MODE  (n={len(questions)} random English questions)")
    print(f"  NOTE: deltas may be small because the fast-path skips rewriting")
    print(f"  for well-formed English queries. Use --mode stress for meaningful deltas.")
    _section("Results")

    if bypass_admin:
        print(f"  [admin bypass ON] custom_notes excluded and feedback lookup disabled.")

    for key in variants_to_run:
        label, runner = VARIANTS[key]
        print(f"\n  Running variant: {label} ...")
        runner_eff = _with_admin_bypass(runner) if bypass_admin else runner
        vr = run_variant(label, runner_eff, questions, chatbot)
        results[key] = vr
        if key == "full":
            baseline_rel  = vr["answer_relevance"]
            baseline_grnd = vr["grounding_score"]
            _row(vr)
        else:
            _row(vr, baseline_rel, baseline_grnd)

    print(f"\n{'='*W}\n")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation study on pipeline stages")
    parser.add_argument("--mode", choices=["stress", "general"], default="stress",
                        help="stress = targeted questions per stage (recommended); general = random subset")
    parser.add_argument("--golden", type=str,
                        default=os.path.join(DATA_DIR, "golden_set.json"))
    parser.add_argument("--stress-set", type=str,
                        default=os.path.join(DATA_DIR, "ablation_stress_set.json"))
    parser.add_argument("--n", type=int, default=20,
                        help="Questions per variant in general mode (default: 20)")
    parser.add_argument("--variants", nargs="+",
                        default=["full", "no_rewrite", "vector_only", "no_rerank"],
                        choices=["full", "no_rewrite", "vector_only", "no_rerank"],
                        help="Variants to run in general mode")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--bypass-admin", action="store_true",
                        help="Exclude custom_notes (faculty_text) and admin feedback "
                             "corrections so retrieval-stage ablations are measurable. "
                             "Without this flag, admin-curated sources short-circuit the "
                             "pipeline and BM25 / rerank deltas collapse to zero.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set."); sys.exit(1)

    init_db()
    chatbot = LibraryChatbot(os.environ["OPENAI_API_KEY"])

    if args.mode == "stress":
        with open(args.stress_set, encoding="utf-8") as f:
            stress_data = json.load(f)
        results = run_stress(chatbot, stress_data["questions"], args.output,
                             bypass_admin=args.bypass_admin)
    else:
        with open(args.golden, encoding="utf-8") as f:
            golden_data = json.load(f)
        results = run_general(chatbot, golden_data["questions"], args.n, args.variants, args.output,
                              bypass_admin=args.bypass_admin)

    output_path = args.output
    if not output_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_ablation_{args.mode}_{ts}.json",
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": args.mode,
                "bypass_admin": args.bypass_admin,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"Full results saved to {output_path}")
    close_db()


if __name__ == "__main__":
    main()
