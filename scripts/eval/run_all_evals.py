#!/usr/bin/env python3
"""
run_all_evals.py
Run every evaluation script in sequence and collect results into one directory.

Prerequisites:
    1. PostgreSQL is running  (docker-compose up -d)
    2. Indices are built      (python scripts/build_index.py)
    3. .env has OPENAI_API_KEY

Usage:
    # Full suite — every eval in the project (~1-2 hr with LLM-judged runs)
    python scripts/run_all_evals.py

    # Quick smoke-test — smaller question counts per script
    python scripts/run_all_evals.py --quick

    # No-LLM evals only — structural / latency / guard / intent / feedback-sim
    python scripts/run_all_evals.py --fast-only

    # Just the design-choice evals (baselines + threshold/rrf/rerank sweeps +
    # red-team + verifier catch-rate + intent + feedback)
    python scripts/run_all_evals.py --design-only

    # Specific scripts by key
    python scripts/run_all_evals.py --only golden baselines threshold

    # Choose output directory
    python scripts/run_all_evals.py --out results/run_20260423

At the end, a consolidated SCORECARD prints headline metrics for every eval
that produced a parseable JSON — one-glance regression dashboard after
any code change.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Pass/fail gates
# ---------------------------------------------------------------------------
# Each (key, metric) listed here is checked against a target. Rows not listed
# get no gate (status=None). Direction:
#   "min"  — value must be >= threshold (e.g. faithfulness)
#   "max"  — value must be <= threshold (e.g. false-positive rate)
#   "abs"  — when True, compare abs(value) instead of value (for symmetric
#            gaps where either direction past the bound is a failure)
# Targets reflect defense thresholds, not currently observed values — gates
# are designed to FAIL on regression so a diff against the last run flags
# real problems. Update intentionally, not to silence a failing run.
THRESHOLDS = {
    ("ragas",          "faithfulness"):     {"min": 0.80},
    ("ragas",          "answer_relevancy"): {"min": 0.60},
    ("golden",         "grounding_score"):  {"min": 0.65},
    ("golden",         "false abst rate"):  {"max": 0.30},
    ("golden",         "EN−AR gap"):        {"max": 0.08, "abs": True},
    ("redteam",        "injection F1"):     {"min": 0.85},
    ("guard",          "inj TPR"):          {"min": 0.80},
    ("guard",          "legit FPR"):        {"max": 0.10},
    ("verifier-catch", "false-pos rate"):   {"max": 0.30},
}


def _gate_status(key: str, metric: str, value):
    """Return 'PASS' / 'FAIL' / None for a (key, metric, value) triple.

    None means no gate is defined for this row, OR the value couldn't be
    interpreted as a number (string/None/missing — render falls back to
    'n/a' rather than failing the run).
    """
    if value is None or not isinstance(value, (int, float)):
        return None
    spec = THRESHOLDS.get((key, metric))
    if not spec:
        return None
    v = abs(value) if spec.get("abs") else value
    if "min" in spec:
        return "PASS" if v >= spec["min"] else "FAIL"
    if "max" in spec:
        return "PASS" if v <= spec["max"] else "FAIL"
    return None


def _gate_label(key: str, metric: str) -> str:
    """Human-readable threshold label, e.g. '≥0.80' or '≤0.30'."""
    spec = THRESHOLDS.get((key, metric))
    if not spec:
        return ""
    if "min" in spec:
        return f"≥{spec['min']:.2f}"
    if "max" in spec:
        prefix = "|·|≤" if spec.get("abs") else "≤"
        return f"{prefix}{spec['max']:.2f}"
    return ""


# ---------------------------------------------------------------------------
# Post-processing scripts
# ---------------------------------------------------------------------------
# Run AFTER the main eval loop because they read other scripts' outputs.
# Each entry takes (out_dir, out_file) as positional argv and writes its
# own JSON. Skipped if any of `needs` didn't run successfully.
POST_PROCESS = [
    {
        "key": "cost",
        "script": "eval_cost.py",
        "needs": ["baselines"],
        "desc": "Cost-quality Pareto + annual projections from baselines.json",
    },
    {
        "key": "bootstrap_ci",
        "script": "eval_bootstrap_ci.py",
        "needs": ["golden", "baselines"],
        "desc": "95% bootstrap CIs for headline metrics + paired baseline deltas",
    },
]

# ---------------------------------------------------------------------------
# Script registry
# ---------------------------------------------------------------------------
# Each entry:
#   key       – short name used with --only
#   script    – path relative to SCRIPTS_DIR
#   args      – extra CLI args (injected at runtime with --output)
#   fast_args – used instead of args when --quick is set
#   needs_llm – True = uses LLM judge (slow, costs money)
#   fast_only – False = excluded when --fast-only is set
#   desc      – one-line description printed in the header
# ---------------------------------------------------------------------------

REGISTRY = [
    {
        "key": "golden",
        "script": "run_golden_eval.py",
        "args": [],
        "fast_args": ["--categories", "faq_direct", "arabic", "adversarial", "out_of_domain"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Golden set — overall metrics, EN/AR parity, per-category, abstention, guard accuracy",
    },
    {
        "key": "ragas",
        "script": "eval_ragas.py",
        "args": [],
        "fast_args": ["--n", "10"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Ragas — faithfulness, answer_relevancy, context_precision (standard RAG benchmark)",
    },
    {
        "key": "ablation",
        "script": "run_ablation.py",
        "args": [],
        "fast_args": ["--n", "10"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Ablation — contribution of rewriting, BM25, and LLM reranking",
    },
    {
        "key": "latency",
        "script": "eval_latency.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Latency — per-stage timing, cold vs cached, P50/P95/P99",
    },
    {
        "key": "guard",
        "script": "eval_guard_accuracy.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Guard accuracy — injection detection, out-of-domain refusal rate",
    },
    {
        "key": "source",
        "script": "eval_source_selection.py",
        "args": [],
        "fast_args": ["--dry-run"],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Source selection — FAQ vs scraped vs database routing correctness",
    },
    {
        "key": "hybrid",
        "script": "eval_hybrid.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Hybrid retrieval — vector+BM25 vs vector-only hit-rate",
    },
    {
        "key": "rewriting",
        "script": "eval_rewriting.py",
        "args": [],
        "fast_args": [],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Query rewriting — follow-up resolution and Arabic translation accuracy",
    },
    {
        "key": "arabic",
        "script": "eval_arabic_gap.py",
        "args": [],
        "fast_args": [],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Arabic gap — cross-lingual retrieval quality before/after rewriting",
    },
    {
        "key": "bilingual",
        "script": "eval_bilingual_consistency.py",
        "args": [],
        "fast_args": [],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Bilingual consistency — same question EN vs AR answer similarity",
    },
    {
        "key": "abstention",
        "script": "eval_abstention.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Abstention calibration — false abstention rate, threshold sensitivity",
    },
    {
        "key": "cache",
        "script": "eval_cache.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Cache — hit rate, semantic match threshold, latency reduction",
    },
    {
        "key": "verifier",
        "script": "eval_verifier_impact.py",
        "args": [],
        "fast_args": [],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Verifier impact — hallucination reduction from claim verification",
    },

    # ─── Design-choice evaluations (added for jury defense) ─────────────
    {
        "key": "baselines",
        "script": "eval_baselines.py",
        "args": ["--n", "20"],
        "fast_args": ["--n", "8", "--baselines", "B0", "B1", "B4"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Baselines — full pipeline vs LLM-only / BM25-only / FAQ lookup / summarize",
    },
    {
        "key": "threshold",
        "script": "eval_threshold_sweep.py",
        "args": ["--n", "40", "--bypass-admin"],
        "fast_args": ["--n", "15", "--bypass-admin"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Confidence threshold sweep — justifies partial=0.45 / confident=0.60",
    },
    {
        "key": "rrf",
        "script": "eval_rrf_sweep.py",
        "args": ["--n", "15", "--bypass-admin"],
        "fast_args": ["--n", "8", "--weights", "0.35", "0.65", "1.0", "--bypass-admin"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "RRF weight sweep — justifies vector=0.65 / keyword=0.35 split",
    },
    {
        "key": "rerank",
        "script": "eval_rerank_threshold.py",
        "args": ["--n", "15", "--bypass-admin"],
        "fast_args": ["--n", "8", "--thresholds", "0.35", "0.45", "0.55", "--bypass-admin"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Rerank min_score sweep — justifies rerank_min_score=0.45",
    },
    {
        "key": "redteam",
        "script": "eval_guard_redteam.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Input guard red-team — injection/scope F1 on 60-item labeled set",
    },
    {
        "key": "verifier-catch",
        "script": "eval_verifier.py",
        "args": ["--n", "15"],
        "fast_args": ["--n", "6"],
        "needs_llm": True,
        "fast_only": False,
        "desc": "Verifier catch-rate — synthetic claim corruption + false-positive rate",
    },
    {
        "key": "intent",
        "script": "eval_intent.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Intent classifier accuracy — regex vs labeled intent set",
    },
    {
        "key": "feedback",
        "script": "eval_feedback_similarity.py",
        "args": [],
        "fast_args": [],
        "needs_llm": False,
        "fast_only": True,
        "desc": "Feedback similarity threshold — justifies FEEDBACK_MIN_SIMILARITY=0.85",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _safe_load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(v, spec=".3f", suffix=""):
    if v is None:
        return "  n/a"
    try:
        return f"{float(v):{spec}}{suffix}"
    except Exception:
        return str(v)


def _bootstrap_cis_for_golden(out_dir: str) -> dict:
    """Compute headline 95% bootstrap CIs from golden.json so we can attach
    a CI column to relevant scorecard rows. Best-effort — returns {} if
    the file is missing, malformed, or the import fails (no numpy etc.).
    """
    try:
        sys.path.insert(0, SCRIPTS_DIR)
        from eval_bootstrap_ci import headline_cis  # type: ignore
        import numpy as _np
        golden_path = os.path.join(out_dir, "golden.json")
        if not os.path.exists(golden_path):
            return {}
        with open(golden_path, encoding="utf-8") as f:
            golden = json.load(f)
        if "raw_results" not in golden:
            return {}
        rng = _np.random.default_rng(seed=42)
        return (headline_cis(golden, n_boot=1000, rng=rng) or {}).get("metrics", {})
    except Exception as e:
        print(f"  [scorecard] bootstrap CI computation skipped: {e}")
        return {}


# Map (scorecard key, scorecard metric) → bootstrap metric key.
# Only golden-set rows have CIs wired up — ragas/redteam/etc. would need
# their own per-row resampling logic to be paired correctly.
_CI_MAP = {
    ("golden", "grounding_score"):  "grounding_score_composite",
    ("golden", "false abst rate"):  "abstention_rate",
}


def build_scorecard(summary: list, out_dir: str) -> dict:
    """Read each per-script output JSON and extract a headline metric.

    The scorecard is a single-glance dashboard: one row per eval with the
    key number you would quote to the jury. Shape differs per script, so
    each extractor is bespoke and falls back to 'n/a' if the shape doesn't
    match (e.g., script failed).

    Each row carries:
      - value:  the headline number
      - gate:   "PASS" / "FAIL" / None depending on THRESHOLDS
      - ci:     {ci_low, ci_high, n} for golden rows that have a bootstrap
                CI (eval_bootstrap_ci.headline_cis); None otherwise

    main() exits non-zero when any gate is FAIL.
    """
    rows = []
    out_by_key = {r["key"]: r for r in summary}
    ci_table = _bootstrap_cis_for_golden(out_dir)

    def _row(key, metric_name, value, note=""):
        gate = _gate_status(key, metric_name, value)
        gate_label = _gate_label(key, metric_name)
        ci_key = _CI_MAP.get((key, metric_name))
        ci = ci_table.get(ci_key) if ci_key else None
        rows.append({
            "key": key,
            "metric": metric_name,
            "value": value,
            "note": note,
            "gate": gate,
            "gate_threshold": gate_label,
            "ci": ci,
        })

    # ── Existing suite ──────────────────────────────────────────────
    g = _safe_load((out_by_key.get("golden") or {}).get("output_file"))
    if g:
        metrics = (g.get("table1_overall") or {}).get("metrics") or {}
        gs = metrics.get("grounding_score_composite") or metrics.get("grounding_score")
        gap = ((g.get("table2_parity") or {}).get("gaps_en_minus_ar") or {}).get("grounding_score")
        abst = (g.get("table4_abstention") or {}).get("false_abstention_rate")
        n_ans = metrics.get("_composite_basis", "")
        _row("golden", "grounding_score", gs, n_ans or "answered-only composite")
        _row("golden", "EN−AR gap",        gap, "negative = AR higher")
        _row("golden", "false abst rate",  abst, "in-scope questions abstained")

        # Truncation warning: groundedness/faithfulness prompts cap context
        # at 5000 chars. If many rows hit that cap, supported claims may
        # have been clipped from the judge's view → suppressed grounding.
        raw = g.get("raw_results", [])
        n_ctx_trunc = sum(1 for r in raw if r.get("context_truncated"))
        n_ans_trunc = sum(1 for r in raw if r.get("answer_truncated"))
        if n_ctx_trunc or n_ans_trunc:
            note = []
            if n_ctx_trunc:
                note.append(f"{n_ctx_trunc} ctx>5000c")
            if n_ans_trunc:
                note.append(f"{n_ans_trunc} ans>2000c")
            _row("golden", "judge truncations", n_ctx_trunc + n_ans_trunc,
                 "⚠ " + " / ".join(note) + " — judge may underscore grounding")

    rg = _safe_load((out_by_key.get("ragas") or {}).get("output_file"))
    if rg:
        if rg.get("status") == "skipped":
            _row("ragas", "status", "skipped", rg.get("reason", ""))
        else:
            ov = rg.get("overall") or rg.get("metrics") or rg.get("scores") or {}
            _row("ragas", "faithfulness",      ov.get("faithfulness"),      "claims grounded in contexts")
            _row("ragas", "answer_relevancy",  ov.get("answer_relevancy"),  "answer addresses question")
            _row("ragas", "context_precision", ov.get("context_precision"), "retrieved contexts relevant")
            if "answer_correctness" in ov:
                _row("ragas", "answer_correctness", ov.get("answer_correctness"), "vs ground_truth")
            if "context_recall" in ov:
                _row("ragas", "context_recall", ov.get("context_recall"), "vs ground_truth")

    a = _safe_load((out_by_key.get("ablation") or {}).get("output_file"))
    if a:
        res = a.get("results") or {}
        rw  = ((res.get("rewriting") or {}).get("stage_delta_grounding"))
        bm  = ((res.get("bm25")      or {}).get("stage_delta_grounding"))
        rk  = ((res.get("reranking") or {}).get("stage_delta_grounding"))
        _row("ablation", "rewriting Δ",  rw, "no_rewrite − full; negative = stage helps")
        _row("ablation", "bm25 Δ",       bm, "vector_only − full")
        _row("ablation", "reranking Δ",  rk, "no_rerank − full")

    lat = _safe_load((out_by_key.get("latency") or {}).get("output_file"))
    if lat:
        agg = lat.get("aggregate") or {}
        _row("latency", "avg cold ms",    agg.get("avg_full_cold_ms"))
        _row("latency", "avg cached ms",  agg.get("avg_full_cached_ms"))
        _row("latency", "cache speedup",  agg.get("cache_speedup"), "cold / cached")

    guard = _safe_load((out_by_key.get("guard") or {}).get("output_file"))
    if guard:
        agg = guard.get("aggregate") or {}
        _row("guard", "inj TPR",   agg.get("true_positive_rate"),  "injections caught")
        _row("guard", "legit FPR", agg.get("false_positive_rate"), "false blocks")
        _row("guard", "scope acc", agg.get("scope_filter_accuracy"))

    src = _safe_load((out_by_key.get("source") or {}).get("output_file"))
    if src:
        md = src.get("metadata") or {}
        if md:
            _row("source", "routing accuracy", md.get("accuracy"),
                 f"{md.get('n_passed','?')}/{md.get('n_tests','?')}")
        elif src.get("mode") == "dry_run":
            _row("source", "status", "dry_run only", f"{src.get('n_tests','?')} tests")

    # ── Design-choice evaluations ──────────────────────────────────
    b = _safe_load((out_by_key.get("baselines") or {}).get("output_file"))
    if b:
        # New format nests baselines under summaries.<subset> (e.g. "answerable").
        # Old format had them flat under summary.<baseline>. Support both.
        s = b.get("summary") or (b.get("summaries") or {}).get("answerable") or {}
        b0 = ((s.get("B0") or {}).get("overall") or {}).get("grounding_score")
        b0_rel = ((s.get("B0") or {}).get("overall") or {}).get("answer_relevance")
        others_gs, others_rel = [], []
        for k in ("B1", "B2", "B3", "B4", "B5"):
            gs = ((s.get(k) or {}).get("overall") or {}).get("grounding_score")
            rel = ((s.get(k) or {}).get("overall") or {}).get("answer_relevance")
            if gs is not None:
                others_gs.append((k, gs))
            if rel is not None:
                others_rel.append((k, rel))
        best_gs = max(others_gs, key=lambda x: x[1]) if others_gs else (None, None)
        best_rel = max(others_rel, key=lambda x: x[1]) if others_rel else (None, None)
        margin_gs = (b0 - best_gs[1]) if (b0 is not None and best_gs[1] is not None) else None
        margin_rel = (b0_rel - best_rel[1]) if (b0_rel is not None and best_rel[1] is not None) else None
        _row("baselines", "B0 grounding",     b0)
        _row("baselines", f"best baseline ({best_gs[0]})", best_gs[1])
        _row("baselines", "margin (gs)",      margin_gs, "B0 − best baseline; biased toward verbatim")
        _row("baselines", "B0 answer_rel",    b0_rel)
        _row("baselines", f"best baseline rel ({best_rel[0]})", best_rel[1])
        _row("baselines", "margin (rel)",     margin_rel, "fair metric — does the answer address the Q")

    t = _safe_load((out_by_key.get("threshold") or {}).get("output_file"))
    if t:
        bb = t.get("best_balance") or {}
        _row("threshold", "best balance T",     bb.get("threshold"))
        _row("threshold", "answer coverage",    bb.get("answer_coverage"))
        _row("threshold", "abstention recall",  bb.get("abstention_recall"))

    r = _safe_load((out_by_key.get("rrf") or {}).get("output_file"))
    if r:
        results = r.get("results") or {}
        best_w, best_gs = None, -1.0
        configured_gs = None
        for label, data in results.items():
            gs = (data or {}).get("grounding_score")
            if gs is None: continue
            w = (data or {}).get("vector_weight")
            if abs((w or 0) - 0.65) < 1e-6:
                configured_gs = gs
            if gs > best_gs:
                best_gs, best_w = gs, w
        _row("rrf", "best vec_weight", best_w)
        _row("rrf", "gs@best",         best_gs)
        _row("rrf", "gs@0.65 (cfg)",   configured_gs)

    rk_ = _safe_load((out_by_key.get("rerank") or {}).get("output_file"))
    if rk_:
        results = rk_.get("results") or {}
        best_T, best_gs = None, -1.0
        for label, data in results.items():
            gs = (data or {}).get("grounding_score")
            T  = (data or {}).get("rerank_min_score")
            if gs is None: continue
            if gs > best_gs:
                best_gs, best_T = gs, T
        _row("rerank", "best min_score", best_T)
        _row("rerank", "gs@best",        best_gs)

    rt = _safe_load((out_by_key.get("redteam") or {}).get("output_file"))
    if rt:
        inj = rt.get("injection_detection") or {}
        gate = rt.get("combined_gate") or {}
        _row("redteam", "injection F1", inj.get("f1"))
        _row("redteam", "gate precision", gate.get("precision"))
        _row("redteam", "gate recall",    gate.get("recall"))

    vc = _safe_load((out_by_key.get("verifier-catch") or {}).get("output_file"))
    if vc:
        sm = vc.get("summary") or {}
        cr = sm.get("catch_rates") or {}
        _row("verifier-catch", "catch fabricate",   (cr.get("fabricate")   or {}).get("catch_rate"))
        _row("verifier-catch", "catch number_swap", (cr.get("number_swap") or {}).get("catch_rate"))
        _row("verifier-catch", "catch entity_swap", (cr.get("entity_swap") or {}).get("catch_rate"))
        _row("verifier-catch", "false-pos rate",    sm.get("false_positive_rate"))

    it = _safe_load((out_by_key.get("intent") or {}).get("output_file"))
    if it:
        _row("intent", "accuracy", it.get("overall_accuracy"))

    fb = _safe_load((out_by_key.get("feedback") or {}).get("output_file"))
    if fb:
        best = fb.get("best_f1") or {}
        _row("feedback", "best T",  best.get("threshold"))
        _row("feedback", "best F1", best.get("f1"))

    # ── Pre-existing scripts that also produce JSON ─────────────────
    hy = _safe_load((out_by_key.get("hybrid") or {}).get("output_file"))
    if hy:
        agg = hy.get("aggregate") or {}
        _row("hybrid", "hybrid recall@5", agg.get("hybrid_recall_at_5"))
        _row("hybrid", "vector recall@5", agg.get("vector_recall_at_5"))
        _row("hybrid", "recall uplift",   agg.get("recall_improvement"), "hybrid − vector")

    rw = _safe_load((out_by_key.get("rewriting") or {}).get("output_file"))
    if rw:
        ov = rw.get("overall") or {}
        _row("rewriting", "rewrite rate", ov.get("rewrite_rate"))
        by_cat = rw.get("by_category") or {}
        ar_rate = (by_cat.get("arabic") or {}).get("rewrite_rate")
        fu_rate = (by_cat.get("followup") or {}).get("rewrite_rate")
        if ar_rate is not None: _row("rewriting", "arabic rewrite %",  ar_rate)
        if fu_rate is not None: _row("rewriting", "followup rewrite %", fu_rate)

    ar = _safe_load((out_by_key.get("arabic") or {}).get("output_file"))
    if ar:
        agg = ar.get("aggregate") or {}
        _row("arabic", "mean gap %",   agg.get("mean_gap_pct"),
             "higher = bigger AR↓ deficit vs EN")
        _row("arabic", "pairs AR lower %", agg.get("pct_pairs_arabic_lower"))

    bl = _safe_load((out_by_key.get("bilingual") or {}).get("output_file"))
    if bl:
        agg = bl.get("aggregate") or {}
        _row("bilingual", "avg EN↔AR sim",     agg.get("avg_answer_similarity"))
        _row("bilingual", "high-consist rate", agg.get("high_consistency_rate"))

    ab = _safe_load((out_by_key.get("abstention") or {}).get("output_file"))
    if ab:
        agg = ab.get("aggregate") or {}
        _row("abstention", "false abst rate",  agg.get("false_abstention_rate"),
             "answerable Qs wrongly refused")
        _row("abstention", "ood handled %",    agg.get("ood_correct_handling_rate"))
        _row("abstention", "unanswer abst %",  agg.get("unanswerable_abstention_rate"))

    ca = _safe_load((out_by_key.get("cache") or {}).get("output_file"))
    if ca:
        agg = ca.get("aggregate") or {}
        _row("cache", "exact hit rate",    agg.get("exact_hit_rate"))
        _row("cache", "semantic hit rate", agg.get("semantic_hit_rate"))
        _row("cache", "speedup semantic",  agg.get("speedup_semantic"))

    vi = _safe_load((out_by_key.get("verifier") or {}).get("output_file"))
    if vi:
        agg = vi.get("aggregate") or {}
        _row("verifier", "modification rate", agg.get("modification_rate"),
             "answers edited by verifier")
        _row("verifier", "avg claims removed", agg.get("avg_claims_removed"))
        _row("verifier", "adversarial abst",   agg.get("adversarial_abstained"),
             "adversarial Qs abstained (of 5)")

    # ── Post-process outputs ───────────────────────────────────────
    co = _safe_load(os.path.join(out_dir, "cost.json"))
    if co:
        per = co.get("per_baseline_cost", {})
        b0 = (per.get("B0") or {}).get("total_per_query_usd")
        annual = co.get("annual_cost_full_pipeline_usd", {})
        _row("cost", "B0 $/query",   b0, "full pipeline (per-stage estimate)")
        _row("cost", "$/yr @ 100k",  annual.get("100,000 queries"))

    # Bootstrap-CI script writes its own JSON file; we also fold its
    # headline CIs onto matching rows above (via _CI_MAP). Surface the n
    # used so a juror can see the sample-size disclosure on the scorecard.
    bs = _safe_load(os.path.join(out_dir, "bootstrap_ci.json"))
    if bs:
        nq = ((bs.get("baselines_paired") or {}).get("n_questions"))
        nh = ((bs.get("headline") or {}).get("n_rows"))
        _row("bootstrap", "headline n", nh, "rows used for golden CIs")
        _row("bootstrap", "paired n",   nq, "rows shared across all baselines")

    # Render
    n_fail = sum(1 for r in rows if r.get("gate") == "FAIL")
    n_pass = sum(1 for r in rows if r.get("gate") == "PASS")

    print(f"\n{'='*88}")
    print(f"  SCORECARD — HEADLINE METRICS PER EVAL")
    print(f"{'='*88}")
    if not rows:
        print("  (no parseable outputs — did any script fail?)")
    else:
        print(f"  {'script':<16} {'metric':<22} {'value':>10}  {'gate':<6} {'95% CI':<22} {'note'}")
        print(f"  {'-'*86}")
        for r in rows:
            val = r["value"]
            if isinstance(val, (int, float)):
                disp = f"{val:>10.3f}"
            elif val is None:
                disp = f"{'n/a':>10}"
            else:
                disp = f"{str(val)[:10]:>10}"

            gate = r.get("gate")
            if gate == "PASS":
                gate_disp = "✓PASS"
            elif gate == "FAIL":
                gate_disp = "✗FAIL"
            else:
                gate_disp = ""
            gate_disp = f"{gate_disp:<6}"

            ci = r.get("ci")
            if ci and ci.get("ci_low") is not None:
                ci_disp = f"[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"
            else:
                ci_disp = ""
            ci_disp = f"{ci_disp:<22}"

            print(f"  {r['key']:<16} {r['metric']:<22} {disp}  {gate_disp} {ci_disp} {r['note']}")
    print(f"{'='*88}")
    if n_pass or n_fail:
        verdict = "PASS" if n_fail == 0 else "FAIL"
        print(f"  Gates: {n_pass} pass, {n_fail} fail → overall {verdict}")
        if n_fail:
            print("  Failing gates:")
            for r in rows:
                if r.get("gate") == "FAIL":
                    print(f"    ✗ {r['key']}/{r['metric']} = {r['value']} "
                          f"(target {r.get('gate_threshold')})")
    print(f"{'='*88}")

    # Save
    sc_path = os.path.join(out_dir, "_scorecard.json")
    with open(sc_path, "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "thresholds": {f"{k[0]}/{k[1]}": v for k, v in THRESHOLDS.items()},
            "gates_summary": {"pass": n_pass, "fail": n_fail},
            "rows": rows,
        }, f, indent=2)
    print(f"  Scorecard saved to {sc_path}\n")
    return {"rows": rows, "path": sc_path, "gates_failed": n_fail}


def run_post_process(out_dir: str, summary: list) -> list:
    """Run post-processing scripts that consume per-eval JSON outputs.

    Skipped if any of an entry's `needs` didn't produce output. Each script
    is invoked as: `python <script> <out_dir> <out_file>`.
    """
    pp_results = []
    available_keys = {r["key"] for r in summary if r.get("output_file")}
    for entry in POST_PROCESS:
        missing = [n for n in entry["needs"] if n not in available_keys]
        out_file = os.path.join(out_dir, f"{entry['key']}.json")
        if missing:
            print(f"\n  [post-process: skip] {entry['key']} "
                  f"— needs {missing} but those evals didn't produce output")
            pp_results.append({
                "key": entry["key"], "status": "SKIPPED",
                "reason": f"missing inputs: {missing}",
                "output_file": None,
            })
            continue
        script_path = os.path.join(SCRIPTS_DIR, entry["script"])
        cmd = [sys.executable, script_path, out_dir, out_file]
        print(f"\n{'─'*70}")
        print(f"  [POST-PROCESS: {entry['key'].upper()}]  {entry['desc']}")
        print(f"{'─'*70}")
        start = time.time()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = round(time.time() - start, 1)
        status = "PASS" if result.returncode == 0 else "FAIL"
        flag = "✓" if status == "PASS" else "✗"
        print(f"\n  {flag} {entry['key']} — {status} in {elapsed}s")
        pp_results.append({
            "key": entry["key"],
            "script": entry["script"],
            "status": status,
            "returncode": result.returncode,
            "elapsed_s": elapsed,
            "output_file": out_file if os.path.exists(out_file) else None,
        })
    return pp_results


def run_script(entry: dict, out_dir: str, quick: bool) -> dict:
    script_path = os.path.join(SCRIPTS_DIR, entry["script"])
    out_file = os.path.join(out_dir, f"{entry['key']}.json")

    extra_args = entry["fast_args"] if quick else entry["args"]

    # Inject --output if the script accepts it (all except eval_human_agreement)
    cmd = [sys.executable, script_path] + extra_args + ["--output", out_file]

    print(f"\n{'─'*70}")
    print(f"  [{entry['key'].upper()}]  {entry['desc']}")
    print(f"  cmd: {' '.join(os.path.basename(a) if i < 2 else a for i, a in enumerate(cmd))}")
    print(f"{'─'*70}")

    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = round(time.time() - start, 1)

    status = "PASS" if result.returncode == 0 else "FAIL"
    return {
        "key": entry["key"],
        "script": entry["script"],
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "output_file": out_file if os.path.exists(out_file) else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation scripts")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: eval_run_YYYYMMDD_HHMMSS/)")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer questions per script — fast smoke-test "
                             "(uses each entry's fast_args)")
    parser.add_argument("--fast-only", action="store_true",
                        help="Skip LLM-judged scripts; run only structural/latency tests")
    parser.add_argument("--design-only", action="store_true",
                        help="Run only the design-choice evaluations (baselines, "
                             "threshold, rrf, rerank, redteam, verifier-catch, intent, "
                             "feedback)")
    parser.add_argument("--only", nargs="+", default=None,
                        metavar="KEY",
                        help=f"Run only these scripts. Keys: {[e['key'] for e in REGISTRY]}")
    parser.add_argument("--scorecard-only", action="store_true",
                        help="Skip running evals; rebuild the scorecard from existing "
                             "{key}.json files in --out (requires --out).")
    args = parser.parse_args()

    if args.scorecard_only:
        if not args.out:
            print("--scorecard-only requires --out pointing at an existing run directory.")
            sys.exit(1)
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
        if not os.path.isdir(out_dir):
            print(f"Directory not found: {out_dir}")
            sys.exit(1)
        summary = []
        for entry in REGISTRY:
            out_file = os.path.join(out_dir, f"{entry['key']}.json")
            if os.path.exists(out_file):
                summary.append({
                    "key": entry["key"],
                    "script": entry["script"],
                    "status": "PASS",
                    "returncode": 0,
                    "elapsed_s": None,
                    "output_file": out_file,
                })
        if not summary:
            print(f"No {{key}}.json files found in {out_dir}")
            sys.exit(1)
        print(f"\n  Rebuilding scorecard from {len(summary)} existing result file(s) in {out_dir}")
        # Re-run post-processing too so cost/CI files refresh against
        # whichever run_dir we're rendering from.
        run_post_process(out_dir, summary)
        sc = build_scorecard(summary, out_dir)
        gates_failed = sc.get("gates_failed", 0) if isinstance(sc, dict) else 0
        sys.exit(0 if not gates_failed else 2)

    DESIGN_KEYS = {"baselines", "threshold", "rrf", "rerank",
                   "redteam", "verifier-catch", "intent", "feedback"}

    # Determine which scripts to run
    if args.only:
        scripts = [e for e in REGISTRY if e["key"] in args.only]
        missing = set(args.only) - {e["key"] for e in scripts}
        if missing:
            print(f"Unknown script keys: {missing}")
            sys.exit(1)
    elif args.design_only:
        scripts = [e for e in REGISTRY if e["key"] in DESIGN_KEYS]
    elif args.fast_only:
        scripts = [e for e in REGISTRY if e["fast_only"]]
    else:
        scripts = REGISTRY

    # Output directory
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(PROJECT_ROOT, f"eval_run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  AUB LIBRARIES ASSISTANT — EVALUATION SUITE")
    print(f"{'='*70}")
    print(f"  Mode:       {'quick' if args.quick else 'fast-only' if args.fast_only else 'full'}")
    print(f"  Scripts:    {len(scripts)}")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*70}")
    for e in scripts:
        tag = "[LLM]" if e["needs_llm"] else "[fast]"
        print(f"  {tag:6}  {e['key']:<16}  {e['desc']}")

    print(f"\n  Starting in 3 seconds… (Ctrl-C to abort)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    # Run
    summary = []
    total_start = time.time()

    for entry in scripts:
        result = run_script(entry, out_dir, args.quick)
        summary.append(result)
        flag = "✓" if result["status"] == "PASS" else "✗"
        print(f"\n  {flag} {entry['key']} — {result['status']} in {result['elapsed_s']}s")

    total_elapsed = round(time.time() - total_start, 1)

    # Summary table
    passed = [r for r in summary if r["status"] == "PASS"]
    failed = [r for r in summary if r["status"] == "FAIL"]

    print(f"\n{'='*70}")
    print(f"  EVALUATION SUITE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:  {total_elapsed}s")
    print(f"  Passed:      {len(passed)} / {len(summary)}")
    if failed:
        print(f"  Failed:      {len(failed)} → {[r['key'] for r in failed]}")
    print(f"  Results in:  {out_dir}/")
    print(f"{'='*70}\n")

    # Post-processing: cost estimate + bootstrap CIs (need other scripts'
    # JSON outputs as inputs, so run after the main loop).
    pp_results = run_post_process(out_dir, summary)

    # Consolidated scorecard across all scripts (reads post-process outputs too)
    sc = build_scorecard(summary, out_dir)
    gates_failed = sc.get("gates_failed", 0) if isinstance(sc, dict) else 0

    # Cost estimate on the manifest: report the per-query estimate from
    # eval_cost.py and the projected total if every question in this run
    # had hit the full B0 pipeline. This is rough — real billing depends
    # on cache hit-rate, abstention, etc. — but good enough to surface
    # an order-of-magnitude $ figure on the headline.
    cost_summary = None
    cost_json = os.path.join(out_dir, "cost.json")
    if os.path.exists(cost_json):
        try:
            co = json.load(open(cost_json))
            b0_per_q = ((co.get("per_baseline_cost") or {}).get("B0") or {}).get("total_per_query_usd")
            cost_summary = {
                "b0_per_query_usd": b0_per_q,
                "annual_projection_usd": co.get("annual_cost_full_pipeline_usd"),
            }
        except Exception:
            pass

    # Save manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "quick" if args.quick else "fast_only" if args.fast_only else "full",
        "total_elapsed_s": total_elapsed,
        "passed": len(passed),
        "failed": len(failed),
        "gates_failed": gates_failed,
        "cost_estimate": cost_summary,
        "results": summary,
        "post_process": pp_results,
    }
    manifest_path = os.path.join(out_dir, "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved to {manifest_path}")

    # Exit non-zero on either eval-script failure OR threshold-gate failure.
    # This is what turns the scorecard into a CI signal.
    exit_code = 0
    if failed:
        exit_code = 1
    elif gates_failed:
        exit_code = 2  # distinguishable from script crashes
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
