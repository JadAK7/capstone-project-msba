#!/usr/bin/env python3
"""
eval_ragas.py
Evaluate the chatbot against the golden set using the Ragas framework.

Runs the production pipeline (LibraryChatbot.answer) over each in-scope
question in data/golden_set.json, captures the generated answer and the
post-rerank retrieved_chunks, then scores with Ragas:

    faithfulness         - are claims in the answer supported by contexts
    answer_relevancy     - does the answer address the question
    context_precision    - are retrieved contexts ranked usefully
    context_recall       - did retrieval cover the reference   (ref-based)
    answer_correctness   - answer vs reference                 (ref-based)

Reference-based metrics are only computed when a question in the golden
set has a "ground_truth" (or "reference_answer") field. Guard-block /
refuse / abstained questions are excluded from scoring but counted in
the summary (Ragas cannot score without retrieved context).

Prerequisites:
    pip install ragas datasets langchain-openai

Usage:
    python scripts/eval_ragas.py                         # full set
    python scripts/eval_ragas.py --n 20                  # sample 20
    python scripts/eval_ragas.py --categories faq_direct arabic
    python scripts/eval_ragas.py --judge gpt-4o-mini
    python scripts/eval_ragas.py --output results/ragas.json
    python scripts/eval_ragas.py --dry-run               # no API calls
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ---------------------------------------------------------------------------
# Sample collection (run the pipeline, capture answer + contexts)
# ---------------------------------------------------------------------------

def _is_abstention(debug: dict, answer: str) -> bool:
    src = (debug.get("chosen_source") or "")
    if src.startswith("none") or src.startswith("refused"):
        return True
    refusal_markers = [
        "could not find", "don't have information",
        "i can only answer", "please contact the library",
        "لم أتمكن", "لا أملك معلومات", "يمكنني فقط", "يرجى التواصل",
    ]
    a = (answer or "").lower()
    return any(m in a for m in refusal_markers)


def collect_samples(questions, chatbot):
    """Run the pipeline per question; return (ragas_rows, excluded_rows)."""
    samples, excluded = [], []
    for i, q in enumerate(questions):
        qid = q["id"]
        q_text = q["question"]
        expected = q.get("expected_behavior", "answer")
        logger.info(f"[{i+1}/{len(questions)}] {qid} — {q_text[:70]}")

        if expected in ("guard_block", "refuse"):
            excluded.append({"id": qid, "reason": "expected_block_or_refuse"})
            continue

        try:
            t0 = time.time()
            answer, debug = chatbot.answer(q_text)
            elapsed_ms = (time.time() - t0) * 1000
        except Exception as e:
            logger.error(f"Pipeline failed for {qid}: {e}")
            excluded.append({"id": qid, "reason": f"pipeline_error: {e}"})
            continue

        chunks = debug.get("retrieved_chunks") or []
        contexts = [c.get("text", "") for c in chunks if c.get("text")]

        if _is_abstention(debug, answer):
            excluded.append({"id": qid, "reason": "abstained"})
            continue
        if not contexts:
            excluded.append({"id": qid, "reason": "no_context"})
            continue

        row = {
            "id": qid,
            "question": q_text,
            "answer": answer,
            "contexts": contexts,
            "category": q.get("category", ""),
            "language": q.get("language", "en"),
            "elapsed_ms": round(elapsed_ms, 1),
            "chosen_source": debug.get("chosen_source", ""),
        }
        gt = q.get("ground_truth") or q.get("reference_answer")
        if gt:
            row["ground_truth"] = gt
        samples.append(row)

    return samples, excluded


# ---------------------------------------------------------------------------
# Ragas invocation (version-tolerant)
# ---------------------------------------------------------------------------

def run_ragas(samples, has_gt: bool, judge_model: str):
    """Build the dataset and call ragas.evaluate; returns pandas DataFrame."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Ragas 0.2 renamed the dataset columns. Populate both old and new
    # names so either version of the library picks them up.
    data = {
        "question":           [s["question"] for s in samples],
        "user_input":         [s["question"] for s in samples],
        "answer":             [s["answer"] for s in samples],
        "response":           [s["answer"] for s in samples],
        "contexts":           [s["contexts"] for s in samples],
        "retrieved_contexts": [s["contexts"] for s in samples],
    }
    if has_gt:
        gt_list = [s.get("ground_truth", "") for s in samples]
        data["ground_truth"] = gt_list
        data["reference"] = gt_list

    dataset = Dataset.from_dict(data)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0.0))
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # Import metric instances — names stable across 0.1.x and 0.2.x
    from ragas.metrics import faithfulness, answer_relevancy
    metrics = [faithfulness, answer_relevancy]
    if has_gt:
        from ragas.metrics import context_precision, context_recall, answer_correctness
        metrics.extend([context_precision, context_recall, answer_correctness])
    else:
        # No ground truth → use reference-free precision (LLM judges
        # whether each retrieved context was useful for the answer).
        from ragas.metrics import LLMContextPrecisionWithoutReference
        metrics.append(LLMContextPrecisionWithoutReference())

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        raise_exceptions=False,
    )
    return result.to_pandas()


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "faithfulness", "answer_relevancy", "context_precision",
    "llm_context_precision_without_reference",
    "context_recall", "answer_correctness",
]


def aggregate(df, samples):
    """Per-metric means overall, by language, by category."""
    present = [m for m in METRIC_COLS if m in df.columns]

    def _mean(sub):
        return {m: round(float(sub[m].mean()), 4)
                for m in present if sub[m].notna().any()}

    overall = _mean(df)

    # Attach metadata from samples (preserves order) for slicing
    df = df.copy()
    df["_language"] = [s["language"] for s in samples]
    df["_category"] = [s["category"] for s in samples]

    by_language = {lang: _mean(df[df["_language"] == lang])
                   for lang in sorted(df["_language"].unique())}
    by_category = {cat: {"n": int((df["_category"] == cat).sum()),
                         **_mean(df[df["_category"] == cat])}
                   for cat in sorted(df["_category"].unique())}

    return overall, by_language, by_category


def print_report(overall, by_language, by_category, excluded, total):
    W = 68
    sep = "=" * W
    print(f"\n{sep}")
    print("  RAGAS EVALUATION — SUMMARY")
    print(f"{sep}")
    print(f"  Scored:   {total - len(excluded)} / {total}")
    print(f"  Excluded: {len(excluded)}")
    if excluded:
        reasons = {}
        for e in excluded:
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
        for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"             {n:>3}  {r}")

    print(f"\n  Overall metrics:")
    for k, v in overall.items():
        print(f"    {k:<22} {v:.1%}")

    print(f"\n  By language:")
    print(f"    {'lang':<6} " + "  ".join(f"{m:<20}" for m in overall.keys()))
    for lang, m in by_language.items():
        vals = "  ".join(f"{m.get(k, 0):<20.1%}" for k in overall.keys())
        print(f"    {lang:<6} {vals}")

    print(f"\n  By category:")
    for cat, stats in by_category.items():
        line = f"    {cat:<26} n={stats['n']:<3}  "
        line += "  ".join(f"{k[:6]}={stats.get(k, 0):.2f}" for k in overall.keys())
        print(line)
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate chatbot with Ragas")
    parser.add_argument("--golden", type=str,
                        default=os.path.join(DATA_DIR, "golden_set.json"),
                        help="Path to golden set JSON")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only these categories (e.g. faq_direct arabic)")
    parser.add_argument("--n", type=int, default=None,
                        help="Random sample of N questions for quick runs")
    parser.add_argument("--judge", type=str, default="gpt-4o-mini",
                        help="OpenAI model used as Ragas judge (default gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: eval_ragas_<ts>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load & filter the set but skip API calls")
    parser.add_argument("--references-only", action="store_true",
                        help="Only score questions that have a reference_answer "
                             "(or ground_truth) field — enables answer_correctness "
                             "and context_recall without requiring all questions "
                             "to be annotated.")
    args = parser.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)

    questions = golden["questions"]
    if args.categories:
        questions = [q for q in questions if q["category"] in args.categories]

    # Exclude expected-block/refuse up-front: Ragas needs a real answer+context
    in_scope = [q for q in questions
                if q.get("expected_behavior", "answer") == "answer"]

    if args.references_only:
        before = len(in_scope)
        in_scope = [q for q in in_scope
                    if q.get("reference_answer") or q.get("ground_truth")]
        logger.info(f"--references-only: {len(in_scope)}/{before} questions have a reference answer")

    if args.n and args.n < len(in_scope):
        random.seed(args.seed)
        in_scope = random.sample(in_scope, args.n)

    logger.info(f"Loaded {len(questions)} questions "
                f"({len(in_scope)} in-scope after filtering)")

    if args.dry_run:
        print(f"DRY RUN — {len(in_scope)} questions would be scored")
        by_cat = {}
        for q in in_scope:
            by_cat.setdefault(q["category"], 0)
            by_cat[q["category"]] += 1
        for c, n in sorted(by_cat.items()):
            print(f"  {c}: {n}")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    # Import ragas early so missing-dep errors surface before we run the pipeline
    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError as e:
        msg = (f"Ragas / datasets / langchain-openai not installed — skipping this eval. "
               f"Install with: pip install ragas datasets langchain-openai  ({e})")
        logger.warning(msg)
        # Write a skip marker so run_all_evals can treat this as a clean skip
        # rather than a failure, and so the scorecard can show 'skipped'.
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump({"status": "skipped",
                               "reason": "missing_dependencies",
                               "missing": str(e)}, f, indent=2)
            except Exception:
                pass
        sys.exit(0)

    init_db()
    chatbot = LibraryChatbot(os.environ["OPENAI_API_KEY"])

    # --- Phase 1: generate answers + collect contexts
    logger.info("Phase 1 — running pipeline and collecting contexts")
    samples, excluded = collect_samples(in_scope, chatbot)
    if args.references_only:
        # Drop rows without a reference so Ragas sees a clean annotated set
        ref_samples = [s for s in samples if "ground_truth" in s]
        if len(ref_samples) < len(samples):
            dropped = len(samples) - len(ref_samples)
            logger.info(f"Dropping {dropped} collected samples that lack a reference "
                        "(system answered but no reference on file)")
        samples = ref_samples
    has_gt = bool(samples) and all("ground_truth" in s for s in samples)

    if not samples:
        logger.error("No scoreable samples. "
                     "All questions abstained or produced no context.")
        close_db()
        sys.exit(1)

    # --- Phase 2: Ragas judge
    logger.info(f"Phase 2 — scoring {len(samples)} samples with Ragas "
                f"(judge={args.judge}, reference_metrics={has_gt})")
    df = run_ragas(samples, has_gt=has_gt, judge_model=args.judge)

    # --- Phase 3: aggregate + report
    overall, by_language, by_category = aggregate(df, samples)
    print_report(overall, by_language, by_category, excluded, len(in_scope))

    # --- Persist
    output_path = args.output or os.path.join(
        PROJECT_ROOT, f"eval_ragas_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    )
    per_question = []
    drop_cols = {"_language", "_category"}
    records = df.drop(columns=[c for c in drop_cols if c in df.columns]).to_dict(orient="records")
    for s, rec in zip(samples, records):
        per_question.append({
            "id": s["id"],
            "question": s["question"],
            "category": s["category"],
            "language": s["language"],
            "answer_preview": s["answer"][:200],
            "num_contexts": len(s["contexts"]),
            "chosen_source": s["chosen_source"],
            "metrics": {k: (None if rec.get(k) is None or (isinstance(rec.get(k), float) and rec[k] != rec[k]) else round(float(rec[k]), 4))
                        for k in METRIC_COLS if k in rec},
        })

    output = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "golden_set": args.golden,
            "judge_model": args.judge,
            "n_requested": len(in_scope),
            "n_scored": len(samples),
            "n_excluded": len(excluded),
            "reference_metrics_enabled": has_gt,
        },
        "overall": overall,
        "by_language": by_language,
        "by_category": by_category,
        "excluded": excluded,
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Full results saved to {output_path}")

    close_db()


if __name__ == "__main__":
    main()
