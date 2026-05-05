#!/usr/bin/env python3
"""
run_ablation_models.py

2x2 factorial ablation: {text-embedding-3-small, text-embedding-3-large}
                       x {gpt-4o-mini, gpt-4.1}

For each of the 4 configurations:
  1. Sets OPENAI_EMBEDDING_MODEL + OPENAI_CHAT_MODEL env vars
  2. Rebuilds the pgvector index when the embedding model changes
     (the database layer auto-migrates the schema on dim mismatch).
  3. Runs the MSBA-relevant eval subset:
        - run_golden_eval.py     (golden.json)
        - eval_ragas.py          (ragas.json)
        - eval_latency.py        (latency.json)
        - eval_bilingual_consistency.py (bilingual.json)
  4. Writes results into  ablation_models/<config>/

Outputs a final summary CSV at ablation_models/_summary.csv with the
headline metrics from each config's _scorecard.

Usage
-----
    # Full run (~2-4 hr, $5-15 in OpenAI fees)
    python scripts/eval/run_ablation_models.py

    # Dry run — print the commands only, no API calls
    python scripts/eval/run_ablation_models.py --dry-run

    # Resume — skip configs whose output dir already has the full set of JSONs
    python scripts/eval/run_ablation_models.py --resume

    # Custom output root
    python scripts/eval/run_ablation_models.py --out results/abl_2026_05

Cost / time notes
-----------------
* text-embedding-3-large is ~6.5x more expensive per token than -small.
* gpt-4.1 (~$2/$8 per 1M) is ~13x pricier than gpt-4o-mini per query.
* Index rebuilds happen twice (once per embedding model). Each rebuild
  re-embeds the full corpus (~$0.05 for large, ~$0.01 for small).
* Final state of the DB after the run: whichever config ran last.
  Pass --restore-default to re-index in production config (small) at the end.

This is a parent script — it shells out to your existing eval scripts so
you can also run any single eval manually with custom env vars.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EVAL_DIR = SCRIPTS_DIR / "eval"

# (label, embedding_model, chat_model)
# Grouped by embedding model so we only rebuild the index twice, not four times.
CONFIGS = [
    ("small_4o-mini", "text-embedding-3-small", "gpt-4o-mini"),
    ("small_4.1",     "text-embedding-3-small", "gpt-4.1"),
    ("large_4o-mini", "text-embedding-3-large", "gpt-4o-mini"),
    ("large_4.1",     "text-embedding-3-large", "gpt-4.1"),
]

# Each tuple: (key, script_path, extra_args). Eval scripts all accept --output.
EVALS = [
    ("golden",    EVAL_DIR / "run_golden_eval.py",            []),
    ("ragas",     EVAL_DIR / "eval_ragas.py",                 []),
    ("latency",   EVAL_DIR / "eval_latency.py",               []),
    ("bilingual", EVAL_DIR / "eval_bilingual_consistency.py", []),
]


def make_env(embedding: str, chat: str) -> dict:
    """Return a copy of the parent env with model selectors overridden."""
    env = os.environ.copy()
    env["OPENAI_EMBEDDING_MODEL"] = embedding
    env["OPENAI_CHAT_MODEL"] = chat
    return env


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def run_cmd(cmd: list[str], env: dict, dry_run: bool, log_path: Path | None = None) -> int:
    pretty = " ".join(str(c) for c in cmd)
    print(f"  $ {pretty}")
    if dry_run:
        return 0
    if log_path is not None:
        with log_path.open("w") as logf:
            logf.write(f"# {datetime.utcnow().isoformat()}Z\n# cmd: {pretty}\n\n")
            logf.flush()
            res = subprocess.run(cmd, env=env, cwd=REPO_ROOT,
                                 stdout=logf, stderr=subprocess.STDOUT)
        return res.returncode
    res = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
    return res.returncode


def rebuild_index(env: dict, dry_run: bool, log_path: Path) -> bool:
    """Truncate + re-embed via scripts/build_index.py. The DB layer handles
    schema migration when the embedding dim changes."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "build_index.py")]
    rc = run_cmd(cmd, env, dry_run, log_path=log_path)
    if rc != 0:
        print(f"  ✗ build_index.py exited with code {rc}")
        return False
    print("  ✓ index rebuilt")
    return True


def run_eval(eval_key: str, script: Path, extra_args: list[str],
             out_file: Path, env: dict, dry_run: bool, log_path: Path) -> dict:
    cmd = [sys.executable, str(script)] + extra_args + ["--output", str(out_file)]
    start = time.time()
    rc = run_cmd(cmd, env, dry_run, log_path=log_path)
    elapsed = round(time.time() - start, 1)
    return {
        "key": eval_key,
        "returncode": rc,
        "elapsed_s": elapsed,
        "output_file": str(out_file) if out_file.exists() else None,
    }


def is_config_complete(out_dir: Path) -> bool:
    """A config is 'complete' if all four eval JSONs exist and parse."""
    for key, _, _ in EVALS:
        p = out_dir / f"{key}.json"
        if not p.exists():
            return False
        try:
            json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return False
    return True


def is_eval_corrupt(json_path: Path, eval_key: str) -> bool:
    """Detect outputs that exist but are likely the result of a failed run
    (e.g. swept by judge connection errors). Returns True if the file should
    be regenerated.
    """
    if not json_path.exists():
        return False  # missing != corrupt; --resume will re-run missing
    try:
        d = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return True

    # Heuristic: golden eval where median grounding is 0 but there are answered
    # rows is almost certainly judge-failure noise. Same for ragas all-zero.
    if eval_key == "golden":
        rows = d.get("raw_results") or []
        scored = [r.get("grounding_score") for r in rows
                  if r.get("grounding_score") is not None]
        if scored:
            zeros = sum(1 for s in scored if s == 0)
            if zeros / len(scored) >= 0.4:
                return True
    elif eval_key == "ragas":
        overall = d.get("overall") or {}
        meta = d.get("metadata") or {}
        # Empty overall = nothing was scored successfully
        if not overall:
            return True
        # If RAGAS produced all near-zero scores, it's corrupt
        nonzero = [v for v in overall.values()
                   if isinstance(v, (int, float)) and v > 0.05]
        if overall and not nonzero:
            return True
        # If too many questions were excluded (>30%), the run was throttled
        n_scored = meta.get("n_scored", 0)
        n_requested = meta.get("n_requested") or meta.get("n_total") or 0
        if n_requested and n_scored / n_requested < 0.7:
            return True
    return False


# ---------------------------------------------------------------------------
# Headline metric extraction (tolerant — missing fields → None)
# ---------------------------------------------------------------------------

def safe_get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def extract_headline(out_dir: Path) -> dict:
    """Return a dict of headline metrics for one config."""
    headline = {}

    # golden.json -> grounding_score, abstention rate, EN-AR gap
    try:
        g = json.loads((out_dir / "golden.json").read_text())
        m = safe_get(g, "table1_overall", "metrics", default={})
        headline["grounding_score"] = m.get("grounding_score_composite")
        headline["faithfulness_golden"] = m.get("faithfulness")
        headline["answer_relevance_golden"] = m.get("answer_relevance")
        headline["false_abstention_rate"] = safe_get(g, "table4_abstention",
                                                      "false_abstention_rate")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ragas.json -> faithfulness, answer_relevancy, context_precision (overall + by_lang)
    try:
        r = json.loads((out_dir / "ragas.json").read_text())
        overall = r.get("overall", {})
        headline["ragas_faithfulness"] = overall.get("faithfulness")
        headline["ragas_answer_relevancy"] = overall.get("answer_relevancy")
        headline["ragas_context_precision"] = overall.get(
            "llm_context_precision_without_reference")
        headline["ragas_relevancy_en"] = safe_get(r, "by_language", "en", "answer_relevancy")
        headline["ragas_relevancy_ar"] = safe_get(r, "by_language", "ar", "answer_relevancy")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # latency.json -> avg cold ms, cache speedup
    try:
        l = json.loads((out_dir / "latency.json").read_text())
        agg = l.get("aggregate", {})
        headline["avg_cold_ms"] = agg.get("avg_full_cold_ms")
        headline["avg_cached_ms"] = agg.get("avg_full_cached_ms")
        headline["cache_speedup"] = agg.get("cache_speedup")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # bilingual.json -> avg EN-AR sim
    try:
        b = json.loads((out_dir / "bilingual.json").read_text())
        headline["bilingual_en_ar_sim"] = safe_get(b, "aggregate", "avg_answer_similarity")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return headline


def write_summary_csv(summary_rows: list[dict], path: Path) -> None:
    fieldnames = ["config", "embedding", "chat_model"] + [
        "grounding_score", "false_abstention_rate",
        "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision",
        "ragas_relevancy_en", "ragas_relevancy_ar",
        "bilingual_en_ar_sim",
        "avg_cold_ms", "avg_cached_ms", "cache_speedup",
        "faithfulness_golden", "answer_relevance_golden",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\n  wrote summary → {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "ablation_models",
                        help="Output root directory (default: ablation_models/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs whose JSONs already exist")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Delete eval JSONs that look corrupt (e.g. golden "
                             "with median grounding=0, indicating judge-throttle "
                             "errors) and re-run them. Implies --resume.")
    parser.add_argument("--restore-default", action="store_true",
                        help="At the end, re-index in the production config "
                             "(text-embedding-3-small) so the DB returns to "
                             "its normal state.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these config labels (e.g. small_4o-mini)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    last_embedding: str | None = None

    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if c[0] in args.only]
        if not configs:
            print(f"No configs matched --only {args.only}", file=sys.stderr)
            sys.exit(2)

    if args.retry_failed:
        args.resume = True  # retry_failed implies resume

    overall_start = time.time()
    for label, embedding, chat in configs:
        banner(f"CONFIG: {label}   (emb={embedding}, llm={chat})")
        out_dir = args.out / label
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sweep corrupt outputs so they get re-run by the resume logic below
        if args.retry_failed:
            for key, _, _ in EVALS:
                p = out_dir / f"{key}.json"
                if is_eval_corrupt(p, key):
                    if args.dry_run:
                        print(f"  ✗ {key}.json is corrupt; would delete (dry-run)")
                    else:
                        print(f"  ✗ {key}.json is corrupt; deleting for retry")
                        p.unlink()

        if args.resume and is_config_complete(out_dir):
            print("  ↺ skipped — all eval JSONs already present")
            headline = extract_headline(out_dir)
            summary_rows.append({"config": label, "embedding": embedding,
                                  "chat_model": chat, **headline})
            last_embedding = embedding
            continue

        env = make_env(embedding, chat)

        # 1. Rebuild index if embedding changed
        if embedding != last_embedding:
            print(f"  → embedding model changed ({last_embedding} → {embedding}), "
                  "rebuilding index...")
            log_path = out_dir / "_build_index.log"
            ok = rebuild_index(env, args.dry_run, log_path)
            if not ok:
                print(f"  ✗ index rebuild failed for {label}; skipping evals")
                continue
            last_embedding = embedding
        else:
            print("  → embedding unchanged, skipping index rebuild")

        # 2. Run each eval (skip ones whose output already exists in resume mode)
        per_eval_results = []
        for key, script, extra in EVALS:
            out_file = out_dir / f"{key}.json"
            log_path = out_dir / f"_{key}.log"
            if args.resume and out_file.exists():
                try:
                    json.loads(out_file.read_text())
                    print(f"\n  [{key.upper()}] ↺ skipped — already exists")
                    continue
                except (json.JSONDecodeError, OSError):
                    pass  # corrupt, fall through to re-run
            print(f"\n  [{key.upper()}]")
            res = run_eval(key, script, extra, out_file, env, args.dry_run, log_path)
            per_eval_results.append(res)
            flag = "✓" if res["returncode"] == 0 else "✗"
            print(f"  {flag} {key} — {res['elapsed_s']}s")

        # 3. Collect headline metrics
        if not args.dry_run:
            headline = extract_headline(out_dir)
            summary_rows.append({"config": label, "embedding": embedding,
                                  "chat_model": chat, **headline})

    # 4. Write summary CSV
    if summary_rows and not args.dry_run:
        write_summary_csv(summary_rows, args.out / "_summary.csv")

    # 5. Optional: restore production config
    if args.restore_default and not args.dry_run:
        banner("RESTORE: rebuilding index in production config (small)")
        env = make_env("text-embedding-3-small", "gpt-4o-mini")
        rebuild_index(env, args.dry_run, args.out / "_restore.log")

    total_elapsed = round((time.time() - overall_start) / 60, 1)
    print(f"\nTotal elapsed: {total_elapsed} min")
    print(f"Outputs in: {args.out.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
