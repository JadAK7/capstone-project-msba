"""
generate_comparison_charts.py

Loads three scorecards (T1 baseline → T2 first fixes → T3 post changes), joins
them on (key, metric), and produces:

  - iteration_improvements.png  — 2x2 panel of the four big wins
  - safety_latency_tradeoff.png — abstention rate falling vs cold latency rising
  - pipeline_maturity_radar.png — T1 vs T3 multi-axis comparison
  - timeline_bars.png           — grouped-bar timeline for headline metrics
  - deltas.csv                  — full joined long-form table

Run from repo root:
    python scripts/eval/generate_comparison_charts.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _d(s: str) -> str:
    """Escape literal dollar signs so matplotlib's mathtext does not eat them
    (only relevant in user-controlled strings; tick formatters still use $...$
    internally and must not be globally disabled)."""
    return s.replace("$", r"\$")

# AUB brand palette (matches backend/chart_generator.py)
MAROON = "#840132"
RED = "#ee3524"
GRAY = "#424242"
LIGHT_GRAY = "#d1d2d2"
GREEN = "#2d6a4f"
BLUE = "#457b9d"

REPO_ROOT = Path(__file__).resolve().parents[2]

SCORECARDS = [
    ("T1\nApr 23\n(before fixes)",
     REPO_ROOT / "baseline_before_fixes/eval_run_20260423_120009/_scorecard.json"),
    ("T2\nApr 24\n(first fixes)",
     REPO_ROOT / "eval_run_20260424_095649/_scorecard.json"),
    ("T3\nApr 28\n(post changes)",
     REPO_ROOT / "scripts/eval/run_postchanges_20260428_174442/_scorecard.json"),
    ("T4\nApr 30\n(latest full run)",
     REPO_ROOT / "eval_run_20260429_214144/_scorecard.json"),
]

# Per-timepoint colors (length must match SCORECARDS)
TIMEPOINT_COLORS = [LIGHT_GRAY, BLUE, RED, MAROON]

# Latest full-run directory used for the single-run breakdown charts
# (baselines, RAGAS by language/category, per-stage latency, sweeps).
LATEST_RUN_DIR = REPO_ROOT / "eval_run_20260429_214144"

# Earlier run directory that contains the only cost.json in the repo.
COST_RUN_DIR = REPO_ROOT / "eval_run_20260423_120009"

OUT_DIR = REPO_ROOT / "scripts/eval/comparison_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scorecard(path: Path) -> dict[tuple[str, str], float | None]:
    """Return {(key, metric): value} for numeric rows; non-numeric skipped."""
    data = json.loads(path.read_text())
    out: dict[tuple[str, str], float | None] = {}
    for row in data["rows"]:
        v = row.get("value")
        if isinstance(v, (int, float)):
            out[(row["key"], row["metric"])] = float(v)
    return out


def write_deltas_csv(scorecards: list[tuple[str, dict]], out_path: Path) -> None:
    """Long-form CSV: one row per (key, metric, timepoint).

    Column count is derived from the number of timepoints to keep the CSV
    in sync if SCORECARDS is extended (e.g. T1-T4).
    """
    all_keys: set[tuple[str, str]] = set()
    for _, sc in scorecards:
        all_keys.update(sc.keys())

    n = len(scorecards)
    timepoint_cols = [f"T{i+1}" for i in range(n)]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "metric", *timepoint_cols, f"T1_to_T{n}_delta"])
        for key, metric in sorted(all_keys):
            row = [key, metric]
            vals = [sc.get((key, metric)) for _, sc in scorecards]
            row.extend(vals)
            if vals[0] is not None and vals[-1] is not None:
                row.append(round(vals[-1] - vals[0], 4))
            else:
                row.append(None)
            w.writerow(row)


def fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%" if v <= 1.0 else f"{v:.0f}%"


def panel_bar(ax, title: str, labels: list[str], values: list[float | None],
              ylim: tuple[float, float], y_fmt: str, annotate_fmt) -> None:
    """One panel of the iteration-improvements grid."""
    xs = np.arange(len(labels))
    plotted_vals = [v if v is not None else 0 for v in values]
    colors = TIMEPOINT_COLORS[:len(labels)]
    bars = ax.bar(xs, plotted_vals, color=colors, width=0.6, edgecolor="white")
    for bar, v in zip(bars, values):
        if v is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02 * (ylim[1] - ylim[0]),
                    "n/a", ha="center", va="bottom", color=GRAY, fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.02 * (ylim[1] - ylim[0]),
                    annotate_fmt(v), ha="center", va="bottom",
                    color=GRAY, fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10, color=MAROON, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def chart_iteration_improvements(scorecards):
    labels = [lbl for lbl, _ in scorecards]
    cards = [sc for _, sc in scorecards]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle(f"Iteration improvements across {len(labels)} eval rounds",
                 fontsize=13, color=MAROON, fontweight="bold", y=0.99)

    # Panel A — false abstention rate (abstention bench)
    abst = [sc.get(("abstention", "false abst rate")) for sc in cards]
    panel_bar(axes[0][0], "False abstention rate (lower is better)",
              labels, abst, (0, 0.4), "%", lambda v: f"{v*100:.0f}%")

    # Panel B — verifier entity_swap catch
    ent = [sc.get(("verifier-catch", "catch entity_swap")) for sc in cards]
    panel_bar(axes[0][1], "Verifier entity-swap catch rate (higher is better)",
              labels, ent, (0, 1.05), "%", lambda v: f"{v*100:.0f}%")

    # Panel C — cache semantic speedup
    speed = [sc.get(("cache", "speedup semantic")) for sc in cards]
    panel_bar(axes[1][0], "Semantic-cache speedup (higher is better; <1 = slower)",
              labels, speed, (0, 1.6), "x", lambda v: f"{v:.2f}x")
    axes[1][0].axhline(y=1.0, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)

    # Panel D — bilingual similarity
    bil = [sc.get(("bilingual", "avg EN↔AR sim")) for sc in cards]
    panel_bar(axes[1][1], "Bilingual EN↔AR answer similarity (higher is better)",
              labels, bil, (0, 0.7), "", lambda v: f"{v:.3f}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "iteration_improvements.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_safety_latency_tradeoff(scorecards):
    labels = [lbl for lbl, _ in scorecards]
    cards = [sc for _, sc in scorecards]
    abst = [sc.get(("abstention", "false abst rate")) for sc in cards]
    cold_ms = [sc.get(("latency", "avg cold ms")) for sc in cards]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(labels))

    bars = ax1.bar(xs - 0.2, [v if v is not None else 0 for v in abst],
                   width=0.4, color=MAROON, label="False abstention rate", alpha=0.9)
    ax1.set_ylabel("False abstention rate", color=MAROON, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=MAROON)
    ax1.set_ylim(0, 0.4)
    for bar, v in zip(bars, abst):
        if v is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f"{v*100:.0f}%", ha="center", va="bottom",
                     color=MAROON, fontweight="bold", fontsize=9)

    ax2 = ax1.twinx()
    line_xs = xs + 0.2
    line_vals = [v / 1000 if v is not None else None for v in cold_ms]
    line_xs_clean = [x for x, v in zip(line_xs, line_vals) if v is not None]
    line_vals_clean = [v for v in line_vals if v is not None]
    ax2.plot(line_xs_clean, line_vals_clean, marker="o", markersize=10,
             color=RED, linewidth=2.5, label="Cold pipeline latency")
    for x, v in zip(line_xs_clean, line_vals_clean):
        ax2.text(x, v + 0.25, f"{v:.1f}s", ha="center", va="bottom",
                 color=RED, fontweight="bold", fontsize=9)
    ax2.set_ylabel("Cold latency (seconds)", color=RED, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.set_ylim(0, max(line_vals_clean) * 1.25)

    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Safety-latency tradeoff: abstention dropped, latency rose",
                  fontsize=12, color=MAROON, fontweight="bold", pad=12)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(axis="y", alpha=0.15)

    fig.text(0.5, -0.02,
             "Each iteration added more thorough verification, raising cold latency "
             "but cutting false abstentions by 5x.",
             ha="center", fontsize=9, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "safety_latency_tradeoff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_pipeline_maturity_radar(scorecards):
    """Spider chart comparing T1 vs latest timepoint across normalized axes."""
    t1_label = scorecards[0][0].split("\n")[0]
    latest_label = scorecards[-1][0].split("\n")[0]
    t1 = scorecards[0][1]
    t3 = scorecards[-1][1]  # 't3' kept as variable name for diff minimality

    # (display_name, (key, metric), normalize_fn)
    # Each metric is normalized to [0,1] where higher = better.
    axes_spec = [
        ("Abstention\nprecision", ("abstention", "false abst rate"),
         lambda v: 1.0 - v if v is not None else None),
        ("Bilingual\nconsistency", ("bilingual", "avg EN↔AR sim"),
         lambda v: v),
        ("Verifier\nentity catch", ("verifier-catch", "catch entity_swap"),
         lambda v: v),
        ("Cache\nspeedup (sem.)", ("cache", "speedup semantic"),
         lambda v: min(v / 2.0, 1.0) if v is not None else None),
        ("Verifier\nprecision", ("verifier-catch", "false-pos rate"),
         lambda v: 1.0 - v if v is not None else None),
        ("Guard\ninjection TPR", ("guard", "inj TPR"),
         lambda v: v),
    ]

    def collect(card):
        out = []
        for _, key, norm in axes_spec:
            raw = card.get(key)
            out.append(norm(raw) if raw is not None else None)
        return out

    t1_vals = collect(t1)
    t3_vals = collect(t3)

    # Drop axes where either is missing (radar can't have nulls)
    keep = [i for i, (a, b) in enumerate(zip(t1_vals, t3_vals))
            if a is not None and b is not None]
    labels = [axes_spec[i][0] for i in keep]
    t1_vals = [t1_vals[i] for i in keep]
    t3_vals = [t3_vals[i] for i in keep]

    if not labels:
        print("  skipped pipeline_maturity_radar (no overlapping axes)")
        return

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    t1_closed = t1_vals + t1_vals[:1]
    t3_closed = t3_vals + t3_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, t1_closed, color=LIGHT_GRAY, linewidth=2,
            label=f"{t1_label} (before fixes)")
    ax.fill(angles_closed, t1_closed, color=LIGHT_GRAY, alpha=0.25)
    ax.plot(angles_closed, t3_closed, color=MAROON, linewidth=2,
            label=f"{latest_label} (latest run)")
    ax.fill(angles_closed, t3_closed, color=MAROON, alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=GRAY, fontsize=8)
    ax.set_title(f"Pipeline maturity: {t1_label} baseline vs {latest_label} latest",
                 fontsize=12, color=MAROON, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)
    ax.grid(alpha=0.3)

    out = OUT_DIR / "pipeline_maturity_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_timeline_bars(scorecards):
    """Grouped bar chart for headline metrics across the three timepoints."""
    labels = [lbl.replace("\n", " ") for lbl, _ in scorecards]
    cards = [sc for _, sc in scorecards]

    metric_specs = [
        ("False abst. (golden)", ("golden", "false abst rate"), True),
        ("False abst. (bench)", ("abstention", "false abst rate"), True),
        ("EN↔AR sim.", ("bilingual", "avg EN↔AR sim"), False),
        ("Verifier entity catch", ("verifier-catch", "catch entity_swap"), False),
        ("Verifier FP rate", ("verifier-catch", "false-pos rate"), True),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    n_groups = len(labels)
    width = min(0.22, 0.85 / n_groups)
    xs = np.arange(len(metric_specs))
    colors = TIMEPOINT_COLORS[:n_groups]

    # Center each timepoint group around the metric tick.
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width

    for i, (label, sc) in enumerate(zip(labels, cards)):
        vals = []
        for _, key, _ in metric_specs:
            v = sc.get(key)
            vals.append(v if v is not None else 0)
        bars = ax.bar(xs + offsets[i], vals, width=width,
                      color=colors[i], label=label, edgecolor="white")
        for bar, v, (_, key, _) in zip(bars, vals, metric_specs):
            if sc.get(key) is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "n/a", ha="center", va="bottom", color=GRAY, fontsize=7)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom",
                        color=GRAY, fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in metric_specs], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value (0-1 scale)", fontsize=10)
    ax.set_title(f"Headline metric trajectory across {n_groups} eval rounds",
                 fontsize=12, color=MAROON, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.02,
             "Bars marked higher-is-worse: false abstention rates, verifier FP rate. "
             "All others higher-is-better.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "timeline_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_baselines_comparison(run_dir: Path) -> None:
    """Grouped bar chart: B0-B5 across grounding, answer_relevance, groundedness."""
    path = run_dir / "baselines.json"
    if not path.exists():
        print(f"  skipped baselines (missing {path.name})")
        return
    data = json.loads(path.read_text())
    summaries = data["summaries"]["answerable"]

    bnames = ["B0", "B1", "B2", "B3", "B4", "B5"]
    labels = {
        "B0": "B0\nFull pipeline",
        "B1": "B1\nLLM-only",
        "B2": "B2\nBM25+raw",
        "B3": "B3\nVector+raw",
        "B4": "B4\nFAQ verbatim",
        "B5": "B5\nRetrieve+\nsummarize",
    }
    metrics = [
        ("grounding_score", "Grounding score", MAROON),
        ("answer_relevance", "Answer relevance", RED),
        ("groundedness", "Groundedness", BLUE),
    ]

    xs = np.arange(len(bnames))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (key, mlabel, color) in enumerate(metrics):
        vals = [summaries[b]["overall"][key] for b in bnames]
        bars = ax.bar(xs + (i - 1) * width, vals, width=width,
                      color=color, label=mlabel, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=GRAY, fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels([labels[b] for b in bnames], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value (0-1 scale)", fontsize=10)
    ax.axhline(summaries["B0"]["overall"]["grounding_score"],
               color=MAROON, linestyle=":", linewidth=1, alpha=0.5)
    ax.set_title("Full system (B0) versus ablated baselines (B1-B5)",
                 fontsize=12, color=MAROON, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.02,
             "B5 wins on grounding by leaning on verbatim copying; B0 wins on answer "
             "relevance because it actually answers the question. No baseline dominates B0.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "baselines_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_ragas_breakdown(run_dir: Path) -> None:
    """Two-panel: RAGAS by language, RAGAS answer_relevancy by category."""
    path = run_dir / "ragas.json"
    if not path.exists():
        print(f"  skipped ragas (missing {path.name})")
        return
    data = json.loads(path.read_text())
    by_lang = data["by_language"]
    by_cat = data["by_category"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: by language (EN vs AR) for three RAGAS metrics
    metrics_lang = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer\nrelevancy"),
        ("llm_context_precision_without_reference", "Context\nprecision"),
    ]
    xs = np.arange(len(metrics_lang))
    width = 0.38
    en_vals = [by_lang["en"][k] for k, _ in metrics_lang]
    ar_vals = [by_lang["ar"][k] for k, _ in metrics_lang]
    en_bars = ax1.bar(xs - width / 2, en_vals, width=width,
                      color=BLUE, label=f"English (n={ _en_n(data)})", edgecolor="white")
    ar_bars = ax1.bar(xs + width / 2, ar_vals, width=width,
                      color=MAROON, label=f"Arabic (n={ _ar_n(data)})", edgecolor="white")
    for bars in (en_bars, ar_bars):
        for bar in bars:
            v = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                     f"{v:.2f}", ha="center", va="bottom",
                     color=GRAY, fontsize=8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([m[1] for m in metrics_lang], fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("RAGAS score", fontsize=10)
    ax1.set_title("RAGAS metrics by language",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: answer_relevancy by category, sorted descending
    cat_label = {
        "faq_direct": "FAQ direct",
        "policy_hours": "Policy &\nhours",
        "database_recommendation": "Database\nrec.",
        "follow_up_ambiguous": "Follow-up\nambiguous",
        "arabic": "Arabic",
    }
    cat_items = sorted(by_cat.items(),
                       key=lambda kv: kv[1]["answer_relevancy"], reverse=True)
    cat_xs = np.arange(len(cat_items))
    cat_vals = [info["answer_relevancy"] for _, info in cat_items]
    cat_n = [info["n"] for _, info in cat_items]
    bars = ax2.bar(cat_xs, cat_vals, width=0.6, color=MAROON, edgecolor="white")
    for bar, v, n in zip(bars, cat_vals, cat_n):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v:.2f}", ha="center", va="bottom",
                 color=GRAY, fontsize=8, fontweight="bold")
        ax2.text(bar.get_x() + bar.get_width() / 2, 0.02,
                 f"n={n}", ha="center", va="bottom",
                 color="white", fontsize=8)
    ax2.set_xticks(cat_xs)
    ax2.set_xticklabels([cat_label.get(k, k) for k, _ in cat_items], fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("RAGAS answer_relevancy", fontsize=10)
    ax2.axhline(0.6, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(len(cat_items) - 0.5, 0.61, "gate ≥ 0.60", ha="right", va="bottom",
             color=GRAY, fontsize=8, style="italic")
    ax2.set_title("Answer-relevancy by question category",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"RAGAS evaluation — {run_dir.name} (n_scored = {data['metadata']['n_scored']})",
        fontsize=12, color=MAROON, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = OUT_DIR / "ragas_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def _en_n(ragas_data: dict) -> int:
    cat = ragas_data.get("by_category", {})
    # English RAGAS items live in non-arabic categories.
    return sum(info.get("n", 0) for k, info in cat.items() if k != "arabic")


def _ar_n(ragas_data: dict) -> int:
    return ragas_data.get("by_category", {}).get("arabic", {}).get("n", 0)


def chart_per_stage_latency(run_dir: Path) -> None:
    """Horizontal stacked bar: cold-pipeline latency broken down by stage."""
    path = run_dir / "latency.json"
    if not path.exists():
        print(f"  skipped per-stage latency (missing {path.name})")
        return
    data = json.loads(path.read_text())
    agg = data["aggregate"]

    stages = [
        ("Input guard", "avg_input_guard_ms", "#999999"),
        ("Query rewriter", "avg_query_rewriting_ms", BLUE),
        ("Embedding", "avg_embedding_ms", "#7e57c2"),
        ("Retrieval (vec+kw+RRF)", "avg_retrieval_ms", GREEN),
        ("LLM reranker", "avg_reranking_ms", RED),
        ("Generation + verifier", "avg_generation_verify_ms", MAROON),
    ]

    fig, ax = plt.subplots(figsize=(12, 3.4))
    left = 0.0
    total = sum(agg[k] for _, k, _ in stages)
    # Only show in-bar text on segments wide enough to fit it without clipping;
    # smaller segments are described by the legend underneath.
    IN_BAR_MIN_SHARE = 0.10
    for label, key, color in stages:
        v = agg[key]
        share = v / total
        ax.barh(0, v, left=left, color=color, edgecolor="white",
                label=f"{label}  ({v:.0f} ms · {share*100:.1f}%)")
        if share >= IN_BAR_MIN_SHARE:
            ax.text(left + v / 2, 0,
                    f"{label}\n{v:.0f} ms\n{share*100:.1f}%",
                    ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold")
        left += v

    ax.set_yticks([])
    ax.set_xlim(0, total * 1.02)
    ax.set_xlabel("Average cold-path latency (ms)", fontsize=10)
    ax.set_title(
        f"Per-stage cold latency — total {total:,.0f} ms "
        f"(cached: {agg['avg_full_cached_ms']:,.0f} ms; "
        f"speedup {agg['cache_speedup']:.1f}x)",
        fontsize=11, color=MAROON, fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4),
              ncol=3, fontsize=8, frameon=False)

    fig.text(0.5, -0.05,
             "LLM-mediated stages (rewriter, reranker, generation + verifier) account "
             "for over 90% of cold time; pure retrieval is under 1%.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "per_stage_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_sweeps(run_dir: Path) -> None:
    """Two-panel: RRF vector-weight sweep and rerank min_score sweep."""
    rrf_path = run_dir / "rrf.json"
    rerank_path = run_dir / "rerank.json"
    if not rrf_path.exists() or not rerank_path.exists():
        print("  skipped sweeps (missing rrf.json or rerank.json)")
        return

    rrf = json.loads(rrf_path.read_text())["results"]
    rer = json.loads(rerank_path.read_text())["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    def _add_headroom(ax, values, pad_frac=0.20):
        """Pad y-limits so legends and labels have room above/below the data."""
        lo, hi = min(values), max(values)
        span = hi - lo or 0.01
        ax.set_ylim(lo - pad_frac * span, hi + pad_frac * span)

    # Panel A: RRF vec-weight sweep
    weights = sorted(float(k.split("_")[1]) for k in rrf.keys())
    rrf_scores = [rrf[f"vw_{w:.2f}"]["grounding_score"] for w in weights]
    best_idx = int(np.argmax(rrf_scores))
    ax1.plot(weights, rrf_scores, marker="o", color=MAROON, linewidth=2,
             markersize=7)
    ax1.scatter([weights[best_idx]], [rrf_scores[best_idx]],
                color=RED, s=140, zorder=5,
                label=f"Best vec_weight = {weights[best_idx]:.2f}  "
                      f"(gs = {rrf_scores[best_idx]:.3f})")
    for w, s in zip(weights, rrf_scores):
        ax1.annotate(f"{s:.3f}", (w, s), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color=GRAY)
    ax1.set_xlabel("RRF vector weight (1.0 = pure vector, 0.0 = pure keyword)",
                   fontsize=9)
    ax1.set_ylabel("Grounding score", fontsize=9)
    ax1.set_title("RRF weight sweep — vector-only is optimal",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    _add_headroom(ax1, rrf_scores)
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.25)

    # Panel B: rerank min_score sweep
    thresholds = sorted(float(k.split("_")[1]) for k in rer.keys())
    rer_scores = [rer[f"T_{t:.3f}"]["grounding_score"] for t in thresholds]
    best_idx = int(np.argmax(rer_scores))
    ax2.plot(thresholds, rer_scores, marker="o", color=MAROON, linewidth=2,
             markersize=7)
    ax2.scatter([thresholds[best_idx]], [rer_scores[best_idx]],
                color=RED, s=140, zorder=5,
                label=f"Best min_score = {thresholds[best_idx]:.2f}  "
                      f"(gs = {rer_scores[best_idx]:.3f})")
    ax2.axvline(0.5, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)
    for t, s in zip(thresholds, rer_scores):
        ax2.annotate(f"{s:.3f}", (t, s), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, color=GRAY)
    ax2.set_xlabel("Reranker minimum-score threshold", fontsize=9)
    ax2.set_ylabel("Grounding score", fontsize=9)
    ax2.set_title("Rerank threshold sweep — production is below optimum",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    _add_headroom(ax2, rer_scores)
    # Place the production marker text in the padded headroom above the data.
    ymin, ymax = ax2.get_ylim()
    ax2.text(0.5, ymax - 0.02 * (ymax - ymin), "current production (0.5)",
             ha="center", va="top", fontsize=8, color=GRAY, style="italic")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out = OUT_DIR / "sweeps_rrf_rerank.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_grounding_by_category(run_dir: Path) -> None:
    """Per-category grounding score with sample sizes annotated."""
    path = run_dir / "golden.json"
    if not path.exists():
        print(f"  skipped grounding-by-category (missing {path.name})")
        return
    data = json.loads(path.read_text())
    cats = data["table3_category"]

    # Only in-scope categories (drop OOD/adversarial which have gs=0 by design).
    label_map = {
        "faq_direct": "FAQ direct",
        "policy_hours": "Policy &\nhours",
        "database_recommendation": "Database\nrec.",
        "arabic": "Arabic",
        "follow_up_ambiguous": "Follow-up\nambiguous",
    }
    items = [(label_map[k], cats[k]) for k in label_map if k in cats]
    items.sort(key=lambda kv: kv[1]["grounding_score"], reverse=True)

    xs = np.arange(len(items))
    gs = [info["grounding_score"] for _, info in items]
    ar = [info["answer_relevance"] for _, info in items]
    grd = [info["groundedness"] for _, info in items]
    ns = [info["n"] for _, info in items]

    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars1 = ax.bar(xs - width, gs, width=width, color=MAROON,
                   label="Grounding score", edgecolor="white")
    bars2 = ax.bar(xs, ar, width=width, color=RED,
                   label="Answer relevance", edgecolor="white")
    bars3 = ax.bar(xs + width, grd, width=width, color=BLUE,
                   label="Groundedness", edgecolor="white")
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=GRAY, fontsize=7)
    for x, n in zip(xs, ns):
        ax.text(x, 0.02, f"n={n}", ha="center", va="bottom",
                color="white", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _ in items], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value (0-1 scale)", fontsize=10)
    ax.axhline(0.65, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(items) - 0.5, 0.66, "grounding gate ≥ 0.65",
            ha="right", va="bottom", color=GRAY, fontsize=8, style="italic")
    ax.set_title(
        f"Grounding metrics by question category — "
        f"{run_dir.name} (n_scored = {data['table1_overall']['n_scored']})",
        fontsize=12, color=MAROON, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "grounding_by_category.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_verifier_catch(run_dir: Path) -> None:
    """Verifier catch-rates by error type with FP-rate and gate line."""
    path = run_dir / "verifier-catch.json"
    if not path.exists():
        print(f"  skipped verifier-catch (missing {path.name})")
        return
    data = json.loads(path.read_text())
    summary = data["summary"]
    catch = summary["catch_rates"]

    types = [
        ("Fabrication\n(unsupported claim)", "fabricate", MAROON),
        ("Number swap\n(e.g. 3d → 7d)", "number_swap", RED),
        ("Entity swap\n(e.g. Jafet → Saab)", "entity_swap", BLUE),
    ]
    fp = summary["false_positive_rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # Panel A: catch rate per corruption type
    xs = np.arange(len(types))
    vals = [catch[k]["catch_rate"] for _, k, _ in types]
    caught = [catch[k]["caught"] for _, k, _ in types]
    appl = [catch[k]["applicable"] for _, k, _ in types]
    bars = ax1.bar(xs, vals, width=0.55,
                   color=[c for _, _, c in types], edgecolor="white")
    for bar, v, c, a in zip(bars, vals, caught, appl):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                 f"{v*100:.0f}%\n({c}/{a})",
                 ha="center", va="bottom",
                 color=GRAY, fontsize=9, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([lbl for lbl, _, _ in types], fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("Catch rate", fontsize=10)
    ax1.set_title("Verifier catch rate by corruption type",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: false-positive rate against gate
    fp_color = MAROON if fp <= 0.30 else RED
    bar = ax2.bar([0], [fp], width=0.5, color=fp_color, edgecolor="white")
    ax2.text(0, fp + 0.02, f"{fp*100:.0f}%",
             ha="center", va="bottom", color=GRAY,
             fontsize=11, fontweight="bold")
    ax2.text(0, 0.02, f"{summary['false_positive_n']}/10",
             ha="center", va="bottom", color="white", fontsize=8)
    ax2.axhline(0.30, color=GRAY, linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(0.45, 0.31, "gate ≤ 30%",
             ha="right", va="bottom", color=GRAY, fontsize=8, style="italic")
    ax2.set_xticks([0])
    ax2.set_xticklabels(["FP rate"], fontsize=9)
    ax2.set_ylim(0, 0.6)
    ax2.set_title("False-positive rate",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "verifier_catch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_threshold_sweep(run_dir: Path) -> None:
    """Confidence-threshold sweep: coverage vs abstention recall vs grounding."""
    path = run_dir / "threshold.json"
    if not path.exists():
        print(f"  skipped threshold sweep (missing {path.name})")
        return
    data = json.loads(path.read_text())
    sweep = sorted(data["sweep"], key=lambda r: r["threshold"])
    thresholds = [r["threshold"] for r in sweep]
    coverage = [r["answer_coverage"] for r in sweep]
    abst_recall = [r["abstention_recall"] for r in sweep]
    abst_prec = [r["abstention_precision"] for r in sweep]
    gs = [r["mean_grounding_score_answered"] for r in sweep]
    cfg_partial = data["metadata"]["configured_partial"]
    cfg_confident = data["metadata"]["configured_confident"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, coverage, marker="o", color=BLUE, linewidth=2,
            markersize=4, label="Answer coverage (answered fraction)")
    ax.plot(thresholds, abst_recall, marker="s", color=MAROON, linewidth=2,
            markersize=4, label="Abstention recall (correctly refused / unanswerable)")
    ax.plot(thresholds, abst_prec, marker="^", color=RED, linewidth=2,
            markersize=4, label="Abstention precision (correct refusals / all refusals)")
    ax.plot(thresholds, gs, marker="d", color=GREEN, linewidth=2,
            markersize=4, label="Grounding score (answered subset)")

    ax.axvline(cfg_partial, color=GRAY, linestyle=":", linewidth=1.5, alpha=0.7)
    ax.axvline(cfg_confident, color=GRAY, linestyle="--", linewidth=1.5, alpha=0.7)
    ymax_text = 1.04
    ax.text(cfg_partial, ymax_text, f"partial\n({cfg_partial})",
            ha="center", va="top", fontsize=8, color=GRAY, style="italic")
    ax.text(cfg_confident, ymax_text, f"confident\n({cfg_confident})",
            ha="center", va="top", fontsize=8, color=GRAY, style="italic")

    ax.set_xlabel("Top-score threshold", fontsize=10)
    ax.set_ylabel("Metric value", fontsize=10)
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"Confidence-threshold sweep on the abstention bench (n = {data['metadata']['n_questions']})",
        fontsize=12, color=MAROON, fontweight="bold", pad=12,
    )
    ax.legend(loc="center right", fontsize=8, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)

    fig.text(0.5, -0.02,
             "The 0.45/0.60 production thresholds split the curve into "
             "abstain / partial / confident; raising them increases abstention "
             "recall at the cost of coverage.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "threshold_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_cache_modes(run_dir: Path) -> None:
    """Three-mode cache comparison: cold vs exact vs semantic."""
    path = run_dir / "cache.json"
    if not path.exists():
        print(f"  skipped cache modes (missing {path.name})")
        return
    data = json.loads(path.read_text())
    agg = data["aggregate"]

    modes = [
        ("Cold\n(no cache)", agg["cold_avg_latency_ms"], 1.0,
         None, GRAY),
        ("Exact-key\nreplay", agg["exact_avg_latency_ms"],
         agg["speedup_exact"], agg["exact_hit_rate"], MAROON),
        ("Semantic\n(cosine ≥ 0.95)", agg["semantic_avg_latency_ms"],
         agg["speedup_semantic"], agg["semantic_hit_rate"], RED),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    xs = np.arange(len(modes))
    lats = [m[1] for m in modes]
    bars = ax1.bar(xs, lats, width=0.55,
                   color=[m[4] for m in modes], edgecolor="white")
    for bar, lat, sp in zip(bars, lats, [m[2] for m in modes]):
        ax1.text(bar.get_x() + bar.get_width() / 2, lat + max(lats) * 0.02,
                 f"{lat:,.0f} ms\n({sp:.2f}x)",
                 ha="center", va="bottom",
                 color=GRAY, fontsize=9, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([m[0] for m in modes], fontsize=9)
    ax1.set_ylabel("Avg latency (ms)", fontsize=10)
    ax1.set_title("Latency by cache mode",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: hit rates
    hit_modes = [(m[0], m[3], m[4]) for m in modes if m[3] is not None]
    hxs = np.arange(len(hit_modes))
    hits = [h[1] for h in hit_modes]
    hbars = ax2.bar(hxs, hits, width=0.55,
                    color=[h[2] for h in hit_modes], edgecolor="white")
    for bar, h in zip(hbars, hits):
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                 f"{h*100:.0f}%", ha="center", va="bottom",
                 color=GRAY, fontsize=10, fontweight="bold")
    ax2.set_xticks(hxs)
    ax2.set_xticklabels([h[0] for h in hit_modes], fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Hit rate", fontsize=10)
    ax2.set_title("Hit rate by cache mode",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Cache effectiveness — {data['metadata']['num_query_pairs']} paraphrase pairs, "
        f"semantic threshold {data['metadata']['semantic_threshold']}",
        fontsize=12, color=MAROON, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = OUT_DIR / "cache_modes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_ablation_deltas(run_dir: Path) -> None:
    """Bar chart of per-stage ablation deltas (full minus ablated)."""
    path = run_dir / "ablation.json"
    if not path.exists():
        print(f"  skipped ablation deltas (missing {path.name})")
        return
    data = json.loads(path.read_text())
    res = data["results"]

    # delta = full − ablated; positive means full pipeline is better.
    stages = [
        ("Query\nrewriting", res["rewriting"]["full"]["grounding_score"]
         - res["rewriting"]["no_rewrite"]["grounding_score"],
         res["rewriting"]["full"]["grounding_score"],
         res["rewriting"]["no_rewrite"]["grounding_score"],
         res["rewriting"]["n"]),
        ("BM25\n(keyword arm)", res["bm25"]["full"]["grounding_score"]
         - res["bm25"]["vector_only"]["grounding_score"],
         res["bm25"]["full"]["grounding_score"],
         res["bm25"]["vector_only"]["grounding_score"],
         res["bm25"]["n"]),
        ("LLM\nreranking", res["reranking"]["full"]["grounding_score"]
         - res["reranking"]["no_rerank"]["grounding_score"],
         res["reranking"]["full"]["grounding_score"],
         res["reranking"]["no_rerank"]["grounding_score"],
         res["reranking"]["n"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Panel A: deltas (full − ablated). Positive => stage helps.
    xs = np.arange(len(stages))
    deltas = [s[1] for s in stages]
    ns = [s[4] for s in stages]
    colors = [GREEN if d > 0 else RED for d in deltas]
    bars = ax1.bar(xs, deltas, width=0.55, color=colors, edgecolor="white")
    # Set y-limits with padding for labels.
    ymax_data = max(deltas + [0])
    ymin_data = min(deltas + [0])
    span = (ymax_data - ymin_data) or 0.02
    pad = span * 0.30
    ax1.set_ylim(ymin_data - pad, ymax_data + pad)
    label_off = span * 0.05
    for bar, d, n in zip(bars, deltas, ns):
        # Δ label: always on the outside of the bar.
        if d >= 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, d + label_off,
                     f"{d:+.3f}", ha="center", va="bottom",
                     color=GRAY, fontsize=10, fontweight="bold")
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, d - label_off,
                     f"{d:+.3f}", ha="center", va="top",
                     color=GRAY, fontsize=10, fontweight="bold")
        # n-count: always near the zero line, on the opposite side of the bar.
        n_y = label_off if d < 0 else -label_off
        n_va = "bottom" if n_y > 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, n_y, f"n={n}",
                 ha="center", va=n_va, color=GRAY, fontsize=8, style="italic")
    ax1.axhline(0, color=GRAY, linewidth=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax1.set_ylabel("Δ grounding score (full − ablated)", fontsize=9)
    ax1.set_title("Stage-removal Δ (positive = stage helps)",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: side-by-side full vs ablated grounding scores
    width = 0.35
    full_vals = [s[2] for s in stages]
    abl_vals = [s[3] for s in stages]
    fbars = ax2.bar(xs - width / 2, full_vals, width=width, color=MAROON,
                    label="Full pipeline", edgecolor="white")
    abars = ax2.bar(xs + width / 2, abl_vals, width=width, color=LIGHT_GRAY,
                    label="Stage removed", edgecolor="white")
    for bar in list(fbars) + list(abars):
        v = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.3f}", ha="center", va="bottom",
                 color=GRAY, fontsize=8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s[0] for s in stages], fontsize=9)
    ax2.set_ylabel("Grounding score", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Side-by-side: full vs ablated",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.02,
             "Deltas under 0.01 are within measurement noise on these "
             "10-15 item per-stage stress sets. BM25 delta is negative on this "
             "run, supporting the production decision to disable the keyword arm.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "ablation_deltas.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def chart_cost_breakdown(cost_run_dir: Path) -> None:
    """Per-stage cost breakdown for B0 + cost-vs-quality scatter for B0-B5."""
    path = cost_run_dir / "cost.json"
    if not path.exists():
        print(f"  skipped cost breakdown (missing {path.name})")
        return
    data = json.loads(path.read_text())
    b0 = data["per_baseline_cost"]["B0"]
    pareto = data["cost_vs_quality_pareto"]
    pareto_optimal = set(data.get("pareto_optimal_baselines", []))
    annual = data.get("annual_cost_full_pipeline_usd", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: B0 per-stage cost (sorted descending). Units: USD per 1k queries.
    stage_costs = [(k, v) for k, v in b0["per_stage_usd"].items() if v > 0]
    stage_costs.sort(key=lambda kv: kv[1], reverse=True)
    stage_label = {
        "input_guard_embed": "Input guard\n(embed)",
        "query_rewriter": "Query\nrewriter",
        "retrieval_embed": "Retrieval\nembed",
        "reranker": "LLM\nreranker",
        "generator": "Generator",
        "verifier": "Verifier",
    }
    xs = np.arange(len(stage_costs))
    vals_per_1k = [v * 1000 for _, v in stage_costs]
    bars = ax1.bar(xs, vals_per_1k, width=0.6, color=MAROON, edgecolor="white")
    for bar, v in zip(bars, vals_per_1k):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + max(vals_per_1k) * 0.02,
                 _d(f"${v:.3f}"),
                 ha="center", va="bottom",
                 color=GRAY, fontsize=8, fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([stage_label.get(k, k) for k, _ in stage_costs],
                        fontsize=8)
    ax1.set_ylabel("Cost (USD per 1,000 queries)", fontsize=10)
    ax1.set_title(
        _d(f"B0 cost breakdown — total ${b0['total_per_query_usd']*1000:.3f} per 1k queries "
           f"(${b0['total_per_query_usd']:.6f}/query)"),
        fontsize=11, color=MAROON, fontweight="bold", pad=10,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # Panel B: cost vs quality scatter for B0-B5
    for entry in pareto:
        b = entry["baseline"]
        x = entry["cost_usd_per_query"]
        y = entry["grounding_score"]
        is_opt = b in pareto_optimal
        is_b0 = b == "B0"
        face = MAROON if is_b0 else (GREEN if is_opt else LIGHT_GRAY)
        edge = RED if is_b0 else GRAY
        size = 220 if is_b0 else 140
        ax2.scatter([x], [y], s=size, color=face, edgecolor=edge,
                    linewidth=1.5, zorder=5)
        ax2.annotate(b, (x, y), textcoords="offset points", xytext=(8, 6),
                     fontsize=9, fontweight="bold", color=GRAY)
    ax2.set_xscale("log")
    ax2.set_xlabel("Cost (USD per query, log scale)", fontsize=10)
    ax2.set_ylabel("Grounding score", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Cost vs grounding score across baselines",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=10,
                   markerfacecolor=MAROON, markeredgecolor=RED,
                   label="B0 (full pipeline)"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=GREEN, markeredgecolor=GRAY,
                   label="Pareto-optimal baseline"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=LIGHT_GRAY, markeredgecolor=GRAY,
                   label="Dominated baseline"),
    ]
    ax2.legend(handles=legend_handles, loc="lower right", fontsize=8,
               framealpha=0.95)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.25)

    if annual:
        annual_text = " · ".join(_d(f"{k}: ${v:,.2f}")
                                  for k, v in annual.items())
        fig.text(0.5, -0.02,
                 f"Annual extrapolation (no cache): {annual_text}",
                 ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "cost_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main():
    print(f"Loading scorecards into {OUT_DIR}/")
    scorecards = []
    for label, path in SCORECARDS:
        if not path.exists():
            raise FileNotFoundError(f"Missing scorecard: {path}")
        scorecards.append((label, load_scorecard(path)))
        print(f"  loaded {path.relative_to(REPO_ROOT)} "
              f"({len(scorecards[-1][1])} numeric rows)")

    write_deltas_csv(scorecards, OUT_DIR / "deltas.csv")
    print(f"  wrote {(OUT_DIR / 'deltas.csv').relative_to(REPO_ROOT)}")

    # Cross-run trend charts
    chart_iteration_improvements(scorecards)
    chart_safety_latency_tradeoff(scorecards)
    chart_pipeline_maturity_radar(scorecards)
    chart_timeline_bars(scorecards)

    # Latest-run breakdown charts (drawn from a single full eval run)
    print(f"\nLatest-run breakdown charts from {LATEST_RUN_DIR.name}/")
    chart_baselines_comparison(LATEST_RUN_DIR)
    chart_ragas_breakdown(LATEST_RUN_DIR)
    chart_per_stage_latency(LATEST_RUN_DIR)
    chart_sweeps(LATEST_RUN_DIR)
    chart_grounding_by_category(LATEST_RUN_DIR)
    chart_verifier_catch(LATEST_RUN_DIR)
    chart_threshold_sweep(LATEST_RUN_DIR)
    chart_cache_modes(LATEST_RUN_DIR)
    chart_ablation_deltas(LATEST_RUN_DIR)

    # Cost charts come from an earlier run — cost.json was not regenerated in T4.
    print(f"\nCost charts from {COST_RUN_DIR.name}/")
    chart_cost_breakdown(COST_RUN_DIR)

    print("done.")


if __name__ == "__main__":
    main()
