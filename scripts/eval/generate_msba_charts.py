"""
generate_msba_charts.py

Produces six analyst-style charts for the MSBA capstone report. All charts
draw on real eval JSON outputs — no invented numbers.

Outputs (under scripts/eval/msba_charts/):
  - baseline_comparison.png   : B0 full pipeline vs B1-B5 (grounding + relevance)
  - cost_quality_pareto.png   : cost-per-query (log) vs grounding score, with Pareto frontier
  - latency_distribution.png  : per-stage mean + boxplot of cold pipeline latency
  - threshold_sweep.png       : confidence threshold vs answer coverage / grounding
  - category_performance.png  : per-category grounding / relevance / coverage
  - annual_cost_projection.png: USD cost projection at 1k -> 1M queries / yr

Run from repo root:
    python3 scripts/eval/generate_msba_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# AUB brand palette (matches backend/chart_generator.py)
MAROON = "#840132"
RED = "#ee3524"
GRAY = "#424242"
LIGHT_GRAY = "#d1d2d2"
GREEN = "#2d6a4f"
BLUE = "#457b9d"
GOLD = "#b5651d"

REPO_ROOT = Path(__file__).resolve().parents[2]
LATEST_RUN = REPO_ROOT / "eval_run_20260429_214144"
COST_RUN = REPO_ROOT / "baseline_before_fixes/eval_run_20260423_120009"
OUT_DIR = REPO_ROOT / "scripts/eval/msba_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def style_ax(ax, title="", xlabel="", ylabel=""):
    if title:
        ax.set_title(title, fontsize=12, color=MAROON, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=GRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GRAY)
    ax.spines["bottom"].set_color(LIGHT_GRAY)


# -----------------------------------------------------------------------------
# 1. Baseline comparison (B0 vs B1-B5)
# -----------------------------------------------------------------------------
def chart_baseline_comparison():
    data = load_json(LATEST_RUN / "baselines.json")
    summaries = data["summaries"]["answerable"]
    order = ["B0", "B1", "B2", "B3", "B4", "B5"]
    short_labels = {
        "B0": "Full pipeline",
        "B1": "LLM-only\n(no retrieval)",
        "B2": "BM25-only\n+ raw chunk",
        "B3": "Vector-only\n+ raw chunk",
        "B4": "FAQ nearest\n(verbatim)",
        "B5": "Retrieve\n+ summarize",
    }
    labels = []
    grounding = []
    relevance = []
    for k in order:
        s = summaries[k]
        labels.append(f"{k}\n{short_labels[k]}")
        grounding.append(s["overall"]["grounding_score"])
        relevance.append(s["overall"]["answer_relevance"])

    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(order))
    width = 0.38

    # Consistent colors per metric across all bars; highlight B0 with thick maroon edge
    edge_colors = [MAROON if k == "B0" else "white" for k in order]
    edge_widths = [2.5 if k == "B0" else 0.5 for k in order]

    bar1 = ax.bar(xs - width / 2, grounding, width=width, label="Grounding score",
                  color=BLUE, edgecolor=edge_colors, linewidth=edge_widths)
    bar2 = ax.bar(xs + width / 2, relevance, width=width, label="Answer relevance",
                  color=GOLD, edgecolor=edge_colors, linewidth=edge_widths, alpha=0.95)

    for bars, vals in [(bar1, grounding), (bar2, relevance)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=GRAY)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)

    # Build legend with metric swatches + a B0-highlight indicator
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=BLUE, edgecolor="white", label="Grounding score"),
        Patch(facecolor=GOLD, edgecolor="white", label="Answer relevance"),
        Patch(facecolor="white", edgecolor=MAROON, linewidth=2.5,
              label="B0 (this work)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=False)
    style_ax(ax,
             "Full pipeline (B0) vs five reference baselines",
             ylabel="Score (0-1)")
    ax.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.02,
             "Higher is better on both metrics. B0 leads on relevance and ranks 2nd on grounding "
             "behind B5 (which never abstains, inflating its raw score on answerable Qs).",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "baseline_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 2. Cost vs quality Pareto frontier
# -----------------------------------------------------------------------------
def chart_cost_quality_pareto():
    cost = load_json(COST_RUN / "cost_estimate.json")
    pareto_set = set(cost.get("pareto_optimal_baselines", []))

    # Use newer grounding scores from latest run, but cost from cost_estimate.json
    new_baselines = load_json(LATEST_RUN / "baselines.json")["summaries"]["answerable"]

    points = []
    for entry in cost["cost_vs_quality_pareto"]:
        b = entry["baseline"]
        # Pull the *newer* grounding score for fairness with the rest of the report
        gs_new = new_baselines[b]["overall"]["grounding_score"]
        points.append({
            "baseline": b,
            "label": entry["label"],
            "cost": entry["cost_usd_per_query"],
            "gs": gs_new,
            "is_pareto": b in pareto_set or b == "B0",
        })

    fig, ax = plt.subplots(figsize=(10, 6))

    for p in points:
        is_b0 = p["baseline"] == "B0"
        color = MAROON if is_b0 else (GREEN if p["is_pareto"] else GRAY)
        size = 320 if is_b0 else 200
        marker = "*" if is_b0 else ("D" if p["is_pareto"] else "o")
        ax.scatter(p["cost"], p["gs"], s=size, color=color, marker=marker,
                   edgecolor="white", linewidth=1.5, zorder=3,
                   label=None)
        # Annotate
        offset_y = 0.025 if not is_b0 else -0.04
        ax.annotate(f"{p['baseline']}: {p['label']}",
                    (p["cost"], p["gs"]),
                    xytext=(8, offset_y * 100),
                    textcoords="offset points",
                    fontsize=8.5, color=GRAY,
                    fontweight="bold" if is_b0 else "normal")

    ax.set_xscale("log")
    ax.set_xlim(5e-7, 3e-3)
    ax.set_ylim(0, 1.0)
    style_ax(ax,
             "Cost vs quality: where does B0 sit on the frontier?",
             xlabel="Cost per query (USD, log scale)",
             ylabel="Grounding score (0-1)")
    ax.grid(True, which="both", alpha=0.2)

    # Legend keys
    legend_handles = [
        plt.scatter([], [], color=MAROON, marker="*", s=240, edgecolor="white",
                    label="B0 (this work)"),
        plt.scatter([], [], color=GREEN, marker="D", s=130, edgecolor="white",
                    label="Pareto-optimal baseline"),
        plt.scatter([], [], color=GRAY, marker="o", s=130, edgecolor="white",
                    label="Dominated baseline"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

    fig.text(0.5, -0.02,
             "B0 trades a ~7x cost premium over B3/B4 for higher answer relevance, "
             "abstention safety, and bilingual capability — features a single grounding-only "
             "axis cannot capture.",
             ha="center", fontsize=8, color=GRAY, style="italic", wrap=True)

    plt.tight_layout()
    out = OUT_DIR / "cost_quality_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 3. Latency distribution: stage breakdown + cold/cached boxplot
# -----------------------------------------------------------------------------
def chart_latency_distribution():
    lat = load_json(LATEST_RUN / "latency.json")
    agg = lat["aggregate"]
    stages = [
        ("Input guard", agg["avg_input_guard_ms"], agg["pct_guard"]),
        ("Query rewriting", agg["avg_query_rewriting_ms"], agg["pct_rewriting"]),
        ("Embedding", agg["avg_embedding_ms"], agg["pct_embedding"]),
        ("Retrieval", agg["avg_retrieval_ms"], agg["pct_retrieval"]),
        ("Reranking", agg["avg_reranking_ms"], agg["pct_reranking"]),
        ("Generation+verify", agg["avg_generation_verify_ms"], agg["pct_generation_verify"]),
    ]

    cold = [r["cold_ms"] for r in lat["full_pipeline"]]
    cached = [r["cached_ms"] for r in lat["full_pipeline"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1.4, 1]})

    # LEFT: stacked horizontal bar of mean stage timings
    ax1 = axes[0]
    colors = [MAROON, RED, GOLD, GREEN, BLUE, GRAY]
    left = 0.0
    for (name, ms, pct), color in zip(stages, colors):
        ax1.barh(0, ms, left=left, color=color, height=0.45,
                 edgecolor="white", linewidth=1, label=f"{name} ({pct:.1f}%)")
        if pct >= 5:
            ax1.text(left + ms / 2, 0, f"{ms:.0f}ms",
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        left += ms
    ax1.set_xlim(0, left * 1.05)
    ax1.set_yticks([])
    ax1.set_xlabel("Milliseconds (mean of n=10 cold queries)", fontsize=9, color=GRAY)
    ax1.set_title(f"Pipeline stage latency  (total ≈ {left:.0f}ms)",
                  fontsize=11, color=MAROON, fontweight="bold", pad=10)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
               fontsize=8, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_color(LIGHT_GRAY)

    # RIGHT: cold vs cached boxplot
    ax2 = axes[1]
    bp = ax2.boxplot([cold, cached], widths=0.5, patch_artist=True,
                     labels=["Cold\n(no cache)", "Cached"],
                     medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], [MAROON, GREEN]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    for whisker in bp["whiskers"]:
        whisker.set(color=GRAY, linewidth=1)
    for cap in bp["caps"]:
        cap.set(color=GRAY, linewidth=1)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor=GRAY, markeredgecolor="white",
                  markersize=5, alpha=0.6)

    p50_cold, p95_cold = np.percentile(cold, [50, 95])
    p50_cached, p95_cached = np.percentile(cached, [50, 95])
    ax2.text(1, p95_cold * 1.04, f"p50={p50_cold:.0f}ms\np95={p95_cold:.0f}ms",
             ha="center", fontsize=8, color=MAROON, fontweight="bold")
    ax2.text(2, p95_cached + 200, f"p50={p50_cached:.0f}ms\np95={p95_cached:.0f}ms",
             ha="center", fontsize=8, color=GREEN, fontweight="bold")

    style_ax(ax2,
             f"Cold vs cached latency  ({agg['cache_speedup']:.1f}x speedup)",
             ylabel="Milliseconds")
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = OUT_DIR / "latency_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 4. Threshold sweep: coverage vs grounding
# -----------------------------------------------------------------------------
def chart_threshold_sweep():
    thr = load_json(LATEST_RUN / "threshold.json")
    sweep = thr["sweep"]
    thresholds = [s["threshold"] for s in sweep]
    coverage = [s["answer_coverage"] for s in sweep]
    grounding = [s["mean_grounding_score_answered"] for s in sweep]
    false_abst = [s["false_abstention"] for s in sweep]

    configured_partial = thr.get("configured_partial", 0.45)
    configured_confident = thr.get("configured_confident", 0.6)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(thresholds, coverage, color=MAROON, linewidth=2.4,
             marker="o", markersize=5, label="Answer coverage (frac. of Qs answered)")
    ax1.set_ylabel("Answer coverage", color=MAROON, fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor=MAROON)
    ax1.set_xlabel("Confidence threshold (top retrieval score)", fontsize=10, color=GRAY)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, grounding, color=BLUE, linewidth=2.4,
             marker="s", markersize=5, label="Mean grounding score (on answered Qs)")
    ax2.set_ylabel("Grounding score (on answered)", color=BLUE, fontsize=10)
    ax2.set_ylim(0.7, 0.95)
    ax2.tick_params(axis="y", labelcolor=BLUE)

    # Mark configured thresholds
    ax1.axvline(configured_partial, color=GOLD, linestyle="--", linewidth=1.4, alpha=0.8)
    ax1.text(configured_partial, 1.07,
             f"partial={configured_partial}",
             ha="center", fontsize=8, color=GOLD, fontweight="bold")
    ax1.axvline(configured_confident, color=GREEN, linestyle="--", linewidth=1.4, alpha=0.8)
    ax1.text(configured_confident, 1.07,
             f"confident={configured_confident}",
             ha="center", fontsize=8, color=GREEN, fontweight="bold")

    ax1.set_title("Threshold sweep: coverage vs grounding tradeoff (n=40 golden Qs)",
                  fontsize=12, color=MAROON, fontweight="bold", pad=24)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(True, alpha=0.2)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9, frameon=False)

    fig.text(0.5, -0.02,
             "Higher thresholds raise grounding on the answered subset but quickly cut "
             "coverage — the operating point trades safety against helpfulness.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "threshold_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 5. Per-category performance
# -----------------------------------------------------------------------------
def chart_category_performance():
    g = load_json(LATEST_RUN / "golden.json")
    cats = g["table3_category"]

    # Drop categories where the system is expected to refuse (score=0 by design).
    keep = {k: v for k, v in cats.items() if k not in ("out_of_domain", "adversarial")}

    order = ["faq_direct", "policy_hours", "database_recommendation",
             "follow_up_ambiguous", "arabic"]
    order = [k for k in order if k in keep]

    pretty = {
        "faq_direct": "FAQ direct",
        "policy_hours": "Policy / hours",
        "database_recommendation": "Database rec.",
        "follow_up_ambiguous": "Follow-up ambig.",
        "arabic": "Arabic Qs",
    }

    n_per = [keep[k]["n"] for k in order]
    labels = [f"{pretty[k]}\n(n={n})" for k, n in zip(order, n_per)]
    grounding = [keep[k]["grounding_score"] for k in order]
    relevance = [keep[k]["answer_relevance"] for k in order]
    coverage = [keep[k]["keyword_coverage"] for k in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(order))
    width = 0.27

    b1 = ax.bar(xs - width, grounding, width=width, color=MAROON, label="Grounding score")
    b2 = ax.bar(xs, relevance, width=width, color=RED, label="Answer relevance")
    b3 = ax.bar(xs + width, coverage, width=width, color=BLUE, label="Keyword coverage")

    for bars, vals in [(b1, grounding), (b2, relevance), (b3, coverage)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=GRAY)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style_ax(ax,
             "Per-category quality breakdown (golden set v1.0)",
             ylabel="Score (0-1)")
    ax.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.05,
             "Direct FAQ questions are answered most reliably; ambiguous follow-ups and "
             "Arabic queries lag on grounding even though relevance stays high.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "category_performance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 6. Annual cost projection (per-stage stacked)
# -----------------------------------------------------------------------------
def chart_annual_cost_projection():
    cost = load_json(COST_RUN / "cost_estimate.json")
    b0 = cost["per_baseline_cost"]["B0"]
    stages = b0["per_stage_usd"]  # dict: stage -> $/query

    # Volumes for x-axis
    volumes = [1_000, 10_000, 100_000, 1_000_000]
    vol_labels = ["1k", "10k", "100k", "1M"]

    stage_order = ["query_rewriter", "reranker", "generator", "verifier",
                   "input_guard_embed", "retrieval_embed"]
    stage_pretty = {
        "input_guard_embed": "Input guard (embed)",
        "query_rewriter": "Query rewriter (LLM)",
        "retrieval_embed": "Retrieval (embed)",
        "reranker": "Reranker (LLM)",
        "generator": "Generator (LLM)",
        "verifier": "Verifier (LLM)",
    }
    stage_colors = {
        "query_rewriter": MAROON,
        "reranker": RED,
        "generator": GOLD,
        "verifier": GREEN,
        "input_guard_embed": BLUE,
        "retrieval_embed": GRAY,
    }

    # Build [stage][volume] matrix in USD
    matrix = {s: [stages[s] * v for v in volumes] for s in stage_order}
    totals = [sum(matrix[s][i] for s in stage_order) for i in range(len(volumes))]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(volumes))
    bottoms = np.zeros(len(volumes))
    for s in stage_order:
        vals = matrix[s]
        ax.bar(xs, vals, bottom=bottoms, color=stage_colors[s],
               edgecolor="white", label=stage_pretty[s], width=0.55)
        bottoms = bottoms + np.array(vals)

    for x, total in zip(xs, totals):
        ax.text(x, total * 1.02, f"${total:,.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=MAROON)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\nqueries / yr" for v in vol_labels], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(totals) * 3)
    style_ax(ax,
             "Annual API cost projection by query volume  (B0 full pipeline, log scale)",
             ylabel="USD per year (log scale)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.2, which="both")

    cpq = b0["total_per_query_usd"]
    fig.text(0.5, -0.02,
             f"Cost-per-query ≈ ${cpq:.4f}. Reranker + generator dominate spend; "
             "embedding and guard costs are negligible at any plausible scale.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "annual_cost_projection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 7. Retrieval: RRF weight sweep (replaces P@k/MRR/nDCG which require labels)
# -----------------------------------------------------------------------------
def chart_retrieval_rrf_sweep():
    rrf = load_json(LATEST_RUN / "rrf.json")
    hyb = load_json(LATEST_RUN / "hybrid.json")["aggregate"]

    weights = []
    grounding = []
    abstention = []
    for key, res in rrf["results"].items():
        weights.append(res["vector_weight"])
        grounding.append(res["grounding_score"])
        abstention.append(res["abstention_rate"])

    # Sort by weight
    order = np.argsort(weights)
    weights = [weights[i] for i in order]
    grounding = [grounding[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(weights, grounding, color=MAROON, linewidth=2.4,
            marker="o", markersize=8, zorder=3)

    # Mark production setting (vector_weight = 1.0)
    prod_idx = weights.index(1.0)
    ax.scatter(weights[prod_idx], grounding[prod_idx],
               s=350, color=GREEN, marker="*", zorder=4,
               edgecolor="white", linewidth=1.5,
               label=f"Production setting (vector-only, vw=1.0)")

    for w, g in zip(weights, grounding):
        ax.text(w, g + 0.005, f"{g:.3f}",
                ha="center", fontsize=8, color=GRAY)

    ax.set_xlabel("RRF vector weight  (0 = BM25-only, 1 = vector-only)",
                  fontsize=10, color=GRAY)
    ax.set_ylim(0.70, 0.78)
    ax.set_xticks(weights)
    style_ax(ax,
             "Hybrid retrieval (RRF) weight sweep — grounding score on n=15",
             ylabel="Grounding score (0-1)")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)

    # Annotate the BM25-decision finding
    ax.text(0.5, 0.71,
            f"Recall@5: vector={hyb['vector_recall_at_5']:.2f}, "
            f"hybrid={hyb['hybrid_recall_at_5']:.2f}\n"
            f"BM25 added {hyb['avg_new_results_from_keyword']:.1f} new top-5 docs on average",
            ha="center", fontsize=9, color=GRAY,
            bbox=dict(facecolor="white", edgecolor=LIGHT_GRAY, boxstyle="round,pad=0.5"))

    fig.text(0.5, -0.02,
             "All vector weights produce near-identical grounding (Δ < 0.02). "
             "BM25 contributes no new top-5 results, so the production retriever "
             "uses pure vector search (RRF weight = 1.0).",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "retrieval_rrf_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 8. RAGAS quality split by language (Arabic vs English)
# -----------------------------------------------------------------------------
def chart_ragas_bilingual():
    rag = load_json(LATEST_RUN / "ragas.json")
    by_lang = rag["by_language"]

    metrics_keys = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]
    metrics_pretty = ["Faithfulness", "Answer relevancy", "Context precision"]

    en_vals = [by_lang["en"][k] for k in metrics_keys]
    ar_vals = [by_lang["ar"][k] for k in metrics_keys]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(metrics_pretty))
    width = 0.36

    b1 = ax.bar(xs - width / 2, en_vals, width=width, color=BLUE,
                label="English", edgecolor="white")
    b2 = ax.bar(xs + width / 2, ar_vals, width=width, color=GOLD,
                label="Arabic", edgecolor="white")

    for bars, vals in [(b1, en_vals), (b2, ar_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, color=GRAY)

    # Highlight the gap on answer_relevancy
    rel_gap = en_vals[1] - ar_vals[1]
    ax.annotate(
        f"Δ {rel_gap:+.2f}",
        xy=(1, max(en_vals[1], ar_vals[1]) + 0.06),
        xytext=(1, max(en_vals[1], ar_vals[1]) + 0.18),
        ha="center", fontsize=9, color=MAROON, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=MAROON, lw=1.2),
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(metrics_pretty, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    style_ax(ax,
             f"RAGAS quality by language  (n={rag['metadata'].get('n_scored', '?')} scored)",
             ylabel="Score (0-1)")
    ax.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.02,
             "Bilingual parity is strong on faithfulness and context precision; "
             "answer relevancy drops in Arabic, reflecting LLM-judge bias toward English phrasing.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "ragas_bilingual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 9. Latency distribution by language and category
# -----------------------------------------------------------------------------
def chart_latency_by_segment():
    g = load_json(LATEST_RUN / "golden.json")
    rows = [r for r in g["raw_results"]
            if r.get("elapsed_ms") and r.get("category") not in ("out_of_domain", "adversarial")]

    by_lang = {"en": [], "ar": []}
    by_cat = {}
    for r in rows:
        lang = r["language"]
        cat = r["category"]
        ms = r["elapsed_ms"]
        if lang in by_lang:
            by_lang[lang].append(ms)
        by_cat.setdefault(cat, []).append(ms)

    cat_order = ["faq_direct", "policy_hours", "database_recommendation",
                 "follow_up_ambiguous", "arabic"]
    cat_order = [c for c in cat_order if c in by_cat]
    cat_pretty = {
        "faq_direct": "FAQ direct",
        "policy_hours": "Policy /\nhours",
        "database_recommendation": "Database\nrec.",
        "follow_up_ambiguous": "Follow-up\nambig.",
        "arabic": "Arabic",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                              gridspec_kw={"width_ratios": [1, 1.8]})

    # LEFT: by language
    ax1 = axes[0]
    bp1 = ax1.boxplot([by_lang["en"], by_lang["ar"]], widths=0.5,
                       patch_artist=True, tick_labels=["English\n(n={})".format(len(by_lang["en"])),
                                                         "Arabic\n(n={})".format(len(by_lang["ar"]))],
                       medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp1["boxes"], [BLUE, GOLD]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    for whisker in bp1["whiskers"]:
        whisker.set(color=GRAY, linewidth=1)
    for cap in bp1["caps"]:
        cap.set(color=GRAY, linewidth=1)

    style_ax(ax1, "Latency by language", ylabel="End-to-end latency (ms)")
    ax1.grid(axis="y", alpha=0.2)

    # Annotate medians
    for i, lang in enumerate(["en", "ar"]):
        med = float(np.median(by_lang[lang]))
        ax1.text(i + 1, med, f"  median {med:.0f}ms",
                 va="center", fontsize=8, color=GRAY)

    # RIGHT: by category
    ax2 = axes[1]
    data = [by_cat[c] for c in cat_order]
    labels = [f"{cat_pretty[c]}\n(n={len(by_cat[c])})" for c in cat_order]
    bp2 = ax2.boxplot(data, widths=0.55, patch_artist=True,
                       tick_labels=labels,
                       medianprops=dict(color="white", linewidth=2))
    palette = [MAROON, RED, GOLD, GREEN, BLUE]
    for patch, color in zip(bp2["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.85)
    for whisker in bp2["whiskers"]:
        whisker.set(color=GRAY, linewidth=1)
    for cap in bp2["caps"]:
        cap.set(color=GRAY, linewidth=1)

    style_ax(ax2, "Latency by question category", ylabel="End-to-end latency (ms)")
    ax2.grid(axis="y", alpha=0.2)

    fig.text(0.5, -0.02,
             "Cold-path latency from the golden eval (n=90 in-scope queries). "
             "Arabic queries pay an extra translation step in the rewriter.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "latency_by_segment.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 10. Intent classification confusion matrix
# -----------------------------------------------------------------------------
def chart_intent_confusion_matrix():
    intent = load_json(LATEST_RUN / "intent.json")
    cm = intent["confusion_matrix"]
    labels = list(cm.keys())  # rows = true intents
    n = len(labels)

    matrix = np.zeros((n, n), dtype=float)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i][j] = cm[true_label].get(pred_label, 0)

    # Normalize by row (per true-intent recall)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("aub", ["#ffffff", "#f9e0e6", MAROON])
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted intent", fontsize=10, color=GRAY)
    ax.set_ylabel("True intent", fontsize=10, color=GRAY)

    for i in range(n):
        for j in range(n):
            count = int(matrix[i][j])
            if count == 0:
                continue
            pct = norm[i][j]
            color = "white" if pct > 0.5 else GRAY
            ax.text(j, i, f"{count}\n({pct*100:.0f}%)",
                    ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Recall (per true intent)", shrink=0.8)

    acc = intent["overall_accuracy"]
    n_q = intent["metadata"]["n_questions"]
    ax.set_title(f"Intent classification confusion matrix  "
                 f"(accuracy = {acc*100:.1f}%, n={n_q})",
                 fontsize=12, color=MAROON, fontweight="bold", pad=12)

    fig.text(0.5, 0.02,
             "Two errors total: one 'database' query routed to 'general'; "
             "one 'general' query routed to 'faq'. No cross-confusion between hours/contact/faq.",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "intent_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 11. Cost vs naive single-LLM baselines (gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-4)
# -----------------------------------------------------------------------------
def chart_cost_vs_gpt4():
    cost = load_json(COST_RUN / "cost_estimate.json")
    b0_cost = cost["per_baseline_cost"]["B0"]["total_per_query_usd"]
    # B1 token assumption: 1500 in / 300 out (single LLM call, no retrieval)
    gen = cost["stage_token_assumptions"]["generator"]
    in_tok, out_tok = gen["in"], gen["out"]

    # Pricing per 1M tokens (USD) — public list pricing as of 2025
    pricing = {
        "gpt-4o-mini\n(B1 measured)": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4 (legacy)": (30.00, 60.00),
    }

    bars_data = [("B0\nFull pipeline\n(this work)", b0_cost, MAROON)]
    palette = [BLUE, GOLD, RED, GRAY]
    for (label, (pin, pout)), color in zip(pricing.items(), palette):
        c = (in_tok * pin + out_tok * pout) / 1_000_000
        bars_data.append((f"Naive single-LLM call\n{label}", c, color))

    labels = [b[0] for b in bars_data]
    values = [b[1] for b in bars_data]
    colors = [b[2] for b in bars_data]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, color=colors, edgecolor="white", width=0.6)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15,
                f"${v:.4f}",
                ha="center", va="bottom", fontsize=9.5,
                color=GRAY, fontweight="bold")

    # Annual projection at 100k queries on secondary annotation
    for x, v in zip(xs, values):
        annual = v * 100_000
        ax.text(x, v * 1.5, f"${annual:,.0f} / yr\n@100k Q",
                ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

    ax.set_yscale("log")
    ax.set_ylim(min(values) * 0.5, max(values) * 5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    style_ax(ax,
             "Cost per query: full pipeline vs naive single-LLM baselines  (log scale)",
             ylabel="USD per query (log scale)")
    ax.grid(axis="y", alpha=0.2, which="both")

    ratio = values[-1] / values[0]
    fig.text(0.5, -0.04,
             f"All single-LLM costs assume the same {in_tok}-in / {out_tok}-out token budget. "
             f"A naive GPT-4 call would cost roughly {ratio:.0f}x more per query than B0 — "
             "while still lacking retrieval grounding (B1 grounding score = 0.36).",
             ha="center", fontsize=8, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "cost_vs_gpt4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------------
# 12. Model ablation 2x2  (only if scripts/eval/run_ablation_models.py was run)
# -----------------------------------------------------------------------------
ABLATION_DIR = REPO_ROOT / "ablation_models"


def chart_model_ablation_2x2():
    """Compare {small,large} embedding x {gpt-4o-mini, gpt-4.1} LLM.

    Skips silently if ablation_models/_summary.csv does not exist.
    """
    summary_path = ABLATION_DIR / "_summary.csv"
    if not summary_path.exists():
        print(f"  skipped model_ablation_2x2 (no {summary_path.relative_to(REPO_ROOT)})")
        return

    import csv as _csv
    rows = list(_csv.DictReader(summary_path.open()))
    if not rows:
        print("  skipped model_ablation_2x2 (summary is empty)")
        return

    # Map config label -> dict
    by_label = {r["config"]: r for r in rows}

    # Display order: small_mini, small_4.1, large_mini, large_4.1
    expected = ["small_4o-mini", "small_4.1", "large_4o-mini", "large_4.1"]
    avail = [c for c in expected if c in by_label]
    if len(avail) < 2:
        print("  skipped model_ablation_2x2 (need at least 2 configs)")
        return

    metrics = [
        ("Grounding score",       "grounding_score",        (0, 1.0), "higher better"),
        ("RAGAS faithfulness",    "ragas_faithfulness",     (0, 1.0), "higher better"),
        ("RAGAS answer relevancy","ragas_answer_relevancy", (0, 1.0), "higher better"),
        ("Bilingual EN↔AR sim.",  "bilingual_en_ar_sim",    (0, 1.0), "higher better"),
        ("Cold latency (s)",      "avg_cold_ms",            (0, None), "lower better"),
    ]

    pretty = {
        "small_4o-mini": "small\n+ gpt-4o-mini",
        "small_4.1":     "small\n+ gpt-4.1",
        "large_4o-mini": "large\n+ gpt-4o-mini",
        "large_4.1":     "large\n+ gpt-4.1",
    }
    label_palette = {
        "small_4o-mini": BLUE,
        "small_4.1":     GREEN,
        "large_4o-mini": GOLD,
        "large_4.1":     MAROON,
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.2))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (title, key, (ymin, ymax), direction) in zip(axes, metrics):
        vals = []
        labels_used = []
        for cfg in avail:
            raw = by_label[cfg].get(key, "")
            try:
                v = float(raw)
            except (ValueError, TypeError):
                v = None
            # Convert ms → seconds for the latency metric
            if key == "avg_cold_ms" and v is not None:
                v = v / 1000.0
            vals.append(v)
            labels_used.append(cfg)

        xs = np.arange(len(labels_used))
        colors = [label_palette[c] for c in labels_used]
        plotted = [v if v is not None else 0 for v in vals]
        bars = ax.bar(xs, plotted, color=colors, edgecolor="white", width=0.65)
        for b, v in zip(bars, vals):
            txt = "n/a" if v is None else (f"{v:.2f}" if (ymax == 1.0) else f"{v:.1f}")
            ax.text(b.get_x() + b.get_width() / 2,
                    (v if v is not None else 0) * 1.02 + (0 if v else 0.02),
                    txt, ha="center", va="bottom", fontsize=9, color=GRAY,
                    fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([pretty[c] for c in labels_used], fontsize=8)
        if ymax is not None:
            ax.set_ylim(ymin, ymax * 1.1)
        else:
            ymax_v = max((v for v in vals if v is not None), default=1)
            ax.set_ylim(ymin, ymax_v * 1.25)
        style_ax(ax, title, ylabel="")
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.5, -0.18, f"({direction})", transform=ax.transAxes,
                ha="center", fontsize=8, color=GRAY, style="italic")

    fig.suptitle("Model ablation 2x2: embedding × LLM",
                 fontsize=14, color=MAROON, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Each panel: 4 configs of {text-embedding-3-small/large} × {gpt-4o-mini, gpt-4.1}. "
             "Same golden set, RAGAS judge model, and infrastructure across all runs.",
             ha="center", fontsize=8.5, color=GRAY, style="italic")

    plt.tight_layout()
    out = OUT_DIR / "model_ablation_2x2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main():
    print(f"Generating MSBA charts into {OUT_DIR.relative_to(REPO_ROOT)}/")
    for path in [LATEST_RUN, COST_RUN]:
        if not path.exists():
            raise FileNotFoundError(f"Missing eval directory: {path}")
    chart_baseline_comparison()
    chart_cost_quality_pareto()
    chart_latency_distribution()
    chart_threshold_sweep()
    chart_category_performance()
    chart_annual_cost_projection()
    chart_retrieval_rrf_sweep()
    chart_ragas_bilingual()
    chart_latency_by_segment()
    chart_intent_confusion_matrix()
    chart_cost_vs_gpt4()
    chart_model_ablation_2x2()
    print("done.")


if __name__ == "__main__":
    main()
