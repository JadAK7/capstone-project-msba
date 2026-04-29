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
]

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
    """Long-form CSV: one row per (key, metric, timepoint)."""
    all_keys: set[tuple[str, str]] = set()
    for _, sc in scorecards:
        all_keys.update(sc.keys())

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "metric", "T1", "T2", "T3", "T1_to_T3_delta"])
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
    colors = [LIGHT_GRAY, RED, MAROON]
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("Iteration improvements across three eval rounds",
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
    """Spider chart comparing T1 vs T3 across multiple normalized axes."""
    t1 = scorecards[0][1]
    t3 = scorecards[-1][1]

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
            label="T1 (before fixes)")
    ax.fill(angles_closed, t1_closed, color=LIGHT_GRAY, alpha=0.25)
    ax.plot(angles_closed, t3_closed, color=MAROON, linewidth=2,
            label="T3 (post changes)")
    ax.fill(angles_closed, t3_closed, color=MAROON, alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color=GRAY, fontsize=8)
    ax.set_title("Pipeline maturity: T1 baseline vs T3 post-changes",
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

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.25
    xs = np.arange(len(metric_specs))
    colors = [LIGHT_GRAY, RED, MAROON]

    for i, (label, sc) in enumerate(zip(labels, cards)):
        vals = []
        for _, key, _ in metric_specs:
            v = sc.get(key)
            vals.append(v if v is not None else 0)
        bars = ax.bar(xs + (i - 1) * width, vals, width=width,
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
    ax.set_title("Headline metric trajectory across three eval rounds",
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

    chart_iteration_improvements(scorecards)
    chart_safety_latency_tradeoff(scorecards)
    chart_pipeline_maturity_radar(scorecards)
    chart_timeline_bars(scorecards)
    print("done.")


if __name__ == "__main__":
    main()
