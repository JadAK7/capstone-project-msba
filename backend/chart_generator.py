"""
chart_generator.py
Server-side chart generation using matplotlib.
Generates base64-encoded PNG images for the admin analytics dashboard.
"""

import io
import base64
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

logger = logging.getLogger(__name__)

# AUB Brand Colors
MAROON = "#840132"
MAROON_LIGHT = "#f9f5f6"
RED = "#ee3524"
GRAY = "#424242"
LIGHT_GRAY = "#d1d2d2"
BG_WHITE = "#ffffff"
PALETTE = [MAROON, RED, "#b5651d", "#2d6a4f", "#457b9d", GRAY]


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG_WHITE)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", color=GRAY, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=GRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT_GRAY)
    ax.spines["bottom"].set_color(LIGHT_GRAY)


def _parse_date(ts: str):
    return ts[:10] if len(ts) >= 10 else None


def _iso_week(ts: str):
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z"))
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except (ValueError, AttributeError):
        return None


class ChartGenerator:
    """Generates matplotlib charts as base64 PNG strings."""

    def __init__(self, entries: list):
        self.entries = entries
        self._now = datetime.utcnow()

    # ===================================================================
    # A. Usage & Volume
    # ===================================================================

    def chart_daily_volume(self) -> str:
        daily = Counter()
        for e in self.entries:
            d = _parse_date(e.get("timestamp", ""))
            if d:
                daily[d] += 1

        days = 30
        dates, counts = [], []
        for i in range(days, -1, -1):
            d = (self._now - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(datetime.strptime(d, "%Y-%m-%d"))
            counts.append(daily.get(d, 0))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, counts, alpha=0.3, color=MAROON)
        ax.plot(dates, counts, color=MAROON, linewidth=2, marker="o", markersize=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        fig.autofmt_xdate(rotation=45)
        _style_ax(ax, "Daily Conversation Volume (Last 30 Days)", ylabel="Conversations")
        ax.set_ylim(bottom=0)
        return _fig_to_base64(fig)

    def chart_hourly_heatmap(self) -> str:
        matrix = np.zeros((7, 24))
        for e in self.entries:
            dow = e.get("day_of_week", -1)
            hour = e.get("hour_of_day", -1)
            if dow == -1 or hour == -1:
                ts = e.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts.rstrip("Z"))
                    dow = dt.weekday()
                    hour = dt.hour
                except (ValueError, AttributeError):
                    continue
            if 0 <= dow <= 6 and 0 <= hour <= 23:
                matrix[dow][hour] += 1

        fig, ax = plt.subplots(figsize=(12, 4))
        cmap = LinearSegmentedColormap.from_list("aub", [BG_WHITE, MAROON_LIGHT, RED, MAROON])
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h}" for h in range(24)], fontsize=7)
        fig.colorbar(im, ax=ax, label="Conversations", shrink=0.8)
        _style_ax(ax, "Usage Heatmap (Day of Week x Hour of Day)")
        return _fig_to_base64(fig)

    def chart_weekly_volume(self) -> str:
        weekly = Counter()
        for e in self.entries:
            w = _iso_week(e.get("timestamp", ""))
            if w:
                weekly[w] += 1

        weeks_sorted = sorted(weekly.keys())[-12:]
        counts = [weekly[w] for w in weeks_sorted]
        labels = [w.split("-")[1] for w in weeks_sorted]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(labels, counts, color=MAROON, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(counts):
            if v > 0:
                ax.text(i, v + 0.3, str(v), ha="center", fontsize=8, color=GRAY)
        _style_ax(ax, "Weekly Conversation Volume (Last 12 Weeks)", xlabel="ISO Week", ylabel="Conversations")
        ax.set_ylim(bottom=0)
        return _fig_to_base64(fig)

    def chart_cumulative_growth(self) -> str:
        daily = Counter()
        for e in self.entries:
            d = _parse_date(e.get("timestamp", ""))
            if d:
                daily[d] += 1

        if not daily:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=GRAY)
            _style_ax(ax, "Cumulative Usage Growth")
            return _fig_to_base64(fig)

        all_dates = sorted(daily.keys())
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates]
        cumulative = []
        total = 0
        for d in all_dates:
            total += daily[d]
            cumulative.append(total)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, cumulative, alpha=0.2, color=MAROON)
        ax.plot(dates, cumulative, color=MAROON, linewidth=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.autofmt_xdate(rotation=45)
        _style_ax(ax, "Cumulative Usage Growth", ylabel="Total Conversations")
        ax.set_ylim(bottom=0)
        return _fig_to_base64(fig)

    # ===================================================================
    # B. Language
    # ===================================================================

    def chart_language_pie(self) -> str:
        lang_counter = Counter(e.get("language", "en") for e in self.entries)
        labels = [{"en": "English", "ar": "Arabic"}.get(k, k) for k in lang_counter.keys()]
        sizes = list(lang_counter.values())
        colors = [MAROON, RED] + list(PALETTE[2:])

        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[: len(sizes)],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white"),
            textprops={"fontsize": 10, "color": GRAY},
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")
        _style_ax(ax, "Language Distribution")
        ax.set_ylabel("")
        return _fig_to_base64(fig)

    def chart_language_trend(self) -> str:
        daily_lang = defaultdict(lambda: Counter())
        for e in self.entries:
            d = _parse_date(e.get("timestamp", ""))
            if d:
                daily_lang[d][e.get("language", "en")] += 1

        days = 30
        dates = []
        en_counts = []
        ar_counts = []
        for i in range(days, -1, -1):
            d = (self._now - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(datetime.strptime(d, "%Y-%m-%d"))
            en_counts.append(daily_lang[d].get("en", 0))
            ar_counts.append(daily_lang[d].get("ar", 0))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.stackplot(dates, en_counts, ar_counts, labels=["English", "Arabic"],
                      colors=[MAROON, RED], alpha=0.7)
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        fig.autofmt_xdate(rotation=45)
        _style_ax(ax, "Language Usage Trend (Last 30 Days)", ylabel="Conversations")
        ax.set_ylim(bottom=0)
        return _fig_to_base64(fig)

    def chart_cross_lingual_scores(self) -> str:
        scores = {"en": {"faq": [], "db": [], "lib": []}, "ar": {"faq": [], "db": [], "lib": []}}
        for e in self.entries:
            lang = e.get("language", "en")
            if lang not in scores:
                lang = "en"
            scores[lang]["faq"].append(e.get("faq_top_score", 0))
            scores[lang]["db"].append(e.get("db_top_score", 0))
            scores[lang]["lib"].append(e.get("library_top_score", 0))

        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0

        categories = ["FAQ", "Database", "Library"]
        en_vals = [_avg(scores["en"]["faq"]), _avg(scores["en"]["db"]), _avg(scores["en"]["lib"])]
        ar_vals = [_avg(scores["ar"]["faq"]), _avg(scores["ar"]["db"]), _avg(scores["ar"]["lib"])]

        x = np.arange(len(categories))
        w = 0.35

        fig, ax = plt.subplots(figsize=(8, 4))
        bars1 = ax.bar(x - w / 2, en_vals, w, label="English", color=MAROON)
        bars2 = ax.bar(x + w / 2, ar_vals, w, label="Arabic", color=RED)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8, color=GRAY)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=8, color=GRAY)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=9)
        _style_ax(ax, "Cross-Lingual Retrieval Score Comparison", ylabel="Avg Similarity Score")
        ax.set_ylim(0, 1)
        return _fig_to_base64(fig)

    # ===================================================================
    # C. Retrieval Quality
    # ===================================================================

    def chart_score_distributions(self) -> str:
        faq = [e.get("faq_top_score", 0) for e in self.entries]
        db = [e.get("db_top_score", 0) for e in self.entries]
        lib = [e.get("library_top_score", 0) for e in self.entries if e.get("library_top_score", 0) > 0]

        fig, ax = plt.subplots(figsize=(10, 4))
        bins = np.linspace(0, 1, 31)
        if faq:
            ax.hist(faq, bins=bins, alpha=0.5, label="FAQ", color=MAROON)
        if db:
            ax.hist(db, bins=bins, alpha=0.5, label="Database", color=RED)
        if lib:
            ax.hist(lib, bins=bins, alpha=0.5, label="Library", color=GRAY)

        ax.axvline(x=0.60, color=MAROON, linestyle="--", alpha=0.7, linewidth=1, label="FAQ threshold (0.60)")
        ax.axvline(x=0.45, color=RED, linestyle="--", alpha=0.7, linewidth=1, label="DB threshold (0.45)")
        ax.axvline(x=0.35, color=GRAY, linestyle="--", alpha=0.7, linewidth=1, label="Library threshold (0.35)")

        ax.legend(fontsize=7, loc="upper right")
        _style_ax(ax, "Confidence Score Distributions", xlabel="Similarity Score", ylabel="Frequency")
        return _fig_to_base64(fig)

    def chart_score_boxplots(self) -> str:
        faq = [e.get("faq_top_score", 0) for e in self.entries]
        db = [e.get("db_top_score", 0) for e in self.entries]
        lib = [e.get("library_top_score", 0) for e in self.entries if e.get("library_top_score", 0) > 0]

        data = [faq, db]
        labels = ["FAQ", "Database"]
        if lib:
            data.append(lib)
            labels.append("Library")

        fig, ax = plt.subplots(figsize=(8, 4))
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color=RED, linewidth=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        colors_box = [MAROON, RED, GRAY]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        _style_ax(ax, "Retrieval Score Box Plots by Source", ylabel="Similarity Score")
        ax.set_ylim(0, 1)
        return _fig_to_base64(fig)

    def chart_scores_over_time(self) -> str:
        weekly = defaultdict(lambda: {"faq": [], "db": [], "lib": []})
        for e in self.entries:
            w = _iso_week(e.get("timestamp", ""))
            if w:
                weekly[w]["faq"].append(e.get("faq_top_score", 0))
                weekly[w]["db"].append(e.get("db_top_score", 0))
                ls = e.get("library_top_score", 0)
                if ls > 0:
                    weekly[w]["lib"].append(ls)

        weeks = sorted(weekly.keys())[-12:]
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0

        faq_avgs = [_avg(weekly[w]["faq"]) for w in weeks]
        db_avgs = [_avg(weekly[w]["db"]) for w in weeks]
        lib_avgs = [_avg(weekly[w]["lib"]) for w in weeks]
        labels = [w.split("-")[1] for w in weeks]

        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(weeks))
        ax.plot(x, faq_avgs, color=MAROON, marker="o", markersize=4, linewidth=2, label="FAQ")
        ax.plot(x, db_avgs, color=RED, marker="s", markersize=4, linewidth=2, label="Database")
        if any(v > 0 for v in lib_avgs):
            ax.plot(x, lib_avgs, color=GRAY, marker="^", markersize=4, linewidth=2, label="Library")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend(fontsize=9)
        _style_ax(ax, "Average Retrieval Scores Over Time", xlabel="ISO Week", ylabel="Avg Score")
        ax.set_ylim(0, 1)
        return _fig_to_base64(fig)

    def chart_score_correlation(self) -> str:
        faq = [e.get("faq_top_score", 0) for e in self.entries]
        db = [e.get("db_top_score", 0) for e in self.entries]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(faq, db, alpha=0.5, color=MAROON, s=20, edgecolors="none")
        ax.axvline(x=0.60, color=MAROON, linestyle="--", alpha=0.4, linewidth=1)
        ax.axhline(y=0.45, color=RED, linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _style_ax(ax, "FAQ vs Database Score Correlation", xlabel="FAQ Score", ylabel="Database Score")
        return _fig_to_base64(fig)

    # ===================================================================
    # D. Intent & Routing
    # ===================================================================

    def chart_intent_pie(self) -> str:
        intent_counter = Counter(e.get("intent_source", "unknown") for e in self.entries)
        label_map = {
            "FAQ": "FAQ",
            "database (keyword intent)": "DB (keyword)",
            "database (semantic)": "DB (semantic)",
            "library pages (scraped)": "Library Pages",
            "none (unclear)": "Unanswered",
        }
        labels = [label_map.get(k, k) for k in intent_counter.keys()]
        sizes = list(intent_counter.values())
        colors = PALETTE[: len(sizes)]

        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white"),
            textprops={"fontsize": 9, "color": GRAY},
        )
        for t in autotexts:
            t.set_fontsize(8)
            t.set_fontweight("bold")
        _style_ax(ax, "Intent Source Distribution")
        ax.set_ylabel("")
        return _fig_to_base64(fig)

    def chart_intent_trend(self) -> str:
        weekly_intent = defaultdict(Counter)
        for e in self.entries:
            w = _iso_week(e.get("timestamp", ""))
            if w:
                src = e.get("intent_source", "unknown")
                weekly_intent[w][src] += 1

        weeks = sorted(weekly_intent.keys())[-12:]
        all_sources = set()
        for w in weeks:
            all_sources.update(weekly_intent[w].keys())
        sources_list = sorted(all_sources)

        x = np.arange(len(weeks))
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(weeks))

        label_map = {
            "FAQ": "FAQ",
            "database (keyword intent)": "DB (keyword)",
            "database (semantic)": "DB (semantic)",
            "library pages (scraped)": "Library Pages",
            "none (unclear)": "Unanswered",
        }

        for i, src in enumerate(sources_list):
            vals = [weekly_intent[w].get(src, 0) for w in weeks]
            label = label_map.get(src, src)
            ax.bar(x, vals, bottom=bottom, label=label,
                   color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([w.split("-")[1] for w in weeks])
        ax.legend(fontsize=7, loc="upper left")
        _style_ax(ax, "Intent Routing Trend (Last 12 Weeks)", xlabel="ISO Week", ylabel="Conversations")
        return _fig_to_base64(fig)

    def chart_keyword_vs_semantic(self) -> str:
        keyword_count = sum(1 for e in self.entries if e.get("keyword_intent_fired", False))
        semantic_count = len(self.entries) - keyword_count

        db_keyword = sum(1 for e in self.entries
                         if e.get("intent_source") == "database (keyword intent)")
        db_semantic = sum(1 for e in self.entries
                          if e.get("intent_source") == "database (semantic)")

        categories = ["Keyword Intent Fired", "DB Routed (keyword)", "DB Routed (semantic)"]
        values = [keyword_count, db_keyword, db_semantic]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(categories, values, color=[MAROON, RED, GRAY])
        for bar in bars:
            w = bar.get_width()
            if w > 0:
                ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                        str(int(w)), va="center", fontsize=9, color=GRAY)
        _style_ax(ax, "Keyword vs Semantic Intent Detection", xlabel="Count")
        return _fig_to_base64(fig)

    # ===================================================================
    # E. Query Analysis
    # ===================================================================

    def chart_top_faq_matches(self) -> str:
        counter = Counter()
        for e in self.entries:
            q = e.get("top_faq_question", "")
            if q:
                q_short = q[:60] + "..." if len(q) > 60 else q
                counter[q_short] += 1

        top = counter.most_common(15)
        if not top:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=GRAY)
            _style_ax(ax, "Top 15 Matched FAQ Questions")
            return _fig_to_base64(fig)

        labels = [t[0] for t in reversed(top)]
        values = [t[1] for t in reversed(top)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(labels)), values, color=MAROON)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        _style_ax(ax, "Top 15 Matched FAQ Questions", xlabel="Match Count")
        fig.tight_layout()
        return _fig_to_base64(fig)

    def chart_top_db_matches(self) -> str:
        counter = Counter()
        for e in self.entries:
            name = e.get("top_db_name", "")
            if name:
                counter[name] += 1

        top = counter.most_common(15)
        if not top:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color=GRAY)
            _style_ax(ax, "Top 15 Matched Databases")
            return _fig_to_base64(fig)

        labels = [t[0][:40] for t in reversed(top)]
        values = [t[1] for t in reversed(top)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(labels)), values, color=RED)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        _style_ax(ax, "Top 15 Matched Databases", xlabel="Match Count")
        fig.tight_layout()
        return _fig_to_base64(fig)

    def chart_query_length_dist(self) -> str:
        wc = [e.get("query_word_count", len(e.get("query", "").split())) for e in self.entries]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(wc, bins=range(1, max(wc or [10]) + 2), color=MAROON, edgecolor="white", alpha=0.8)
        if wc:
            avg = sum(wc) / len(wc)
            ax.axvline(x=avg, color=RED, linestyle="--", linewidth=1.5, label=f"Mean ({avg:.1f})")
            ax.legend(fontsize=9)
        _style_ax(ax, "Query Length Distribution", xlabel="Word Count", ylabel="Frequency")
        return _fig_to_base64(fig)

    # ===================================================================
    # F. Performance
    # ===================================================================

    def chart_response_time_dist(self) -> str:
        times = [e.get("response_time_ms", 0) for e in self.entries
                 if e.get("response_time_ms", 0) > 0 and not e.get("cache_hit", False)]

        if not times:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No response time data yet", ha="center", va="center",
                    transform=ax.transAxes, color=GRAY)
            _style_ax(ax, "Response Time Distribution (excl. cache hits)")
            return _fig_to_base64(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(times, bins=30, color=MAROON, edgecolor="white", alpha=0.8)
        median = sorted(times)[len(times) // 2]
        p95 = sorted(times)[min(int(len(times) * 0.95), len(times) - 1)]
        ax.axvline(x=median, color=RED, linestyle="--", linewidth=1.5, label=f"Median ({median:.0f}ms)")
        ax.axvline(x=p95, color=GRAY, linestyle="--", linewidth=1.5, label=f"P95 ({p95:.0f}ms)")
        ax.legend(fontsize=9)
        _style_ax(ax, "Response Time Distribution (excl. cache hits)", xlabel="Time (ms)", ylabel="Frequency")
        return _fig_to_base64(fig)

    def chart_response_time_trend(self) -> str:
        daily_times = defaultdict(list)
        for e in self.entries:
            rt = e.get("response_time_ms", 0)
            if rt > 0 and not e.get("cache_hit", False):
                d = _parse_date(e.get("timestamp", ""))
                if d:
                    daily_times[d].append(rt)

        if not daily_times:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No response time data yet", ha="center", va="center",
                    transform=ax.transAxes, color=GRAY)
            _style_ax(ax, "Response Time Trend")
            return _fig_to_base64(fig)

        days_sorted = sorted(daily_times.keys())[-30:]
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in days_sorted]
        medians = [sorted(daily_times[d])[len(daily_times[d]) // 2] for d in days_sorted]
        p25 = [sorted(daily_times[d])[max(0, int(len(daily_times[d]) * 0.25))] for d in days_sorted]
        p75 = [sorted(daily_times[d])[min(int(len(daily_times[d]) * 0.75), len(daily_times[d]) - 1)]
               for d in days_sorted]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, p25, p75, alpha=0.2, color=MAROON, label="P25-P75")
        ax.plot(dates, medians, color=MAROON, linewidth=2, marker="o", markersize=3, label="Median")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.autofmt_xdate(rotation=45)
        ax.legend(fontsize=9)
        _style_ax(ax, "Response Time Trend (Last 30 Days)", ylabel="Time (ms)")
        return _fig_to_base64(fig)

    def chart_cache_hit_rate(self) -> str:
        daily = defaultdict(lambda: {"hits": 0, "total": 0})
        for e in self.entries:
            d = _parse_date(e.get("timestamp", ""))
            if d:
                daily[d]["total"] += 1
                if e.get("cache_hit", False):
                    daily[d]["hits"] += 1

        days_sorted = sorted(daily.keys())[-30:]
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in days_sorted]
        rates = [daily[d]["hits"] / daily[d]["total"] * 100 if daily[d]["total"] > 0 else 0
                 for d in days_sorted]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, rates, alpha=0.3, color=MAROON)
        ax.plot(dates, rates, color=MAROON, linewidth=2, marker="o", markersize=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.autofmt_xdate(rotation=45)
        ax.set_ylim(0, 100)
        _style_ax(ax, "Cache Hit Rate Over Time", ylabel="Hit Rate (%)")
        return _fig_to_base64(fig)

    # ===================================================================
    # G. Knowledge Gaps
    # ===================================================================

    def chart_unanswered_trend(self) -> str:
        weekly_total = Counter()
        weekly_unanswered = Counter()
        for e in self.entries:
            w = _iso_week(e.get("timestamp", ""))
            if w:
                weekly_total[w] += 1
                if e.get("intent_source") == "none (unclear)":
                    weekly_unanswered[w] += 1

        weeks = sorted(weekly_total.keys())[-12:]
        rates = [
            weekly_unanswered.get(w, 0) / weekly_total[w] * 100 if weekly_total[w] > 0 else 0
            for w in weeks
        ]
        labels = [w.split("-")[1] for w in weeks]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(labels, rates, color=RED, linewidth=2, marker="o", markersize=5)
        ax.fill_between(range(len(labels)), rates, alpha=0.15, color=RED)
        _style_ax(ax, "Unanswered Rate Trend (Last 12 Weeks)", xlabel="ISO Week", ylabel="Unanswered %")
        ax.set_ylim(0, max(rates or [10]) * 1.3)
        return _fig_to_base64(fig)

    def chart_near_miss_scores(self) -> str:
        unclear = [e for e in self.entries if e.get("intent_source") == "none (unclear)"]
        if not unclear:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No unanswered queries", ha="center", va="center",
                    transform=ax.transAxes, color=GRAY)
            _style_ax(ax, "Near-Miss Score Distribution (Unanswered Queries)")
            return _fig_to_base64(fig)

        max_scores = [
            max(e.get("faq_top_score", 0), e.get("db_top_score", 0), e.get("library_top_score", 0))
            for e in unclear
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(max_scores, bins=20, color=RED, edgecolor="white", alpha=0.8, range=(0, 1))

        threshold = 0.35
        ax.axvline(x=threshold, color=MAROON, linestyle="--", linewidth=1.5,
                   label=f"Lowest threshold ({threshold})")

        near_miss_count = sum(1 for s in max_scores if s >= threshold - 0.05)
        pct = near_miss_count / len(max_scores) * 100 if max_scores else 0
        ax.annotate(
            f"{pct:.0f}% within 5% of threshold",
            xy=(threshold, 0), xytext=(threshold + 0.15, ax.get_ylim()[1] * 0.7),
            fontsize=9, color=MAROON, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=MAROON),
        )

        ax.legend(fontsize=9)
        _style_ax(ax, "Near-Miss Score Distribution (Unanswered Queries)",
                  xlabel="Best Score Across All Sources", ylabel="Frequency")
        return _fig_to_base64(fig)

    # ===================================================================
    # Generate All
    # ===================================================================

    def generate_all(self) -> dict:
        if not self.entries:
            return {}

        generators = {
            "usage_volume": {
                "daily_volume": ("Daily Conversation Volume", self.chart_daily_volume),
                "hourly_heatmap": ("Usage Heatmap by Day & Hour", self.chart_hourly_heatmap),
                "weekly_volume": ("Weekly Conversation Volume", self.chart_weekly_volume),
                "cumulative_growth": ("Cumulative Usage Growth", self.chart_cumulative_growth),
            },
            "language": {
                "language_pie": ("Language Distribution", self.chart_language_pie),
                "language_trend": ("Language Usage Trend", self.chart_language_trend),
                "cross_lingual_scores": ("Cross-Lingual Score Comparison", self.chart_cross_lingual_scores),
            },
            "retrieval_quality": {
                "score_distributions": ("Confidence Score Distributions", self.chart_score_distributions),
                "score_boxplots": ("Score Box Plots by Source", self.chart_score_boxplots),
                "scores_over_time": ("Average Scores Over Time", self.chart_scores_over_time),
                "score_correlation": ("FAQ vs DB Score Correlation", self.chart_score_correlation),
            },
            "intent_routing": {
                "intent_pie": ("Intent Source Distribution", self.chart_intent_pie),
                "intent_trend": ("Intent Routing Trend", self.chart_intent_trend),
                "keyword_vs_semantic": ("Keyword vs Semantic Intent", self.chart_keyword_vs_semantic),
            },
            "query_analysis": {
                "top_faq_matches": ("Top Matched FAQ Questions", self.chart_top_faq_matches),
                "top_db_matches": ("Top Matched Databases", self.chart_top_db_matches),
                "query_length_dist": ("Query Length Distribution", self.chart_query_length_dist),
            },
            "performance": {
                "response_time_dist": ("Response Time Distribution", self.chart_response_time_dist),
                "response_time_trend": ("Response Time Trend", self.chart_response_time_trend),
                "cache_hit_rate": ("Cache Hit Rate Over Time", self.chart_cache_hit_rate),
            },
            "knowledge_gaps": {
                "unanswered_trend": ("Unanswered Rate Trend", self.chart_unanswered_trend),
                "near_miss_scores": ("Near-Miss Score Distribution", self.chart_near_miss_scores),
            },
        }

        result = {}
        for category, charts in generators.items():
            result[category] = {}
            for key, (title, func) in charts.items():
                try:
                    result[category][key] = {"title": title, "image": func()}
                except Exception as e:
                    logger.error(f"Chart generation failed for {category}/{key}: {e}")
                    result[category][key] = {"title": title, "image": None}
        return result
