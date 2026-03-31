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
            },
            "language": {
                "language_pie": ("Language Distribution", self.chart_language_pie),
                "language_trend": ("Language Usage Trend", self.chart_language_trend),
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
