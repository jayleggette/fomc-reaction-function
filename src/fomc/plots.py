"""Static figures for the README (matplotlib).

Visual rules: one y-axis per panel, thin 2px lines, hairline solid grid,
categorical colours in a fixed validated order, a legend whenever two or more
series share a panel, text in neutral ink (never in series colours), and a
CSV twin of every plotted series in outputs/tables.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3

NBER = [("1990-07-01", "1991-03-31"), ("2001-03-01", "2001-11-30"),
        ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30")]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _shade_recessions(ax, start=None, end=None) -> None:
    for a, b in NBER:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if (start is None or b >= start) and (end is None or a <= end):
            ax.axvspan(a, b, color=BAND, lw=0, zorder=0)


def _end_label(ax, s: pd.Series, text: str, dy: float = 0.0) -> None:
    s = s.dropna()
    ax.annotate(text, (s.index[-1], s.iloc[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2)


def _finish(fig, path: Path, source: str) -> None:
    fig.text(0.01, -0.01, source, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def rules_vs_actual(df: pd.DataFrame, path: Path) -> None:
    """Top: actual rate vs rules. Bottom: actual minus Taylor (1993)."""
    style()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
    for a in (ax, bx):
        _shade_recessions(a, df.index[0], df.index[-1])
        a.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.plot(df.index, df["ffr"], color=S1, label="Effective fed funds rate", zorder=3)
    ax.plot(df.index, df["taylor_1993"], color=S2, label="Taylor (1993) rule, r* = 2", zorder=2)
    ax.plot(df.index, df["balanced_approach"], color=S3, label="Balanced-approach rule, r* = 2", zorder=2)
    lo, hi = -8.0, 12.0
    ax.set_ylim(lo, hi)
    off_scale = df["balanced_approach"].min()
    _end_label(ax, df["ffr"], f"{df['ffr'].dropna().iloc[-1]:.2f}%", dy=-9)
    _end_label(ax, df["taylor_1993"], f"{df['taylor_1993'].dropna().iloc[-1]:.1f}%", dy=7)
    ax.set_ylabel("Percent")
    ax.set_title("The Fed against two textbook policy rules, 1990 to 2026")
    ax.legend(loc="lower left", ncol=1)
    gap = df["ffr"] - df["taylor_1993"]
    bx.fill_between(gap.index, 0, gap, color=S2, alpha=0.10, lw=0)
    bx.plot(gap.index, gap, color=S2)
    bx.set_ylabel("Actual minus rule, pp")
    for when, label, y in [("2004-06-01", "2003-05: below rule", -4.6),
                           ("2021-10-01", "2021-22: far below rule", -6.6)]:
        bx.annotate(label, (pd.Timestamp(when), y), fontsize=8, color=INK2, ha="center")
    bx.xaxis.set_major_locator(mdates.YearLocator(4))
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _finish(fig, path, "Monthly. Inflation = core PCE, 12-month. Slack = unemployment minus CBO noncyclical rate. "
                       f"Both rules leave the scale in 2020 (balanced approach reaches {off_scale:.0f}%).\n"
                       "Shaded: NBER recessions. Source: FRED; author's calculations.")


def rolling_coefficients(roll: pd.DataFrame, path: Path) -> None:
    style()
    roll = roll.reindex(pd.date_range(roll.index.min(), roll.index.max(), freq="MS"))  # gaps -> NaN, line breaks
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(roll.index, roll["beta_pi_lo"], roll["beta_pi_hi"], color=S1, alpha=0.10, lw=0)
    ax.plot(roll.index, roll["beta_pi"], color=S1)
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.annotate("Taylor principle: 1.0", (roll.index[2], 1.0), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK2)
    gap = roll["beta_pi"].isna()
    if gap.any():
        mid = roll.index[gap][len(roll.index[gap]) // 2]
        ax.annotate("Too few months off the lower bound\nto estimate (2012 to 2023)", (mid, 2.4),
                    ha="center", fontsize=8, color=INK2)
    ax.set_ylabel("Response of policy rate to inflation")
    ax.set_title("Inflation response, rolling 10-year windows (95% band)")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _finish(fig, path, "OLS of the effective rate on core PCE inflation and the unemployment gap; months at the "
                       "lower bound excluded; Newey-West bands. Windows with fewer than 72 usable months are skipped.")


def oos_probabilities(probs: pd.DataFrame, y: pd.Series, at_elb: pd.Series, meeting: pd.Series, path: Path) -> None:
    """Forecasts for months followed by a scheduled meeting, with actual moves as a rug."""
    from matplotlib.lines import Line2D

    style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"hspace": 0.22})
    on = meeting.reindex(probs.index).fillna(0).astype(bool)
    pm = probs[on]
    for ax, cls, color, val, title in [
        (axes[0], "hike", S2, 1.0, "Probability of a hike at the next scheduled meeting"),
        (axes[1], "cut", S1, -1.0, "Probability of a cut at the next scheduled meeting"),
    ]:
        elb = at_elb.astype(bool)
        for _, grp in elb[elb].groupby((~elb).cumsum()):
            ax.axvspan(grp.index[0], grp.index[-1] + pd.offsets.MonthEnd(1), color=BAND, lw=0, zorder=0)
        ax.fill_between(pm.index, 0, pm[cls], color=color, alpha=0.10, lw=0)
        ax.plot(pm.index, pm[cls], color=color, lw=1.4)
        hits = y[y == val].index
        ax.vlines(hits, 1.03, 1.09, color=INK2, lw=1.1, clip_on=False, zorder=4)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_title(title)
        key = [Line2D([0], [0], color=color, lw=1.6, label=f"Forecast P({cls})"),
               Line2D([0], [0], color=INK2, lw=0, marker="|", markersize=9, markeredgewidth=1.4,
                      label=f"Month followed by an actual {cls}")]
        ax.legend(handles=key, loc="upper left", bbox_to_anchor=(0.36, 0.95))
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _finish(fig, path, "Full model, expanding-window out-of-sample forecasts made at each month end, shown for months "
                       "followed by a scheduled FOMC meeting. Shaded: policy rate at the lower bound.")


def skill_bars(scores: pd.DataFrame, path: Path) -> None:
    """Brier skill vs climatology by model (off-ELB sample)."""
    style()
    s = scores[scores["model"] != "Climatology"].set_index("model")["brier_skill_vs_climatology"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 2.9))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ypos = np.arange(len(s))
    ax.barh(ypos, s.values * 100, height=0.42, color=S1)
    for yv, v in zip(ypos, s.values):
        ax.annotate(f"{v * 100:.0f}%", (v * 100, yv), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(s.index)
    ax.set_xlabel("Brier skill vs calendar-aware climatology, %")
    ax.set_xlim(0, max(s.values * 100) * 1.18)
    ax.set_title("Out-of-sample skill, months off the lower bound, 2000 to 2026")
    _finish(fig, path, "Higher is better. 0% = no better than historical move frequencies.")


def calibration(rel_hike: pd.DataFrame, rel_cut: pd.DataFrame, path: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.grid(axis="x", visible=True)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, zorder=1)
    for rel, color, label in [(rel_hike, S2, "Hike"), (rel_cut, S1, "Cut")]:
        ax.plot(rel["mean_forecast"], rel["realised"], color=color, zorder=2)
        ax.scatter(rel["mean_forecast"], rel["realised"], s=36, color=color, edgecolor=SURFACE, lw=1.6,
                   zorder=3, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Realised frequency")
    ax.set_title("Calibration, full model")
    ax.legend(loc="upper left")
    _finish(fig, path, "Bins: 0-10, 10-30, 30-50, 50-70, 70-100%. Diagonal = perfect calibration.")


def nowcast_bar(nc: dict[str, float], title: str, path: Path) -> None:
    """Tall segmented bar of cut / hold / hike probabilities."""
    style()
    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.axis("off")
    colors = {"cut": S1, "hold": MUTED, "hike": S2}
    left = 0.0
    for k in ("cut", "hold", "hike"):
        w = nc[k]
        if w <= 0:
            continue
        ax.barh(0, w - 0.004 if w > 0.01 else w, left=left, height=0.5, color=colors[k])
        if w >= 0.08:
            ax.text(left + 0.012, 0, f"{k.capitalize()} {w * 100:.0f}%", va="center", ha="left",
                    fontsize=10, color="#ffffff", fontweight="semibold")
        left += w
    small = [f"{k} {nc[k] * 100:.1f}%" for k in ("cut", "hold", "hike") if 0 < nc[k] < 0.08]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(title, loc="left")
    if small:
        ax.text(1.0, -0.48, "Not labelled in bar: " + ", ".join(small), ha="right", fontsize=8, color=INK2)
    _finish(fig, path, "Full model fitted on all months with known outcomes; inputs as of the latest data.")
