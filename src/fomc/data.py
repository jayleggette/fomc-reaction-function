"""Load FRED data and build the monthly panel used throughout the project.

Timing convention
-----------------
A row stamped month t describes the world at the END of month t.

* Market data (Treasury yields) and the policy rate are known in real time,
  so they enter at month t.
* Macro data (inflation, unemployment, payrolls) for month t are published
  during month t+1. Every macro FEATURE is therefore lagged one month, so a
  forecast made at the end of month t only uses numbers that had been
  released by then.

Descriptive work (policy rules, reaction-function estimates) uses the
contemporaneous, final-vintage values. That is standard in the literature and
the README is explicit about it. The forecasting model uses only the lagged
features built in :func:`build_features`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

#: Target-range upper bound at the effective lower bound (ELB).
ELB_UPPER = 0.25
#: Effective fed funds rate at or below this level counts as "at the ELB".
ELB_EFFECTIVE = 0.25
#: Inflation target (PCE, percent).
PI_STAR = 2.0
#: Months of publication delay applied to macro features.
MACRO_LAG = 1


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_fred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # FRED marks missing values as "." in older exports and blanks in newer ones.
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


def load_monthly(path: Path | None = None) -> pd.DataFrame:
    return _read_fred_csv(path or RAW / "fred_monthly.csv")


def load_quarterly(path: Path | None = None) -> pd.DataFrame:
    return _read_fred_csv(path or RAW / "fred_quarterly.csv")


def load_daily(path: Path | None = None) -> pd.DataFrame | None:
    """Daily target rate and Treasury yields. Returns None if the file is absent."""
    p = path or RAW / "fred_daily.csv"
    return _read_fred_csv(p) if p.exists() else None


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------
def fill_isolated_gaps(s: pd.Series, log: bool = False) -> pd.Series:
    """Interpolate single missing months that sit between two valid months.

    The October 2025 federal shutdown stopped BLS data collection, so FRED has
    no October 2025 CPI or unemployment rate. Without a fill, every
    year-over-year rate that needs October 2025 (including October 2026)
    would be missing. Only isolated one-month holes are filled; longer gaps
    and the ends of the series are left untouched.
    """
    s = s.astype(float)
    hole = s.isna() & s.shift(1).notna() & s.shift(-1).notna()
    if not hole.any():
        return s
    x = np.log(s) if log else s
    mid = (x.shift(1) + x.shift(-1)) / 2.0
    filled = x.where(~hole, mid)
    return np.exp(filled) if log else filled


def quarterly_to_monthly(q: pd.Series) -> pd.Series:
    """Linear interpolation of a quarterly series to months.

    Quarterly values are stamped on the middle month of their quarter before
    interpolating, so the monthly path is centred correctly.
    """
    q = q.dropna().copy()
    q.index = q.index + pd.DateOffset(months=1)
    m = q.resample("MS").asfreq().interpolate("linear")
    return m


def policy_rate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily policy target.

    Before 16 Dec 2008 the FOMC set a point target (DFEDTAR). Since then it
    sets a range (DFEDTARL to DFEDTARU). Returns the upper bound, lower bound
    and midpoint as one continuous daily series.
    """
    upper = daily["DFEDTARU"].combine_first(daily["DFEDTAR"])
    lower = daily["DFEDTARL"].combine_first(daily["DFEDTAR"])
    out = pd.DataFrame({"upper": upper, "lower": lower}).dropna(how="all").ffill()
    out["mid"] = (out["upper"] + out["lower"]) / 2.0
    return out.dropna()


def yoy(s: pd.Series) -> pd.Series:
    return 100.0 * (s / s.shift(12) - 1.0)


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
def build_panel(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Monthly panel of contemporaneous (final-vintage) values."""
    m = monthly.copy()
    for col in ("CPIAUCSL", "CPILFESL", "PCEPILFE", "PAYEMS"):
        m[col] = fill_isolated_gaps(m[col], log=True)
    m["UNRATE"] = fill_isolated_gaps(m["UNRATE"], log=False)

    p = pd.DataFrame(index=m.index)
    p["ffr"] = m["FEDFUNDS"]
    p["cpi_yoy"] = yoy(m["CPIAUCSL"])
    p["core_cpi_yoy"] = yoy(m["CPILFESL"])
    p["core_pce_yoy"] = yoy(m["PCEPILFE"])
    p["u"] = m["UNRATE"]
    p["nrou"] = quarterly_to_monthly(quarterly["NROU"]).reindex(p.index)
    p["u_gap"] = p["u"] - p["nrou"]
    p["nfp_3m"] = m["PAYEMS"].diff().rolling(3).mean()
    p["tb3"] = m["TB3MS"]
    p["gs2"] = m["GS2"]
    p["gs5"] = m["GS5"]
    p["gs10"] = m["GS10"]
    p["gs30"] = m["GS30"]

    if daily is not None:
        pr = policy_rate_daily(daily)
        eom = pr.resample("MS").last()
        ylds = daily[["DGS3MO", "DGS2", "DGS10"]].resample("MS").last()
        # Extend the index to the current (possibly incomplete) month so a live
        # forecast row exists even before that month's macro data are out.
        p = p.reindex(p.index.union(eom.index))
        p["target_upper"] = eom["upper"]
        p["target_mid"] = eom["mid"]
        p["y3m_eom"] = ylds["DGS3MO"]
        p["y2_eom"] = ylds["DGS2"]
        p["y10_eom"] = ylds["DGS10"]

    return p.loc["1990-01-01":]


# --------------------------------------------------------------------------
# Forecasting features (real-time safe)
# --------------------------------------------------------------------------
def months_since_last_move(rate: pd.Series, tol: float = 1e-9) -> pd.Series:
    """Months elapsed since the policy rate last changed (0 in a move month)."""
    moved = rate.diff().abs() > tol
    out, count = [], np.nan
    for flag in moved:
        if flag:
            count = 0
        elif not np.isnan(count):
            count += 1
        out.append(count)
    return pd.Series(out, index=rate.index, dtype=float)


def build_features(
    p: pd.DataFrame,
    r_star: float = 2.0,
    asof: pd.Timestamp | None = None,
    meetings: pd.DatetimeIndex | None = None,
    tone: pd.Series | None = None,
) -> pd.DataFrame:
    """Feature matrix and target for the hike / hold / cut model.

    Row t holds what was knowable at the end of month t and the target
    y_t = sign(policy rate at end of t+1 minus policy rate at end of t).

    ``asof`` is the last date in the data. Targets whose month t+1 has not
    finished by then are left missing, so an incomplete month is never scored.
    Macro features carry the latest released value forward for at most two
    months, which only matters at the live edge of the sample.
    """
    from .rules import taylor_1993  # local import keeps modules decoupled

    if "target_mid" not in p:
        raise ValueError("Daily target-rate data are required for the forecasting model.")

    lag = lambda s: s.shift(MACRO_LAG).ffill(limit=2)  # noqa: E731
    infl = lag(p["core_pce_yoy"])
    gap = lag(p["u_gap"])
    rate = p["target_mid"]

    f = pd.DataFrame(index=p.index)
    # Macro block (published with a lag)
    f["infl_gap"] = infl - PI_STAR
    f["infl_mom"] = infl - infl.shift(3)
    f["headline_minus_core"] = lag(p["cpi_yoy"] - p["core_cpi_yoy"])
    f["u_gap"] = gap
    f["du_3m"] = lag(p["u"] - p["u"].shift(3))
    f["nfp_3m"] = lag(p["nfp_3m"]) / 100.0  # hundreds of thousands of jobs
    # Policy state (known in real time)
    f["rate"] = rate
    f["rule_gap"] = taylor_1993(infl, gap, r_star=r_star) - rate
    move = np.sign(rate.diff().where(rate.diff().abs() > 1e-9))
    f["last_move"] = move.ffill().fillna(0.0)
    f["months_since_move"] = np.minimum(months_since_last_move(rate), 24.0)
    f["at_elb"] = (p["target_upper"] <= ELB_UPPER + 1e-9).astype(float)
    if meetings is not None:
        from .calendar import meeting_next_month
        f["meeting_next"] = meeting_next_month(p.index, meetings)
    if tone is not None:
        f["statement_tone"] = tone.reindex(p.index)
    # Market block (known in real time)
    f["bill_spread"] = p["y3m_eom"] - rate
    f["two_spread"] = p["y2_eom"] - rate
    f["slope"] = p["y10_eom"] - p["y3m_eom"]

    nxt = rate.shift(-1) - rate
    y = np.sign(nxt.where(nxt.abs() > 1e-9, 0.0))
    known = nxt.notna()
    if asof is not None:
        known &= (p.index + pd.offsets.MonthEnd(2)) <= pd.Timestamp(asof)
    f["y"] = y.where(known)
    return f


MACRO_FEATURES = [
    "infl_gap", "infl_mom", "headline_minus_core", "u_gap", "du_3m", "nfp_3m",
]
POLICY_FEATURES = ["rate", "rule_gap", "last_move", "months_since_move", "at_elb"]
MARKET_FEATURES = ["bill_spread", "two_spread", "slope"]
