import numpy as np
import pandas as pd
import pytest

from fomc import calendar as CAL
from fomc import data as D


def test_fill_isolated_gaps_only_fills_single_holes():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    out = D.fill_isolated_gaps(s)
    assert out[1] == pytest.approx(2.0)
    assert np.isnan(out[3]) and np.isnan(out[4])


def test_months_since_last_move():
    rate = pd.Series([1.0, 1.0, 1.25, 1.25, 1.25, 1.0])
    out = D.months_since_last_move(rate)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert list(out[2:]) == [0, 1, 2, 0]


@pytest.fixture(scope="module")
def panel_and_features():
    daily = D.load_daily()
    p = D.build_panel(D.load_monthly(), D.load_quarterly(), daily)
    meetings = CAL.load_calendar(D.RAW / "fomc_scheduled_meetings.csv")
    f = D.build_features(p, asof=daily.index.max(), meetings=meetings)
    return daily, p, f, meetings


def test_macro_features_do_not_peek_ahead(panel_and_features):
    """Scrambling macro data from month T on must not change features dated T or earlier."""
    daily, _, f, meetings = panel_and_features
    cutoff = pd.Timestamp("2015-06-01")
    monthly = D.load_monthly()
    scrambled = monthly.copy()
    cols = ["CPIAUCSL", "CPILFESL", "PCEPILFE", "UNRATE", "PAYEMS"]
    scrambled.loc[scrambled.index >= cutoff, cols] *= 3.0
    p2 = D.build_panel(scrambled, D.load_quarterly(), daily)
    f2 = D.build_features(p2, asof=daily.index.max(), meetings=meetings)
    feats = D.MACRO_FEATURES + ["rule_gap"]
    pd.testing.assert_frame_equal(f.loc[:cutoff, feats], f2.loc[:cutoff, feats])
    assert not f.loc["2015-07-01", "infl_gap"] == f2.loc["2015-07-01", "infl_gap"]


def test_target_is_next_month_change(panel_and_features):
    _, p, f, _ = panel_and_features
    # September 2026: the target range rose 25bp on 17 September, so August's target is a hike.
    assert f.loc["2026-08-01", "y"] == 1.0
    # The latest month has no known outcome yet.
    assert np.isnan(f["y"].iloc[-1])


def test_calendar_has_eight_scheduled_meetings_every_year():
    cal = CAL.load_calendar(D.RAW / "fomc_scheduled_meetings.csv")
    counts = pd.Series(1, index=cal).groupby(cal.year).size()
    assert (counts.loc[1990:2026] == 8).all()


def test_every_scheduled_move_lands_on_an_effective_date():
    """Each target change within three days after a scheduled meeting must fall on
    that meeting's effective date (decision day before March 2017, next day after)."""
    daily = D.load_daily()
    t = daily["DFEDTARU"].combine_first(daily["DFEDTAR"]).dropna()
    changes = t[t.diff().abs() > 1e-9].loc["1994-01-01":].index
    cal = CAL.load_calendar(D.RAW / "fomc_scheduled_meetings.csv")
    eff = CAL.effective_dates(cal)
    checked = 0
    for c in changes:
        near = cal[(cal <= c) & (cal >= c - pd.Timedelta(days=3))]
        if len(near):
            assert c == eff[cal.get_loc(near[-1])], c
            checked += 1
    assert checked > 60


def test_meeting_next_month_uses_effective_dates():
    idx = pd.DatetimeIndex(["2019-06-01", "2019-07-01"])
    flag = CAL.meeting_next_month(idx, pd.DatetimeIndex(["2019-07-31"]))
    assert list(flag) == [0.0, 1.0]  # decided 31 July, effective 1 August


def test_meeting_next_month_flag():
    idx = pd.date_range("2026-10-01", periods=3, freq="MS")
    meetings = pd.DatetimeIndex(["2026-10-28", "2026-12-09"])
    flag = CAL.meeting_next_month(idx, meetings)
    assert list(flag) == [0.0, 1.0, 0.0]  # Nov: no meeting; Dec: meeting; Jan: none listed
