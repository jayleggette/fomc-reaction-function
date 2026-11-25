import numpy as np
import pandas as pd
import pytest

from fomc import calendar as CAL
from fomc import data as D
from fomc import direction as M
from fomc import text as T


def test_brier_and_log_loss():
    y = np.array([0.0])
    perfect = np.array([[0.0, 1.0, 0.0]])
    uniform = np.full((1, 3), 1 / 3)
    assert M.brier_per_obs(perfect, y)[0] == pytest.approx(0.0)
    assert M.brier_per_obs(uniform, y)[0] == pytest.approx(2 / 3)
    assert M.log_loss_per_obs(uniform, y)[0] == pytest.approx(np.log(3))


def test_elb_constraint_removes_cut_and_renormalises():
    P = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
    out = M.apply_elb_constraint(P, np.array([1.0, 0.0]))
    np.testing.assert_allclose(out[0], [0.0, 0.6, 0.4])
    np.testing.assert_allclose(out[1], P[1])


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    worse = rng.uniform(0.3, 0.5, 300)
    better = worse - 0.05 + rng.normal(0, 0.01, 300)
    stat, p = M.diebold_mariano(better, worse)
    assert stat < 0 and p < 0.01


def test_climatology_is_conditional_on_calendar():
    train = pd.DataFrame({"y": [0, 0, 0, 1, 1, -1, 0, 0], "meeting_next": [0, 0, 0, 1, 1, 1, 1, 0]}, dtype=float)
    test = pd.DataFrame({"meeting_next": [0.0, 1.0]})
    P = M.climatology(train, test)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert P[0, 1] > P[1, 1]  # holds are likelier when no meeting is scheduled


def test_text_tone_direction():
    hawk = "Inflation remains elevated. The Committee anticipates that ongoing increases in the target range will be appropriate."
    dove = "The Committee decided to lower the target range. Downside risks to employment rose. Policy remains accommodative."
    assert T.tone(hawk)[0] > 0 > T.tone(dove)[0]


def test_backtest_forecasts_use_only_the_past():
    """Rewriting everything after 2010 must not change any forecast made before 2010."""
    daily = D.load_daily()
    p = D.build_panel(D.load_monthly(), D.load_quarterly(), daily)
    meetings = CAL.load_calendar(D.RAW / "fomc_scheduled_meetings.csv")
    f = D.build_features(p, asof=daily.index.max(), meetings=meetings)
    specs = {"Climatology": None, "Full model": M.MODEL_SPECS["Full model"]}
    a = M.expanding_backtest(f, specs, start="2000-01-01")

    g = f.copy()
    late = g.index >= "2010-01-01"
    rng = np.random.default_rng(1)
    g.loc[late, "y"] = rng.choice([-1.0, 0.0, 1.0], size=late.sum())
    g.loc[late, "bill_spread"] = rng.normal(0, 2, late.sum())
    b = M.expanding_backtest(g, specs, start="2000-01-01")
    for name in specs:
        pd.testing.assert_frame_equal(a.probs[name].loc[:"2009-12-01"], b.probs[name].loc[:"2009-12-01"])
