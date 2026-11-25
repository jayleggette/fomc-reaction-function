import numpy as np
import pandas as pd
import pytest

from fomc import rules as R


def test_taylor_1993_arithmetic():
    # r* + pi + 0.5 (pi - 2) - 1.0 * (u - u*)
    assert R.taylor_1993(2.0, 0.0, r_star=2.0) == pytest.approx(4.0)
    assert R.taylor_1993(3.0, -0.5, r_star=2.0) == pytest.approx(6.0)
    assert R.balanced_approach(3.0, -0.5, r_star=2.0) == pytest.approx(6.5)


def test_implied_r_star_round_trip():
    # alpha = r* + (1 - b_pi) * 2  ->  r* = alpha - (1 - b_pi) * 2
    assert R.implied_r_star(alpha=1.0, beta_pi=1.5) == pytest.approx(2.0)


def _simulate(n=600, rho=0.8, alpha=1.0, b_pi=1.5, b_u=-2.0, sigma=0.15, lower=None, seed=0):
    rng = np.random.default_rng(seed)
    pi = 2.5 + np.cumsum(rng.normal(0, 0.25, n)) * 0.15
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.9 * gap[t - 1] + rng.normal(0, 0.4)
    i = np.zeros(n)
    i[0] = alpha + b_pi * pi[0] + b_u * gap[0]
    for t in range(1, n):
        latent = rho * i[t - 1] + (1 - rho) * (alpha + b_pi * pi[t] + b_u * gap[t]) + rng.normal(0, sigma)
        i[t] = latent if lower is None else max(latent, lower)
    idx = pd.date_range("1950-01-01", periods=n, freq="QS")
    return pd.DataFrame({"ffr": i, "core_pce_yoy": pi, "u_gap": gap}, index=idx)


def test_inertial_ols_recovers_long_run_coefficients():
    df = _simulate()
    est = R.estimate_inertial_ols(df, "sim", drop_elb=False)
    assert est.rho == pytest.approx(0.8, abs=0.05)
    assert est.beta_pi == pytest.approx(1.5, abs=0.25)
    assert est.beta_u == pytest.approx(-2.0, abs=0.3)


def test_tobit_handles_censoring_better_than_dropping():
    # Shift the latent rate down so a meaningful share of quarters hit the bound.
    df = _simulate(alpha=-2.5, lower=0.25, seed=3)
    share = (df["ffr"] <= 0.25 + 1e-12).mean()
    assert 0.05 < share < 0.6
    tob = R.estimate_inertial_tobit(df, "sim", lower=0.25)
    ols = R.estimate_inertial_ols(df, "sim", drop_elb=True)
    assert tob.n_censored > 0
    assert abs(tob.beta_u - (-2.0)) < abs(ols.beta_u - (-2.0)) + 0.1
    assert tob.beta_pi == pytest.approx(1.5, abs=0.4)
