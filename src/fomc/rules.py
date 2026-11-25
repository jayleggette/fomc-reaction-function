"""Policy rules and estimated reaction functions.

Two families:

1. Prescriptive rules in the form the Federal Reserve Board uses in its
   Monetary Policy Report, with the unemployment gap standing in for the
   output gap (Okun coefficient of 2 folded into the gap coefficient):

       R = r* + pi + 0.5 (pi - pi*) + g (u* - u)

   g = 1 gives the Taylor (1993) rule and g = 2 the balanced-approach rule.

2. An estimated inertial (partial-adjustment) rule in the spirit of
   Clarida, Gali and Gertler (2000):

       i_t = rho i_{t-1} + (1 - rho) (alpha + b_pi pi_t + b_u gap_t) + e_t

   estimated two ways: OLS on the linear form with Newey-West errors, and a
   Tobit that treats effective-lower-bound quarters as censored instead of
   dropping them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

from .data import ELB_EFFECTIVE, PI_STAR


# --------------------------------------------------------------------------
# Prescriptive rules
# --------------------------------------------------------------------------
def taylor_1993(infl, u_gap, r_star: float = 2.0, pi_star: float = PI_STAR, gap_coef: float = 1.0):
    """Taylor (1993) in unemployment-gap form. ``u_gap`` is u minus u*."""
    return r_star + infl + 0.5 * (infl - pi_star) - gap_coef * u_gap


def balanced_approach(infl, u_gap, r_star: float = 2.0, pi_star: float = PI_STAR):
    """Balanced-approach rule: twice the Taylor (1993) weight on slack."""
    return taylor_1993(infl, u_gap, r_star=r_star, pi_star=pi_star, gap_coef=2.0)


# --------------------------------------------------------------------------
# Estimated inertial rule
# --------------------------------------------------------------------------
@dataclass
class RuleEstimate:
    method: str
    sample: str
    n: int
    rho: float
    rho_se: float
    alpha: float
    alpha_se: float
    beta_pi: float
    beta_pi_se: float
    beta_u: float
    beta_u_se: float
    sigma: float
    implied_r_star: float
    n_censored: int = 0
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "method": self.method, "sample": self.sample, "n": self.n,
            "n_censored": self.n_censored,
            "rho": self.rho, "rho_se": self.rho_se,
            "beta_pi": self.beta_pi, "beta_pi_se": self.beta_pi_se,
            "beta_u": self.beta_u, "beta_u_se": self.beta_u_se,
            "alpha": self.alpha, "alpha_se": self.alpha_se,
            "implied_r_star": self.implied_r_star, "sigma": self.sigma,
        }


def to_quarterly(p: pd.DataFrame, cols=("ffr", "core_pce_yoy", "u_gap")) -> pd.DataFrame:
    """Quarterly averages of monthly columns; drops incomplete quarters."""
    q = p[list(cols)].resample("QS").agg(["mean", "count"])
    out = pd.DataFrame({c: q[(c, "mean")].where(q[(c, "count")] == 3) for c in cols})
    return out.dropna()


def _long_run(params: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Map short-run (c, rho, a, b) to long-run (alpha, b_pi, b_u) with delta-method SEs."""
    c, rho, a, b = params
    k = 1.0 - rho
    lr = np.array([c / k, a / k, b / k])
    # Jacobian rows: d(theta/k)/d(c, rho, a, b)
    jac = np.array([
        [1 / k, c / k**2, 0.0, 0.0],
        [0.0, a / k**2, 1 / k, 0.0],
        [0.0, b / k**2, 0.0, 1 / k],
    ])
    lr_cov = jac @ cov @ jac.T
    return lr, np.sqrt(np.diag(lr_cov)), float(rho), float(np.sqrt(cov[1, 1]))


def _design(df: pd.DataFrame, y: str, infl: str, gap: str) -> tuple[pd.Series, pd.DataFrame]:
    d = pd.DataFrame({
        "y": df[y],
        "const": 1.0,
        "lag": df[y].shift(1),
        "infl": df[infl],
        "gap": df[gap],
    }).dropna()
    return d["y"], d[["const", "lag", "infl", "gap"]]


def estimate_inertial_ols(
    df: pd.DataFrame,
    sample: str,
    y: str = "ffr",
    infl: str = "core_pce_yoy",
    gap: str = "u_gap",
    drop_elb: bool = True,
    hac_lags: int = 4,
) -> RuleEstimate:
    """Linear partial-adjustment rule, Newey-West standard errors.

    With ``drop_elb`` the quarters with the policy rate at the effective lower
    bound are removed, which biases the sample toward non-binding periods;
    :func:`estimate_inertial_tobit` is the fix.
    """
    endog, exog = _design(df, y, infl, gap)
    if drop_elb:
        keep = endog > ELB_EFFECTIVE
        endog, exog = endog[keep], exog[keep]
    res = sm.OLS(endog, exog).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    lr, lr_se, rho, rho_se = _long_run(res.params.values, res.cov_params().values)
    alpha, b_pi, b_u = lr
    return RuleEstimate(
        method="OLS, ELB dropped" if drop_elb else "OLS",
        sample=sample, n=int(res.nobs),
        rho=rho, rho_se=rho_se,
        alpha=alpha, alpha_se=lr_se[0],
        beta_pi=b_pi, beta_pi_se=lr_se[1],
        beta_u=b_u, beta_u_se=lr_se[2],
        sigma=float(np.sqrt(res.scale)),
        implied_r_star=implied_r_star(alpha, b_pi),
        extra={"r2": float(res.rsquared)},
    )


class _CensoredLinear(GenericLikelihoodModel):
    """Linear model with left-censoring at a known lower bound (Tobit type I)."""

    def __init__(self, endog, exog, lower: float, **kwds):
        self.lower = lower
        super().__init__(endog, exog, extra_params_names=["log_sigma"], **kwds)

    def loglikeobs(self, params):
        beta, log_sigma = params[:-1], params[-1]
        sigma = np.exp(log_sigma)
        xb = self.exog @ beta
        y = self.endog
        censored = y <= self.lower + 1e-12
        ll_c = stats.norm.logcdf((self.lower - xb) / sigma)
        ll_u = stats.norm.logpdf((y - xb) / sigma) - log_sigma
        return np.where(censored, ll_c, ll_u)


def estimate_inertial_tobit(
    df: pd.DataFrame,
    sample: str,
    y: str = "ffr",
    infl: str = "core_pce_yoy",
    gap: str = "u_gap",
    lower: float = ELB_EFFECTIVE,
) -> RuleEstimate:
    """Partial-adjustment rule with ELB quarters treated as censored.

    The observed rate is max(latent rate, lower). Quarters at the bound tell
    us the rule wanted a rate at or below the bound, which is information the
    OLS-with-drops estimate throws away.
    """
    endog, exog = _design(df, y, infl, gap)
    endog = endog.clip(lower=lower)
    ols = sm.OLS(endog, exog).fit()
    start = np.r_[ols.params.values, np.log(np.sqrt(ols.scale))]
    model = _CensoredLinear(endog, exog, lower=lower)
    res = model.fit(start_params=start, method="bfgs", maxiter=5000, disp=False)
    cov = res.cov_params()
    cov = cov.values if hasattr(cov, "values") else np.asarray(cov)
    params = np.asarray(res.params)
    lr, lr_se, rho, rho_se = _long_run(params[:4], cov[:4, :4])
    alpha, b_pi, b_u = lr
    return RuleEstimate(
        method="Tobit, ELB censored",
        sample=sample, n=int(len(endog)),
        rho=rho, rho_se=rho_se,
        alpha=alpha, alpha_se=lr_se[0],
        beta_pi=b_pi, beta_pi_se=lr_se[1],
        beta_u=b_u, beta_u_se=lr_se[2],
        sigma=float(np.exp(params[-1])),
        implied_r_star=implied_r_star(alpha, b_pi),
        n_censored=int((endog <= lower + 1e-12).sum()),
        extra={"converged": bool(res.mle_retvals.get("converged", False)), "llf": float(res.llf)},
    )


def implied_r_star(alpha: float, beta_pi: float, pi_star: float = PI_STAR) -> float:
    """Back out r* from alpha = r* + (1 - b_pi) pi*, the Taylor-form intercept."""
    return float(alpha - (1.0 - beta_pi) * pi_star)


def inertial_path(df: pd.DataFrame, est: RuleEstimate, y: str = "ffr",
                  infl: str = "core_pce_yoy", gap: str = "u_gap") -> pd.DataFrame:
    """Long-run target i* and one-step partial-adjustment fit from an estimate."""
    target = est.alpha + est.beta_pi * df[infl] + est.beta_u * df[gap]
    fitted = est.rho * df[y].shift(1) + (1 - est.rho) * target
    return pd.DataFrame({"long_run_target": target, "one_step_fit": fitted})


# --------------------------------------------------------------------------
# Rolling static rule
# --------------------------------------------------------------------------
def rolling_static(
    p: pd.DataFrame,
    window: int = 120,
    min_obs: int = 72,
    y: str = "ffr",
    infl: str = "core_pce_yoy",
    gap: str = "u_gap",
    hac_lags: int = 12,
) -> pd.DataFrame:
    """Rolling OLS of the policy rate on inflation and the unemployment gap.

    Monthly, ``window`` months wide, ELB months excluded. Reports the
    inflation and slack coefficients with 95% Newey-West intervals. This is
    descriptive: it shows how the Fed's systematic response drifted, not a
    structural estimate.
    """
    d = p[[y, infl, gap]].dropna()
    rows = []
    for end in range(window, len(d) + 1):
        w = d.iloc[end - window:end]
        w = w[w[y] > ELB_EFFECTIVE]
        if len(w) < min_obs:
            continue
        X = sm.add_constant(w[[infl, gap]])
        res = sm.OLS(w[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        b, se = res.params, res.bse
        rows.append({
            "date": d.index[end - 1], "n": len(w),
            "beta_pi": b[infl], "beta_pi_lo": b[infl] - 1.96 * se[infl], "beta_pi_hi": b[infl] + 1.96 * se[infl],
            "beta_u": b[gap], "beta_u_lo": b[gap] - 1.96 * se[gap], "beta_u_hi": b[gap] + 1.96 * se[gap],
        })
    return pd.DataFrame(rows).set_index("date")
