"""Out-of-sample model of the next policy move: cut, hold or hike.

Design choices
--------------
* Target: sign of the change in the policy rate (target or target-range
  midpoint) from the end of month t to the end of month t+1.
* Features: only what was knowable at the end of month t
  (see :func:`fomc.data.build_features`).
* Model: L2-regularised multinomial logit on standardised features. The
  penalty is picked by time-series cross-validation inside each training
  window, so the test period never informs tuning.
* Evaluation: expanding window, refit every January from 2000 on. Each
  forecast for month t uses a model trained only on rows whose outcomes
  were known before month t began.
* Scoring: multiclass Brier score and log loss (both proper scoring rules),
  accuracy, and Brier skill relative to climatology. Model comparisons use a
  Diebold-Mariano test with Newey-West variance.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MACRO_FEATURES, MARKET_FEATURES, POLICY_FEATURES

CLASSES = np.array([-1.0, 0.0, 1.0])
CLASS_NAMES = ("cut", "hold", "hike")

CAL = ["meeting_next"]  # every model, including climatology, knows the public calendar

MODEL_SPECS: dict[str, list[str] | None] = {
    "Climatology": None,
    "Policy momentum": ["last_move", "months_since_move", "rate", "at_elb"] + CAL,
    "Macro rule": MACRO_FEATURES + POLICY_FEATURES + CAL,
    "Market only": MARKET_FEATURES + CAL,
    "Full model": MACRO_FEATURES + POLICY_FEATURES + MARKET_FEATURES + CAL,
}

TEXT_SPECS: dict[str, list[str] | None] = {
    "Climatology": None,
    "Full model": MODEL_SPECS["Full model"],
    "Full model + statement tone": MODEL_SPECS["Full model"] + ["statement_tone"],
}

C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def one_hot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return (y[:, None] == CLASSES[None, :]).astype(float)


def brier_per_obs(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiclass Brier score per observation: sum over classes of (p - 1{y=k})^2."""
    return np.sum((np.asarray(P) - one_hot(y)) ** 2, axis=1)


def log_loss_per_obs(P: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p_true = np.sum(np.asarray(P) * one_hot(y), axis=1)
    return -np.log(np.clip(p_true, eps, 1.0))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 3) -> tuple[float, float]:
    """DM statistic for H0: equal expected loss. Negative means model a is better.

    Newey-West long-run variance with Bartlett weights and the
    Harvey-Leybourne-Newbold small-sample correction (h = 1).
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    dm = d.mean() / np.sqrt(lrv / n)
    dm *= np.sqrt((n - 1.0) / n)  # HLN correction for one-step forecasts
    p = 2.0 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def _align_proba(model, X: np.ndarray) -> np.ndarray:
    """predict_proba with columns forced into CLASSES order (missing classes = 0)."""
    raw = model.predict_proba(X)
    fitted = model.classes_ if not hasattr(model, "steps") else model.steps[-1][1].classes_
    out = np.zeros((len(X), len(CLASSES)))
    for j, c in enumerate(fitted):
        out[:, int(np.where(CLASSES == c)[0][0])] = raw[:, j]
    return out


class ClipToTrainingRange(BaseEstimator, TransformerMixin):
    """Clip each feature to the min/max seen in training.

    Logit is linear in the features, so an input far outside the training
    range (April 2020 payrolls were roughly nine times worse than anything
    before) produces probabilities of exactly 0 or 1. Clipping keeps the
    model from extrapolating beyond its evidence.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanmin(X, axis=0)
        self.hi_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _make_model(C: float):
    return make_pipeline(
        ClipToTrainingRange(),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000),
    )


def apply_elb_constraint(P: np.ndarray, at_elb: np.ndarray) -> np.ndarray:
    """A cut is infeasible when the target range is already 0 to 0.25%.

    The FOMC has never set a negative target, so at the lower bound the cut
    probability is set to zero and the rest renormalised. Applied identically
    to every model, including climatology.
    """
    P = np.array(P, dtype=float, copy=True)
    elb = np.asarray(at_elb, dtype=float) > 0.5
    P[elb, 0] = 0.0
    P /= P.sum(axis=1, keepdims=True)
    return P


def choose_C(X: np.ndarray, y: np.ndarray, grid=C_GRID, n_splits: int = 4) -> float:
    """Pick the L2 penalty by forward-chaining CV on the training window only."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {C: [] for C in grid}
    for tr, va in tscv.split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        for C in grid:
            m = _make_model(C).fit(X[tr], y[tr])
            P = _align_proba(m, X[va])
            # A class unseen in the fold gets a small floor so log loss stays finite.
            P = (P + 1e-3) / (1 + 3e-3)
            scores[C].append(log_loss_per_obs(P, y[va]).mean())
    means = {C: np.mean(v) for C, v in scores.items() if v}
    return min(means, key=means.get) if means else 1.0


@dataclass
class Backtest:
    probs: dict[str, pd.DataFrame]   # model -> DataFrame[cut, hold, hike]
    y: pd.Series                     # realised class
    at_elb: pd.Series                # policy at the lower bound at forecast time
    chosen_C: pd.DataFrame           # refit year x model


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Historical class frequencies, conditional on whether a meeting is scheduled.

    Laplace-smoothed so no class ever gets probability exactly zero.
    """
    def freq(y):
        counts = np.array([(y == c).sum() for c in CLASSES], dtype=float) + 1.0
        return counts / counts.sum()

    if "meeting_next" not in train:
        return np.tile(freq(train["y"].values), (len(test), 1))
    out = np.zeros((len(test), len(CLASSES)))
    for flag in (0.0, 1.0):
        rows = test["meeting_next"].values == flag
        out[rows] = freq(train.loc[train["meeting_next"] == flag, "y"].values)
    return out


def expanding_backtest(
    f: pd.DataFrame,
    specs: dict[str, list[str] | None] = MODEL_SPECS,
    start: str = "2000-01-01",
    refit_every: int = 12,
    train_start: str | None = None,
) -> Backtest:
    """Expanding-window backtest.

    Each model trains on every earlier row that has its own features, so a
    feature that only exists from 2000 does not shorten the other models'
    training samples. All models are scored on the same test months.
    """
    data = f.dropna(subset=["y"])
    if train_start is not None:
        data = data[data.index >= pd.Timestamp(train_start)]
    needed = sorted({c for cols in specs.values() if cols for c in cols})
    test_idx = data.dropna(subset=needed).index
    test_idx = test_idx[test_idx >= pd.Timestamp(start)]
    blocks = [test_idx[i:i + refit_every] for i in range(0, len(test_idx), refit_every)]

    probs = {name: [] for name in specs}
    chosen = []
    for block in blocks:
        train_all = data[data.index < block[0]]
        test = data.loc[block]
        row = {"refit": block[0]}
        for name, cols in specs.items():
            if cols is None:
                P = climatology(train_all, test)
            else:
                train = train_all.dropna(subset=cols)
                ytr = train["y"].values
                Xtr, Xte = train[cols].values, test[cols].values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    C = choose_C(Xtr, ytr)
                    m = _make_model(C).fit(Xtr, ytr)
                P = _align_proba(m, Xte)
                row[name] = C
            P = apply_elb_constraint(P, test["at_elb"].values)
            probs[name].append(pd.DataFrame(P, index=test.index, columns=CLASS_NAMES))
        chosen.append(row)

    probs = {k: pd.concat(v) for k, v in probs.items()}
    y = data.loc[test_idx, "y"]
    return Backtest(probs=probs, y=y, at_elb=data.loc[test_idx, "at_elb"],
                    chosen_C=pd.DataFrame(chosen).set_index("refit"))


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def score_table(bt: Backtest, mask: pd.Series | None = None, label: str = "All test months") -> pd.DataFrame:
    y = bt.y if mask is None else bt.y[mask]
    clim = bt.probs["Climatology"].loc[y.index].values
    clim_brier = brier_per_obs(clim, y.values).mean()
    rows = []
    for name, P in bt.probs.items():
        Pm = P.loc[y.index].values
        b = brier_per_obs(Pm, y.values)
        ll = log_loss_per_obs(Pm, y.values)
        pred = CLASSES[Pm.argmax(axis=1)]
        moves = y.values != 0
        rows.append({
            "sample": label, "model": name, "n": len(y),
            "brier": b.mean(), "brier_skill_vs_climatology": 1 - b.mean() / clim_brier,
            "log_loss": ll.mean(), "accuracy": (pred == y.values).mean(),
            "moves_called": int(((pred == y.values) & moves).sum()), "moves": int(moves.sum()),
        })
    return pd.DataFrame(rows)


def dm_table(bt: Backtest, against: str = "Full model", mask: pd.Series | None = None) -> pd.DataFrame:
    y = bt.y if mask is None else bt.y[mask]
    base = brier_per_obs(bt.probs[against].loc[y.index].values, y.values)
    rows = []
    for name, P in bt.probs.items():
        if name == against:
            continue
        other = brier_per_obs(P.loc[y.index].values, y.values)
        dm, p = diebold_mariano(base, other)
        rows.append({"comparison": f"{against} vs {name}", "mean_brier_diff": (base - other).mean(),
                     "dm_stat": dm, "p_value": p})
    return pd.DataFrame(rows)


def reliability(bt: Backtest, model: str, cls: str,
                bins=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0001)) -> pd.DataFrame:
    """Forecast probability vs realised frequency for one class."""
    p = bt.probs[model][cls]
    hit = (bt.y == CLASSES[CLASS_NAMES.index(cls)]).astype(float)
    cut = pd.cut(p, bins=list(bins), right=False)
    g = pd.DataFrame({"p": p, "hit": hit, "bin": cut}).groupby("bin", observed=True)
    return g.agg(mean_forecast=("p", "mean"), realised=("hit", "mean"), n=("hit", "size")).reset_index()


def fit_final(f: pd.DataFrame, cols: list[str]) -> tuple[object, float]:
    """Fit on every row with a known outcome (for the live nowcast)."""
    data = f.dropna(subset=cols + ["y"])
    X, y = data[cols].values, data["y"].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = choose_C(X, y)
        m = _make_model(C).fit(X, y)
    return m, C


def nowcast(model, row: pd.Series, cols: list[str]) -> dict[str, float]:
    P = _align_proba(model, row[cols].values.reshape(1, -1).astype(float))
    P = apply_elb_constraint(P, np.array([row["at_elb"]]))[0]
    return dict(zip(CLASS_NAMES, map(float, P)))
