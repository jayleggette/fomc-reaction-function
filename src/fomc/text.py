"""A transparent hawkish / dovish tone score for FOMC statements.

Deliberately simple: a short lexicon of phrases the Committee uses when it
leans toward tightening or easing, counted per statement. The point is not a
state-of-the-art language model; it is a measure anyone can audit, which is
then put through the same out-of-sample test as every other feature. If the
tone of the last statement adds nothing beyond market prices, that is a
finding, not a failure.

Tone = (hawkish hits - dovish hits) / (hawkish hits + dovish hits + 1),
which lies in (-1, 1).
"""
from __future__ import annotations

import re

import pandas as pd

HAWKISH = [
    r"inflation (remains|has remained|is) elevated", r"elevated inflation",
    r"upside risks to (the )?inflation", r"inflation pressures? (remain|have increased|picked up)",
    r"risks? (of|to) (higher|rising) inflation", r"(additional|further) (policy )?firming",
    r"raise the target range", r"increases? in the target range", r"ongoing increases",
    r"remov(e|al of) (policy )?accommodation", r"reduce (policy )?accommodation",
    r"tighten", r"deliver price stability", r"strongly committed to returning inflation",
    r"inflation (has )?moved up", r"inflation .{0,40}(picked up|rose|increased)",
]

DOVISH = [
    r"accommodative", r"lower the target range", r"reduc(e|tion in) the target range",
    r"downside risks", r"(has|have) (weakened|slowed|softened|moderated)",
    r"\bpatient\b", r"support the economy", r"(running|run) (persistently )?below (its |the Committee's )?2 percent",
    r"low inflation", r"exceptionally low", r"economic weakness", r"labor market (has )?(softened|weakened)",
    r"downside risks to employment", r"(slack|underutilization)", r"maintain .{0,30}at (the )?current (low )?level",
]

_H = [re.compile(p, re.I) for p in HAWKISH]
_D = [re.compile(p, re.I) for p in DOVISH]


def _clean(text: str) -> str:
    text = re.split(r"Voting for (the|this) (FOMC )?(monetary policy )?action", text, maxsplit=1)[0]
    text = re.split(r"Implementation Note", text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text)


def tone(text: str) -> tuple[float, int, int]:
    t = _clean(str(text))
    h = sum(len(p.findall(t)) for p in _H)
    d = sum(len(p.findall(t)) for p in _D)
    return (h - d) / (h + d + 1.0), h, d


def statement_tone(comm: pd.DataFrame) -> pd.DataFrame:
    """Tone of every policy statement, stamped on its release date."""
    s = comm[comm["Type"] == "Statement"].copy()
    s["release"] = pd.to_datetime(s["Release Date"])
    scored = s["Text"].map(tone)
    s["tone"] = [x[0] for x in scored]
    s["hawk_hits"] = [x[1] for x in scored]
    s["dove_hits"] = [x[2] for x in scored]
    return s[["release", "tone", "hawk_hits", "dove_hits"]].sort_values("release").reset_index(drop=True)


def monthly_tone(st: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Tone of the latest statement released on or before the end of month t."""
    eom = index + pd.offsets.MonthEnd(0)
    s = st.set_index("release")["tone"].sort_index()
    pos = s.index.searchsorted(eom, side="right") - 1
    vals = [s.iloc[i] if i >= 0 else float("nan") for i in pos]
    return pd.Series(vals, index=index)
