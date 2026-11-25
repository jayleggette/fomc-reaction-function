"""Scheduled FOMC meeting calendar, 1990 to 2027.

The FOMC publishes its meeting schedule about a year ahead, so whether a
scheduled meeting falls in the coming month is public information at
forecast time. Unscheduled calls (for example 3 January 2001, 7 October 2008
or 15 March 2020) are NOT known in advance and are excluded.

Sources
-------
* 1990-1999: Federal Reserve Board, "FOMC historical materials by year"
  (federalreserve.gov/monetarypolicy/fomchistoricalYYYY.htm), regular
  meetings only, conference calls excluded.
* 2000 onward: meetings that produced both a statement and minutes in the
  public FOMC communications archive maintained at
  github.com/vtasca/fed-statement-scraping (scraped from federalreserve.gov),
  minus the unscheduled meetings listed below, plus the archive's list of
  announced future meetings.

Every year reconciles to exactly eight scheduled meetings; a unit test
checks this.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Decision day (second day of two-day meetings), regular meetings only.
PRE_2000 = """
1990-02-07 1990-03-27 1990-05-15 1990-07-03 1990-08-21 1990-10-02 1990-11-13 1990-12-18
1991-02-06 1991-03-26 1991-05-14 1991-07-03 1991-08-20 1991-10-01 1991-11-05 1991-12-17
1992-02-05 1992-03-31 1992-05-19 1992-07-01 1992-08-18 1992-10-06 1992-11-17 1992-12-22
1993-02-03 1993-03-23 1993-05-18 1993-07-07 1993-08-17 1993-09-21 1993-11-16 1993-12-21
1994-02-04 1994-03-22 1994-05-17 1994-07-06 1994-08-16 1994-09-27 1994-11-15 1994-12-20
1995-02-01 1995-03-28 1995-05-23 1995-07-06 1995-08-22 1995-09-26 1995-11-15 1995-12-19
1996-01-31 1996-03-26 1996-05-21 1996-07-03 1996-08-20 1996-09-24 1996-11-13 1996-12-17
1997-02-05 1997-03-25 1997-05-20 1997-07-02 1997-08-19 1997-09-30 1997-11-12 1997-12-16
1998-02-04 1998-03-31 1998-05-19 1998-07-01 1998-08-18 1998-09-29 1998-11-17 1998-12-22
1999-02-03 1999-03-30 1999-05-18 1999-06-30 1999-08-24 1999-10-05 1999-11-16 1999-12-21
""".split()

# Meetings in the archive that were unscheduled calls (not known in advance).
UNSCHEDULED = {
    "2001-01-03", "2001-04-18", "2001-09-17",
    "2007-08-10", "2007-08-16",
    "2008-01-21", "2008-03-10", "2008-10-07",
    "2010-05-09",
    "2019-10-04",
    "2020-03-03", "2020-03-15",
}

# Scheduled meetings that were pre-empted but were on the public calendar.
# The 17-18 March 2020 meeting was replaced by the 15 March emergency meeting.
SCHEDULED_BUT_REPLACED = {"2020-03-18"}

# Entries in the archive's future schedule that are not on the Fed's published
# calendar (2027: Jan 26-27, Mar 16-17, Apr 27-28, Jun 8-9, Jul 27-28,
# Sep 14-15, Oct 26-27, Dec 7-8).
ARCHIVE_ERRORS = {"2027-11-17"}


def meetings_from_archive(comm: pd.DataFrame) -> pd.DatetimeIndex:
    """Scheduled meeting dates from the communications archive."""
    d = comm.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    statements = set(d.loc[d["Type"] == "Statement", "Date"])
    minutes = set(d.loc[d["Type"] == "Minute", "Date"])
    future = set(d.loc[d["Type"] == "Scheduled Meeting", "Date"])
    held = {x for x in statements & minutes if x.strftime("%Y-%m-%d") not in UNSCHEDULED}
    future = {x for x in future if x.strftime("%Y-%m-%d") not in ARCHIVE_ERRORS}
    extra = {pd.Timestamp(x) for x in SCHEDULED_BUT_REPLACED}
    return pd.DatetimeIndex(sorted(held | future | extra))


def build_calendar(comm: pd.DataFrame) -> pd.DataFrame:
    pre = pd.DatetimeIndex(pd.to_datetime(PRE_2000))
    post = meetings_from_archive(comm)
    dates = pre.union(post)
    src = ["Fed historical materials" if x.year < 2000 else "FOMC communications archive" for x in dates]
    return pd.DataFrame({"date": dates, "source": src})


def load_calendar(path: Path) -> pd.DatetimeIndex:
    cal = pd.read_csv(path, parse_dates=["date"])
    return pd.DatetimeIndex(cal["date"])


#: From the March 2017 meeting on, a change in the target range takes effect
#: the day after the decision (FRED records it on that day). Before that it was
#: recorded on the decision day. Derived from the target series itself; a unit
#: test re-checks every change against this convention.
NEXT_DAY_EFFECTIVE_FROM = pd.Timestamp("2017-03-15")


def effective_dates(meetings: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Date on which a decision at each scheduled meeting shows up in the target series."""
    shift = (meetings >= NEXT_DAY_EFFECTIVE_FROM).astype(int)
    return pd.DatetimeIndex(meetings + pd.to_timedelta(shift, unit="D"))


def meeting_next_month(index: pd.DatetimeIndex, meetings: pd.DatetimeIndex) -> pd.Series:
    """1 if a scheduled meeting's decision takes effect in month t+1.

    Uses effective dates, so a decision announced on 31 July 2019 (effective
    1 August) counts as an August meeting, matching when the target moved.
    """
    months = set(effective_dates(meetings).to_period("M"))
    nxt = (index + pd.DateOffset(months=1)).to_period("M")
    return pd.Series([float(m in months) for m in nxt], index=index)
