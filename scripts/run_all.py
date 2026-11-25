"""Reproduce every table, figure and number in the README.

    python scripts/run_all.py

Writes outputs/tables/*.csv, outputs/figures/*.png and outputs/results.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fomc import calendar as CAL  # noqa: E402
from fomc import data as D  # noqa: E402
from fomc import direction as M  # noqa: E402
from fomc import plots as P  # noqa: E402
from fomc import rules as R  # noqa: E402
from fomc import text as T  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs"
TAB, FIG = OUT / "tables", OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def main() -> None:
    # ------------------------------------------------------------------ data
    daily = D.load_daily()
    asof = daily.index.max()
    p = D.build_panel(D.load_monthly(), D.load_quarterly(), daily)
    meetings = CAL.load_calendar(D.RAW / "fomc_scheduled_meetings.csv")
    st = T.statement_tone(pd.read_csv(D.RAW / "fomc_statements.csv"))
    st.to_csv(TAB / "statement_tone.csv", index=False)
    f = D.build_features(p, asof=asof, meetings=meetings, tone=T.monthly_tone(st, p.index))
    f.to_csv(TAB / "features.csv")

    # ------------------------------------------------------------ rules
    q = R.to_quarterly(p.dropna(subset=["ffr"]))
    ests = [
        R.estimate_inertial_ols(q.loc[:"2008-07-01"], "1990Q1-2008Q3"),
        R.estimate_inertial_ols(q, f"1990Q1-{q.index[-1].year}Q{q.index[-1].quarter}"),
        R.estimate_inertial_tobit(q, f"1990Q1-{q.index[-1].year}Q{q.index[-1].quarter}"),
    ]
    est_tab = pd.DataFrame([e.as_row() for e in ests])
    est_tab.to_csv(TAB / "reaction_function_estimates.csv", index=False)
    tobit = ests[-1]

    m = p.dropna(subset=["ffr"]).copy()
    m["taylor_1993"] = R.taylor_1993(m["core_pce_yoy"], m["u_gap"], r_star=2.0)
    m["balanced_approach"] = R.balanced_approach(m["core_pce_yoy"], m["u_gap"], r_star=2.0)
    m["estimated_rule"] = tobit.alpha + tobit.beta_pi * m["core_pce_yoy"] + tobit.beta_u * m["u_gap"]
    m[["ffr", "taylor_1993", "balanced_approach", "estimated_rule"]].to_csv(TAB / "rules_vs_actual.csv")
    P.rules_vs_actual(m, FIG / "fig1_rules_vs_actual.png")

    roll = R.rolling_static(p.dropna(subset=["ffr"]))
    roll.to_csv(TAB / "rolling_coefficients.csv")
    P.rolling_coefficients(roll, FIG / "fig2_rolling_inflation_response.png")

    last = p.dropna(subset=["core_pce_yoy", "u_gap"]).iloc[-1]
    target_mid = float(p["target_mid"].dropna().iloc[-1])
    presc = []
    for r_star in (2.0, 1.0, 0.5):
        presc.append({"rule": "Taylor (1993)", "r_star": r_star,
                      "prescription": R.taylor_1993(last.core_pce_yoy, last.u_gap, r_star)})
        presc.append({"rule": "Balanced approach", "r_star": r_star,
                      "prescription": R.balanced_approach(last.core_pce_yoy, last.u_gap, r_star)})
    presc.append({"rule": "Estimated rule, long-run target (Tobit)", "r_star": tobit.implied_r_star,
                  "prescription": tobit.alpha + tobit.beta_pi * last.core_pce_yoy + tobit.beta_u * last.u_gap})
    lr_target = presc[-1]["prescription"]
    presc.append({"rule": "Estimated rule, one quarter of partial adjustment", "r_star": tobit.implied_r_star,
                  "prescription": tobit.rho * target_mid + (1 - tobit.rho) * lr_target})
    presc = pd.DataFrame(presc)
    presc["vs_current_midpoint_pp"] = presc["prescription"] - target_mid
    presc.to_csv(TAB / "current_rule_prescriptions.csv", index=False)

    # ------------------------------------------------------- direction model
    bt = M.expanding_backtest(f)
    off = bt.at_elb == 0
    scores = pd.concat([M.score_table(bt), M.score_table(bt, off, "Off the lower bound")])
    scores.to_csv(TAB / "oos_scores.csv", index=False)
    dms = pd.concat([M.dm_table(bt).assign(sample="All test months"),
                     M.dm_table(bt, mask=off).assign(sample="Off the lower bound")])
    dms.to_csv(TAB / "diebold_mariano.csv", index=False)
    probs = bt.probs["Full model"].join(bt.y.rename("actual"))
    probs.to_csv(TAB / "oos_probabilities_full_model.csv")
    rel_h = M.reliability(bt, "Full model", "hike")
    rel_c = M.reliability(bt, "Full model", "cut")
    pd.concat([rel_h.assign(cls="hike"), rel_c.assign(cls="cut")]).to_csv(TAB / "calibration.csv", index=False)
    bt.chosen_C.to_csv(TAB / "chosen_penalty_by_refit.csv")

    P.oos_probabilities(bt.probs["Full model"], bt.y, bt.at_elb, f["meeting_next"], FIG / "fig3_oos_probabilities.png")
    P.skill_bars(scores[scores["sample"] == "Off the lower bound"], FIG / "fig4_skill_by_model.png")
    P.calibration(rel_h, rel_c, FIG / "fig5_calibration.png")

    # Text layer: same training data and test months, with and without tone.
    btt = M.expanding_backtest(f, M.TEXT_SPECS, start="2008-01-01", train_start="2000-01-01")
    off_t = btt.at_elb == 0
    text_scores = pd.concat([M.score_table(btt), M.score_table(btt, off_t, "Off the lower bound")])
    text_scores.to_csv(TAB / "text_layer_scores.csv", index=False)
    text_dm = M.dm_table(btt, against="Full model + statement tone", mask=off_t)
    text_dm.to_csv(TAB / "text_layer_dm.csv", index=False)

    # Live forecast for the next month (the December meeting when run in late November).
    live_row = f.index[-1]
    nowcasts = {}
    for name in ("Full model", "Market only", "Macro rule"):
        mdl, C = M.fit_final(f, M.MODEL_SPECS[name])
        nowcasts[name] = {**M.nowcast(mdl, f.loc[live_row], M.MODEL_SPECS[name]), "C": C}
    next_meeting = meetings[meetings > asof].min()
    P.nowcast_bar(nowcasts["Full model"],
                  f"Model odds for the {next_meeting:%B %-d, %Y} FOMC decision",
                  FIG / "fig6_next_meeting_odds.png")

    # ------------------------------------------------------------ results
    def score(model, sample):
        r = scores[(scores["model"] == model) & (scores["sample"] == sample)].iloc[0]
        return {k: r3(r[k]) if isinstance(r[k], (float, np.floating)) else int(r[k])
                for k in ("n", "brier", "brier_skill_vs_climatology", "log_loss", "accuracy", "moves_called", "moves")}

    def dm(comp, sample):
        r = dms[(dms["comparison"] == comp) & (dms["sample"] == sample)].iloc[0]
        return {"dm_stat": r3(r["dm_stat"]), "p_value": r3(r["p_value"])}

    results = {
        "data_as_of": str(asof.date()),
        "latest_inputs": {
            "core_pce_yoy": r3(last.core_pce_yoy), "core_pce_month": str(last.name.date()),
            "unemployment": r3(last.u), "nrou": r3(last.nrou), "u_gap": r3(last.u_gap),
            "target_range_midpoint": target_mid,
            "live_row": str(live_row.date()),
            "y3m": r3(p["y3m_eom"].dropna().iloc[-1]), "y2": r3(p["y2_eom"].dropna().iloc[-1]),
            "y10": r3(p["y10_eom"].dropna().iloc[-1]),
            "statement_tone_latest": r3(st["tone"].iloc[-1]),
        },
        "reaction_function": [{k: (r3(v) if isinstance(v, float) else v) for k, v in e.as_row().items()} for e in ests],
        "prescriptions": [{k: (r3(v) if isinstance(v, float) else v) for k, v in row.items()}
                          for row in presc.to_dict("records")],
        "oos": {
            "test_period": f"{bt.y.index[0]:%Y-%m} to {bt.y.index[-1]:%Y-%m}",
            "all": {mname: score(mname, "All test months") for mname in M.MODEL_SPECS},
            "off_elb": {mname: score(mname, "Off the lower bound") for mname in M.MODEL_SPECS},
            "dm_full_vs_market": {"all": dm("Full model vs Market only", "All test months"),
                                  "off_elb": dm("Full model vs Market only", "Off the lower bound")},
        },
        "text_layer": {
            "test_period": f"{btt.y.index[0]:%Y-%m} to {btt.y.index[-1]:%Y-%m}",
            "scores_off_elb": text_scores[text_scores["sample"] == "Off the lower bound"]
            .set_index("model")[["brier", "brier_skill_vs_climatology", "log_loss"]].round(3).to_dict("index"),
            "dm_tone_vs_no_tone_off_elb": {"dm_stat": r3(text_dm.iloc[-1]["dm_stat"]),
                                           "p_value": r3(text_dm.iloc[-1]["p_value"])},
        },
        "nowcast": {"next_meeting": str(next_meeting.date()),
                    "models": {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in nowcasts.items()}},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
