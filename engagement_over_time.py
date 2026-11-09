"""Engagement-over-time metrics for OULAD.

OULAD logs clicks per student per site per *day*, not per session, so "session"
here means an active day: session length = clicks per active day.
Accuracy = TMA/CMA assessment scores (0-100).

Outputs (processed/):
  weekly_engagement.csv        one row per enrolment x course week
  assessment_engagement.csv    one row per coursework submission, with engagement
                               in the 14 days before the deadline
figures/: engagement_by_week.png, score_vs_engagement.png, pass_by_early_activity.png
"""
from pathlib import Path

import numpy as np
import pandas as pd

from oulad_clean import (COURSE, KEY, OUT_DIR, build_all, load_student_vle)

WINDOW_DAYS = 14


def weekly_panel(sv, features):
    sv = sv.assign(week=sv["date"] // 7)
    g = sv.groupby(KEY + ["week"], observed=True)
    wk = pd.DataFrame({"clicks": g["sum_click"].sum(), "active_days": g["date"].nunique()}).reset_index()
    for c in COURSE:
        wk[c] = wk[c].astype(str)

    # Full grid (zeros included) from week -2 to the last week of each presentation.
    base = features[KEY + ["final_result", "module_presentation_length", "date_unregistration"]].copy()
    base["n_weeks"] = base["module_presentation_length"] // 7 + 1
    grid = base.loc[base.index.repeat(base["n_weeks"] + 2)].copy()
    grid["week"] = grid.groupby(KEY).cumcount() - 2
    panel = grid.merge(wk, on=KEY + ["week"], how="left")
    panel[["clicks", "active_days"]] = panel[["clicks", "active_days"]].fillna(0).astype(int)
    panel["active"] = panel["active_days"] > 0
    panel["clicks_per_active_day"] = panel["clicks"] / panel["active_days"].replace(0, np.nan)
    panel["unregistered_by_week"] = panel["date_unregistration"] < panel["week"] * 7
    return panel.drop(columns=["module_presentation_length", "date_unregistration", "n_weeks"])


def assessment_engagement(sv, sa):
    """Clicks and active days in the WINDOW_DAYS before each coursework deadline."""
    cw = sa[(sa["assessment_type"] != "Exam") & ~sa["is_banked"] & sa["score"].notna()].copy()
    for c in COURSE:
        cw[c] = cw[c].astype(str)
    daily = (sv.groupby(KEY + ["date"], observed=True)["sum_click"].sum().reset_index())
    for c in COURSE:
        daily[c] = daily[c].astype(str)
    m = cw[KEY + ["id_assessment", "date"]].rename(columns={"date": "deadline"}).merge(daily, on=KEY)
    m = m[(m["date"] < m["deadline"]) & (m["date"] >= m["deadline"] - WINDOW_DAYS)]
    agg = m.groupby(["id_assessment", "id_student"]).agg(
        window_clicks=("sum_click", "sum"), window_active_days=("date", "nunique"))
    out = cw.merge(agg, on=["id_assessment", "id_student"], how="left")
    out[["window_clicks", "window_active_days"]] = out[["window_clicks", "window_active_days"]].fillna(0)
    out["window_clicks_per_active_day"] = out["window_clicks"] / out["window_active_days"].replace(0, np.nan)
    return out


def within_student_corr(df, x, y="score"):
    """Spearman correlation after removing each enrolment's own mean (rank-transformed
    within assessment first so module difficulty doesn't leak in)."""
    d = df.dropna(subset=[x, y]).copy()
    for c in (x, y):
        d[c + "_r"] = d.groupby("id_assessment")[c].rank(pct=True)
        d[c + "_w"] = d[c + "_r"] - d.groupby(KEY)[c + "_r"].transform("mean")
    d = d[d.groupby(KEY)[y].transform("size") >= 3]
    return d[x + "_w"].corr(d[y + "_w"])


def auc(score, label):
    r = score.rank()
    n1 = label.sum(); n0 = len(label) - n1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def drop_off(panel, features):
    res = {}
    # 1. Share of each outcome group active each week.
    active = (panel.groupby(["final_result", "week"])["active"].mean().unstack(0))
    res["active_share"] = active

    # 2. Individual drop-off week = last week with any activity, for students whose
    #    activity stops before the final four weeks.
    last = panel[panel["active"]].groupby(KEY)["week"].max().rename("last_active_week")
    f = features.set_index(KEY).join(last)
    f["n_weeks"] = f["module_presentation_length"] // 7
    f["dropped_off"] = f["last_active_week"].isna() | (f["last_active_week"] < f["n_weeks"] - 4)
    res["dropoff_by_result"] = f.groupby("final_result")["dropped_off"].mean()
    dropped = f[f["dropped_off"] & f["last_active_week"].notna()]
    res["last_week_hist"] = dropped.groupby("final_result")["last_active_week"].describe()
    # Lag between going quiet and formally withdrawing.
    w = f[(f["final_result"] == "Withdrawn") & f["date_unregistration"].notna() & f["last_active_week"].notna()]
    res["withdraw_lag_days"] = (w["date_unregistration"] - w["last_active_week"] * 7).describe()

    # 3. Week at which cumulative engagement separates eventual pass vs. not
    #    (AUC of cumulative clicks / active days up to week w, among students still registered).
    p = panel.sort_values(KEY + ["week"]).copy()
    p["cum_clicks"] = p.groupby(KEY)["clicks"].cumsum()
    p["cum_active_days"] = p.groupby(KEY)["active_days"].cumsum()
    p["passed"] = p["final_result"].isin(["Pass", "Distinction"])
    rows = []
    for wk, g in p[p["week"].between(-2, 38) & ~p["unregistered_by_week"]].groupby("week"):
        rows.append({"week": wk, "n": len(g),
                     "auc_cum_clicks": auc(g["cum_clicks"], g["passed"]),
                     "auc_cum_active_days": auc(g["cum_active_days"], g["passed"])})
    res["auc_by_week"] = pd.DataFrame(rows).set_index("week")

    # 4. Pass rate by early engagement (first 4 weeks, active days).
    early = panel[panel["week"].between(0, 3)].groupby(KEY)["active_days"].sum().rename("early_active_days")
    f = f.join(early)
    f["passed"] = f["final_result"].isin(["Pass", "Distinction"])
    res["pass_by_early_days"] = f.groupby("early_active_days")["passed"].agg(["mean", "size"])

    # 5. Course-wide weekly decline: share of all still-registered students active.
    reg = panel[~panel["unregistered_by_week"]]
    res["active_share_registered"] = reg.groupby("week")["active"].mean()
    return res, f


# Categorical slots 1-4 of the reference palette, in fixed order.
RESULT_COLORS = {"Distinction": "#2a78d6", "Pass": "#eb6834", "Fail": "#1baf7a", "Withdrawn": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# Weeks where every outcome group dips together (term breaks), not individual drop-off.
BREAKS = [(9.5, 11.5), (23.5, 26.5)]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def _mpl():
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


def plot_engagement_by_week(do):
    """Weekly share of students active, by final result."""
    matplotlib, plt = _mpl()
    share = do["active_share"].loc[-2:38]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    for a, b in BREAKS:
        ax.axvspan(a, b, color="#f0efec", zorder=0)
    ax.text(10.5, 1.02, "term break", ha="center", fontsize=8, color=INK2)
    ax.text(25, 1.02, "term break", ha="center", fontsize=8, color=INK2)
    for r, c in RESULT_COLORS.items():
        ax.plot(share.index, share[r], color=c, lw=2, label=r)
        nudge = {"Fail": 0.025, "Withdrawn": -0.005}.get(r, 0)
        ax.text(38.6, share[r].iloc[-1] + nudge, r, color=INK, fontsize=9, va="center")
    w0, w8 = share.loc[0, "Withdrawn"], share.loc[8, "Withdrawn"]
    ax.annotate(f"Withdrawn students: active share halves\nbetween week 0 ({w0:.0%}) and week 8 ({w8:.0%})",
                xy=(8, w8), xytext=(-1.5, 0.05), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(0, 1.06); ax.set_xlim(-2, 41)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9)
    _style(ax, "Share of students active on the VLE each week, by final result",
           "Course week (0 = module start)", "Students active that week")
    fig.tight_layout()
    return fig


def plot_score_vs_engagement(ae):
    """Score vs. active days and vs. clicks per active day (separate panels)."""
    matplotlib, plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    d = ae.groupby(ae["window_active_days"].clip(upper=14))["score"].mean()
    a1.plot(d.index, d.values, color=RESULT_COLORS["Distinction"], lw=2, marker="o", ms=4)
    a1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    _style(a1, "More active days before a deadline → higher score",
           f"Active days in the {WINDOW_DAYS} days before the deadline", "Mean score")
    q = pd.qcut(ae["window_clicks_per_active_day"], 10, duplicates="drop")
    d2 = ae.groupby(q, observed=True).agg(x=("window_clicks_per_active_day", "median"), y=("score", "mean"))
    a2.plot(d2["x"], d2["y"], color=RESULT_COLORS["Distinction"], lw=2, marker="o", ms=4)
    a2.set_xscale("log")
    a2.set_xticks([3, 5, 10, 20, 50, 100]); a2.set_xticklabels(["3", "5", "10", "20", "50", "100"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(a2, "Longer sessions help less, flat past ~20 clicks/day",
           "Clicks per active day (decile median, log scale)", "Mean score")
    for a in (a1, a2):
        a.set_ylim(68, 86)
    fig.tight_layout()
    return fig


def plot_pass_by_early_activity(do):
    """Pass rate by active days in the first four weeks."""
    matplotlib, plt = _mpl()
    p = do["pass_by_early_days"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.bar(p.index, p["mean"], color=RESULT_COLORS["Distinction"], width=0.8)
    for x in (0, 28):
        ax.text(x, p.loc[x, "mean"] + 0.015, f"{p.loc[x, 'mean']:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Pass rate by active days in weeks 0–3",
           "Active days in the first four weeks (max 28)", "Passed or Distinction")
    fig.tight_layout()
    return fig


def make_figures(ae, do, fig_dir):
    _, plt = _mpl()
    fig_dir.mkdir(exist_ok=True)
    for name, fig in [("engagement_by_week", plot_engagement_by_week(do)),
                      ("score_vs_engagement", plot_score_vs_engagement(ae)),
                      ("pass_by_early_activity", plot_pass_by_early_activity(do))]:
        fig.savefig(fig_dir / f"{name}.png")
        plt.close(fig)


def main():
    sv = load_student_vle()
    tables = build_all(sv=sv)
    features, sa = tables["student_features"], tables["student_assessments"]

    panel = weekly_panel(sv, features)
    ae = assessment_engagement(sv, sa)
    do, enrol = drop_off(panel, features)

    OUT_DIR.mkdir(exist_ok=True)
    panel.to_csv(OUT_DIR / "weekly_engagement.csv", index=False)
    ae.to_csv(OUT_DIR / "assessment_engagement.csv", index=False)
    make_figures(ae, do, OUT_DIR.parent / "figures")
    return panel, ae, do, enrol


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
