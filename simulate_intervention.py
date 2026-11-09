"""
SIMULATED, ILLUSTRATIVE ONLY -- NOT EVIDENCE.

Builds a lightweight "after" dataset for learners who receive three drop-off
interventions, and charts it against the "before" baseline:

  C2 "One small thing"  - one 8-minute first task in weeks 0-3 (aimed at Jordan,
                          the early withdrawer)
  C6 "Deadline runway"  - 15 min/day over the 14 days before each assessment
                          (aimed at Sam, the gradual fader)
  C7 "Welcome back"     - one task after each term break, backlog hidden
                          (aimed at Sam)

Baseline
--------
The OULAD clickstream (studentVle.csv) is not in the project folder, so the
weekly "share active" curves are transcribed by eye from engagement_by_week.png
(the earlier analysis of the full OULAD data). Group sizes come from the real
studentInfo.csv; assessment weeks come from the real assessments.csv (median
week of each TMA across module presentations).

Model
-----
For each baseline outcome group g and week w >= 0:

    share_active(w) = S(w) * q(w)

  S(w)  share of learners still "engaged" (not yet disengaged for good).
        Non-increasing. Its week-to-week drop is the dropout hazard.
  q(w)  chance an engaged learner is active that week (captures term-break
        dips and seasonality). q has the same shape as the Pass curve.

Each simulated learner gets fixed random draws, so the "after" record is the
SAME learner with the intervention: a reached learner's dropout hazard is cut
and their in-week activity nudged up in the intervention windows. Learners who
are not reached keep their "before" record exactly.

All effect sizes live in ASSUMPTIONS below. Edit them and re-run.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "simulated"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/mnt/project-files/OULAD Learning Analytics Dataset")

# ---------------------------------------------------------------------------
# ASSUMPTIONS (adjustable). Modest by design.
# hazard_cut: relative reduction in the weekly dropout hazard while active.
# activity_lift: relative lift in the chance an engaged learner is active.
# reach: share of each targeted group who actually see and act on the screen.
# ---------------------------------------------------------------------------
ASSUMPTIONS = {
    "C2_one_small_thing": {
        "weeks": [0, 1, 2, 3],
        "targets": {"Withdrawn": 0.60, "Fail": 0.50},   # reach by group
        "hazard_cut": 0.40,
        "activity_lift": 0.10,
    },
    "C6_deadline_runway": {
        "weeks_before_each_assessment": 2,              # 14-day runway
        "targets": {"Withdrawn": 0.40, "Fail": 0.50},
        "hazard_cut": 0.25,
        "activity_lift": 0.08,
    },
    "C7_welcome_back": {
        "weeks": [10, 11, 12, 13, 24, 25, 26, 27],      # breaks + return week
        "targets": {"Withdrawn": 0.40, "Fail": 0.50,
                    "Pass": 0.30, "Distinction": 0.30},
        "hazard_cut": 0.35,
        # recover this share of the gap between the break-week activity rate
        # and the pre-break rate (week 9 / week 23)
        "break_dip_recovered": 0.30,
    },
    # Outcome projection: earlier analysis found ~2 pass-rate points per extra
    # active DAY in weeks 0-3. We only simulate active WEEKS, so we assume an
    # extra active week is worth this many active days.
    "pass_points_per_active_day": 2.0,
    "active_days_per_extra_active_week": 2.0,
    "seed": 7,
}

BREAKS = [(10, 11), (24, 26)]

# ---------------------------------------------------------------------------
# Baseline curves, % active, weeks -2..38, transcribed from engagement_by_week.png
# ---------------------------------------------------------------------------
WEEKS = np.arange(-2, 39)
BASE = {
    "Distinction": [68, 80, 90, 91, 91, 89, 90, 89, 88.5, 88, 89, 86, 77, 72, 82,
                    86.5, 86, 88, 86, 89, 88, 87, 88.5, 90.5, 87.5, 87.5, 84, 80,
                    79, 82.5, 84.5, 84, 84, 82.5, 81, 71, 59, 50, 45, 37, 29],
    "Pass": [60, 73, 86.5, 87, 90, 86, 86.5, 84, 85.5, 85.5, 80.5, 78, 66, 63, 76,
             80.5, 79, 80.5, 79.5, 81, 82, 79.5, 80, 81.5, 80, 80, 74, 69, 68.5,
             71, 73.5, 75, 73, 73.5, 74, 66, 52, 39, 34, 29, 21],
    "Fail": [43, 55, 71, 68, 72, 65, 62, 56, 56.5, 54, 47, 44, 34, 32.5, 39, 39.5,
             36.5, 37, 34, 34, 34, 30.5, 30, 30, 27, 26.5, 22.5, 18.5, 18.5, 20,
             19, 19.5, 17, 16.5, 16.5, 14, 11.5, 7.5, 7, 5.5, 3],
    "Withdrawn": [37, 44, 51.5, 48, 48, 42, 38.5, 33, 30, 28.5, 24, 22, 16, 14.5,
                  16.3, 16, 14.5, 13.2, 11.7, 10.5, 9.8, 8.5, 7.6, 6.6, 5.7, 4.8,
                  3.8, 2.9, 2.4, 2.3, 2.3, 1.9, 1.5, 1.0, 0.9, 0.6, 0.3, 0.2, 0.1,
                  0.1, 0.1],
}
BASE = {g: np.array(v) / 100 for g, v in BASE.items()}
GROUPS = ["Withdrawn", "Fail", "Pass", "Distinction"]


def assessment_weeks():
    a = pd.read_csv(DATA / "assessments.csv")
    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    t = a[a.assessment_type == "TMA"].dropna(subset=["date"]).sort_values("date")
    t["n"] = t.groupby(["code_module", "code_presentation"]).cumcount()
    return sorted((t.groupby("n")["date"].median() // 7).astype(int).tolist())


def group_sizes():
    s = pd.read_csv(DATA / "studentInfo.csv")
    return s["final_result"].value_counts().to_dict()


def survival_and_q(p):
    """Split a share-active curve (weeks >= 0) into S (engaged) and q."""
    w0 = WEEKS >= 0
    shape = BASE["Pass"] / BASE["Pass"][w0].max()
    q0 = p[WEEKS == 0][0] / shape[WEEKS == 0][0]
    S = np.ones_like(p)
    S[w0] = np.minimum.accumulate(np.clip(p[w0] / (q0 * shape[w0]), 0, 1))
    q = np.clip(p / np.maximum(S, 1e-9), 0, 1)  # reproduces p exactly
    return S, q


def after_curves(group, S, q, tma_weeks):
    """Per-week S and q for a learner REACHED by each intervention."""
    A = ASSUMPTIONS
    idx = {w: i for i, w in enumerate(WEEKS)}
    windows = {}
    c2 = A["C2_one_small_thing"]
    windows["C2"] = (set(c2["weeks"]), c2)
    c6 = A["C6_deadline_runway"]
    k = c6["weeks_before_each_assessment"]
    windows["C6"] = ({w - j for w in tma_weeks for j in range(1, k + 1)}, c6)
    c7 = A["C7_welcome_back"]
    windows["C7"] = (set(c7["weeks"]), c7)

    out = {}
    for name, (wks, cfg) in windows.items():
        if group not in cfg["targets"]:
            continue
        # hazard: h(w) = 1 - S(w)/S(w-1); cut it inside the window
        S_new = S.copy()
        q_new = q.copy()
        for i in range(1, len(WEEKS)):
            w = WEEKS[i]
            h = 0.0 if S[i - 1] <= 0 else 1 - S[i] / S[i - 1]
            if w in wks:
                h *= 1 - cfg["hazard_cut"]
            S_new[i] = S_new[i - 1] * (1 - h) if w >= 0 else S[i]
            if w in wks and "activity_lift" in cfg:
                q_new[i] = min(1.0, q[i] * (1 + cfg["activity_lift"]))
        if name == "C7":
            for b0, b1 in BREAKS:
                ref = q[idx[b0 - 1]]
                for w in range(b0, b1 + 2):
                    i = idx[w]
                    if q[i] < ref:
                        q_new[i] = q[i] + cfg["break_dip_recovered"] * (ref - q[i])
        out[name] = (S_new, q_new, cfg["targets"][group])
    return out


def simulate():
    rng = np.random.default_rng(ASSUMPTIONS["seed"])
    tma = assessment_weeks()
    sizes = group_sizes()
    learners, weekly = [], []
    sid = 0
    for g in GROUPS:
        n = sizes[g]
        S, q = survival_and_q(BASE[g])
        effects = after_curves(g, S, q, tma)
        u = rng.random(n)                      # engagement draw (fixed)
        v = rng.random((n, len(WEEKS)))        # weekly activity draws (fixed)
        before = (u[:, None] < S[None, :]) & (v < q[None, :])

        # Each learner is independently reached (or not) by each intervention.
        # Combine effects multiplicatively on hazard, take the max q lift.
        S_after = np.tile(S, (n, 1))
        q_after = np.tile(q, (n, 1))
        reached = {}
        for name, (S_new, q_new, reach) in effects.items():
            r = rng.random(n) < reach
            reached[name] = r
            ratio = np.divide(S_new, S, out=np.ones_like(S), where=S > 0)
            S_after[r] *= ratio
            q_after[r] = np.maximum(q_after[r], q_new)
        S_after = np.clip(S_after, 0, 1)
        after = (u[:, None] < S_after) & (v < q_after)

        def last_active(m):
            any_ = m[:, WEEKS >= 0].any(axis=1)
            last = np.where(m[:, ::-1].argmax(axis=1) >= 0,
                            WEEKS[::-1][m[:, ::-1].argmax(axis=1)], np.nan)
            return np.where(any_, last, np.nan)

        early = (WEEKS >= 0) & (WEEKS <= 3)
        df = pd.DataFrame({
            "student_id": np.arange(sid, sid + n),
            "baseline_result": g,
            **{f"reached_{k}": reached.get(k, np.zeros(n, bool))
               for k in ["C2", "C6", "C7"]},
            "active_weeks_0_3_before": before[:, early].sum(1),
            "active_weeks_0_3_after": after[:, early].sum(1),
            "active_weeks_total_before": before[:, WEEKS >= 0].sum(1),
            "active_weeks_total_after": after[:, WEEKS >= 0].sum(1),
            "last_active_week_before": last_active(before),
            "last_active_week_after": last_active(after),
        })
        learners.append(df)
        any_r = np.zeros(n, bool)
        for r in reached.values():
            any_r |= r
        for cohort, mask in [("all", np.ones(n, bool)), ("reached", any_r)]:
            for cond, m in [("before", before), ("after", after)]:
                weekly.append(pd.DataFrame({
                    "baseline_result": g, "cohort": cohort, "condition": cond,
                    "week": WEEKS, "share_active": m[mask].mean(0),
                    "n": int(mask.sum())}))
        sid += n
    return pd.concat(learners), pd.concat(weekly), tma


def summarise(learners, weekly):
    A = ASSUMPTIONS
    rows = []
    for g in GROUPS:
        d_all = learners[learners.baseline_result == g]
        reached_any = d_all[["reached_C2", "reached_C6", "reached_C7"]].any(axis=1)
        d = d_all[reached_any]
        if d.empty:
            continue
        wk = weekly[(weekly.baseline_result == g)
                    & (weekly.cohort == "reached")].pivot(
            index="week", columns="condition", values="share_active")
        extra_wk = (d.active_weeks_0_3_after - d.active_weeks_0_3_before).mean()
        rows.append({
            "baseline_result": g,
            "n_group": len(d_all),
            "n_reached": len(d),
            "share_active_all_wk8_before": weekly.query(
                "baseline_result == @g and cohort == 'all' and week == 8 "
                "and condition == 'before'").share_active.iloc[0],
            "share_active_all_wk8_after": weekly.query(
                "baseline_result == @g and cohort == 'all' and week == 8 "
                "and condition == 'after'").share_active.iloc[0],
            "active_wk0_before": wk.loc[0, "before"],
            "active_wk8_before": wk.loc[8, "before"],
            "active_wk8_after": wk.loc[8, "after"],
            "active_wk14_before": wk.loc[14, "before"],
            "active_wk14_after": wk.loc[14, "after"],
            "median_last_active_before": d.last_active_week_before.median(),
            "median_last_active_after": d.last_active_week_after.median(),
            "extra_active_weeks_0_3": extra_wk,
            "projected_pass_rate_points": extra_wk
            * A["active_days_per_extra_active_week"]
            * A["pass_points_per_active_day"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
BEFORE_C = "#8a8985"
AFTER_C = "#2a78d6"
INK, INK2 = "#0b0b0b", "#52514e"


def chart(weekly, tma, summary, path):
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    panels = [
        ("Withdrawn", "Would-be withdrawers (\u201cJordan\u201d)"),
        ("Fail", "Would-be fails (\u201cSam\u201d)"),
    ]
    runway = sorted({w - j for w in tma for j in (1, 2)})
    for ax, (g, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        d = weekly[(weekly.baseline_result == g)
                   & (weekly.cohort == "reached")].pivot(
            index="week", columns="condition", values="share_active") * 100
        for b0, b1 in BREAKS:
            ax.axvspan(b0 - 0.5, b1 + 0.5, color="#efeeea", zorder=0)
            ax.text((b0 + b1) / 2, 97, "term break\n(C7 window)", ha="center",
                    va="top", fontsize=8, color=INK2)
        ax.axvspan(-0.5, 3.5, color=AFTER_C, alpha=0.07, zorder=0)
        ax.text(1.5, 97, "C2 window", ha="center", va="top", fontsize=8,
                color=INK2)
        ax.plot(runway, [1.5] * len(runway), "|", color=INK2, ms=7, ls="none",
                label="C6 runway weeks (2 weeks before each assessment)")
        ax.fill_between(d.index, d["before"], d["after"], color=AFTER_C,
                        alpha=0.15, lw=0)
        ax.plot(d.index, d["before"], color=BEFORE_C, lw=2, ls="--",
                label="Before (baseline)")
        ax.plot(d.index, d["after"], color=AFTER_C, lw=2,
                label="After (simulated intervention)")
        s = summary.set_index("baseline_result").loc[g]
        ax.annotate(
            f"Active in week 8: {s.active_wk8_before:.0%} \u2192 {s.active_wk8_after:.0%}\n"
            f"Active in week 14: {s.active_wk14_before:.0%} \u2192 {s.active_wk14_after:.0%}\n"
            f"Median last active week: {s.median_last_active_before:.0f} \u2192 "
            f"{s.median_last_active_after:.0f}",
            xy=(0.98, 0.72), xycoords="axes fraction", ha="right", va="top",
            fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#dddcd7"))
        ax.set_title(f"{title}: reached by \u22651 intervention "
                     f"(n={int(s.n_reached):,})", loc="left", fontsize=11,
                     color=INK)
        ax.set_xlim(-2, 38.5)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Course week (0 = module start)", color=INK2)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
        ax.grid(axis="x", visible=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("Share active that week", color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("SIMULATED: how C2, C6 and C7 could flatten the drop-off curve",
                 x=0.06, y=0.985, ha="left", fontsize=14, fontweight="bold",
                 color=INK)
    fig.text(0.06, 0.885,
             "Illustrative only, not evidence. The same simulated learners, before "
             "and after.\nBaseline rebuilt from the OULAD engagement curves; "
             "\u201cafter\u201d uses assumed, modest effect sizes "
             "(see assumptions.json).",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_all(weekly, path):
    """Supporting chart: all four groups, before vs after."""
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, g in zip(axes, GROUPS):
        d = weekly[(weekly.baseline_result == g)
                   & (weekly.cohort == "all")].pivot(
            index="week", columns="condition", values="share_active") * 100
        for b0, b1 in BREAKS:
            ax.axvspan(b0 - 0.5, b1 + 0.5, color="#efeeea", zorder=0)
        ax.plot(d.index, d["before"], color=BEFORE_C, lw=1.8, ls="--",
                label="Before")
        ax.plot(d.index, d["after"], color=AFTER_C, lw=1.8,
                label="After (simulated)")
        ax.set_title(g, loc="left")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
        ax.set_xlabel("Course week")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("SIMULATED, illustrative only: share active by week, whole "
                 "baseline group (reached and not reached)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    learners, weekly, tma = simulate()
    summary = summarise(learners, weekly)
    learners.to_csv(OUT / "simulated_learners_before_after.csv", index=False)
    weekly.to_csv(OUT / "simulated_weekly_share_active.csv", index=False)
    summary.to_csv(OUT / "simulated_summary.csv", index=False)
    (OUT / "assumptions.json").write_text(json.dumps(
        {"NOTE": "SIMULATED, illustrative only. Not evidence.",
         "assessment_weeks_used": tma, **ASSUMPTIONS}, indent=2))
    chart(weekly, tma, summary, OUT / "before_after_dropoff.png")
    chart_all(weekly, OUT / "before_after_all_groups.png")
    # sanity: simulated baseline vs transcribed baseline
    for g in GROUPS:
        sim = weekly[(weekly.baseline_result == g) & (weekly.cohort == "all")
                     & (weekly.condition == "before")]
        err = np.abs(sim.share_active.values - BASE[g]).max()
        print(f"{g:12s} max |sim before - transcribed| = {err:.3f}")
    print(f"TMA weeks: {tma}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
