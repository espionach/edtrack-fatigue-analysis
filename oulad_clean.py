"""Load, clean and merge the OULAD tables.

Produces one row per student enrolment (code_module, code_presentation, id_student)
combining demographics, registration, VLE clickstream aggregates and assessment
aggregates. Raw CSVs are only read, never modified.

Usage:
    python oulad_clean.py                 # writes processed/*.csv
    from oulad_clean import build_all     # in a notebook: tables = build_all()

The raw OULAD CSVs are looked for in $OULAD_DATA_DIR, then ./data/, then next
to this file.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _find_data_dir():
    env = os.environ.get("OULAD_DATA_DIR")
    if env:
        return Path(env).expanduser()
    for cand in (HERE / "data", HERE):
        if (cand / "studentInfo.csv").exists():
            return cand
    return HERE / "data"


DATA_DIR = _find_data_dir()
OUT_DIR = HERE / "processed"
KEY = ["code_module", "code_presentation", "id_student"]
COURSE = ["code_module", "code_presentation"]


def load_courses():
    return pd.read_csv(DATA_DIR / "courses.csv")


def load_assessments(courses):
    a = pd.read_csv(DATA_DIR / "assessments.csv")
    # 11 exams have no date; OULAD docs say the final exam is at the end of the presentation.
    a = a.merge(courses, on=COURSE, how="left")
    a["date_imputed"] = a["date"].isna()
    a["date"] = a["date"].fillna(a["module_presentation_length"])
    return a.drop(columns="module_presentation_length")


def load_student_info():
    si = pd.read_csv(DATA_DIR / "studentInfo.csv")
    # One band is written "10-20" instead of "10-20%".
    si["imd_band"] = si["imd_band"].replace({"10-20": "10-20%"})
    imd_order = [f"{i}-{i + 10}%" for i in range(0, 100, 10)]
    si["imd_band"] = pd.Categorical(si["imd_band"], categories=imd_order, ordered=True)
    si["age_band"] = pd.Categorical(si["age_band"], categories=["0-35", "35-55", "55<="], ordered=True)
    si["highest_education"] = pd.Categorical(
        si["highest_education"],
        categories=["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                    "HE Qualification", "Post Graduate Qualification"],
        ordered=True,
    )
    for c in ["gender", "region", "disability", "final_result", "code_module", "code_presentation"]:
        si[c] = si[c].astype("category")
    return si


def load_registration():
    r = pd.read_csv(DATA_DIR / "studentRegistration.csv")
    r["unregistered"] = r["date_unregistration"].notna()
    return r


def load_student_assessment(assessments):
    sa = pd.read_csv(DATA_DIR / "studentAssessment.csv")
    sa = sa.merge(assessments, on="id_assessment", how="left", validate="many_to_one")
    sa["is_banked"] = sa["is_banked"].astype(bool)
    # Banked results carry a submission date from an earlier presentation, so lateness is undefined.
    sa["days_early"] = np.where(sa["is_banked"], np.nan, sa["date"] - sa["date_submitted"])
    sa["is_late"] = sa["days_early"] < 0
    return sa


def load_student_vle():
    dtypes = {"code_module": "category", "code_presentation": "category",
              "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"}
    sv = pd.read_csv(DATA_DIR / "studentVle.csv", dtype=dtypes)
    vle = pd.read_csv(DATA_DIR / "vle.csv", usecols=["id_site", "activity_type"])
    sv["activity_type"] = sv["id_site"].map(vle.set_index("id_site")["activity_type"]).astype("category")
    return sv


def vle_features(sv):
    """Per-enrolment clickstream aggregates. Repeated (site, date) rows are separate
    interaction records, so they are summed rather than de-duplicated."""
    g = sv.groupby(KEY, observed=True)
    feats = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "n_active_days": g["date"].nunique(),
        "n_sites_visited": g["id_site"].nunique(),
        "first_active_day": g["date"].min(),
        "last_active_day": g["date"].max(),
    })
    feats["clicks_pre_start"] = sv[sv["date"] < 0].groupby(KEY, observed=True)["sum_click"].sum()
    feats["clicks_first_4_weeks"] = sv[sv["date"].between(0, 27)].groupby(KEY, observed=True)["sum_click"].sum()
    by_type = sv.pivot_table(index=KEY, columns="activity_type", values="sum_click",
                             aggfunc="sum", observed=True)
    by_type.columns = [f"clicks_{c}" for c in by_type.columns]
    feats = feats.join(by_type)
    click_cols = [c for c in feats.columns if c.startswith("clicks_")]
    feats[click_cols] = feats[click_cols].fillna(0).astype("int64")
    feats["clicks_per_active_day"] = feats["total_clicks"] / feats["n_active_days"]
    return feats.reset_index()


def assessment_features(sa, assessments):
    """Per-enrolment assessment aggregates. Coursework = TMA + CMA; exams kept separate."""
    cw = sa[sa["assessment_type"] != "Exam"].copy()
    cw["w_score"] = cw["score"] * cw["weight"]
    cw["w_scored"] = cw["weight"].where(cw["score"].notna())
    g = cw.groupby(KEY)
    feats = pd.DataFrame({
        "n_coursework_submitted": g["id_assessment"].count(),
        "mean_coursework_score": g["score"].mean(),
        "mean_tma_score": cw[cw["assessment_type"] == "TMA"].groupby(KEY)["score"].mean(),
        "mean_cma_score": cw[cw["assessment_type"] == "CMA"].groupby(KEY)["score"].mean(),
        "n_late_submissions": g["is_late"].sum(),
        "mean_days_early": g["days_early"].mean(),
        "n_banked": g["is_banked"].sum(),
        "_w_score": g["w_score"].sum(),
        "_w_scored": g["w_scored"].sum(),
    })
    # Weighted mean over submitted, weighted items. NaN where all weights are 0 (module GGG).
    feats["weighted_coursework_score"] = feats["_w_score"] / feats["_w_scored"].replace(0, np.nan)
    feats = feats.drop(columns=["_w_score", "_w_scored"])

    ex = sa[sa["assessment_type"] == "Exam"].groupby(KEY)["score"]
    feats = feats.join(pd.DataFrame({"exam_score": ex.mean(), "n_exams_taken": ex.count()}), how="outer")

    n_due = (assessments[assessments["assessment_type"] != "Exam"]
             .groupby(COURSE).size().rename("n_coursework_in_course").reset_index())
    return feats.reset_index().merge(n_due, on=COURSE, how="left")


def build_student_features(si, reg, courses, vle_f, asmt_f, assessments):
    df = (si.merge(reg, on=KEY, how="left", validate="one_to_one")
            .merge(courses, on=COURSE, how="left", validate="many_to_one"))
    for c in ["code_module", "code_presentation"]:
        df[c] = df[c].astype(str)
    df = df.merge(vle_f, on=KEY, how="left", validate="one_to_one")
    df = df.merge(asmt_f.drop(columns="n_coursework_in_course"), on=KEY, how="left", validate="one_to_one")

    df["has_vle_activity"] = df["total_clicks"].notna()
    zero_fill = ([c for c in df.columns if c.startswith("clicks_") and c != "clicks_per_active_day"]
                 + ["total_clicks", "n_active_days", "n_sites_visited",
                    "n_coursework_submitted", "n_late_submissions", "n_banked", "n_exams_taken"])
    df[zero_fill] = df[zero_fill].fillna(0).astype("int64")

    n_due = (assessments[assessments["assessment_type"] != "Exam"]
             .groupby(COURSE).size().rename("n_coursework_in_course").reset_index())
    df = df.merge(n_due, on=COURSE, how="left")
    df["coursework_submission_rate"] = df["n_coursework_submitted"] / df["n_coursework_in_course"]
    df["passed"] = df["final_result"].isin(["Pass", "Distinction"])
    return df


def build_all(write=False, sv=None):
    """Build the cleaned tables. Pass an already-loaded clickstream as `sv` to
    avoid reading the 450 MB studentVle.csv twice."""
    courses = load_courses()
    assessments = load_assessments(courses)
    si = load_student_info()
    reg = load_registration()
    sa = load_student_assessment(assessments)
    if sv is None:
        sv = load_student_vle()

    vle_f = vle_features(sv)
    asmt_f = assessment_features(sa, assessments)
    features = build_student_features(si, reg, courses, vle_f, asmt_f, assessments)

    tables = {"student_features": features, "student_assessments": sa, "assessments": assessments}
    if write:
        OUT_DIR.mkdir(exist_ok=True)
        for name, t in tables.items():
            t.to_csv(OUT_DIR / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = build_all(write=True)
    f = t["student_features"]
    print(f"student_features: {f.shape[0]:,} rows x {f.shape[1]} cols -> {OUT_DIR}")
