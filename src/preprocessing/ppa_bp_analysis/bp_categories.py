"""
Blood-pressure category from the project's NHBPEP tables (blood_pressure_children.pdf), and a
comparison with the `ppa` stored in the file.

Rule (thresholds come straight from the table: P90, P95, P99 for age, sex and height percentile):
    NORMAL            BP <  P90
    PRE-HIPERTENSAO   P90 <= BP < P95
    HAS-1             P95 <= BP < P99
    HAS-2             BP >= P99
applied to systolic (PAS) and diastolic (PAD) separately; the overall category is the higher of the
two, and the label names the component that drives it (PAS on ties, as in the stored `ppa`).

Choices (all tested against the stored ppa, see compare_with_ppa):
    age     rounded to the nearest year, must be 1 to <18 (the table covers 1-17)
    height  the BP thresholds are linearly interpolated between the 5/10/25/50/75/90/95th height
            percentile columns (clamped at 5th and 95th); percentile = `hapct` from growth_scores
    not used: the 120/80 mmHg rule for adolescents and the 'P99 + 5 mmHg' limit for stage 2 are not in
            the table, and using them agrees less with the stored ppa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.ppa_bp_analysis.bp_tables import load_bp_tables, HEIGHT_PCT

LEVELS = ["NORMAL", "PRE-HIPERTENSAO", "HAS-1", "HAS-2"]


def _thresholds(T: pd.DataFrame, sex, age, hpct, kind):
    """Interpolated P90/P95/P99 thresholds for each row (arrays)."""
    out = {p: np.full(len(sex), np.nan) for p in (90, 95, 99)}
    sub = T.set_index(["sex", "age", "bp_pct"])
    cols = [f"{kind}_h{h}" for h in HEIGHT_PCT]
    for (s, a), idx in pd.Series(range(len(sex))).groupby([sex, age]).groups.items():
        if pd.isna(s) or pd.isna(a):
            continue
        h = hpct[list(idx)]
        for p in (90, 95, 99):
            ys = sub.loc[(int(s), int(a), p), cols].to_numpy(float)
            out[p][list(idx)] = np.interp(h, HEIGHT_PCT, ys)
    return out


def add_bp_category(df: pd.DataFrame, *, sex_col="sexo", age_col="age_years", hpct_col="hapct",
                    sbp_col="pa_sistolica", dbp_col="pa_diastolica") -> pd.DataFrame:
    T = load_bp_tables()
    out = df.copy()
    sex = out[sex_col].map({"M": 1, "F": 2, 1: 1, 2: 2}).to_numpy(float)
    age_raw = pd.to_numeric(out[age_col], errors="coerce")
    age = np.where((age_raw >= 1) & (age_raw < 18), np.minimum(np.round(age_raw), 17), np.nan)
    hp = pd.to_numeric(out[hpct_col], errors="coerce").to_numpy(float)
    sbp = pd.to_numeric(out[sbp_col], errors="coerce").to_numpy(float)
    dbp = pd.to_numeric(out[dbp_col], errors="coerce").to_numpy(float)

    status = np.select(
        [np.isnan(sbp) | np.isnan(dbp), np.isnan(sex), np.isnan(age), np.isnan(hp)],
        ["no_bp", "no_sex", "age_out_of_range", "no_height"], default="ok")
    ok = status == "ok"
    lv = {}
    for kind, val in (("sbp", sbp), ("dbp", dbp)):
        th = _thresholds(T, np.where(ok, sex, np.nan), np.where(ok, age, np.nan), np.where(ok, hp, 0.0), kind)
        l = np.where(val >= th[99], 3, np.where(val >= th[95], 2, np.where(val >= th[90], 1, 0)))
        lv[kind] = np.where(ok, l, -1)
        out[f"{kind}_p90"], out[f"{kind}_p95"], out[f"{kind}_p99"] = th[90], th[95], th[99]
    top = np.maximum(lv["sbp"], lv["dbp"])
    out["bp_status"] = status
    out["sbp_level"] = np.where(ok, lv["sbp"], np.nan)
    out["dbp_level"] = np.where(ok, lv["dbp"], np.nan)
    out["bp_level"] = np.where(ok, top, np.nan)
    out["bp_cat"] = pd.Series(np.where(ok, np.array(LEVELS + ["?"])[np.clip(top, 0, 4)], None), index=out.index)
    comp = np.where(lv["dbp"] > lv["sbp"], "PAD", "PAS")
    out["bp_label"] = np.where(ok & (top > 0), out["bp_cat"].astype(str) + " " + comp,
                               np.where(ok, "NORMAL", None))
    return out


def stored_level(ppa: pd.Series) -> pd.Series:
    """Level (0-3) of the stored ppa text; NaN for 'NAO CALCULADO' / missing."""
    p = ppa.astype("string").fillna("")
    lv = np.select([p.str.startswith("NORMAL").to_numpy(bool), p.str.startswith("PRE").to_numpy(bool),
                    p.str.startswith("HAS-1").to_numpy(bool), p.str.startswith("HAS-2").to_numpy(bool)],
                   [0, 1, 2, 3], default=-1).astype(float)
    return pd.Series(np.where(lv < 0, np.nan, lv), index=ppa.index)


def compare_with_ppa(df: pd.DataFrame) -> None:
    d = df.copy()
    d["ppa_level"] = stored_level(d["ppa"])
    both = d["ppa_level"].notna() & d["bp_level"].notna()
    print("Stored ppa:", d["ppa"].fillna("(missing)").value_counts().to_dict())
    print("Computed status:", d["bp_status"].value_counts().to_dict())
    print(f"\nRows with BOTH a stored and a computed category: {int(both.sum())}")
    x = d[both]
    print(f"Agreement: {(x.ppa_level == x.bp_level).mean():.1%}")
    ct = pd.crosstab(x.ppa_level.map(dict(enumerate(LEVELS))), x.bp_level.map(dict(enumerate(LEVELS))),
                     rownames=["stored"], colnames=["computed"]).reindex(index=LEVELS, columns=LEVELS, fill_value=0)
    print(ct.to_string())
    print(f"computed higher than stored: {int((x.bp_level > x.ppa_level).sum())}, "
          f"lower: {int((x.bp_level < x.ppa_level).sum())}")
    new = d["ppa_level"].isna() & d["bp_level"].notna()
    print(f"\nRows with NO usable stored ppa but now classified: {int(new.sum())}")
    print(d.loc[new, "bp_cat"].value_counts().to_dict())
