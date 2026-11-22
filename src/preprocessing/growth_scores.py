"""
growth_scores.py
================
Python port of the two SAS programs used to compute growth z-scores, percentiles and
"biologically implausible value" (BIV) flags:

  * who-source-code.sas  + WHOref_d.sas7bdat  -> WHO Child Growth Standards, 0 to <2 years
  * cdc-source-code.sas  + CDCref_d.sas7bdat  -> CDC 2000 Growth Charts (+2022 extended BMI),
                                                 2 to <20 years

`add_growth_scores()` computes age in days, decides per child which reference to use, runs the
right calculation and returns the original table plus new columns.  Original columns are never
modified.

Which reference is used
-----------------------
    age_days (rounded) <  731                      -> WHO   (same rule as who-source-code.sas)
    age_days >= 731 and age < 240 months           -> CDC   (CDC code only scores >= 24 months)
    anything else (no dates, negative age, >=20 y, sex missing/indeterminate) -> not scored,
    with the reason written to `growth_status`.

    731 days = 24.02 months, so there is no gap and no overlap between the two references.

Typical use
-----------
    import pandas as pd
    from growth_scores import load_refs, add_growth_scores, growth_report

    refs = load_refs("WHOref_d.sas7bdat", "CDCref_d.sas7bdat")
    df   = pd.read_excel("UCMF5.xls")
    out  = add_growth_scores(df, refs)          # column names follow the data dictionary
    growth_report(out)

New columns
-----------
    age_days, age_months, age_years        age from DN and Atendimento
    growth_ref                             'WHO' or 'CDC' (NaN if not scored)
    growth_status                          'scored' or the reason the row was not scored
    bmi_calc                               weight / height(m)^2, recomputed from Peso and Altura
    haz hapct     height-for-age z / percentile
    waz wapct     weight-for-age
    bmiz bmipct   BMI-for-age       (WHO: from birth; CDC: extended BMI z for BMI >= P95)
    whz whpct     weight-for-length (WHO) / weight-for-stature (CDC, 77-122 cm)
    mod_haz mod_waz mod_bmiz mod_whz        CDC modified z-scores (NaN for WHO rows)
    biv_ht biv_wt biv_bmi biv_wh            -1 = implausibly low, 0 = ok, +1 = implausibly high
    biv_low biv_high                        1 if any of the four flags is -1 / +1
    CDC only: bmi50 bmi95 bmip95 bmi120 original_bmiz original_bmipct

BIV cutoffs (applied by each reference to its own scale; do NOT mix them)
    WHO (adjusted z):  height <-6 / >=+6 ; weight <-6 / >=+5 ; BMI and weight-for-length  <-5 / >+5
    CDC (modified z):  height <-5 / >+4  ; weight <-5 / >+8  ; BMI and weight-for-height  <-4 / >+8
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

DAYS_PER_MONTH = 30.4375      # CDC convention: days / 30.4375 = months
WHO_MAX_DAYS = 731            # WHO is used for round(age_days) < 731
CDC_MAX_MONTHS = 240          # CDC code excludes >= 240 months (20 years)

WHO_OUT = ["haz", "hapct", "waz", "wapct", "bmiz", "bmipct", "whz", "whpct",
           "biv_ht", "biv_wt", "biv_bmi", "biv_wh"]
CDC_OUT = WHO_OUT + ["mod_haz", "mod_waz", "mod_bmiz", "mod_whz",
                     "bmi50", "bmi95", "bmip95", "bmi120", "original_bmiz", "original_bmipct"]


# --------------------------------------------------------------------------- reference data
def load_refs(who_path: str, cdc_path: str) -> dict:
    """Read the two SAS reference files (pandas can read .sas7bdat directly)."""
    who = pd.read_sas(who_path, encoding="latin1")
    cdc = pd.read_sas(cdc_path, encoding="latin1")
    who.columns = [c.lower() for c in who.columns]
    cdc.columns = [c.lower() for c in cdc.columns]
    return {"who": who, "cdc": cdc}


# --------------------------------------------------------------------------- small helpers
def _arr(x):
    return np.asarray(x, dtype=float)


def _pct(z):
    """percentile = 100 * Phi(z)   (SAS: 100*probnorm(z))"""
    return 100.0 * norm.cdf(_arr(z))


def _sd_value(M, L, S, k):
    """Measurement value at z = k for LMS parameters (inverse of the LMS formula)."""
    M, L, S = _arr(M), _arr(L), _arr(S)
    with np.errstate(all="ignore"):
        return np.where(np.abs(L) < 1e-12, M * np.exp(S * k), M * (1 + L * S * k) ** (1.0 / L))


def _lms_z(x, L, M, S, small):
    """z = ((x/M)^L - 1) / (L*S); log form when |L| < `small`.
    WHO program uses the power form always (small ~ 0), CDC program switches below 0.01."""
    x, L, M, S = _arr(x), _arr(L), _arr(M), _arr(S)
    with np.errstate(all="ignore"):
        z = np.where(np.abs(L) >= small, ((x / M) ** L - 1) / (L * S), np.log(x / M) / S)
    return np.where(x > 0, z, np.nan)


def _cuts(v, lo, hi):
    """-1 if v < lo, +1 if v > hi, 0 otherwise, NaN stays NaN   (SAS macro _cuts)."""
    v = _arr(v)
    out = np.where(v > hi, 1.0, np.where(v < lo, -1.0, 0.0))
    return np.where(np.isnan(v), np.nan, out)


def _any_biv(df, cols):
    f = df[cols]
    low = (f == -1).any(axis=1).astype(float)
    high = (f == 1).any(axis=1).astype(float)
    none = f.isna().all(axis=1)
    low[none] = np.nan
    high[none] = np.nan
    return low, high


# --------------------------------------------------------------------------- WHO (0 to <2 y)
def _who_tail(z, x, L, M, S):
    """WHO adjustment for |z| >= 3: beyond +-3 SD, z grows linearly using the SD2-SD3 distance."""
    z, x = _arr(z), _arr(x)
    sd2p, sd3p = _sd_value(M, L, S, 2), _sd_value(M, L, S, 3)
    sd2n, sd3n = _sd_value(M, L, S, -2), _sd_value(M, L, S, -3)
    with np.errstate(all="ignore"):
        hi = 3 + (x - sd3p) / (sd3p - sd2p)
        lo = -3 + (x - sd3n) / (sd2n - sd3n)
    return np.where(z >= 3, hi, np.where(z <= -3, lo, z))


def score_who(d: pd.DataFrame, who: pd.DataFrame) -> pd.DataFrame:
    """d needs columns: sex (1/2), age_days, height (cm), weight (kg), bmi."""
    d = d.copy()
    d["_ad"] = d["age_days"].round().astype(int)

    fa = who[(who["_denom"] == "forage") & (who["_agedays"] < WHO_MAX_DAYS)].copy()
    fa["sex"] = fa["sex"].astype(int)
    fa["_ad"] = fa["_agedays"].astype(int)
    fa = fa[["sex", "_ad", "_len_l", "_len_m", "_len_s", "_wei_l", "_wei_m", "_wei_s",
             "_bmi_l", "_bmi_m", "_bmi_s"]]
    m = d.merge(fa, on=["sex", "_ad"], how="left", validate="m:1")
    m.index = d.index

    # height-for-age (no tail adjustment for height, as in the SAS code)
    m["haz"] = _lms_z(m["height"], m["_len_l"], m["_len_m"], m["_len_s"], small=1e-12)
    m["hapct"] = _pct(m["haz"])
    # weight-for-age and BMI-for-age (with SD2/SD3 tail adjustment)
    for zc, pc, x, p in (("waz", "wapct", "weight", "_wei"), ("bmiz", "bmipct", "bmi", "_bmi")):
        L, M, S = m[p + "_l"], m[p + "_m"], m[p + "_s"]
        z = _lms_z(m[x], L, M, S, small=1e-12)
        m[zc] = _who_tail(z, m[x], L, M, S)
        m[pc] = _pct(m[zc])

    # weight-for-length: reference indexed by length in 0.01 cm steps (45 - 110 cm)
    fl = who[who["_denom"] == "forlen"][["sex", "_len", "_wfl_l", "_wfl_m", "_wfl_s"]].copy()
    fl["sex"] = fl["sex"].astype(int)
    fl["_hc"] = np.round(fl["_len"] * 100).astype(int)
    fl = fl.drop(columns="_len")
    m["_hc"] = np.where(m["height"].notna(), np.round(m["height"] * 100), -1).astype(int)
    w = m[["sex", "_hc"]].merge(fl, on=["sex", "_hc"], how="left", validate="m:1")
    w.index = m.index
    z = _lms_z(m["weight"], w["_wfl_l"], w["_wfl_m"], w["_wfl_s"], small=1e-12)
    m["whz"] = _who_tail(z, m["weight"], w["_wfl_l"], w["_wfl_m"], w["_wfl_s"])
    m["whpct"] = _pct(m["whz"])

    # BIV flags (WHO cutoffs)
    m["biv_ht"] = np.where(m["haz"].isna(), np.nan, np.where(m["haz"] < -6, -1, np.where(m["haz"] >= 6, 1, 0)))
    m["biv_wt"] = np.where(m["waz"].isna(), np.nan, np.where(m["waz"] < -6, -1, np.where(m["waz"] >= 5, 1, 0)))
    m["biv_bmi"] = _cuts(m["bmiz"], -5, 5)
    m["biv_wh"] = _cuts(m["whz"], -5, 5)
    for c in ("mod_haz", "mod_waz", "mod_bmiz", "mod_whz", "bmi50", "bmi95", "bmip95",
              "bmi120", "original_bmiz", "original_bmipct"):
        m[c] = np.nan
    return m[CDC_OUT]


# --------------------------------------------------------------------------- CDC (2 to <20 y)
def _cdc_z(x, L, M, S):
    """CDC %_zscore macro: z, percentile and 'modified z' (half the 0-to-+-2 SD distance)."""
    x, L, M, S = _arr(x), _arr(L), _arr(M), _arr(S)
    z = _lms_z(x, L, M, S, small=0.01)
    pct = _pct(z)
    sdl = (M - _sd_value(M, L, S, -2)) / 2.0
    sdh = (_sd_value(M, L, S, 2) - M) / 2.0
    with np.errstate(all="ignore"):
        mod = np.where(x < M, (x - M) / sdl, (x - M) / sdh)
    mod = np.where(x > 0, mod, np.nan)
    return z, pct, mod


def _interp(m, base, tags):
    """linear interpolation of L, M, S between the two bracketing reference ages (or heights)."""
    ageint = m["_x2"] - m["_x1"]
    dx = m["_x"] - m["_x1"]
    for t in tags:
        for letter in "lms":
            c = f"_{letter}{t}"
            m[c] = m[c + "1"] + dx * (m[c + "2"] - m[c + "1"]) / ageint


def score_cdc(d: pd.DataFrame, cdc: pd.DataFrame) -> pd.DataFrame:
    """d needs columns: sex (1/2), age_months (24 <= x < 240), height (cm), weight (kg), bmi."""
    d = d.copy()
    d["_agecat"] = np.floor(d["age_months"] + 0.5) - 0.5

    tags = ["ht", "wt", "bmi"]
    ra = cdc[cdc["denom"] == "age"].copy()
    ra["sex"] = ra["sex"].astype(int)
    keep = ["sex", "_agecat", "_agemos1", "_agemos2"] + [f"_{l}{t}{k}" for t in tags for l in "lms" for k in "12"]
    m = d.merge(ra[keep], on=["sex", "_agecat"], how="left", validate="m:1")
    m.index = d.index
    m["_x"], m["_x1"], m["_x2"] = m["age_months"], m["_agemos1"], m["_agemos2"]
    _interp(m, "age", tags)

    out = pd.DataFrame(index=d.index)
    # height-for-age (standing height, >= 24 months)
    z, p, mod = _cdc_z(m["height"], m["_lht"], m["_mht"], m["_sht"])
    out["haz"], out["hapct"], out["mod_haz"] = z, p, mod
    out["biv_ht"] = _cuts(mod, -5, 4)
    # weight-for-age
    z, p, mod = _cdc_z(m["weight"], m["_lwt"], m["_mwt"], m["_swt"])
    out["waz"], out["wapct"], out["mod_waz"] = z, p, mod
    out["biv_wt"] = _cuts(mod, -5, 8)
    # BMI-for-age (original LMS first; the BIV flag uses the modified z of the LMS value)
    bmi = m["bmi"]
    z, p, mod = _cdc_z(bmi, m["_lbmi"], m["_mbmi"], m["_sbmi"])
    out["mod_bmiz"] = mod
    out["biv_bmi"] = _cuts(mod, -4, 8)
    out["original_bmiz"], out["original_bmipct"] = z, p
    L, M, S = _arr(m["_lbmi"]), _arr(m["_mbmi"]), _arr(m["_sbmi"])
    with np.errstate(all="ignore"):
        bmi50 = M * (1 + L * S * norm.ppf(0.50)) ** (1 / L)
        bmi95 = M * (1 + L * S * norm.ppf(0.95)) ** (1 / L)
    out["bmi50"], out["bmi95"] = bmi50, bmi95
    out["bmip95"] = 100 * _arr(bmi) / bmi95
    out["bmi120"] = 1.2 * bmi95

    # 2022 extended BMI percentile / z-score for BMI >= P95 (half-normal model above P95)
    agey = _arr(m["age_months"]) / 12.0
    sigma = np.where(m["sex"] == 1,
                     0.3728 + 0.5196 * agey - 0.0091 * agey ** 2,
                     0.8334 + 0.3712 * agey - 0.0011 * agey ** 2)
    ext = _arr(p) > 95
    with np.errstate(all="ignore"):
        pct_ext = 90 + 10 * norm.cdf((_arr(bmi) - bmi95) / sigma)
        z_ext = norm.ppf(pct_ext / 100)
    # percentiles so close to 100 that probit overflows -> CDC sets z to 8.21
    z_ext = np.where(np.isfinite(z_ext) & (z_ext < 8.21), z_ext, 8.21)
    out["bmipct"] = np.where(ext, pct_ext, p)
    out["bmiz"] = np.where(ext, z_ext, z)

    # weight-for-stature, standing height 77 - 122 cm
    rh = cdc[cdc["denom"] == "height"].copy()
    rh["sex"] = rh["sex"].astype(int)
    rh["_htcat"] = rh["_ht1"]
    keep_h = ["sex", "_htcat", "_ht1", "_ht2"] + [f"_{l}wht{k}" for l in "lms" for k in "12"]
    h = _arr(m["height"])
    m["_htcat"] = np.where(h >= 77.5, np.floor(h + 0.5) - 0.5, np.where((h >= 77) & (h < 77.5), 77.0, np.nan))
    w = m[["sex", "_htcat", "height", "weight"]].merge(rh[keep_h], on=["sex", "_htcat"], how="left", validate="m:1")
    w.index = m.index
    w["_x"], w["_x1"], w["_x2"] = w["height"], w["_ht1"], w["_ht2"]
    _interp(w, "ht", ["wht"])
    valid = (h > 77) & (h < 122)
    z, p, mod = _cdc_z(w["weight"], w["_lwht"], w["_mwht"], w["_swht"])
    out["whz"] = np.where(valid, z, np.nan)
    out["whpct"] = np.where(valid, p, np.nan)
    out["mod_whz"] = np.where(valid, mod, np.nan)
    out["biv_wh"] = _cuts(out["mod_whz"], -4, 8)
    return out[CDC_OUT]


# --------------------------------------------------------------------------- dispatcher
def _to_date(s: pd.Series, fmt: str) -> pd.Series:
    """Parse a messy date column.
    * real datetimes are kept;
    * numbers are treated as Excel serial dates (what Excel stores when a cell is date-formatted);
    * text must match `fmt` exactly - anything else (typos such as '25/06/996') becomes NaT
      instead of being guessed, and is reported as growth_status = 'no_age'."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    num = pd.to_numeric(s, errors="coerce")
    from_num = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce").astype("datetime64[ns]")
    txt = s.where(num.isna() & s.notna()).astype("string")
    from_txt = pd.to_datetime(txt, format=fmt, errors="coerce").astype("datetime64[ns]")
    return from_num.fillna(from_txt)


def _sex_code(s: pd.Series) -> pd.Series:
    """M/Masculino -> 1, F/Feminino -> 2, anything else (Indeterminado, blank) -> NaN."""
    k = s.astype("string").str.strip().str.lower()
    return k.map({"m": 1, "masculino": 1, "f": 2, "feminino": 2}).astype(float)


def add_growth_scores(df: pd.DataFrame, refs: dict, *, sex_col="SEXO", birth_col="DN",
                      visit_col="Atendimento", height_col="Altura", weight_col="Peso",
                      date_format="%d/%m/%y") -> pd.DataFrame:
    """Return a copy of df with age and WHO/CDC growth columns added (see module docstring)."""
    out = df.copy()

    # 1. age
    birth, visit = _to_date(out[birth_col], date_format), _to_date(out[visit_col], date_format)
    out["age_days"] = (visit - birth).dt.days.astype(float)
    out["age_months"] = out["age_days"] / DAYS_PER_MONTH
    out["age_years"] = out["age_days"] / 365.25

    # 2. clean inputs for the calculation (raw columns untouched)
    sex = _sex_code(out[sex_col])
    h = pd.to_numeric(out[height_col], errors="coerce").where(lambda s: s > 0)
    w = pd.to_numeric(out[weight_col], errors="coerce").where(lambda s: s > 0)
    out["bmi_calc"] = w / (h / 100) ** 2

    # 3. which reference?
    rd = out["age_days"].round()
    status = pd.Series("scored", index=out.index, dtype="object")
    status[sex.isna()] = "no_sex"
    status[rd >= CDC_MAX_MONTHS * DAYS_PER_MONTH] = "age_20y_or_more"
    status[rd < 0] = "negative_age"
    status[out["age_days"].isna()] = "no_age"
    out["growth_status"] = status
    use_who = (status == "scored") & (rd < WHO_MAX_DAYS)
    use_cdc = (status == "scored") & (rd >= WHO_MAX_DAYS)
    out["growth_ref"] = np.where(use_who, "WHO", np.where(use_cdc, "CDC", None))

    # 4. run the matching calculation on each subset
    inp = pd.DataFrame({"sex": sex.astype("Int64"), "age_days": out["age_days"],
                        "age_months": out["age_months"], "height": h, "weight": w,
                        "bmi": out["bmi_calc"]})
    res = pd.DataFrame(np.nan, index=out.index, columns=CDC_OUT)
    if use_who.any():
        sub = inp[use_who].copy()
        sub["sex"] = sub["sex"].astype(int)
        res.loc[use_who, WHO_OUT] = score_who(sub, refs["who"])[WHO_OUT]
    if use_cdc.any():
        sub = inp[use_cdc].copy()
        sub["sex"] = sub["sex"].astype(int)
        res.loc[use_cdc, CDC_OUT] = score_cdc(sub, refs["cdc"])
    out = pd.concat([out, res], axis=1)
    out["biv_low"], out["biv_high"] = _any_biv(out, ["biv_ht", "biv_wt", "biv_bmi", "biv_wh"])
    return out


def growth_report(df: pd.DataFrame) -> None:
    """Print the counts you need for the report: who was scored, with which reference, and
    how many rows each BIV rule flags (nothing is removed here)."""
    print("Rows:", len(df))
    print("\nStatus:\n", df["growth_status"].value_counts(dropna=False).to_string())
    print("\nReference used:\n", df["growth_ref"].value_counts(dropna=False).to_string())
    rows = []
    for ref in ("WHO", "CDC"):
        s = df[df["growth_ref"] == ref]
        for f in ("biv_ht", "biv_wt", "biv_bmi", "biv_wh"):
            rows.append({"ref": ref, "rule": f, "scored": int(s[f].notna().sum()),
                         "low (-1)": int((s[f] == -1).sum()), "high (+1)": int((s[f] == 1).sum())})
    print("\nBIV flags:\n", pd.DataFrame(rows).to_string(index=False))