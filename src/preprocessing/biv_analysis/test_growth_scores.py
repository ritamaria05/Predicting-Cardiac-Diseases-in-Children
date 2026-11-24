"""
Sanity checks for growth_scores.py.  Run:  python test_growth_scores.py

Expected layout (edit the three paths below if yours differs):
    test_growth_scores.py
    growth_scores.py
    WHOref_d.sas7bdat
    CDCref_d.sas7bdat
    data/UCMF_labelled.csv

Part 1-2 use properties the references must satisfy (median -> z = 0, P95 value -> z = 1.645,
CDC's own published example).  Part 3 tests the WHO/CDC dispatcher on made-up children.
Part 4 runs your real file and checks the plumbing.  None of this compares against SAS output,
so also spot-check a few rows against CDC's / WHO's official programs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from preprocessing.biv_analysis.growth_scores import (load_refs, add_growth_scores, growth_report, score_who, score_cdc,
                           _sd_value)

HERE = Path(__file__).resolve().parent
WHO_PATH = HERE / "WHOref_d.sas7bdat"
CDC_PATH = HERE / "CDCref_d.sas7bdat"
DATA_PATH = HERE.parent.parent / "data" / "UCMF_labelled.csv"

refs = load_refs(str(WHO_PATH), str(CDC_PATH))
who, cdc = refs["who"], refs["cdc"]


def close(a, b, tol, msg):
    assert abs(a - b) <= tol, f"{msg}: got {a}, expected {b}"
    print("ok  ", msg)


# ======================= 1. WHO (0 to <2 years) =========================================
row = who[(who["_denom"] == "forage") & (who["sex"] == 1) & (who["_agedays"] == 365)].iloc[0]
d = pd.DataFrame({"sex": [1], "age_days": [365.0], "height": [row["_len_m"]],
                  "weight": [row["_wei_m"]], "bmi": [row["_bmi_m"]]})
r = score_who(d, who)
close(r["haz"].iloc[0], 0, 1e-9, "WHO height z at median")
close(r["waz"].iloc[0], 0, 1e-9, "WHO weight z at median")
close(r["bmiz"].iloc[0], 0, 1e-9, "WHO BMI z at median")
close(r["wapct"].iloc[0], 50, 1e-9, "WHO weight percentile at median")

L, M, S = row["_wei_l"], row["_wei_m"], row["_wei_s"]
for k in (3, -3):
    d["weight"] = _sd_value(M, L, S, k)
    close(score_who(d, who)["waz"].iloc[0], k, 1e-9, f"WHO weight at SD{k:+d} -> z = {k}")
d["weight"] = _sd_value(M, L, S, 3) + 5
assert score_who(d, who)["waz"].iloc[0] > 3
print("ok   WHO z keeps increasing beyond +3")

fl = who[(who["_denom"] == "forlen") & (who["sex"] == 2)]
r0 = fl[np.isclose(fl["_len"], 70.0)].iloc[0]
d = pd.DataFrame({"sex": [2], "age_days": [365.0], "height": [70.0], "weight": [r0["_wfl_m"]], "bmi": [15.0]})
close(score_who(d, who)["whz"].iloc[0], 0, 1e-9, "WHO weight-for-length z at median (70 cm girl)")

# ======================= 2. CDC (2 to <20 years) ========================================
d = pd.DataFrame({"sex": [1], "age_months": [60.0], "height": [110.0], "weight": [20.0], "bmi": [17.9]})
r = score_cdc(d, cdc)
close(r["bmi95"].iloc[0], 17.9, 0.05, "CDC P95 BMI, boy, 60 months (CDC page says 17.9)")
close(r["bmipct"].iloc[0], 95, 0.7, "BMI 17.9 is about P95")

d = pd.DataFrame({"sex": [2, 2], "age_months": [150.0, 150.0], "height": [150.0, 150.0],
                  "weight": [40.0, 40.0], "bmi": [20.0, 20.0]})
r0 = score_cdc(d, cdc)
d["bmi"] = [r0["bmi50"].iloc[0], r0["bmi95"].iloc[0]]
r = score_cdc(d, cdc)
close(r["bmiz"].iloc[0], 0, 1e-6, "CDC BMI z at median")
close(r["bmiz"].iloc[1], norm.ppf(0.95), 1e-6, "CDC BMI z at P95 = 1.645")

d = pd.DataFrame({"sex": [1, 1], "age_months": [120.0, 120.0], "height": [140.0, 140.0],
                  "weight": [70.0, 400.0], "bmi": [35.7, 204.0]})
r = score_cdc(d, cdc)
assert np.isfinite(r["bmiz"]).all() and (r["bmiz"] <= 8.21).all()
print("ok   extended BMI z finite and <= 8.21:", r["bmiz"].round(2).tolist())

d = pd.DataFrame({"sex": [1, 1, 1], "age_months": [120.0] * 3, "height": [140.0] * 3,
                  "weight": [30.0, 160.0, 5.0], "bmi": [30 / 1.96, 160 / 1.96, 5 / 1.96]})
assert score_cdc(d, cdc)["biv_bmi"].tolist() == [0, 1, -1]
print("ok   CDC BIV BMI flags (normal, huge, tiny) = [0, 1, -1]")

# ======================= 3. Dispatcher on made-up children ==============================
birth = pd.Timestamp("2010-01-01")
visits = [birth + pd.Timedelta(days=n) for n in (0, 365, 730, 731, 3000, 7400)]
df = pd.DataFrame({"DN": [birth.strftime("%d/%m/%Y")] * 6,
                   "Atendimento": [v.strftime("%d/%m/%Y") for v in visits],
                   "SEXO": ["M", "F", "Masculino", "feminino", "M", "M"],
                   "Peso": [3.5, 9.0, 12.0, 12.0, 25.0, 60.0], "Altura": [50, 74, 86, 86, 128, 170]})
out = add_growth_scores(df, refs, date_format="%d/%m/%Y")
assert out["growth_ref"].fillna("none").tolist() == ["WHO", "WHO", "WHO", "CDC", "CDC", "none"]
assert out["growth_status"].iloc[5] == "age_20y_or_more"
print("ok   WHO/CDC boundary: 730 days -> WHO, 731 days -> CDC, 20 y+ not scored")

df2 = pd.DataFrame({"DN": ["01/01/2010"] * 3, "Atendimento": ["01/01/2015", None, "01/01/2005"],
                    "SEXO": ["Indeterminado", "M", "M"], "Peso": [20.0] * 3, "Altura": [110] * 3})
s = add_growth_scores(df2, refs, date_format="%d/%m/%Y")["growth_status"].tolist()
assert s == ["no_sex", "no_age", "negative_age"], s
print("ok   status codes:", s)

# ======================= 4. Your real file ==============================================
if DATA_PATH.exists():
    print(f"\n--- real data: {DATA_PATH.name} ---")
    real = pd.read_csv(DATA_PATH)
    res = add_growth_scores(real, refs, sex_col="sexo", birth_col="data_nascimento",
                            visit_col="atendimento", height_col="altura",
                            weight_col="peso", date_format="%Y-%m-%d")
    assert len(res) == len(real)
    scored = res["growth_status"] == "scored"
    assert res.loc[scored, "growth_ref"].isin(["WHO", "CDC"]).all()
    assert res.loc[~scored, "growth_ref"].isna().all()
    print("ok   every scored row has a reference; unscored rows have none")

    # age computed from the dates agrees with the file's own `idade` column
    both = res["age_years"].notna() & res["idade"].notna()
    diff = (res.loc[both, "age_years"] - res.loc[both, "idade"]).abs().max()
    assert diff < 0.02, diff
    print(f"ok   age from dates matches `idade` (max diff {diff:.3f} years)")

    # no lookup failures: input present -> score present
    for ref, need, col in (("WHO", "altura", "haz"), ("WHO", "peso", "waz"),
                           ("CDC", "altura", "haz"), ("CDC", "peso", "waz")):
        sub = res[(res["growth_ref"] == ref) & (res[need] > 0)]
        assert sub[col].notna().all(), (ref, col)
    print("ok   no missing lookups for height-for-age / weight-for-age")

    growth_report(res)           # summary counts only (nothing is removed)
else:
    print(f"\n(skipped real-data checks: {DATA_PATH} not found)")

print("\nAll checks passed.")