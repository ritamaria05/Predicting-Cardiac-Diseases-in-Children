"""
New, derived variables.  Nothing is overwritten or deleted: the originals stay, the new columns are
added at the end, so any cut-off can be changed and recomputed.

    python src/preprocessing/add_features.py          data/UCMF_bp.csv  ->  data/UCMF_features.csv  (everything)
                                                                          data/UCMF_analysis.csv  (simplified)

UCMF_features.csv keeps every column, so each calculation can be checked.  UCMF_analysis.csv keeps only
what is useful for the analysis (see ANALYSIS_COLS): it drops the calculated age in days/months, the
status of the calculation, the intermediate z-scores and reference values, the BP thresholds, the
dates and the percentiles (they are the z-scores in another form).

New columns
-----------
age_group      age_years in intervals: '<1', '1-<2', '2-<6', '6-<12', '12-<20'
                 1 and 2 y: where the WHO and CDC references (and the BP table, from 1 y) change;
                 6 y: school age; 12 y: start of adolescence.  The cuts are a choice: edit AGE_BINS.
bmi_cat        'baixo_peso' (<P5), 'normal' (P5 to <P85), 'excesso_peso' (P85 to <P95), 'obesidade' (>=P95)
                 from the CDC BMI-for-age percentile (bmipct), only for 2 to <20 years (the DGS BMI curves
                 start at 2 y).  P85 and P95 are the DGS circular 05/DSMIA definitions; the P5 limit for
                 low weight is NOT in the DGS circular, it is the CDC definition.
whz_cat        weight-for-length category for UNDER 2 YEARS (WHO reference), from whz:
                 'baixo_peso_grave' (< -3), 'baixo_peso' (-3 to < -2), 'normal' (-2 to +1),
                 'risco_excesso_peso' (> +1 to +2), 'excesso_peso' (> +2 to +3), 'obesidade' (> +3).
                 Cut-offs: WHO Child Growth Standards (0-5 y), training course Module C.  WHO standards are for
                 term infants and the file has no gestational age, so preterm babies cannot be identified.
sintomas       'sintomatico' / 'assintomatico' / NaN, from hda_1 and hda_2.  NaN when hda_1 is 'NAO REGISTADO'
                 (the same 4174 rows in both columns: not recorded, not 'no symptoms')
motivo2_grupo  the 18 values of motivo_2 in 8 groups (motivo_1 already has only 6 values and is not changed)
convenio_grupo the 62 insurers: the ones with >= MIN_CONVENIO rows keep their name, the rest are 'OUTRO';
                 'DESCONHECIDO' becomes NaN
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AGE_BINS = [0, 1, 2, 6, 12, 20]
AGE_LABELS = ["<1", "1-<2", "2-<6", "6-<12", "12-<20"]
BMI_P_LOW, BMI_P_OVERWEIGHT, BMI_P_OBESE = 5, 85, 95
MIN_CONVENIO = 400

MOTIVO2_GROUPS = {
    "1 - CARDIOPATIA CONGENICA": "cardiopatia",
    "1 - CARDIOPATIA ADQUIRIDA": "cardiopatia",
    "5 - CIRURGIA": "cirurgia",
    "5 - ATIVIDADE FISICA": "atividade_fisica",
    "6 - SOPRO": "sopro",
    "6 - DOR PRECORDIAL": "sintomas",
    "6 - PALPITACAO TAQUICARDIA ARRITMIA": "sintomas",
    "6 - DISPNEIA": "sintomas",
    "6 - CIANOSE": "sintomas",
    "6 - CIANOSE E DISPNEIA": "sintomas",
    "6 - CANSACO": "sintomas",
    "6 - ALTERACOES DE PULSO PERFUSAO": "sintomas",
    "6 - HAS DISLIPIDEMIA OBESIDADE": "fatores_risco",
    "6 - CARDIOPATIA NA FAMILIA": "fatores_risco",
    "OUTRO": "outro",
    "5 - USO DE CISAPRIDA": "outro",
    "NENHUM": "nenhum",
}  # 'NAO REGISTADO' -> NaN


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    age = out["age_years"].fillna(out["idade"]) if "idade" in out else out["age_years"]
    out["age_group"] = pd.cut(age, AGE_BINS, labels=AGE_LABELS, right=False).astype("object")

    pct = out["bmipct"]
    cat = np.select([pct < BMI_P_LOW, pct < BMI_P_OVERWEIGHT, pct < BMI_P_OBESE, pct >= BMI_P_OBESE],
                    ["baixo_peso", "normal", "excesso_peso", "obesidade"], default=None)
    cdc = (out["growth_ref"] == "CDC") & (out["age_years"] >= 2)       # DGS curves: 2 to 20 years
    out["bmi_cat"] = pd.Series(np.where(cdc, cat, None), index=out.index, dtype="object")

    z = out["whz"]
    wcat = np.select([z < -3, z < -2, z <= 1, z <= 2, z <= 3, z > 3],
                     ["baixo_peso_grave", "baixo_peso", "normal", "risco_excesso_peso", "excesso_peso", "obesidade"],
                     default=None)
    out["whz_cat"] = pd.Series(np.where(out["growth_ref"] == "WHO", wcat, None), index=out.index, dtype="object")

    h1, h2 = out["hda_1"], out["hda_2"].fillna("NENHUM")
    symptom = ~h1.isin(["ASSINTOMATICO", "NAO REGISTADO"]) | ~h2.isin(["NENHUM", "NAO REGISTADO"])
    out["sintomas"] = np.where(h1.isna() | (h1 == "NAO REGISTADO"), None,
                               np.where(symptom, "sintomatico", "assintomatico"))

    out["motivo2_grupo"] = out["motivo_2"].map(MOTIVO2_GROUPS)

    counts = out["convenio"].value_counts()
    keep = set(counts[counts >= MIN_CONVENIO].index) - {"DESCONHECIDO"}
    out["convenio_grupo"] = np.where(out["convenio"].isna() | (out["convenio"] == "DESCONHECIDO"), None,
                                     np.where(out["convenio"].isin(keep), out["convenio"], "OUTRO"))
    return out


ANALYSIS_COLS = [
    # patient and measurements
    "sexo", "idade", "peso", "altura", "imc", "fc", "pa_sistolica", "pa_diastolica",
    # clinical findings and reason for referral (originals)
    "pulsos", "b2", "sopro", "hda_1", "hda_2", "motivo_1", "motivo_2", "convenio",
    # target and the clinic's own blood-pressure category
    "normal_x_anormal", "ppa",
    # growth: reference used and z-scores (age and sex adjusted)
    "growth_ref", "haz", "waz", "bmiz", "whz",
    # blood pressure category computed from the project's tables
    "bp_cat",
    # new variables
    "age_group", "bmi_cat", "whz_cat", "sintomas", "motivo2_grupo", "convenio_grupo",
]

NEW_COLS = ["age_group", "bmi_cat", "whz_cat", "sintomas", "motivo2_grupo", "convenio_grupo"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    data = Path(__file__).resolve().parent.parent.parent / "data"
    ap.add_argument("--data-dir", default=str(data))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    src, dst = Path(a.data_dir), Path(a.out_dir or a.data_dir)
    dst.mkdir(parents=True, exist_ok=True)
    out = add_features(pd.read_csv(src / "UCMF_bp.csv", low_memory=False))
    for c in NEW_COLS:
        print(f"\n{c}:", out[c].value_counts(dropna=False).to_dict())
    out.to_csv(dst / "UCMF_features.csv", index=False)
    print(f"\nWrote {dst / 'UCMF_features.csv'} ({len(out)} rows, {out.shape[1]} columns)")
    slim = out[ANALYSIS_COLS]
    slim.to_csv(dst / "UCMF_analysis.csv", index=False)
    print(f"Wrote {dst / 'UCMF_analysis.csv'} ({len(slim)} rows, {slim.shape[1]} columns)")
