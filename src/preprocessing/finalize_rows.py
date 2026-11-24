"""
Last row-level clean-up, AFTER the BIV step.  Reads data/UCMF_clean.csv, writes
    data/UCMF_final.csv          the rows that stay
    data/UCMF_final_removed.csv  the rows that left, with the reason and their line in UCMF_clean.csv

Why here and not in clean_data.py: data/biv_decisions.csv refers to row positions in UCMF_labelled.csv.
Removing rows earlier would shift those positions and the manual BIV decisions would point at the wrong children.

Rules (a row gets the first one that applies)
---------------------------------------------
1. sem_informacao          none of idade, sexo, peso, altura, imc, PAS, PAD is filled.  Such rows only carry heart
                           rate / murmur / label: no age to adjust for, and they repeat in a few identical patterns.
2. alvo_contraditorio      rows with the same birth date, visit date, sex, height and weight (the same child, same
                           visit) whose normal_x_anormal DISAGREES: the target is what we predict, so which copy is
                           right cannot be known and all the copies are removed.
3. duplicado_menos_completo  same key, same target: keep the row with most filled original columns (the first on a
                           tie) and remove the others.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CORE = ["idade", "sexo", "peso", "altura", "imc", "pa_sistolica", "pa_diastolica"]
KEY = ["data_nascimento", "atendimento", "sexo", "altura", "peso"]
ORIGINAL = ["peso", "altura", "imc", "atendimento", "data_nascimento", "idade", "convenio", "pulsos",
            "pa_sistolica", "pa_diastolica", "ppa", "normal_x_anormal", "b2", "sopro", "fc", "hda_1",
            "hda_2", "sexo", "motivo_1", "motivo_2"]
TARGET = "normal_x_anormal"


def finalize_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, removed).  `removed` has a `motivo_remocao` column; the index of df is kept."""
    reason = pd.Series("", index=df.index, dtype="object")

    reason[df[[c for c in CORE if c in df]].isna().all(axis=1)] = "sem_informacao"

    cand = df[reason.eq("") & df[KEY].notna().all(axis=1)]
    grp = cand.groupby(KEY, sort=False).ngroup()
    size = grp.map(grp.value_counts())
    cand, grp = cand[size > 1], grp[size > 1]

    n_target = cand.groupby(grp)[TARGET].transform("nunique")
    reason[cand.index[n_target > 1]] = "alvo_contraditorio"

    rest = cand[n_target <= 1]
    filled = rest[[c for c in ORIGINAL if c in rest]].notna().sum(axis=1)
    order = pd.DataFrame({"g": grp[rest.index], "filled": filled, "pos": range(len(rest))}, index=rest.index)
    best = order.sort_values(["g", "filled", "pos"], ascending=[True, False, True]).groupby("g").head(1).index
    reason[rest.index.difference(best)] = "duplicado_menos_completo"

    removed = df[reason.ne("")].copy()
    removed.insert(0, "motivo_remocao", reason[reason.ne("")])
    removed.insert(0, "linha_em_clean", removed.index)
    return df[reason.eq("")].copy(), removed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent.parent.parent / "data"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    src, dst = Path(a.data_dir), Path(a.out_dir or a.data_dir)
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src / "UCMF_clean.csv", low_memory=False)
    kept, removed = finalize_rows(df)

    print(f"UCMF_clean.csv: {len(df)} rows")
    print("Removed by rule:", removed["motivo_remocao"].value_counts().to_dict())
    print(f"Kept: {len(kept)} rows ({len(removed)} removed)")
    print("Target in kept rows:", kept[TARGET].value_counts().to_dict())
    kept.to_csv(dst / "UCMF_final.csv", index=False)
    removed.to_csv(dst / "UCMF_final_removed.csv", index=False)
    print(f"Wrote {dst / 'UCMF_final.csv'} and {dst / 'UCMF_final_removed.csv'}")
