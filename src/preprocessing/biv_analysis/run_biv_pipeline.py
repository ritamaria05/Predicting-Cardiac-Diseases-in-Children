"""
Growth z-scores + BIV cleaning, end to end.

    python src/preprocessing/run_biv_pipeline.py            (from the project root, any folder works)

Reads    data/UCMF_labelled.csv  and  data/biv_decisions.csv (your manual review)
Writes   data/UCMF_annotated.csv  every row, with z-scores, BIV flags and the removal decision
         data/UCMF_clean.csv      only the rows that are kept, without the bookkeeping columns
         data/biv_to_review.csv   only if data/biv_decisions.csv does not exist yet
"""
import argparse
from pathlib import Path

import pandas as pd

from preprocessing.biv_analysis.growth_scores import load_refs, add_growth_scores
from preprocessing.biv_analysis.biv_removal import (add_biv_decision, biv_summary, export_review,
                         blank_manual_values, apply_manual_decisions)

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(HERE.parent.parent / "data"))
ap.add_argument("--out-dir", default=None, help="where to write the CSVs (default: data dir)")
a = ap.parse_args()
DATA = Path(a.data_dir)
OUT = Path(a.out_dir) if a.out_dir else DATA
OUT.mkdir(parents=True, exist_ok=True)
decisions = DATA / "biv_decisions.csv"

df = pd.read_csv(DATA / "UCMF_labelled.csv")
refs = load_refs(str(HERE / "WHOref_d.sas7bdat"), str(HERE / "CDCref_d.sas7bdat"))

if decisions.exists():
    df = blank_manual_values(df, str(decisions))      # 1. blank impossible single values (before scoring)

out = add_growth_scores(df, refs, sex_col="sexo", birth_col="data_nascimento",
                        visit_col="atendimento", height_col="altura", weight_col="peso",
                        date_format="%Y-%m-%d")
out = add_biv_decision(out)                           # 2. automatic rules

if decisions.exists():
    out = apply_manual_decisions(out, str(decisions)) # 3. your manual removals
else:
    export_review(out, str(OUT / "biv_to_review.csv"))
    print("No biv_decisions.csv yet: review biv_to_review.csv first. UCMF_clean.csv not written.")

biv_summary(out)
out.to_csv(OUT / "UCMF_annotated.csv", index=False)

if decisions.exists():
    bookkeeping = ["biv_ht", "biv_wt", "biv_bmi", "biv_wh", "biv_low", "biv_high", "biv_any",
                   "biv_reason", "biv_action", "biv_removed", "biv_rules_matched"]
    clean = out[~out["biv_removed"]].drop(columns=bookkeeping)
    clean.to_csv(OUT / "UCMF_clean.csv", index=False)
    print(f"\nUCMF_clean.csv: {len(clean)} rows ({int(out['biv_removed'].sum())} removed)")
