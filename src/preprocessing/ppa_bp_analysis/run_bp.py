"""Adds the computed BP category to the cleaned data and compares it with the stored ppa.
    python src/preprocessing/run_bp.py        -> data/UCMF_bp.csv"""
from pathlib import Path
import pandas as pd
from preprocessing.ppa_bp_analysis.bp_categories import add_bp_category, compare_with_ppa

DATA = Path(__file__).resolve().parent.parent.parent / "data"
df = pd.read_csv(DATA / "UCMF_clean.csv", low_memory=False)
out = add_bp_category(df)
compare_with_ppa(out)
out.to_csv(DATA / "UCMF_bp.csv", index=False)
print(f"\nWrote {DATA / 'UCMF_bp.csv'} ({len(out)} rows)")
