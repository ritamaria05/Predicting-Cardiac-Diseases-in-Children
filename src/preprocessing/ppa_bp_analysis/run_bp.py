"""Adds the computed BP category to the cleaned data and compares it with the stored ppa.
    python src/preprocessing/ppa_bp_analysis/run_bp.py        -> data/UCMF_bp.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, so 'preprocessing' is importable
import pandas as pd
from preprocessing.ppa_bp_analysis.bp_categories import add_bp_category, compare_with_ppa

DATA = Path(__file__).resolve().parents[3] / "data"
df = pd.read_csv(DATA / "UCMF_final.csv", low_memory=False)
out = add_bp_category(df)
compare_with_ppa(out)
out.to_csv(DATA / "UCMF_bp.csv", index=False)
print(f"\nWrote {DATA / 'UCMF_bp.csv'} ({len(out)} rows)")
