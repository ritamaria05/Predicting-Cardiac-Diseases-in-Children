"""
Summary table of the numerical variables, side by side for several files
(before and after cleaning), with a check against expected intervals.

Usage:
    python3 src/data_description/numerical_summary.py
        -> compares: raw Excel | data/UCMF_cleaned.csv | data/UCMF_final.csv
    python3 src/data_description/numerical_summary.py "UCMF(5).xls" data/UCMF_final.csv
        -> compares the files you choose
Output: printed table + results/numerical_summary.csv
"""
import os
import sys
import pandas as pd

from load_data import ROOT, load_dataset, tag_for

DEFAULT_FILES = ["UCMF(5).xls", "data/UCMF_cleaned.csv", "data/UCMF_final.csv"]

RANGES = {
    "idade": (0, 19),            # years (project definition)
    "peso": (1, 150),            # kg
    "altura": (30, 200),         # cm
    "imc": (8, 50),              # kg/m²
    "pa_sistolica": (50, 200),   # mmHg
    "pa_diastolica": (20, 130),  # mmHg
    "fc": (40, 220),             # bpm
}


def resolve(path):
    if os.path.exists(path):
        return path
    alt = os.path.join(ROOT, path)
    if os.path.exists(alt):
        return alt
    raise SystemExit(f"File not found: {path}")


def summarise(df, col, low, high):
    original = df[col]
    s = pd.to_numeric(original, errors="coerce")
    n = int(s.count())
    out = int(((s < low) | (s > high)).sum())
    return {
        "rows": len(df),
        "n": n,
        "missing_%": round(s.isna().mean() * 100, 2),
        "non_numeric": int((original.notna() & s.isna()).sum()),
        "min": s.min(), "p25": s.quantile(0.25), "median": s.median(),
        "p75": s.quantile(0.75), "max": s.max(),
        "mean": s.mean(), "std": s.std(), "skew": s.skew(),
        "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()),
        "out_of_range": out, "out_%": round(out / n * 100, 2) if n else 0.0,
    }


files = [resolve(p) for p in (sys.argv[1:] or DEFAULT_FILES)]
datasets = {tag_for(p): load_dataset(p) for p in files}

rows = {}
for col, (low, high) in RANGES.items():
    for tag, df in datasets.items():
        if col not in df.columns:
            print(f"[WARNING] column '{col}' not found in {tag}")
            continue
        rows[(col, tag)] = summarise(df, col, low, high)

table = pd.DataFrame(rows).T
int_cols = ["rows", "n", "non_numeric", "zeros", "negatives", "out_of_range"]
table[int_cols] = table[int_cols].astype(int)
table = table.round(2)
table.index.names = ["variable", "file"]

print("Expected intervals used for 'out_of_range':")
for col, (low, high) in RANGES.items():
    print(f"  {col}: [{low}, {high}]")
print()
print(table.to_string())

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out_path = os.path.join(ROOT, "results", "numerical_summary.csv")
table.to_csv(out_path)
print(f"\nSaved: {out_path}")