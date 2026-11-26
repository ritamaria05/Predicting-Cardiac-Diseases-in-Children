"""
Inventory of the files in data/: what each CSV contains and how it differs
from the previous one (files are listed in the order they were created).

Usage:  python3 src/analysis/data_inventory.py
"""
import os
import glob
import datetime
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGETS = ['normal_x_anormal', 'NORMAL X ANORMAL']

# --- raw file -------------------------------------------------------
raw_path = os.path.join(ROOT, 'UCMF(5).xls')
if os.path.exists(raw_path):
    raw = pd.read_excel(raw_path)
    print(f"UCMF(5).xls (raw) | rows: {len(raw)} | columns: {raw.shape[1]}")
else:
    print("UCMF(5).xls not found in the project root")

# --- cleaned/processed files ----------------------------------------
files = sorted(glob.glob(os.path.join(ROOT, 'data', '*.csv')), key=os.path.getmtime)
prev_cols = None
for f in files:
    df = pd.read_csv(f, low_memory=False)
    cols = list(df.columns)
    modified = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    print(f"\n{os.path.basename(f)} | created/modified: {modified:%Y-%m-%d %H:%M}")
    print(f"  rows: {len(df)} | columns: {len(cols)}")

    if prev_cols is None:
        print(f"  columns: {cols}")
    else:
        added = [c for c in cols if c not in prev_cols]
        removed = [c for c in prev_cols if c not in cols]
        print(f"  added vs previous file: {added if added else 'none'}")
        print(f"  removed vs previous file: {removed if removed else 'none'}")
    prev_cols = cols

    target = next((t for t in TARGETS if t in df.columns), None)
    if target:
        counts = df[target].value_counts(dropna=False).to_dict()
        print(f"  target ({target}): {counts}")
    else:
        print("  target column: not present")