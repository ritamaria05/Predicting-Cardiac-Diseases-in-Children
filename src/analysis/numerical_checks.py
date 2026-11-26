"""
Manual checks on the values that are still outside the expected intervals
in UCMF_final.csv (and one cross-variable consistency check).

Usage: python3 src/analysis/numerical_checks.py
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
df = pd.read_csv(os.path.join(ROOT, "data", "UCMF_final.csv"))
show = [c for c in ['id', 'idade', 'peso', 'altura', 'imc', 'pa_sistolica',
                    'pa_diastolica', 'fc', 'normal_x_anormal'] if c in df.columns]

print("1. idade between 19 and 20 (still '19 years old'):",
      ((df['idade'] > 19) & (df['idade'] < 20)).sum())

print("\n2. idade exactly 0:", (df['idade'] == 0).sum())
if 'age_days' in df.columns:
    print("   age_days for those rows (most frequent values):")
    print(df.loc[df['idade'] == 0, 'age_days'].value_counts().head(5).to_string())

print("\n3. peso above 150 kg:")
print(df.loc[df['peso'] > 150, show].to_string())

print("\n4. fc above 220 bpm:")
print(df.loc[df['fc'] > 220, show].to_string())

print("\n5. pa_diastolica below 30 mmHg:")
print(df.loc[df['pa_diastolica'] < 30, show].to_string())

both = df['pa_sistolica'].notna() & df['pa_diastolica'].notna()
inconsistent = both & (df['pa_sistolica'] <= df['pa_diastolica'])
print(f"\n6. systolic <= diastolic (impossible): {inconsistent.sum()} of {both.sum()} rows with both")
if inconsistent.any():
    print(df.loc[inconsistent, show].head(10).to_string())
