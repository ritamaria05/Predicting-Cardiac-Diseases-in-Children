import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
df = pd.read_csv(os.path.join(ROOT, 'data', 'UCMF_cleaned.csv'),
                 parse_dates=['atendimento', 'data_nascimento'])

num_cols = ['idade', 'peso', 'altura', 'imc', 'pa_sistolica', 'pa_diastolica', 'fc']
cat_cols = ['sexo', 'normal_x_anormal', 'pulsos', 'b2', 'sopro', 'ppa', 'convenio',
            'motivo_1', 'motivo_2', 'hda_1', 'hda_2', 'idade_origem']
date_cols = ['atendimento', 'data_nascimento']

print("1. SHAPE AND TYPES")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
print(df.dtypes.to_string())

print("\n2. MISSING VALUES AND UNIQUE VALUES PER COLUMN")
summary = pd.DataFrame({
    'non_null': df.count(),
    'missing': df.isna().sum(),
    'missing_%': (df.isna().mean() * 100).round(2),
    'unique': df.nunique(),
})
print(summary.to_string())

print("\n3. NUMERICAL VARIABLES")
print(df[num_cols].describe().round(2).to_string())

print("\n4. CATEGORICAL VARIABLES (top 10 values)")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10).to_string())

print("\n5. DATES")
for col in date_cols:
    print(f"{col}: min {df[col].min()} | max {df[col].max()}")

print("\n6. SANITY CHECKS")
print(f"Fully duplicated rows: {df.duplicated().sum()}")
print(f"Duplicated ids: {df['id'].duplicated().sum()}")
print(f"idade outside [0, 20]: {((df['idade'] < 0) | (df['idade'] > 20)).sum()}")
print(f"Rows without target (normal_x_anormal): {df['normal_x_anormal'].isna().sum()}")
print("Target distribution (%):")
print((df['normal_x_anormal'].value_counts(normalize=True) * 100).round(1).to_string())
print(f"Rows with at least one missing value: {df.isna().any(axis=1).sum()}")
print(f"Rows with no missing value (complete cases): {df.notna().all(axis=1).sum()}")