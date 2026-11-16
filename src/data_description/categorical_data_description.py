import unicodedata
import re
import pandas as pd
import numpy as np
import warnings


df = pd.read_excel('UCMF(5).xls')


cat_cols = ['Convenio', 'PULSOS', 'PPA', 'NORMAL X ANORMAL', 'B2', 'SOPRO', 'HDA 1', 'HDA2', 'SEXO', 'MOTIVO1', 'MOTIVO2']

dt_cols = ['Atendimento', 'DN']

print("CATEGORICAL VARIABLES")
for col in cat_cols: 
    s = df[col]
    print(f"\n{col} Column Description:")
    print(f"Column defined type: {s.dtype}")
    print(f"Total number of lines in the column: {s.size}")
    print(f"Number of non-null values in the column: {s.count()}")
    print(f"Number of null values in the column: {s.isna().sum()}")
    print(f"Percentage of null values in the column: {s.isna().mean() * 100:.2f}%")
    
    print(f"Number of unique values: {s.nunique()}")
    print("Values and their frequencies:")
    print(s.value_counts().head(s.nunique()).to_string())

# ---------------------------------------------------------

EXCEL_ORIGIN = '1899-12-30'
TODAY = pd.Timestamp.today()

print("\nDATE VARIABLES")
for col in dt_cols:
    raw = df[col]
 
    # Excel serial numbers 
    numeric = pd.to_numeric(raw, errors='coerce')
    excel_mask = numeric.notna() & (numeric > 10000)
    excel_dates = pd.to_datetime(numeric[excel_mask], unit='D', origin=EXCEL_ORIGIN)
 
    # Text dates
    text_dates = pd.to_datetime(raw[~excel_mask], errors='coerce',
                                dayfirst=True, format='mixed')
    
    future_text_mask = text_dates > pd.Timestamp('2026-01-01')
    text_dates.loc[future_text_mask] = text_dates.loc[future_text_mask] - pd.DateOffset(years=100)

    # Restrict to plausible years
    text_dates = text_dates.where(text_dates.dt.year.between(1900, 2100))

    s = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
    s.loc[excel_mask] = excel_dates.astype('datetime64[ns]')
    s.loc[~excel_mask] = text_dates.astype('datetime64[ns]')
 
    print(f"\n{col} Column Description:")
    print(f"Column defined type: {raw.dtype} (Converted to datetime64)")
    print(f"Total number of lines in the column: {s.size}")
    print(f"Number of non-null values in the column: {s.count()}")
    print(f"Number of null values in the column: {s.isna().sum()}")
    print(f"Percentage of null values in the column: {s.isna().mean() * 100:.2f}%")
    print(f"Excel serial dates converted: {excel_mask.sum()}")
 
    if s.count() > 0:
        print(f"Earliest date (Min): {s.min().date()}")
        print(f"Latest date (Max): {s.max().date()}")
        

def normalize_text_key(series: pd.Series) -> pd.Series:
    """Standardize string casing, accents, punctuation, and whitespace."""
    def _clean(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        s = "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
        s = re.sub(r"[./;]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return np.nan if s in {"", "NAN", "NONE", "NULL", "NA", "?"} else s

    return series.map(_clean)


print("\nPOST-NORMALIZATION UNIQUE VALUES (READ-ONLY DIAGNOSTIC)")
for col in cat_cols:
    norm_series = normalize_text_key(df[col])
    print(f"\n--- {col} (Unique after normalization: {norm_series.nunique()}) ---")
    print(norm_series.value_counts(dropna=False).to_string())
    
numeric_mask = df["Convenio"].astype(str).str.strip().str.isdigit()
print(df.loc[numeric_mask, ["ID", "Atendimento", "DN", "Convenio", "SEXO", "IDADE"]])