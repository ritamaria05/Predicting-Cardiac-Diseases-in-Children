'''Data description for categorical and date columns in the UCMF dataset.'''
import unicodedata
import re
import pandas as pd
import numpy as np

df = pd.read_excel('UCMF(5).xls')


cat_cols = ['Convenio', 'PULSOS', 'PPA', 'NORMAL X ANORMAL', 'B2', 'SOPRO', 'HDA 1',
            'HDA2', 'SEXO', 'MOTIVO1', 'MOTIVO2']

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
        st = str(val).strip().upper()
        st = "".join(
            c for c in unicodedata.normalize("NFD", st)
            if unicodedata.category(c) != "Mn"
        )
        st = re.sub(r"[./;]", " ", st)
        st = re.sub(r"\s+", " ", st).strip()
        return np.nan if st in {"", "NAN", "NONE", "NULL", "NA", "?"} else st

    return series.map(_clean)


print("\nPOST-NORMALIZATION UNIQUE VALUES (READ-ONLY DIAGNOSTIC)")
for col in cat_cols:
    norm_series = normalize_text_key(df[col])
    print(f"\n--- {col} (Unique after normalization: {norm_series.nunique()}) ---")
    print(norm_series.value_counts(dropna=False).to_string())

numeric_mask = df["Convenio"].astype(str).str.strip().str.isdigit()
print(df.loc[numeric_mask, ["ID", "Atendimento", "DN", "Convenio", "SEXO", "IDADE"]])


def inspect_date_patterns(series: pd.Series) -> pd.Series:
    '''Inspect the patterns of date strings in a series.'''
    st = series.dropna().astype(str).str.strip()
    patterns = (
        st.str.replace(r"\d", "9", regex=True)
         .str.replace(r"[a-zA-Z]", "A", regex=True)
    )
    return patterns.value_counts()

print("--- Atendimento ---")
print(inspect_date_patterns(df["Atendimento"]))

print("\n--- DN ---")
print(inspect_date_patterns(df["DN"]))

s_dn = df["DN"].dropna().astype(str).str.strip()

# Padrões válidos: DD/MM/AA, D/M/AAAA ou serial de 5 dígitos
valid_mask = s_dn.str.match(r"^(\d{1,2}/\d{1,2}/\d{2}(\d{2})?|\d{5})$")

anomalous_idx = s_dn[~valid_mask].index
print(df.loc[anomalous_idx, ["Atendimento", "DN", "IDADE"]].to_string())

conv_nums = pd.to_numeric(df["Convenio"].astype(str).str.strip(), errors="coerce")
serial_mask = conv_nums.between(20000, 45000)

recovered_dates = pd.to_datetime(conv_nums[serial_mask], unit="D", origin="1899-12-30")
debug_df = df.loc[serial_mask, ["Atendimento", "DN", "IDADE", "Convenio"]].copy()
debug_df["data_recuperada"] = recovered_dates
print(debug_df.to_string())

#check if ID is unique
id_counts = df["ID"].value_counts()
duplicate_ids = id_counts[id_counts > 1]
if not duplicate_ids.empty:
    print("\nDuplicate IDs found:")
    print(duplicate_ids.to_string())
# check for abnormal ids
if not df["ID"].between(1, 100000).all():
    print("\nAbnormal IDs found:")
    print(df.loc[~df["ID"].between(1, 100000), ["ID", "Atendimento", "DN"]].to_string())
