import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(ROOT, 'UCMF(5).xls')
OUT_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(OUT_DIR, "UCMF_cleaned.csv")


# SECTION A: NUMERICAL VARIABLES
# Columns: IDADE, Peso, Altura, IMC, PA SISTOLICA, PA DIASTOLICA, FC
def clean_numerical(df):
    print("\nCleaning numerical variables")
    return df


# SECTION B: CATEGORICAL AND DATE VARIABLES (you)
# Columns: PULSOS, B2, SOPRO, HDA 1, HDA2, SEXO, MOTIVO1, MOTIVO2, Convenio, DN, Atendimento, PPA, NORMAL X ANORMAL
def standardize_labels(df, col, mapping):
    original = df[col]
    key = original.astype('string').str.strip().str.lower()
    new = key.map(mapping)

    unmapped = original.notna() & new.isna()
    if unmapped.any():
        print(f"  [WARNING] {col}: values not in the mapping (kept as they are): "
              f"{original[unmapped].unique().tolist()}")
        new[unmapped] = original[unmapped]

    changed = (original.notna() & (new != original)).fillna(False)
    print(f"  {col}: {original.nunique()} unique values -> {new.nunique()}; "
          f"{changed.sum()} values changed")
    df[col] = new
    return df

def clean_dates(df):
    # TODO phase 2
    return df

def clean_binary(df):      # SEXO, NORMAL X ANORMAL
    print("\nBinary variables")

    sex_map = {'m': 'M', 'masculino': 'M',
               'f': 'F', 'feminino': 'F',
               'indeterminado': 'Indeterminado'}
    df = standardize_labels(df, 'SEXO', sex_map)

    target_map = {'normal': 'Normal', 'normais': 'Normal',
                  'anormal': 'Anormal'}
    df = standardize_labels(df, 'NORMAL X ANORMAL', target_map)

    for col in ['SEXO', 'NORMAL X ANORMAL']:
        print(df[col].value_counts(dropna=False).to_string())
    return df

def clean_nominal(df):     # PULSOS, B2, SOPRO, HDA 1, HDA2, MOTIVO1, MOTIVO2, Convenio
    # TODO phase 2
    return df


def clean_ordinal(df):     # PPA
    # TODO phase 2
    return df


def clean_categorical(df):
    print("\nCleaning categorical and date variables")
    df = clean_dates(df)
    df = clean_binary(df)
    df = clean_nominal(df)
    df = clean_ordinal(df)
    return df


def main():
    df = pd.read_excel(RAW_PATH)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_numerical(df.copy())
    df = clean_categorical(df.copy())

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nCleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()