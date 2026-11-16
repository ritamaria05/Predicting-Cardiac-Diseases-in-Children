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
    print("\n\nCleaning numerical variables")
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

def make_map(labels, variants=None):
    """Build the mapping from a list of standard labels (and optional extra variants).
    variants: {lowercase variant: standard label}"""
    mapping = {label.lower(): label for label in labels}
    if variants:
        mapping.update(variants)
    return mapping

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
    print("\nNominal variables")

    pulsos = ['Normais', 'Amplos', 'Outro', 'Femorais diminuidos', 'Diminuídos']

    b2 = ['Normal', 'Hiperfonética', 'Desdob fixo', 'Outro', 'Única']

    sopro = ['Ausente', 'Sistólico', 'Contínuo', 'Diastólico', 'Sistólico e diastólico']
    sopro_variants = {'sistolico e diastólico': 'Sistólico e diastólico'}  # missing accent

    hda = ['Assintomático', 'Dor precordial', 'Dispneia', 'Palpitacao', 'Cianose',
           'Desmaio/tontura', 'Outro', 'Ganho de peso']

    motivo1 = ['5 - Parecer cardiológico', '6 - Suspeita de cardiopatia',
               '1 - Cardiopatia já estabelecida', '2 - Check-up', '7 - Outro']

    motivo2 = ['5 - Cirurgia', '6 - Sopro', '1 - Cardiopatia congenica',
               '5 - Atividade física', '7 - Outro', '6 - Dor precordial',
               '6 - Palpitação/taquicardia/arritmia', '6 - HAS/dislipidemia/obesidade',
               '6 - Dispnéia', '6 - Cianose', '1 - Cardiopatia adquirida',
               '6 - Cardiopatia na familia', '6 - Cansaço', '5 - Uso de cisaprida',
               '6 - Alterações de pulso/perfusão', '6 - Cianose e dispnéia']
    motivo2_variants = {'outro': '7 - Outro'}

    mappings = {
        'PULSOS':  make_map(pulsos),
        'B2':      make_map(b2),
        'SOPRO':   make_map(sopro, sopro_variants),
        'HDA 1':   make_map(hda),
        'HDA2':    make_map(hda),
        'MOTIVO1': make_map(motivo1),
        'MOTIVO2': make_map(motivo2, motivo2_variants),
    }
    for col, mapping in mappings.items():
        df = standardize_labels(df, col, mapping)
    return df


def clean_ordinal(df):     # PPA
    # TODO phase 2
    return df


def clean_categorical(df):
    print("\n\nCleaning categorical and date variables")
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