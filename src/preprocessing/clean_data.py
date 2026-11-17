'''Script for the data cleaning process'''
import os
import unicodedata
import re
import pandas as pd
import numpy as np
from mappings import ANOMALY_LOOKUP, CONVENIO_LOOKUP, SEX_LOOKUP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_PATH = os.path.join(ROOT, 'UCMF(5).xls')
OUT_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(OUT_DIR, "UCMF_cleaned.csv")

# resolving inconsistencies
def rename_columns(df):
    '''Changing feature names to avoid spaces and special characters'''
    df = df.rename(columns={
        'ID': 'id',
        'Peso': 'peso',
        'Altura': 'altura',
        'IMC': 'imc',
        'Atendimento': 'atendimento',
        'DN': 'data_nascimento',
        'IDADE': 'idade',
        'Convenio': 'convenio',
        'PULSOS': 'pulsos',
        'PA SISTOLICA': 'pa_sistolica',
        'PA DIASTOLICA': 'pa_diastolica',
        'PPA': 'ppa',
        'NORMAL X ANORMAL': 'normal_x_anormal',
        'B2': 'b2',
        'SOPRO': 'sopro',
        'FC': 'fc',
        'HDA 1': 'hda_1',
        'HDA2': 'hda_2',
        'SEXO': 'sexo',
        'MOTIVO1': 'motivo_1',
        'MOTIVO2': 'motivo_2',
    })
    return df

# dealing with duplicate rows/objects
def remove_duplicates(df):
    '''Removing exact duplicate rows'''
    initial_rows = len(df)
    # step 1: check for 100% duplicate rows
    print(f"  Removing {df.duplicated().sum()} duplicate rows")
    df = df.drop_duplicates()
    # step 2: check for duplicate IDs on the same date (keep the first one)
    if {"id", "atendimento"}.issubset(df.columns):
        valid_visit_key = df["id"].notna() & df["atendimento"].notna()
        same_visit_mask = df.duplicated(subset=["id", "atendimento"],
                                        keep="first") & valid_visit_key
        print(f"  Removing {same_visit_mask.sum()} duplicate visits (same id and atendimento)")
        df = df[~same_visit_mask]
    # step 3: check for duplicates where two patients share
    # identical clinical and demographic data, with different or missing 'id'
    non_id_cols = [col for col in df.columns if col != "id"]
    quasi_id_cols = [c for c in ["data_nascimento", "atendimento",
                                 "peso", "altura", "pa_sistolica", "pa_diastolica",
                                 "fc"] if c in df.columns]
    has_quasi_id = df[quasi_id_cols].notna().any(axis=1)

    feature_dups_mask = df.duplicated(subset=non_id_cols, keep="first") & has_quasi_id
    print(f"  Removing {feature_dups_mask.sum()} non-ID full-record duplicate rows")
    df = df[~feature_dups_mask].reset_index(drop=True)

    print(f"  Total duplicate rows removed: {initial_rows - len(df)}")
    return df

# text normalization for categorical variables
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

# convenio normalization: group rare insurers into "Outro" and standardize common ones
def clean_convenio(df):
    '''Normalize text, group insurers according to name similarity and remove numbers.'''
    s = normalize_text_key(df["convenio"])
    s = s.where(~s.str.isdigit().fillna(False), np.nan)
    df["convenio"] = s.replace(CONVENIO_LOOKUP)
    return df
    # numbers in convenio should be analysed separately,
    # as they may represent dates that are wrongly misplaced in the convenio column
    # the best option is to treat the dates before the convenio,
    # so that the numbers present are not misinterpreted as convenio values

# pulsos normalization: applying text normalization
def clean_pulsos(df):
    '''Normalize text in the 'pulsos' column.'''
    df["pulsos"] = normalize_text_key(df["pulsos"])
    return df

# ppa normalization: applying text normalization
def clean_ppa(df):
    '''Normalize text in the 'ppa' column and remove #VALUE! entries.'''
    s = normalize_text_key(df["ppa"])
    df["ppa"] = s.replace("#VALUE!", np.nan)
    return df

# normal x anormal normalization: applying text normalization
def clean_normal_x_anormal(df):
    '''Normalize text in the 'normal_x_anormal' column.'''
    s = normalize_text_key(df["normal_x_anormal"])
    s = s.replace(ANOMALY_LOOKUP)
    df["normal_x_anormal"] = s
    return df

# b2 normalization: applying text normalization
def clean_b2(df):
    '''Normalize text in the 'b2' column.'''
    df["b2"] = normalize_text_key(df["b2"])
    return df

# sopro normalization: applying text normalization
def clean_sopro(df):
    '''Normalize text in the 'sopro' column.'''
    df["sopro"] = normalize_text_key(df["sopro"])
    return df

# hda normalization: applying text normalization and specific rules for missing values
def clean_hda(df):
    '''Normalize text in the 'hda_1' and 'hda_2' columns
    and apply specific rules to handle missing values and inconsistencies.'''
    df["hda_1"] = normalize_text_key(df["hda_1"])
    df["hda_2"] = normalize_text_key(df["hda_2"])
    # se hda_2 == NaN  and hda_1 != NaN -> hda_2 = "NENHUM"
    # Se hda_1 == NaN e hda_2 != NaN -> promove hda_2 para hda_1
    mask = df["hda_1"].isna() & df["hda_2"].notna()
    df.loc[mask, "hda_1"] = df.loc[mask, "hda_2"]
    df.loc[mask, "hda_2"] = np.nan

    # hda_1 == "ASSINTOMATICO" mas hda_2 tem sintoma real -> promove hda_2
    mask = (
        (df["hda_1"] == "ASSINTOMATICO")
        & df["hda_2"].notna()
        & (df["hda_2"] != "ASSINTOMATICO")
    )
    df.loc[mask, "hda_1"] = df.loc[mask, "hda_2"]
    df.loc[mask, "hda_2"] = np.nan

    # Remover redundâncias em hda_2 (igual a "ASSINTOMATICO" ou igual a hda_1)
    redundant = (df["hda_2"] == "ASSINTOMATICO") | (df["hda_2"] == df["hda_1"])
    df.loc[redundant, "hda_2"] = np.nan

    # Preencher "NENHUM" em hda_2 sempre que hda_1 esteja preenchido
    mask = df["hda_1"].notna() & df["hda_2"].isna()
    df.loc[mask, "hda_2"] = "NENHUM"
    return df

# sexo normalization: applying text normalization
def clean_sexo(df):
    '''Normalize text in the 'sexo' column.'''
    s = normalize_text_key(df["sexo"])
    s = s.replace(SEX_LOOKUP)
    df["sexo"] = s
    return df

# motivo normalization: applying text normalization
def clean_motivo(df):
    '''Normalize text in the 'motivo_1' and 'motivo_2' columns.'''
    df["motivo_1"] = normalize_text_key(df["motivo_1"])
    df["motivo_2"] = normalize_text_key(df["motivo_2"])
    # se motivo_1 == NaN e motivo_2 != NaN, motivo_1 = motivo_2 e motivo_2 = "NENHUM"
    mask = df["motivo_1"].isna() & df["motivo_2"].notna()
    if mask.any():
        df.loc[mask, "motivo_1"] = df.loc[mask, "motivo_2"]
        df.loc[mask, "motivo_2"] = "NENHUM"
    # se motivo_1 == motivo_2 , motivo_1 != NaN, motivo_2 = "NENHUM"
    mask = df["motivo_1"] == df["motivo_2"]
    if mask.any():
        df.loc[mask & df["motivo_1"].notna(), "motivo_2"] = "NENHUM"
    # se motivo_1 != NaN e motivo_2 == NaN, motivo_2 = "NENHUM"
    mask = df["motivo_1"].notna() & df["motivo_2"].isna()
    if mask.any():
        df.loc[mask, "motivo_2"] = "NENHUM"
    return df

# NUMERICAL VARIABLES
# Columns: IDADE, Peso, Altura, IMC, PA SISTOLICA, PA DIASTOLICA, FC
def clean_numerical(df):
    print("\n\nCleaning numerical variables...")
    return df


# CATEGORICAL AND DATE VARIABLES
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

def compact_text(value):
    """Lowercase, no accents, no spaces or punctuation: 'REAL S.' -> 'reals'"""
    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r'[^a-z0-9]', '', text)

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
    print("\n-Ordinal variable (PPA)")
    ppa = df['PPA'].astype('string').str.strip()

    status = pd.Series('Calculated', index=df.index, dtype='string')
    status[ppa == 'Não Calculado'] = 'Not calculated'
    status[ppa == '#VALUE!'] = 'Excel error'
    status[ppa.isna()] = 'Empty'
    df['PPA_status'] = status

    # 'Não Calculado' and '#VALUE!' are not blood pressure classes: set to missing
    not_available = ppa.isin(['Não Calculado', '#VALUE!'])
    df.loc[not_available, 'PPA'] = pd.NA
    print(f"  'Não Calculado' and '#VALUE!' set to missing: {not_available.sum()}")

    valid = ['Normal', 'Pre-Hipertensão PAS', 'Pre-Hipertensão PAD',
             'HAS-1 PAS', 'HAS-1 PAD', 'HAS-2 PAS', 'HAS-2 PAD']
    df = standardize_labels(df, 'PPA', make_map(valid))

    print(f"  PPA missing after cleaning: {df['PPA'].isna().sum()} "
          f"({df['PPA'].isna().mean() * 100:.2f}%)")
    print(df['PPA_status'].value_counts().to_string())
    print(df['PPA'].value_counts(dropna=False).to_string())
    return df


def clean_categorical(df):
    print("\n\nCleaning categorical and date variables")
    df = clean_dates(df)
    df = clean_binary(df)
    df = clean_nominal(df)
    df = clean_ordinal(df)
    df = clean_convenio(df)
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