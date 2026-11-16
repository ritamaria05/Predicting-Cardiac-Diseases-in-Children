import os
import pandas as pd
import numpy as np
import unicodedata
import re

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

CONVENIO_RULES = [
    (r'^cass', 'Cassi'),
    (r'^(camed|acmed)', 'Camed'),
    (r'^(amep|campe)', 'Amepe/Campe'),
    (r'^media', 'Medial'),
    (r'^(medi?s|mede?s|mers)', 'Mediservice'),
    (r'^nor[cl]', 'Norclínicas'),
    (r'^real', 'Real Saúde'),
    (r'^(reemb|ree)', 'Reembolso'),
    (r'^(ideal|odeal)', 'Ideal'),
    (r'^fusex', 'Fusex'),
    (r'^semep', 'Semepe'),
    (r'^fun[cs]ef', 'Funcef'),
    (r'^fac[hj]a?es', 'Fachesf'),
    (r'^celpos', 'Celpos'),
    (r'^(santa|sta)cl', 'Santa Clara'),
    (r'^(santa|sta)lu', 'Santa Lúcia'),
    (r'^(santa|sta|sant|st|s)h(e|er)', 'Santa Helena'),
    (r'^(circ|ccor)', 'Círculo'),
    (r'^cort', 'Cortesia'),
    (r'^(corr|core)', 'Correios'),
    (r'^(part|patrt)', 'Particular'),
    (r'^petro(b|f$|$)', 'Petrobras'),
    (r'^embr', 'Embratel'),
    (r'^blu', 'Bluelife'),
    (r'^(caixa|saudecaixa|scaixa|scx)', 'Caixa'),
    (r'^gama', 'Gama'),
    (r'^assef', 'Assefaz'),
    (r'^fisco', 'Fisco'),
    (r'^(tab|tba)', 'Tabela'),
    (r'^(sul|samer|ameri)', 'Sul América'),
    (r'^((saude|s)rec|saudere)', 'Saúde Recife'),
    (r'^(saude|s)?b[ar]+desco', 'Bradesco Saúde'),
    (r'^(urec|unrec|unimrec|unimedr)', 'Unimed Recife'),
]
MIN_ROWS_CONVENIO = 30

def clean_convenio(df):
    print("\nConvenio")
    original = df['Convenio']
    text = original.astype('string').str.strip()

    # numbers are not insurers
    is_number = text.str.fullmatch(r'\d+(\.0+)?').fillna(False)

    # normalise and group the families
    def standardize(value):
        key = compact_text(value)
        for pattern, label in CONVENIO_RULES:
            if re.match(pattern, key):
                return label
        return key.upper()

    new = text.map(standardize, na_action='ignore').astype('string')
    new[is_number] = pd.NA

    print(f"  Convenio: {original.nunique()} unique values -> {new.nunique()}")
    print(f"  Numbers (not insurers) set to missing: {is_number.sum()}")
    print(f"  Missing after cleaning: {new.isna().sum()} ({new.isna().mean() * 100:.2f}%)")
    df['Convenio'] = new

    # new column with the rare insurers grouped
    counts = new.value_counts()
    rare = counts[counts < MIN_ROWS_CONVENIO].index
    df['Convenio_group'] = new.where(~new.isin(rare), 'Outro')
    print(f"  Convenio_group: {df['Convenio_group'].nunique()} unique values "
          f"(insurers with < {MIN_ROWS_CONVENIO} rows -> 'Outro': "
          f"{(df['Convenio_group'] == 'Outro').sum()} rows)")
    print(df['Convenio_group'].value_counts(dropna=False).head(15).to_string())
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