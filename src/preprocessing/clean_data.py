'''Script for the data cleaning process'''
import os
import unicodedata
import re
import pandas as pd
import numpy as np
from mappings import (
    ANOMALY_LOOKUP, BOUNDS_BY_AGE, CONVENIO_LOOKUP,
    DN_MANUAL_FIXES, DN_TO_AGE_MAPPING, SEX_LOOKUP,
    get_bounds_by_age
)

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

def normalize_dates(series: pd.Series, max_valid_year: int = 2025) -> pd.Series:
    '''Parse a pandas Series of dates,
    handling Excel serial numbers, two-digit years,
    and future dates.'''
    raw = (series.astype('string')
           .str.strip()
           .str.replace(r"[\\,]", "/", regex=True)
           .str.replace(r"[*?]", "", regex=True)
           .str.replace(r"/-", "/", regex=True)
           .str.replace(r"/+", "/", regex=True)
    )
    # convert excel serial numbers to datetime
    nums = pd.to_numeric(raw, errors='coerce')
    serial_dates = pd.to_datetime(
        nums.where(nums.between(10000,50000)),
        unit='D',
        origin='1899-12-30',
        errors='coerce'
    )
    #DD/MM/AA to DD/MM/AAAA
    text_dates = pd.to_datetime(
        raw,
        format='%d/%m/%y',
        errors='coerce'
    )
    text_dates_4y = pd.to_datetime(
        raw,
        format='%d/%m/%Y',
        errors='coerce'
    )
    parsed = serial_dates.fillna(text_dates).fillna(text_dates_4y)
    # future dates
    future_mask = parsed.dt.year > max_valid_year
    parsed.loc[future_mask] = parsed.loc[future_mask] - pd.DateOffset(years=100)
    return parsed

def recover_dates_from_convenio(df):
    '''Recover dates from convenio column if they are Excel serial numbers,
    and fill missing dates in other columns.'''
    df = df.copy()
    conv_nums = pd.to_numeric(df["convenio"].astype("string").str.strip(), errors="coerce")
    serial_mask = conv_nums.between(20000, 45000)
    if not serial_mask.any():
        return df
    recovered_dates = pd.to_datetime(conv_nums.where(serial_mask), unit="D",
                                     origin="1899-12-30", errors="coerce")

    #1 data_nascimento missing, convenio has a date -> fill data_nascimento
    fill_dn = (serial_mask & df["data_nascimento"].isna() & (df["atendimento"].isna() | (
        df["atendimento"] >= recovered_dates)))
    df.loc[fill_dn, "data_nascimento"] = recovered_dates.loc[fill_dn]

    #2 atendimento missing, convenio has a date -> fill atendimento
    fill_atendimento = (serial_mask & df["atendimento"].isna() & (df["data_nascimento"].isna() | (
        df["data_nascimento"] <= recovered_dates)))
    df.loc[fill_atendimento, "atendimento"] = recovered_dates.loc[fill_atendimento]
    df.loc[serial_mask, "convenio"] = np.nan
    return df

def clean_dates(df):
    '''apply mapping, normalization, and fix birth dates into age'''
    df = df.copy()
    dn_clean = df['data_nascimento'].astype("string").str.strip()
    extract_age = dn_clean.map(DN_TO_AGE_MAPPING)
    current_age = pd.to_numeric(df['idade'], errors='coerce')
    df['idade'] = extract_age.fillna(current_age)
    dn_clean = dn_clean.replace(DN_MANUAL_FIXES)
    df['data_nascimento'] = normalize_dates(dn_clean)
    df['atendimento'] = normalize_dates(df['atendimento'])
    return df

def clean_age(df, max_age=20):
    '''Clean age column, converting to numeric and handling invalid values.'''
    df = df.copy()
    original_age = pd.to_numeric(df['idade'], errors='coerce')
    original_age = normalize_numbers(original_age)
    calc = (df['atendimento'] - df['data_nascimento']).dt.days / 365.25
    use_calc = calc.between(0, max_age)
    merged = original_age.where(~use_calc, calc)
    merged = merged.where(~(calc < 0))          # visit before birth: dates are wrong
    df['idade'] = merged.where(merged.between(0, max_age)).round(2)
    df['idade_origem'] = np.where(use_calc, 'calculada', np.where(df[
        'idade'].notna(), 'declarada', 'ausente'))

    return df

def normalize_numbers(series: pd.Series, allow_ranges: bool = False) -> pd.Series:
    '''convert commas into dots and remove non-numeric characters from numeric columns
    turn intervals into its mean value.'''
    if pd.api.types.is_numeric_dtype(series): # already numeric, return as float
        return series.astype(float)

    s = series.astype(str).str.strip().str.lower() # convert to string and lowercase for uniformity
    # null tokens to np.nan
    null_tokens = {"", "nan", "none", "null", "na", "n/a", "-", "--", "nd", "s/d", "#value!"}
    s = s.where(~s.isin(null_tokens), other=np.nan)
    # comma to dots
    s = s.str.replace(",", ".", regex=False)
    # ranges that are present in FC
    if allow_ranges:
        range_pattern = r"^\s*(\d+(?:\.\d+)?)\s*(?:a|-)\s*(\d+(?:\.\d+)?)\s*$"
        extracted = s.str.extract(range_pattern)
        is_range = extracted[0].notna() & extracted[1].notna()
        if is_range.any():
            mean_vals = (extracted.loc[is_range, 0].astype(float
            ) + extracted.loc[is_range, 1].astype(float)) / 2.0
            s.loc[is_range] = mean_vals.astype(str)

    # Remove units and trailing whitespace
    s = s.str.replace(r"\s*(kg|cm|m|mmhg|cmhg|bpm)\s*$", "", regex=True)

    return pd.to_numeric(s, errors="coerce")

def clean_peso_altura(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weight, height, and BMI columns, converting to numeric and handling invalid values."""
    df = df.copy()

    peso = normalize_numbers(df["peso"])
    altura = normalize_numbers(df["altura"])
    imc = normalize_numbers(df["imc"])
    idade = normalize_numbers(df["idade"])
    # apply biological plausibility bounds
    p_min, p_max = get_bounds_by_age(idade, BOUNDS_BY_AGE["peso"])
    a_min, a_max = get_bounds_by_age(idade, BOUNDS_BY_AGE["altura"])
    bad_p = peso.notna() & ~peso.between(p_min, p_max)
    bad_a = altura.notna() & ~altura.between(a_min, a_max)
    imc_calc = peso / (altura / 100) ** 2
    g_p = BOUNDS_BY_AGE["peso"]["unknown"]
    g_a = BOUNDS_BY_AGE["altura"]["unknown"]
    age_suspect = (bad_p | bad_a) & peso.gt(0) & altura.gt(0) & imc_calc.between(
        10, 40) & peso.between(*g_p) & altura.between(*g_a)
    df.loc[age_suspect, "idade"] = np.nan
    df["idade_suspeita"] = age_suspect
    peso[bad_p & ~age_suspect] = np.nan
    altura[bad_a & ~age_suspect] = np.nan
    imc.loc[imc.notna() & ~imc.between(6.0, 100.0)] = np.nan

    # recalculate IMC consistently whenever both cleaned weight and height are valid
    valid_pa = peso.notna() & altura.notna()
    imc.loc[valid_pa] = (peso.loc[valid_pa] / ((altura.loc[valid_pa] / 100.0) ** 2)).round(2)
    isolated_imc = ~valid_pa & altura.notna() & ~imc.between(6.0, 100.0)
    imc.loc[isolated_imc] = np.nan

    df["peso"] = peso
    df["altura"] = altura
    imc = imc.where(peso.notna() & altura.notna())
    df["imc"] = imc

    return df

def clean_pa_fc(df: pd.DataFrame) -> pd.DataFrame:
    '''Clean blood pressure and heart rate columns, applying biological
    plausibility bounds and consistency checks.'''
    df = df.copy()
    pa_sistolica = normalize_numbers(df["pa_sistolica"])
    pa_diastolica = normalize_numbers(df["pa_diastolica"])
    fc = normalize_numbers(df["fc"], allow_ranges=True)
    idade = normalize_numbers(df["idade"])

    pas_min, pas_max = get_bounds_by_age(idade, BOUNDS_BY_AGE["pa_sistolica"])
    pad_min, pad_max = get_bounds_by_age(idade, BOUNDS_BY_AGE["pa_diastolica"])
    fc_min, fc_max = get_bounds_by_age(idade, BOUNDS_BY_AGE["fc"])

    # biological plausibility bounds
    pa_sistolica.loc[pa_sistolica.notna() & ~pa_sistolica.between(pas_min, pas_max)] = np.nan
    pa_diastolica.loc[pa_diastolica.notna() & ~pa_diastolica.between(pad_min, pad_max)] = np.nan
    fc.loc[fc.notna() & ~fc.between(fc_min, fc_max)] = np.nan

    # pulse pressure (PP): DBP <= SBP - 5 and SBP - DBP <= 160
    both_bp = pa_sistolica.notna() & pa_diastolica.notna()
    invalid_pulse = both_bp & (
        (pa_diastolica > (pa_sistolica - 5.0))
        | ((pa_sistolica - pa_diastolica) > 160.0)
    )
    pa_sistolica.loc[invalid_pulse] = np.nan
    pa_diastolica.loc[invalid_pulse] = np.nan

    # pediatric shock index (PSI): 0.15 <= PSI <= 5.0
    psi = fc / pa_sistolica
    invalid_psi = psi.notna() & ~psi.between(0.15, 5.0)
    pa_sistolica.loc[invalid_psi] = np.nan
    pa_diastolica.loc[invalid_psi] = np.nan
    fc.loc[invalid_psi] = np.nan

    df["pa_sistolica"] = pa_sistolica
    df["pa_diastolica"] = pa_diastolica
    df["fc"] = fc

    return df

def remove_empty_rows(df, min_fields=3):
    '''Remove rows that have fewer than a specified number of non-null fields in key columns.'''
    # id is always present
    # sexo isn't that useful in the records so that it can't be missing
    # imc comes from weight and height
    # pa_diastolica pairs with pa_sistolica, so if one is present the other should be too
    # ppa is derived from bp and also holds "NAO CALCULADO"
    # hda_2 and motivo 2 dont carry extra information if hda_1 and motivo_1 are present
    # dates are summarized by age
    info_cols = ["idade", "peso", "altura", "pa_sistolica", "fc", "normal_x_anormal",
                 "sopro", "b2", "pulsos", "convenio", "hda_1", "motivo_1"]
    empty = df[info_cols].notna().sum(axis=1) < min_fields
    print(f"  Removing {empty.sum()} nearly empty rows")
    return df[~empty].reset_index(drop=True)

# dealing with duplicate rows/objects
def remove_duplicates(df):
    '''Removing exact duplicate rows'''
    initial_rows = len(df)
    # step 1: check for 100% duplicate rows
    print(f"  Removing {df.duplicated().sum()} duplicate rows")
    df = df.drop_duplicates()
    # step 2: check for duplicate rows with same features but different IDs
    non_id_cols = [col for col in df.columns if col != "id"]
    strong_key = df["data_nascimento"].notna() & df["atendimento"].notna()
    feature_dups_mask = df.duplicated(subset=non_id_cols, keep="first") & strong_key
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
        return np.nan if s in {"", "NAN", "NONE", "NULL", "NA", "?", "#VALUE!"} else s

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
    df["sexo"] = s.where(s.isin({"M", "F"}))
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




def main():
    '''Main function to execute the data cleaning process.'''
    df = pd.read_excel(RAW_PATH)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    #1 Rename columns
    df = rename_columns(df)
    #2 Clean dates
    df = clean_dates(df)
    #3 Recover dates from convenio
    df = recover_dates_from_convenio(df)
    #4 Clean age
    df = clean_age(df)
    #5 Clean numeric features
    df = clean_peso_altura(df)
    df = clean_pa_fc(df)
    #6 Clean categorical features
    df = clean_convenio(df)
    df = clean_pulsos(df)
    df = clean_ppa(df)
    df = clean_normal_x_anormal(df)
    df = clean_b2(df)
    df = clean_sopro(df)
    df = clean_hda(df)
    df = clean_sexo(df)
    df = clean_motivo(df)
    #7 remove empty rows
    df = remove_empty_rows(df)
    #8 Remove duplicates
    df = remove_duplicates(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nCleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
