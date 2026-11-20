"""
Works with the raw Excel file AND with the cleaned CSV, so the same
scripts can be reused: the file is chosen in the terminal!
"""
import os
import re
import argparse
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(ROOT, "figures")
DEFAULT_INPUT = "UCMF(5).xls"

ALIASES = {'dn': 'data_nascimento', 'hda2': 'hda_2',
           'motivo1': 'motivo_1', 'motivo2': 'motivo_2'}
DATE_COLS = ['atendimento', 'data_nascimento']


def get_input_path(description):
    """Read the input file from the terminal. """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input', nargs='?', default=DEFAULT_INPUT,
                        help="raw .xls/.xlsx or cleaned .csv (default: %(default)s)")
    path = parser.parse_args().input
    if not os.path.exists(path):                   
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {parser.parse_args().input}")
    return path


def _normalise_name(name):
    name = re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')
    return ALIASES.get(name, name)


def is_raw(path):
    return path.lower().endswith(('.xls', '.xlsx'))


def load_dataset(path):
    """Load the file and give the columns the same names in both versions."""
    if is_raw(path):
        df = pd.read_excel(path)
    else:
        header = [_normalise_name(c) for c in pd.read_csv(path, nrows=0).columns]
        df = pd.read_csv(path, parse_dates=[c for c in DATE_COLS if c in header])
    df.columns = [_normalise_name(c) for c in df.columns]
    return df


def parse_dates(series):
    """Dates -> datetime. Already-clean dates are returned as they are; for raw data,
    Excel serial numbers are converted and impossible text (typos) becomes missing."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    numeric = pd.to_numeric(series, errors='coerce')
    excel_mask = numeric.notna() & (numeric > 10000) & (numeric < 100000)
    excel_dates = pd.to_datetime(numeric[excel_mask], unit='D', origin='1899-12-30')
    text_dates = pd.to_datetime(series[~excel_mask], errors='coerce',
                                dayfirst=True, format='mixed')
    text_dates = text_dates.where(text_dates.dt.year.between(1900, 2100))
    out = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    out.loc[excel_mask] = excel_dates.astype('datetime64[ns]')
    out.loc[~excel_mask] = text_dates.astype('datetime64[ns]')
    return out


def tag_for(path):
    """'raw' for the Excel file, otherwise the file name (e.g. 'UCMF_cleaned')."""
    if is_raw(path):
        return 'raw'
    return re.sub(r'[^A-Za-z0-9_]+', '', os.path.splitext(os.path.basename(path))[0])


def data_label(tag):
    return "Raw Data" if tag == 'raw' else "Cleaned Data"


def figure_path(name, tag):
    """figures/<name>_<tag>.png (the folder is created if needed)"""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, f"{name}_{tag}.png")