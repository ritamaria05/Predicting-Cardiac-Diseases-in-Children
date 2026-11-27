"""
Summary of the categorical variables, side by side for several files
(before and after cleaning), plus target distribution and class balance.

Usage:
    python3 src/data_description/categorical_summary.py
        -> compares: raw Excel | data/UCMF_cleaned.csv | data/UCMF_final.csv
    python3 src/data_description/categorical_summary.py "UCMF(5).xls" data/UCMF_final.csv
        -> compares the files you choose
Output: printed tables + results/categorical_summary.csv, results/target_balance.csv
        + figures/target_distribution.png
"""
import os
import sys
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import ROOT, FIG_DIR, load_dataset, tag_for

DEFAULT_FILES = ["UCMF(5).xls", "data/UCMF_cleaned.csv", "data/UCMF_final.csv"]
TARGET = "normal_x_anormal"
BLUE, ORANGE, GREY = "#2b5c8f", "#d95f02", "#9e9e9e"

CAT_COLS = ["sexo", "normal_x_anormal", "pulsos", "b2", "sopro", "ppa",
            "convenio", "motivo_1", "motivo_2", "hda_1", "hda_2"]

# Standard labels of the cleaned data (compared after removing case, accents and punctuation).
# 'needs_recoding' = rows whose label is NOT one of these. It should be 0 after cleaning.
HDA = ["ASSINTOMATICO", "DOR PRECORDIAL", "DISPNEIA", "PALPITACAO", "CIANOSE",
       "DESMAIO TONTURA", "OUTRO", "GANHO DE PESO"]
EXPECTED = {
    "sexo": ["M", "F"],
    "normal_x_anormal": ["NORMAL", "ANORMAL"],
    "pulsos": ["NORMAIS", "AMPLOS", "OUTRO", "FEMORAIS DIMINUIDOS", "DIMINUIDOS"],
    "b2": ["NORMAL", "HIPERFONETICA", "DESDOB FIXO", "OUTRO", "UNICA"],
    "sopro": ["AUSENTE", "SISTOLICO", "CONTINUO", "DIASTOLICO", "SISTOLICO E DIASTOLICO"],
    "ppa": ["NAO CALCULADO", "NORMAL", "PRE-HIPERTENSAO PAS", "PRE-HIPERTENSAO PAD",
            "HAS-1 PAS", "HAS-1 PAD", "HAS-2 PAS", "HAS-2 PAD"],
    "hda_1": HDA,
    "hda_2": HDA + ["NENHUM"],
    "motivo_1": ["5 - PARECER CARDIOLOGICO", "6 - SUSPEITA DE CARDIOPATIA",
                 "1 - CARDIOPATIA JA ESTABELECIDA", "2 - CHECK-UP", "7 - OUTRO"],
    "motivo_2": ["NENHUM", "OUTRO", "5 - CIRURGIA", "6 - SOPRO", "1 - CARDIOPATIA CONGENICA",
                 "5 - ATIVIDADE FISICA", "6 - DOR PRECORDIAL",
                 "6 - PALPITACAO TAQUICARDIA ARRITMIA", "6 - HAS DISLIPIDEMIA OBESIDADE",
                 "6 - DISPNEIA", "6 - CIANOSE", "1 - CARDIOPATIA ADQUIRIDA",
                 "6 - CARDIOPATIA NA FAMILIA", "6 - CANSACO",
                 "5 - USO DE CISAPRIDA", "6 - ALTERACOES DE PULSO PERFUSAO",
                 "6 - CIANOSE E DISPNEIA"],
}
# convenio has no fixed list (many insurers): only unique counts are reported


def resolve(path):
    if os.path.exists(path):
        return path
    alt = os.path.join(ROOT, path)
    if os.path.exists(alt):
        return alt
    raise SystemExit(f"File not found: {path}")


def _strip_accents(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def norm(series):
    """Uppercase, no accents, punctuation -> space ('Palpitação/taquicardia' -> 'PALPITACAO TAQUICARDIA')."""
    s = series.astype("string").str.strip().str.upper()
    s = s.map(_strip_accents, na_action="ignore")
    s = s.str.replace(r"[^A-Z0-9\-]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def summarise(df, col):
    s = df[col]
    n = int(s.count())
    ns = norm(s)
    vc = ns.value_counts()
    row = {
        "rows": len(df),
        "non_null": n,
        "missing_%": round(s.isna().mean() * 100, 2),
        "unique": int(s.nunique()),
        "unique_ignoring_case": int(ns.nunique()),
        "top_value": vc.index[0] if len(vc) else None,
        "top_%": round(vc.iloc[0] / n * 100, 1) if n else None,
        "rare_<1%": int((vc / n < 0.01).sum()) if n else 0,
    }
    if col in EXPECTED:
        row["needs_recoding"] = int((ns.notna() & ~ns.isin(EXPECTED[col])).sum())
    else:
        row["needs_recoding"] = None
    return row


files = [resolve(p) for p in (sys.argv[1:] or DEFAULT_FILES)]
datasets = {tag_for(p): load_dataset(p) for p in files}

# ---------------------------------------------------------------------
# 1. Summary per variable and file
# ---------------------------------------------------------------------
rows = {}
for col in CAT_COLS:
    for tag, df in datasets.items():
        if col not in df.columns:
            print(f"[WARNING] column '{col}' not found in {tag}")
            continue
        rows[(col, tag)] = summarise(df, col)

table = pd.DataFrame(rows).T
for c in ["rows", "non_null", "unique", "unique_ignoring_case", "rare_<1%"]:
    table[c] = table[c].astype(int)
table.index.names = ["variable", "file"]

print("CATEGORICAL VARIABLES, before vs after")
print("(unique_ignoring_case: unique labels ignoring case, accents and punctuation;")
print(" needs_recoding: rows whose label is not one of the standard labels, should be 0 after cleaning)\n")
print(table.to_string())

# Unexpected labels in the cleaned files (should be none)
print("\nUNEXPECTED LABELS in the cleaned files:")
found = False
for tag, df in datasets.items():
    if tag == "raw":
        continue
    for col, expected in EXPECTED.items():
        if col not in df.columns:
            continue
        ns = norm(df[col])
        bad = ns[ns.notna() & ~ns.isin(expected)].value_counts().head(5)
        if len(bad):
            found = True
            print(f"  [WARNING] {tag} | {col}: {bad.to_dict()}")
if not found:
    print("  none")

# ---------------------------------------------------------------------
# 2. Target distribution and class balance
# ---------------------------------------------------------------------
balance = {}
for tag, df in datasets.items():
    if TARGET not in df.columns:
        continue
    t = norm(df[TARGET]).replace({"NORMAIS": "NORMAL"})
    n_normal, n_abnormal = int((t == "NORMAL").sum()), int((t == "ANORMAL").sum())
    n_missing = int(t.isna().sum())
    labelled = n_normal + n_abnormal
    balance[tag] = {
        "rows": len(df), "NORMAL": n_normal, "ANORMAL": n_abnormal, "missing": n_missing,
        "other_labels": len(df) - labelled - n_missing,
        "NORMAL_%": round(n_normal / labelled * 100, 1),
        "ANORMAL_%": round(n_abnormal / labelled * 100, 1),
        "imbalance_ratio": round(max(n_normal, n_abnormal) / min(n_normal, n_abnormal), 2),
        "baseline_accuracy_%": round(max(n_normal, n_abnormal) / labelled * 100, 1),
    }
balance = pd.DataFrame(balance).T
for c in ["rows", "NORMAL", "ANORMAL", "missing", "other_labels"]:
    balance[c] = balance[c].astype(int)
balance.index.name = "file"

print("\n\nTARGET DISTRIBUTION AND CLASS BALANCE (NORMAL x ANORMAL)")
print("(percentages are over the labelled rows; baseline_accuracy = always predicting the majority class)\n")
print(balance.to_string())

# Figure: counts per file
sns.set_theme(style="whitegrid", context="notebook")
fig, ax = plt.subplots(figsize=(9, 5))
for i, (tag, r) in enumerate(balance.iterrows()):
    bottom = 0
    for label, color in [("NORMAL", BLUE), ("ANORMAL", ORANGE), ("missing", GREY)]:
        ax.bar(i, r[label], bottom=bottom, color=color, alpha=0.8, edgecolor="white", width=0.6,
               label=label if i == 0 else None)
        if r[label] > 0:
            pct = f"\n({r[label + '_%']}%)" if label != "missing" else ""
            ax.text(i, bottom + r[label] / 2, f"{int(r[label]):,}{pct}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        bottom += r[label]
ax.set_xticks(range(len(balance)))
ax.set_xticklabels(balance.index)
ax.set_ylabel("Rows")
ax.set_title("Target distribution (NORMAL x ANORMAL) before and after cleaning",
             fontsize=12, fontweight="bold", loc="left")
ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
os.makedirs(FIG_DIR, exist_ok=True)
fig_path = os.path.join(FIG_DIR, "target_distribution.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
table.to_csv(os.path.join(ROOT, "results", "categorical_summary.csv"))
balance.to_csv(os.path.join(ROOT, "results", "target_balance.csv"))
print(f"\nSaved: results/categorical_summary.csv, results/target_balance.csv, {fig_path}")
