import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format")

df = pd.read_excel('UCMF(5).xls')
sns.set_theme(style="whitegrid", context="notebook")

cat_cols = ['Convenio', 'PULSOS', 'PPA', 'NORMAL X ANORMAL', 'B2', 'SOPRO','HDA 1', 'HDA2', 'SEXO', 'MOTIVO1', 'MOTIVO2']
dt_cols = ['Atendimento', 'DN']
BLUE, ORANGE = "#2b5c8f", "#d95f02"

# --- Categorical Variables ---
n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)

fig = plt.figure(figsize=(18, 4.5 * n_rows))
grid = gridspec.GridSpec(n_rows, n_cols, wspace=0.45, hspace=0.45)

for i, col in enumerate(cat_cols):
    ax = fig.add_subplot(grid[i])

    s = df[col].fillna("MISSING").astype(str)
    val_counts = s.value_counts().head(15)
    labels = [l[:35] for l in val_counts.index]
    colors = [ORANGE if l == "MISSING" else BLUE for l in val_counts.index]

    ax.barh(labels, val_counts.values, color=colors, alpha=0.7, edgecolor="white")
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], padding=4, fontsize=9, color="#4d4d4d")

    valid = df[col].notna().sum()
    miss_pct = df[col].isna().mean() * 100
    ax.set_title(
        f"{col} — n={valid:,} | missing {miss_pct:.1f}% | unique={df[col].nunique()}",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.set_xlabel("Count", fontsize=9.5)
    ax.set_ylabel("")
    ax.margins(x=0.12)

# Hide the empty subplot if there's an odd number of variables
if len(cat_cols) % n_cols != 0:
    fig.add_subplot(grid[-1]).axis("off")

fig.suptitle("Visual Analysis of Categorical Variables with Raw Data", fontsize=14.5, fontweight="bold", y=0.93 if n_rows > 3 else 0.97)
plt.savefig("../../figures/categorical_preliminary_analysis.png", dpi=200, bbox_inches="tight")
plt.close()


# --- Date Variables ---
def parse_dates(raw):
    numeric = pd.to_numeric(raw, errors='coerce')
    excel_mask = numeric.notna() & (numeric > 10000) & (numeric < 100000)
    excel_dates = pd.to_datetime(numeric[excel_mask], unit='D', origin='1899-12-30')

    text_dates = pd.to_datetime(raw[~excel_mask], errors='coerce', dayfirst=True, format='mixed')
    text_dates = text_dates.where(text_dates.dt.year.between(1900, 2100))

    s = pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns]')
    s.loc[excel_mask] = excel_dates.astype('datetime64[ns]')
    s.loc[~excel_mask] = text_dates.astype('datetime64[ns]')
    return s

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
for ax, col in zip(axes, dt_cols):
    s = parse_dates(df[col]).dropna()
    years = s.dt.year.value_counts().sort_index()
    ax.bar(years.index, years.values, color=BLUE, alpha=0.7, edgecolor="white", width=0.8)
    ax.set_yscale("log")
    
    # Calculate median timestamp to extract the date and year safely
    median_ts = s.median()
    ax.axvline(median_ts.year, color=ORANGE, linestyle="--", linewidth=1.3, label=f"Median: {median_ts.date()}")
    
    ax.set_title(f"{col} — Raw Data, parsed dates per year (n={len(s):,})", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count (log scale)")
    ax.legend(fontsize=8, loc="upper left", frameon=True)

fig.suptitle("Visual Analysis of Date Variables with Raw Data", fontsize=13.5, fontweight="bold", y=1.02)
plt.savefig("../../figures/date_preliminary_analysis.png", dpi=200, bbox_inches="tight")
plt.close()