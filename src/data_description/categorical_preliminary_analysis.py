'''Categorical and Date Variables Preliminary Analysis'''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

from load_data import (get_input_path, load_dataset, parse_dates,tag_for, data_label, figure_path, DATE_COLS)

path = get_input_path("Preliminary analysis of categorical and date variables")
df = load_dataset(path)
tag = tag_for(path)
label = data_label(tag)
sns.set_theme(style="whitegrid", context="notebook")
BLUE, ORANGE = "#2b5c8f", "#d95f02"

cat_cols = ['convenio', 'pulsos', 'ppa', 'normal_x_anormal', 'b2', 'sopro','hda_1', 'hda_2', 'sexo', 'motivo_1', 'motivo_2', 'idade_origem']
cat_cols = [c for c in cat_cols if c in df.columns] 

# --- FIGURE 1: Categorical Variables ---
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

fig.suptitle(f"Visual Analysis of Categorical Variables with {label} (Top 15 Frequencies)",fontsize=14.5, fontweight="bold", y=0.93 if n_rows > 3 else 0.97)
out1 = figure_path("categorical_preliminary_analysis", tag)
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()


# --- FIGURE 2: Date Variables ---
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
for ax, col in zip(axes, DATE_COLS):
    s = parse_dates(df[col]).dropna()
    years = s.dt.year.value_counts().sort_index()
    ax.bar(years.index, years.values, color=BLUE, alpha=0.7, edgecolor="white", width=0.8)
    ax.set_yscale("log")
    ax.axvline(s.median().year, color=ORANGE, linestyle="--", linewidth=1.3,
               label=f"Median: {s.median().date()}")
    ax.set_title(f"{col} — {label}, dates per year (n={len(s):,})",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count (log scale)")
    ax.legend(fontsize=8, loc="upper left", frameon=True)

fig.suptitle(f"Visual Analysis of Date Variables with {label}",
             fontsize=13.5, fontweight="bold", y=1.02)
out2 = figure_path("date_preliminary_analysis", tag)
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()

print(f"Input: {path}")
print(f"Saved: {out1}")
print(f"Saved: {out2}")