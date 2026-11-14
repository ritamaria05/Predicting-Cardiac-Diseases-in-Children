import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import os

df = pd.read_excel('UCMF(5).xls')
sns.set_theme(style="whitegrid", context="notebook")
os.makedirs("figures", exist_ok=True)

cat_cols = ['Convenio', 'PULSOS', 'PPA', 'NORMAL X ANORMAL', 'B2', 'SOPRO', 'HDA 1', 'HDA2', 'SEXO', 'MOTIVO1', 'MOTIVO2']

num_vars = len(cat_cols)
cols = 2
rows = math.ceil(num_vars / cols)

fig = plt.figure(figsize=(18, 4.5 * rows))
outer_grid = gridspec.GridSpec(rows, cols, wspace=0.35, hspace=0.45)

for i, col in enumerate(cat_cols):
    ax = fig.add_subplot(outer_grid[i])
    
    s = df[col].fillna("MISSING").astype(str)
    
    # Extract the top 15 most frequent values to spot typos and main categories
    val_counts = s.value_counts().head(15)
    
    # Horizontal bar plot
    sns.barplot(
        y=val_counts.index,
        x=val_counts.values,
        color="#2b5c8f",
        alpha=0.65,
        edgecolor="white",
        ax=ax
    )
    
    # Title showing the column name and valid (non-null) count
    valid_count = df[col].notna().sum()
    ax.set_title(
        f"{col} — Raw Data Frequencies (n={valid_count:,})",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
    )
    
    ax.set_xlabel("Count", fontsize=9.5)
    ax.set_ylabel("")
    
    # Annotate the exact counts on the ends of the bars
    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=9, color="#4d4d4d")

# 3. If there is an odd number of variables, hide the leftover empty subplot grid
if num_vars % cols != 0:
    ax_empty = fig.add_subplot(outer_grid[-1])
    ax_empty.axis("off")

# Adjust the title position slightly based on how many rows there are
y_pos = 0.91 if rows > 3 else 0.96
fig.suptitle(
    "Visual Analysis of Categorical Variables (Top 15 Frequencies)", 
    fontsize=14.5, 
    fontweight="bold", 
    y=y_pos
)

plt.savefig("../../figures/categorical_preliminary_analysis.png", dpi=200, bbox_inches="tight")
plt.close()