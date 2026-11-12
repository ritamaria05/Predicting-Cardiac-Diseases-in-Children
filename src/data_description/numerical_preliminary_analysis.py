import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

df = pd.read_excel('UCMF(5).xls')
sns.set_theme(style="whitegrid", context="notebook")
os.makedirs("figures", exist_ok=True)


num_cols = ["IDADE", "Peso", "Altura", "IMC", "PA SISTOLICA", "PA DIASTOLICA", "FC"]
units = {
    "IDADE": "years",
    "Peso": "kg",
    "Altura": "cm",
    "IMC": "kg/m²",
    "PA SISTOLICA": "mmHg",
    "PA DIASTOLICA": "mmHg",
    "FC": "bpm",    
}

# Convert columns to numeric, coercing errors to NaN (there are some in the dataset)
df_num = pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce") for col in num_cols})

fig = plt.figure(figsize=(16, 15))
outer_grid = gridspec.GridSpec(4, 2, wspace=0.22, hspace=0.45)

for i, col in enumerate(num_cols):
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_grid[i], height_ratios=[3.0, 1.1], hspace=0.06
    )
    ax_hist = fig.add_subplot(inner_grid[0])
    ax_box = fig.add_subplot(inner_grid[1], sharex=ax_hist)

    s = df_num[col].dropna()

    # histogram + density plot
    sns.histplot(
        s,
        bins=50,
        kde=True,
        kde_kws={"cut": 0},
        stat="count",
        color="#2b5c8f",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.5,
        ax=ax_hist,
    )

    # log scale for y-axis to better visualize the distribution of counts, especially for skewed data
    ax_hist.set_yscale("log")
    ax_hist.set_ylim(0.8, len(s) * 2.5)
    ax_hist.set_xlabel("")

    med = s.median()
    ax_hist.axvline(med, color="#d95f02", linestyle="--", linewidth=1.3, label=f"Median: {med:.1f}")
    ax_hist.axvline(s.min(), color="#e7298a", linestyle=":", linewidth=1.2, label=f"Min: {s.min():.1f}")
    ax_hist.axvline(s.max(), color="#7570b3", linestyle=":", linewidth=1.2, label=f"Max: {s.max():.1f}")

    ax_hist.set_title(
        f"{col} ({units[col]}) — Raw Data (n={len(s):,})",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    ax_hist.set_ylabel("Count (log scale)", fontsize=8.5)
    ax_hist.legend(fontsize=8, loc="upper right", frameon=True)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

    # boxplot 
    sns.boxplot(
        x=s,
        ax=ax_box,
        color="#5b8db8",
        width=0.55,
        fliersize=3.5,
        flierprops=dict(marker="o", markerfacecolor="#d95f02", markeredgecolor="none", alpha=0.75),
    )
    ax_box.set_xlabel(f"{col} ({units[col]})  |  Range: [{s.min():.1f}, {s.max():.1f}]", fontsize=9)
    ax_box.set_yticks([])

ax_empty = fig.add_subplot(outer_grid[7])
ax_empty.axis("off")

fig.suptitle("Visual Analysis of Numerical Variables with Raw Data (W/ Outliers)", fontsize=13.5, fontweight="bold", y=0.995)
plt.savefig("figures/numerical_preliminary_analysis.png", dpi=200, bbox_inches="tight")
plt.close()