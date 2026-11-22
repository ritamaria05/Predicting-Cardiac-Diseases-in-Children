'''Numerical Variables Preliminary Analysis'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from load_data import get_input_path, load_dataset, tag_for, data_label, figure_path

path = get_input_path("Preliminary analysis of numerical variables")
df = load_dataset(path)
tag = tag_for(path)
label = data_label(tag)
sns.set_theme(style="whitegrid", context="notebook")

num_cols = ["idade", "peso", "altura", "imc", "pa_sistolica", "pa_diastolica", "fc"]
units = {
    "idade": "years",
    "peso": "kg",
    "altura": "cm",
    "imc": "kg/m²",
    "pa_sistolica": "mmHg",
    "pa_diastolica": "mmHg",
    "fc": "bpm",
}

num_cols = [col for col in num_cols if col in df.columns]
df_num = pd.DataFrame(
    {col: pd.to_numeric(df[col], errors="coerce") for col in num_cols}
)

fig = plt.figure(figsize=(16, 15))
outer_grid = gridspec.GridSpec(4, 2, wspace=0.22, hspace=0.45)

for i, col in enumerate(num_cols):
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_grid[i], height_ratios=[3.0, 1.1], hspace=0.06
    )
    ax_hist = fig.add_subplot(inner_grid[0])
    ax_box = fig.add_subplot(inner_grid[1], sharex=ax_hist)

    s = df_num[col].dropna()

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

    ax_hist.set_yscale("log")
    ax_hist.set_ylim(0.8, len(s) * 2.5)
    ax_hist.set_xlabel("")

    med = s.median()
    ax_hist.axvline(med, color="#d95f02", linestyle="--", linewidth=1.3, label=f"Median: {med:.1f}")
    ax_hist.axvline(s.min(), color="#e7298a", linestyle=":", linewidth=1.2, label=f"Min: {s.min():.1f}")
    ax_hist.axvline(s.max(), color="#7570b3", linestyle=":", linewidth=1.2, label=f"Max: {s.max():.1f}")

    ax_hist.set_title(
        f"{col} ({units[col]}) — {label} (n={len(s):,})",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    ax_hist.set_ylabel("Count (log scale)", fontsize=8.5)
    ax_hist.legend(fontsize=8, loc="upper right", frameon=True)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

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

suffix = " (W/ Outliers)" if tag == "raw" else ""
fig.suptitle(
    f"Visual Analysis of Numerical Variables with {label}{suffix}",
    fontsize=13.5,
    fontweight="bold",
    y=0.995,
)
out = figure_path("numerical_preliminary_analysis", tag)
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()

print(f"Input: {path}")
print(f"Saved: {out}")