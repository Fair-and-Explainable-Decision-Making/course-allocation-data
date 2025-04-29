import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from plotly.colors import qualitative

df = pd.read_csv(
    "/scratch3/paula/course-allocation-data/experiments/reduced_experiment_results.csv"
)
palette = qualitative.Plotly
# palette = [sns.color_palette("colorblind")[i] for i in [3,0,2]]
palette = [palette[i] for i in [0, 1, 2, 4]]
faded_palette = [color + "66" for color in palette]

metrics = [
    "seats",
    "nash",
    "zeros",
    "PMMS_violations",
    "EF_violations",
    "EF1_violations",
    "runtime",
]
metric_names = [
    "Seats",
    "NSW",
    "Empty Bundles",
    "PMMS violations",
    "EF violations",
    "EF-1 violations",
    "Runtime (s)",
]
higher = [1, 1, 0, 0, 0, 0, 0]
desired_column_order = ["SD", "RR", "YS", "ILP"]  # Specify the desired column order

fig, axs = plt.subplots(2, 4, figsize=(12, 5))
axs = axs.flatten()

for num, metric in enumerate(metrics):
    plt_num = num
    if num > 2:
        plt_num = num + 1

    filtered_df = df[["seed", "alg", metric]]

    pivot_df = filtered_df.pivot(index="seed", columns="alg", values=metric)

    pivot_df.dropna(inplace=True)

    pivot_df = pivot_df[desired_column_order]

    vals, names, xs = [], [], []
    for i, col in enumerate(pivot_df.columns):
        vals.append(pivot_df[col].values)
        names.append(col)
        xs.append(
            np.random.normal(i + 0.65, 0.03, pivot_df[col].values.shape[0])
        )  # adds jitter to the data points - can be adjusted

    box = axs[plt_num].boxplot(vals, labels=names, patch_artist=True, widths=0.35)

    for whisker, cap, color in zip(
        box["whiskers"],
        box["caps"],
        [
            palette[0],
            palette[0],
            palette[1],
            palette[1],
            palette[2],
            palette[2],
            palette[3],
            palette[3],
        ],
    ):
        whisker.set_color(color)  # Set whisker color
        whisker.set_linewidth(1.5)  # Make whiskers thicker
        cap.set_color(color)  # Set cap color
        cap.set_linewidth(1.5)  # Make caps thicker

    # Customize the boxes
    for patch, color in zip(box["boxes"], faded_palette):
        patch.set_facecolor(color)  # Set face color
    # Customize the boxes
    for patch, color in zip(box["boxes"], palette):
        patch.set_edgecolor(color)  # Set vibrant edge color (no alpha)
        patch.set_linewidth(1.5)  # Edge thickness
        # patch.set_alpha(0.4)                   # Fade the face color (fill only)

    # Customize the median line
    for median, color in zip(box["medians"], palette):
        median.set_color(color)  # Set median line color
        median.set_linewidth(1.5)  # Make the line thicker

    for x, val, c in zip(xs, vals, palette):
        # plt.scatter(x, val, alpha=0.4, color=c)
        axs[plt_num].scatter(x, val, color=c, s=2)

    axs[plt_num].set_xticks([])
    axs[plt_num].set_xlabel(metric_names[num])
    axs[plt_num].set_xlim([0, 4.8])
    if higher[num] == 1:
        axs[plt_num].text(
            0.3,
            0.92,
            "↑ Higher is better",
            ha="center",
            va="center",
            transform=axs[plt_num].transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )
    elif num == 6:
        axs[plt_num].text(
            0.3,
            0.92,
            "↓ Lower is better",
            ha="center",
            va="center",
            transform=axs[plt_num].transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

    else:
        axs[plt_num].text(
            0.715,
            0.92,
            "↓ Lower is better",
            ha="center",
            va="center",
            transform=axs[plt_num].transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )


axs[3].axis("off")


legend_elements = [
    Patch(
        facecolor=faded_palette[0], edgecolor=palette[0], label="Serial Dictatorship"
    ),
    Patch(facecolor=faded_palette[1], edgecolor=palette[1], label="Round Robin"),
    Patch(facecolor=faded_palette[2], edgecolor=palette[2], label="Yankee Swap"),
    Patch(
        facecolor=faded_palette[3], edgecolor=palette[3], label="Integer Linear Program"
    ),
]
axs[3].legend(
    handles=legend_elements,
    ncol=1,
    bbox_to_anchor=(-0.05, 0.45),
    loc="upper left",
    fontsize=11,
)


plt.tight_layout()
plt.savefig(f"./experiments/figs/reduced_boxplot.jpg", dpi=300)
plt.savefig(f"./experiments/figs/reduced_boxplot.pdf", format="pdf", dpi=300)
print(pivot_df)
