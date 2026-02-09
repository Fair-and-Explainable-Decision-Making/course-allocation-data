import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from plotly.colors import qualitative

df = pd.read_csv("./reduced_experiment_results.csv")
palette = qualitative.Plotly
# palette = [sns.color_palette("colorblind")[i] for i in [3,0,2]]
palette = [palette[i] for i in [0, 1, 2, 4, 3, 5]]
faded_palette = [color + "66" for color in palette]

metrics = [
    "USW",
    "seats",
    "nash",
    "zeros",
    "total_envy",
    "status_envy",
    "downward_envy",
]
metric_names = [
    "USW",
    "Seats",
    "NSW",
    "Empty Bundles",
    "Total EF violations",
    "Status EF violations",
    "Downward EF violations",
]
higher = [1, 1, 1, 0, 0, 0, 0]
desired_column_order = [
    "ILP",
    "SD",
    "RR",
    "YS",
]  # Specify the desired column order
desired_column_order_names = [
    "Integer Linear Program",
    "Serial Dictatorship",
    "Round Robin",
    "Yankee Swap",
]  # Specify the desired column order

fig, axs = plt.subplots(2, 4, figsize=(12, 7))
axs = axs.flatten()

for num, metric in enumerate(metrics):
    plt_num = num
    if num > 2:
        plt_num = num + 1

    filtered_df = df[["seed", "alg", metric]]

    pivot_df = filtered_df.pivot(index="seed", columns="alg", values=metric)

    pivot_df.dropna(inplace=True)

    pivot_df = pivot_df[desired_column_order]

        # ---- CONNECT SEEDS ACROSS ALGORITHMS ----
    x_positions = np.arange(1, len(desired_column_order) + 1)

    # for _, row in pivot_df.iterrows():
    #     axs[plt_num].plot(
    #         x_positions,
    #         row.values,
    #         color="gray",
    #         alpha=0.15,
    #         linewidth=0.7,
    #         zorder=0,   # keep behind points & boxes
    #     )
    rr_idx = desired_column_order.index("RR") + 1  # +1 because boxplot x starts at 1
    ys_idx = desired_column_order.index("YS") + 1

    for _, row in pivot_df.iterrows():
        rr_val = row["RR"]
        ys_val = row["YS"]

        # Expected: YS > RR if higher is better, else YS < RR
        violation = (
            ys_val < rr_val if higher[num] == 1 else ys_val > rr_val
        )

        axs[plt_num].plot(
            [rr_idx, ys_idx],
            [rr_val, ys_val],
            color="red" if violation else "gray",
            alpha=0.5 if violation else 0.15,
            linewidth=1.2 if violation else 0.7,
            zorder=1,
        )


    vals, names, xs = [], [], []
    for i, col in enumerate(pivot_df.columns):
        vals.append(pivot_df[col].values)
        names.append(col)
        xs.append(
            np.random.normal(i + 0.75, 0.02, pivot_df[col].values.shape[0])
        )  # adds jitter to the data points - can be adjusted

    box = axs[plt_num].boxplot(vals, labels=names, patch_artist=True, widths=0.28)

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
            palette[4],
            palette[4],
        ],
    ):
        whisker.set_color(color)  # Set whisker color
        whisker.set_linewidth(1.3)  # Make whiskers thicker
        cap.set_color(color)  # Set cap color
        cap.set_linewidth(1.3)  # Make caps thicker

    # Customize the boxes
    for patch, color in zip(box["boxes"], faded_palette):
        patch.set_facecolor(color)  # Set face color
    # Customize the boxes
    for patch, color in zip(box["boxes"], palette):
        patch.set_edgecolor(color)  # Set vibrant edge color (no alpha)
        patch.set_linewidth(1.3)  # Edge thickness
        # patch.set_alpha(0.4)                   # Fade the face color (fill only)

    # Customize the median line
    for median, color in zip(box["medians"], palette):
        median.set_color(color)  # Set median line color
        median.set_linewidth(1.3)  # Make the line thicker

    for x, val, c in zip(xs, vals, palette):
        # plt.scatter(x, val, alpha=0.4, color=c)
        axs[plt_num].scatter(x, val, color=c, s=2)

    axs[plt_num].set_xticks([])
    axs[plt_num].set_xlabel(metric_names[num])
    axs[plt_num].set_xlim([0, 5.0])
    if higher[num] == 1:
        axs[plt_num].text(
            0.71,
            0.95,
            "↑ Higher is better",
            ha="center",
            va="center",
            transform=axs[plt_num].transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )
    elif num == 6:
        axs[plt_num].text(
            0.72,
            0.95,
            "↓ Lower is better",
            ha="center",
            va="center",
            transform=axs[plt_num].transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

    else:
        axs[plt_num].text(
            0.72,
            0.95,
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
        facecolor=faded_palette[i],
        edgecolor=palette[i],
        label=desired_column_order_names[i],
    )
    for i in range(len(desired_column_order))
]
axs[3].legend(
    handles=legend_elements,
    ncol=1,
    bbox_to_anchor=(-0.055, 0.33),
    loc="upper left",
    fontsize=11,
)


plt.tight_layout()
plt.savefig(f"./reduced_boxplot.jpg", dpi=300)
plt.savefig(f"./reduced_boxplot.pdf", format="pdf", dpi=300)
print(pivot_df)
