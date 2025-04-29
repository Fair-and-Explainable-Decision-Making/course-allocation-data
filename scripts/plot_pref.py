import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


status_color_map = {
    1: "lightsteelblue",
    2: "blue",
    3: "forestgreen",
    4: "darkkhaki",
    5: "darkorange",
    6: "red",
}

status_list = {
    1: "Freshmen",
    2: "Sophmore",
    3: "Junior",
    4: "Senior",
    5: "MS",
    6: "MS/PhD",
}


all_courses = []

fig = plt.subplots(figsize=(12, 3.5))
vals, names, xs = [], [],[]

for status in range(1, 7):
    # data_real = np.load(f"./course-allocation-data/experiments/preferences/preferences_real_{status}.npz")
    data_real = np.load(f"./experiments/preferences/preferences_real_{status}.npz")
    real_student_counts = data_real["real_student_counts"]

    data_synth = np.load(f"./experiments/preferences/preferences_synth_0_{status}.npz")
    synth_student_counts = data_synth["synth_student_counts"]


    vals.append(real_student_counts)
    vals.append(synth_student_counts)
    names.append(f"Real")
    names.append(f"Synthetic")
    xs.append(
    np.random.normal(2*(status-1) + 0.65, 0.03, len(real_student_counts))
    )
    xs.append(
    np.random.normal(2*(status-1)+1 + 0.65, 0.03, len(synth_student_counts))
    )


box = plt.boxplot(vals, labels=names, patch_artist=True, widths=0.35)
colors = [status_color_map[i] for i in [1,1,2,2,3,3,4,4,5,5,6,6]]
colorsw = [status_color_map[i] for i in [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6]]
for whisker, cap, color in zip(
    box["whiskers"],
    box["caps"],
    colorsw,
):
    whisker.set_color(color)  # Set whisker color
    whisker.set_linewidth(1.5)  # Make whiskers thicker
    cap.set_color(color)  # Set cap color
    cap.set_linewidth(1.5)  # Make caps thicker

# Customize the boxes
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)  # Set face color
# Customize the boxes
for patch, color in zip(box["boxes"], colors):
    patch.set_edgecolor(color)  # Set vibrant edge color (no alpha)
    patch.set_linewidth(1.5)  # Edge thickness
    patch.set_alpha(0.4)                   # Fade the face color (fill only)

# Customize the median line
for median, color in zip(box["medians"], colors):
    median.set_color(color)  # Set median line color
    median.set_linewidth(1.5)  # Make the line thicker




for x, val,c in zip(xs, vals, colors):
    plt.scatter(x, val, color=c, s=2)

# plt.xticks(rotation=45)


legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=status_list[6],
        markerfacecolor=status_color_map[6],
        alpha=0.7,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=status_list[5],
        markerfacecolor=status_color_map[5],
        alpha=0.7,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=status_list[4],
        markerfacecolor=status_color_map[4],
        alpha=0.7,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=status_list[3],
        markerfacecolor=status_color_map[3],
        alpha=0.7,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=status_list[2],
        markerfacecolor=status_color_map[2],
        alpha=0.7,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=status_list[1],
        markerfacecolor=status_color_map[1],
        alpha=0.7,
        markersize=10,
    ),
]

plt.legend(handles=legend_elements, ncol=2, loc="upper right", fontsize=12)
plt.ylim([0,50])

plt.tight_layout()
# plt.savefig(f"./course-allocation-data/experiments/figs/boxplot_pref.jpg", dpi=300)
plt.savefig(f"./experiments/figs/boxplot_pref.jpg", dpi=300)


print("asd")