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
    2: "Sophomore",
    3: "Junior",
    4: "Senior",
    5: "MS",
    6: "PhD",
}


all_courses = []
for status in range(1, 7):
    data_real = np.load(f"./experiments/preferences/preferences_real_{status}.npz")
    courses = data_real["status_all_courses"]
    print(f"{status_list[status]}: {len(courses)}")
    all_courses = [*all_courses, *courses]

all_courses = sorted(list(set(all_courses)))
element = all_courses.pop(0)
all_courses.insert(8, element)
element = all_courses.pop(44)
all_courses.insert(26, element)
element = all_courses.pop(52)
all_courses.insert(27, element)
element = all_courses.pop(57)
all_courses.insert(28, element)
element = all_courses.pop(67)
all_courses.insert(66, element)

course_dict = {course: i for i, course in enumerate(all_courses)}

fig, axs = plt.subplots(nrows=6, figsize=(12, 4.5), sharex=True)

seeds = [*range(28), *range(30, 35), *range(40, 45)]
for status_inv, ax in enumerate(axs, start=1):
    status = 7 - status_inv
    data_real = np.load(f"./experiments/preferences/preferences_real_{status}.npz")
    courses = data_real["status_all_courses"]
    real_percentages = data_real["real_percentages"]

    real_percentages_dict = {
        courses[i]: real_percentages[i] for i in range(len(courses))
    }
    course_order = [course for course in all_courses if course in courses]
    real_percentages = [real_percentages_dict[course] for course in course_order]

    # print([(i,course) for i,course in enumerate(all_courses) if course not in courses])

    synthetic_percentages_per_seed = []
    for seed in seeds:
        data_synth = np.load(f"./experiments/preferences/preferences_synth_{seed}_{status}.npz")
        synth_courses = data_synth["status_all_courses"]
        synth_percentages = data_synth["synth_percentages"]

        synt_percentages_dict = {
            synth_courses[i]: synth_percentages[i] for i in range(len(synth_courses))
        }

        not_courses = [
            i for i, course in enumerate(synth_courses) if course not in courses
        ]
        not_courses.reverse()

        for course in not_courses:
            synth_courses = np.delete(synth_courses, course)
            synth_percentages = np.delete(synth_percentages, course)

        synt_percentages_dict = {
            synth_courses[i]: synth_percentages[i] for i in range(len(synth_courses))
        }

        course_order = [course for course in all_courses if course in synth_courses]
        synth_percentages = [synt_percentages_dict[course] for course in course_order]
        synthetic_percentages_per_seed.append(synth_percentages)

    synthetic_percentages_per_seed = np.asarray(synthetic_percentages_per_seed)

    x = sorted([course_dict[course] for course in courses])

    ax.plot(x, real_percentages, color=status_color_map[status], linewidth=2)

    synth_max = [
        max(synthetic_percentages_per_seed[:, i]) for i, course in enumerate(courses)
    ]
    synth_min = [
        min(synthetic_percentages_per_seed[:, i]) for i, course in enumerate(courses)
    ]

    ax.fill_between(x, synth_max, synth_min, color=status_color_map[status], alpha=0.3)
    if status > 1:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_yticks([40, 80])
    ax.set_ylim([0,80])
    ax.set_xlim(x[0] - 1, len(all_courses))


legend_elements = [
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
        label=status_list[6],
        markerfacecolor=status_color_map[6],
        alpha=0.7,
        markersize=10,
    ),
]

axs[0].legend(handles=legend_elements, ncol=3, loc="upper left", fontsize=10.5)
axs[-1].set_xticks(range(len(all_courses)))
# axs[-1].set_xticklabels(all_courses, rotation=45, ha="right")
axs[-1].set_xticklabels(all_courses, rotation=90, ha="right")
axs[-1].set_xticks([])
ax.annotate('100 Level', xy=(0.062, -0.15), xytext=(0.062, -0.4), xycoords='axes fraction', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle='-[, widthB=4.3, lengthB=0.8', lw=1.0, color='k'))

ax.annotate('200 Level', xy=(0.191, -0.15), xytext=(0.191, -0.4), xycoords='axes fraction', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle='-[, widthB=5.55, lengthB=0.8', lw=1.0, color='k'))


ax.annotate('300 Level', xy=(0.433, -0.15), xytext=(0.433, -0.4), xycoords='axes fraction', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle='-[, widthB=13.4, lengthB=0.8', lw=1.0, color='k'))

ax.annotate('400 Level', xy=(0.66, -0.15), xytext=(0.66, -0.4), xycoords='axes fraction', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle='-[, widthB=4.4, lengthB=0.8', lw=1.0, color='k'))

ax.annotate('500 Level', xy=(0.763, -0.15), xytext=(0.763, -0.4), xycoords='axes fraction', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle='-[, widthB=3.4, lengthB=0.8', lw=1.0, color='k'))

ax.annotate('600 Level', xy=(0.902, -0.15), xytext=(0.902, -0.4), xycoords='axes fraction', 
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle='-[, widthB=7.2, lengthB=0.8', lw=1.0, color='k'))

fig.supylabel('Percentage of Students', fontsize=10.5)

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"./experiments/pref_freq.jpg", dpi=300)
plt.savefig(f"./experiments/pref_freq.pdf", format="pdf", dpi=300)
