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
for status in range(1, 7):
    data_real = np.load(f"./experiments/preferences_real_{status}.npz")
    courses = data_real["status_all_courses"]
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

fig, axs = plt.subplots(nrows=6, figsize=(20, 8), sharex=True)

seeds = [*range(28), *range(30,35), *range(40,45)]
for status_inv, ax in enumerate(axs, start=1):
    status = 7 - status_inv
    data_real = np.load(f"./experiments/preferences_real_{status}.npz")
    courses = data_real["status_all_courses"]
    real_percentages = data_real["real_percentages"]

    real_percentages_dict={courses[i]:real_percentages[i] for i in range(len(courses))}
    course_order = [course for course in all_courses if course in courses]
    real_percentages=[real_percentages_dict[course] for course in course_order]

    # print([(i,course) for i,course in enumerate(all_courses) if course not in courses])


    synthetic_percentages_per_seed = []
    for seed in seeds:
        data_synth = np.load(f"./experiments/preferences_synth_{seed}_{status}.npz")
        synth_courses= data_synth["status_all_courses"]
        synth_percentages= data_synth["synth_percentages"]

        synt_percentages_dict={synth_courses[i]:synth_percentages[i] for i in range(len(synth_courses))}


        not_courses = [i for i,course in enumerate(synth_courses) if course not in courses]
        not_courses.reverse()

        for course in not_courses:
            synth_courses = np.delete(synth_courses, course)
            synth_percentages = np.delete(synth_percentages, course)

        synt_percentages_dict={synth_courses[i]:synth_percentages[i] for i in range(len(synth_courses))}

        course_order = [course for course in all_courses if course in synth_courses]
        synth_percentages=[synt_percentages_dict[course] for course in course_order]
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
    ax.set_yticks([20, 40])
    ax.set_xlim(x[0] - 1, len(all_courses))


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

axs[0].legend(handles=legend_elements,ncol=2, loc="upper left", fontsize=14)
axs[-1].set_xticks(range(len(all_courses)))
axs[-1].set_xticklabels(all_courses, rotation=45, ha="right")

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"./experiments/pref_freq.jpg", dpi=300)
plt.savefig(f"./experiments/pref_freq.pdf", format="pdf",dpi=300)