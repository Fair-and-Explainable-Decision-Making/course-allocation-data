import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Assuming student_status_map = {student_index: status}
student_status_map = {0: "Freshman", 1: "Sophomore", 2: "Junior", 3: "Senior", 4: "Graduate", 5: "PhD"}
statuses = set(student_status_map.values())

# Example inputs for real and synthetic student preferences
real_student_preferred_courses = [['101a', '101b', '203', '205a'], ['101b', '106', '205b', '403']]
synthetic_student_preferences_seeds = [
    [['101a', '106', '203', '205a'], ['101b', '205b', '403']],
    [['101a', '101b', '205a'], ['203', '205b', '403']],
    [['101b', '106', '205a'], ['101a', '203', '205b', '403']],
]

# Organize data by status
status_real_preferences = {
    status: [real_student_preferred_courses[i] for i, s in student_status_map.items() if s == status]
    for status in statuses
}
status_synthetic_preferences_seeds = {
    status: [
        [
            synthetic_student_preferences_seeds[seed][i]
            for i, s in student_status_map.items() if s == status
        ]
        for seed in range(len(synthetic_student_preferences_seeds))
    ]
    for status in statuses
}

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for idx, status in enumerate(statuses):
    # Real preferences for this status
    real_courses = [course for sublist in status_real_preferences[status] for course in sublist]
    real_student_count = len(status_real_preferences[status])

    # Unique courses
    all_courses = sorted(
        set(real_courses) | set(
            course for seed in status_synthetic_preferences_seeds[status]
            for sublist in seed for course in sublist
        )
    )

    # Calculate proportions for real students
    real_course_counts = {course: real_courses.count(course) for course in all_courses}
    real_percentages = [
        real_course_counts.get(course, 0) / real_student_count * 100 for course in all_courses
    ]

    # Calculate proportions for synthetic students
    synthetic_percentages_per_seed = []
    for seed_preferences in status_synthetic_preferences_seeds[status]:
        synthetic_courses = [course for sublist in seed_preferences for course in sublist]
        synthetic_course_counts = {
            course: synthetic_courses.count(course) for course in all_courses
        }
        synthetic_percentages = [
            synthetic_course_counts.get(course, 0) / len(seed_preferences) * 100
            for course in all_courses
        ]
        synthetic_percentages_per_seed.append(synthetic_percentages)

    # Calculate mean and SE for synthetic data
    synthetic_percentages_mean = np.mean(synthetic_percentages_per_seed, axis=0)
    synthetic_percentages_se = np.std(synthetic_percentages_per_seed, axis=0) / np.sqrt(
        len(synthetic_percentages_per_seed)
    )

    # Plotting
    ax = axes[idx]
    bar_width = 0.35
    index = np.arange(len(all_courses))

    # Bars for real students
    ax.bar(index, real_percentages, bar_width, label="Real Students", color="blue", alpha=0.7)

    # Bars for synthetic students
    ax.bar(
        index + bar_width, synthetic_percentages_mean, bar_width,
        label="Synthetic Students", color="orange", alpha=0.7
    )

    # Error bars for synthetic data
    ax.errorbar(
        index + bar_width, synthetic_percentages_mean, yerr=synthetic_percentages_se,
        fmt="none", color="black", capsize=5
    )

    # Formatting
    ax.set_title(f"Status: {status}")
    ax.set_xlabel("Courses")
    ax.set_ylabel("Percentage of Students (%)")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(all_courses, rotation=45, ha="right")
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()
