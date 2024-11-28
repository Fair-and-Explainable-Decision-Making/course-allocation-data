import numpy as np
from fair.stats.survey import Corpus, SingleTopicSurvey
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import qsurvey

import matplotlib.pyplot as plt
from collections import Counter

NUM_SUB_KERNELS = 3
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True
K = 5

status_max_course_map = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 4,
    6: 4,
}
NUM_STUDENTS_PER_STATUS = {
    1: 239,
    2: 327,
    3: 408,
    4: 573,
    5: 613,
    6: 148,
}
status_crs_prefix_map = {
    1: ["1", "2", "3"],
    2: ["1", "2", "3", "4"],
    3: ["1", "2", "3", "4", "5"],
    4: ["2", "3", "4", "5", "6"],
    5: ["5", "6"],
    6: ["5", "6"],
}

survey_file = "resources/survey_data.csv"
schedule_file = "resources/anonymized_courses.xlsx"
mapping_file = "resources/survey_column_mapping.csv"

mp = qsurvey.QMapper(mapping_file)
qd = qsurvey.QSchedule(schedule_file)
crs_sec_cap_map = qd.capacities()
qs = qsurvey.QSurvey(survey_file, mp, list(crs_sec_cap_map.keys()))
course_map = mp.mapping(qs.all_courses)
all_courses = [crs for crs in course_map.keys()]
features = mp.features(course_map)
course, slot, weekday, _ = features
schedule = mp.schedule(course_map, crs_sec_cap_map, features)
students, responses, statuses = qs.students(
    course_map, all_courses, features, schedule, K, SPARSE
)
student_status_map = {students[i]: status for i, status in enumerate(statuses)}
student_resp_map = {students[i]: response for i, response in enumerate(responses)}
course_cap_map = {
    crs: crs_sec_cap_map[course_map[crs]["course num"]][int(course_map[crs]["section"])]
    for crs in all_courses
}
students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

real_student_preferred_courses = [[item.values[0]+'-'+item.values[3] for item in student.student.preferred_courses] for student in students]

n_responses_per_status=np.zeros(6)
for student in students:
    student_status=int(student_status_map[student])
    n_responses_per_status[student_status-1]+=1
NUM_RAND_SAMP = {i+1: NUM_STUDENTS_PER_STATUS[i+1]- int(n_responses_per_status[i]) for i in range(6)}


synthetic_student_preferences_seeds=[]
for seed in range(1):
    RNG = np.random.default_rng(seed)
    status_mbeta_map = {}
    status_surveys_map = {}
    status_students_map = {}
    for status in range(1, 7):
        relevant_idxs = qsurvey.get_status_relevant(
            status, all_courses, course_map, status_crs_prefix_map
        )
        status_students = [
            student for student in students if student_status_map[student] == status
        ]
        status_students_map[status] = status_students
        status_surveys = [
            SingleTopicSurvey(
                [sch for i, sch in enumerate(schedule) if i in relevant_idxs],
                student_resp_map[student][relevant_idxs],
                student.student.total_courses,
                1,
                8,
            )
            for student in status_students
        ]
        status_corpus = Corpus(status_surveys, RNG)
        status_mbeta = status_corpus.kde_distribution(SAMPLE_PER_STUDENT, NUM_SUB_KERNELS)
        status_mbeta_map[status] = status_mbeta
        status_surveys_map[status] = status_surveys


    status_synth_students_map = {}
    status_data_map = {}
    synth_students= []
    for status in qsurvey.STATUS_LABEL_MAP.keys():
        synth_students_status, data = qsurvey.synthesize_students(
            NUM_RAND_SAMP[status],
            course,
            features,
            schedule,
            qs,
            status_surveys_map[status],
            status_mbeta_map[status],
            course_map,
            status_max_course_map[status],
            qsurvey.get_status_relevant(
                status, all_courses, course_map, status_crs_prefix_map
            ),
            rng=RNG,
            k=K,
        )
        status_synth_students_map[status] = synth_students_status
        status_data_map[status] = data
        synth_students = [*synth_students, *synth_students_status]
    # print([[item.values[0]+'-'+item.values[3] for item in student.preferred_courses] for student in synth_students])
    synthetic_student_preferences_seeds.append([[item.values[0]+'-'+item.values[3] for item in student.preferred_courses] for student in synth_students])
# Flatten the lists to count occurrences of each course
real_courses = [course for sublist in real_student_preferred_courses for course in sublist]

# Get unique courses for plotting
all_courses = sorted(set(real_courses + [course for seed in synthetic_student_preferences_seeds for sublist in seed for course in sublist]))

# Calculate the percentage of real and synthetic students who like each course
real_course_counts = {course: real_courses.count(course) for course in all_courses}
real_student_count = len(real_student_preferred_courses)
real_percentages = [real_course_counts[course] / real_student_count * 100 for course in all_courses]

# For synthetic students with different seeds
synthetic_percentages_per_seed = []
for seed_preferences in synthetic_student_preferences_seeds:
    synthetic_courses = [course for sublist in seed_preferences for course in sublist]
    synthetic_course_counts = {course: synthetic_courses.count(course) for course in all_courses}
    synthetic_percentages = [synthetic_course_counts[course] / len(seed_preferences) * 100 for course in all_courses]
    synthetic_percentages_per_seed.append(synthetic_percentages)

# Calculate average and standard error for synthetic percentages
synthetic_percentages_mean = np.mean(synthetic_percentages_per_seed, axis=0)
synthetic_percentages_std = np.std(synthetic_percentages_per_seed, axis=0)
synthetic_percentages_se = synthetic_percentages_std / np.sqrt(len(synthetic_percentages_per_seed))

# Plotting the frequency per class with error bars
fig, ax = plt.subplots(figsize=(20, 6))
bar_width = 0.35
index = np.arange(len(all_courses))

# Plot bars for real student percentages
ax.bar(index, real_percentages, bar_width, label='Real Students', color='blue', alpha=0.7)

# Plot bars for synthetic student percentages
bars = ax.bar(index + bar_width, synthetic_percentages_mean, bar_width, label='Synthetic Students', color='orange', alpha=0.7)

# Add error bars to synthetic students' bars
ax.errorbar(index + bar_width, synthetic_percentages_mean, yerr=synthetic_percentages_se, fmt='none', color='black', capsize=5)

# Adding labels and title
ax.set_xlabel('Courses')
ax.set_ylabel('Percentage of Students (%)')
ax.set_title('Comparison of Course Preferences: Real vs Synthetic Students')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(all_courses, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(f"real_vs_synthetic_preferences_K={K}_{seed}.png", dpi=300)
plt.close()



# Calculate how many courses each student likes
real_student_counts = [len(pref) for pref in real_student_preferred_courses]
synthetic_student_counts_seeds = [[len(pref) for pref in seed] for seed in synthetic_student_preferences_seeds]

# Count the frequency of number of liked items for real students
real_freq = Counter(real_student_counts)

# Count the frequency of number of liked items for synthetic students across all seeds
synthetic_freqs_seeds = [Counter(seed) for seed in synthetic_student_counts_seeds]

# Get the union of keys across real and synthetic data for consistent x-axis
all_liked_counts = sorted(set(real_freq.keys()).union(*[freq.keys() for freq in synthetic_freqs_seeds]))

# Create frequency arrays for plotting
real_freq_values = [real_freq.get(count, 0) for count in all_liked_counts]
synthetic_freqs_values_seeds = [
    [freq.get(count, 0) for count in all_liked_counts] for freq in synthetic_freqs_seeds
]

# Calculate mean and standard error for synthetic frequencies
synthetic_freqs_mean = np.mean(synthetic_freqs_values_seeds, axis=0)
synthetic_freqs_se = np.std(synthetic_freqs_values_seeds, axis=0) / np.sqrt(len(synthetic_student_preferences_seeds))

# Normalize frequencies to percentages
real_total = len(real_student_preferred_courses)
synthetic_totals = [len(seed) for seed in synthetic_student_preferences_seeds]

real_percentages = [freq / real_total * 100 for freq in real_freq_values]
synthetic_percentages_mean = [mean / np.mean(synthetic_totals) * 100 for mean in synthetic_freqs_mean]
synthetic_percentages_se = [se / np.mean(synthetic_totals) * 100 for se in synthetic_freqs_se]

# Plotting
fig, ax = plt.subplots(figsize=(20, 6))
bar_width = 0.35
index = np.arange(len(all_liked_counts))

# Plot bars for real students
ax.bar(index, real_percentages, bar_width, label='Real Students', color='blue', alpha=0.7)

# Plot bars for synthetic students
ax.bar(index + bar_width, synthetic_percentages_mean, bar_width, label='Synthetic Students (Average)', color='orange', alpha=0.7)

# Add error bars for synthetic students
ax.errorbar(index + bar_width, synthetic_percentages_mean, yerr=synthetic_percentages_se, fmt='none', color='black', capsize=5)

# Adding labels and title
ax.set_xlabel('Number of Classes Liked')
ax.set_ylabel('Percentage of Students (%)')
ax.set_title('Frequency of Number of Classes Liked: Real vs Synthetic Students')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(all_liked_counts)
ax.legend()

plt.tight_layout()
plt.show()
plt.savefig(f"real_vs_synthetic_frequencies__K={K}_{seed}.png", dpi=300)