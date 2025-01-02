import numpy as np
from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E, round_robin, serial_dictatorship
from fair.metrics import utilitarian_welfare, nash_welfare
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import qsurvey

NUM_RAND_SAMP = 20
NUM_SUB_KERNELS = 3
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True
seed=0
RNG = np.random.default_rng(seed)
pref_thresh = 5

status_color_map = {
    1: "lightsteelblue",
    2: "blue",
    3: "forestgreen",
    4: "darkkhaki",
    5: "darkorange",
    6: "red",
}
status_max_course_map = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 4,
    6: 4,
}
status_crs_prefix_map = {
    1: ["1", "2", "3"],
    2: ["1", "2", "3", "4"],
    3: ["1", "2", "3", "4", "5"],
    4: ["2", "3", "4", "5", "6"],
    5: ["5", "6"],
    6: ["5", "6"],
}

survey_file = "resources/random_survey.csv"
schedule_file = "resources/anonymized_courses.xlsx"
mapping_file = "resources/survey_column_mapping.csv"

mp = qsurvey.QMapper(mapping_file)
qd = qsurvey.QSchedule(schedule_file)
crs_sec_cap_map = qd.capacities()
qs = qsurvey.QSurvey(survey_file, mp, list(crs_sec_cap_map.keys()))
course_map = mp.mapping(qs.all_courses)
all_courses = [crs for crs in course_map.keys()]
features = mp.features(course_map)
course, slot, weekday, section = features
schedule = mp.schedule(course_map, crs_sec_cap_map, features)
students, responses, statuses = qs.students(
    course_map, all_courses, features, schedule, pref_thresh, SPARSE
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


def project_data(status_data_map, course_map):
    pca = PCA(n_components=2)
    data_matrix = np.vstack(
        [status_data_map[status] for status in qsurvey.STATUS_LABEL_MAP.keys()]
    )
    # sign posts
    import re

    signs = []
    for i in range(1, 7):
        signs.append(
            [
                1 if re.match(r"^" + str(i) + r"\d{2}.*?", crs["course num"]) else 0
                for crs in course_map.values()
            ]
        )

    data_matrix = np.concatenate([np.vstack(signs), data_matrix])
    data = pca.fit_transform(data_matrix)
    sign_data = data[0:7]

    proj_data_map = {}
    start = 7
    for status in qsurvey.STATUS_LABEL_MAP.keys():
        stop = start + len(status_data_map[status])
        proj_data_map[status] = data[start:stop][:]
        start = stop

    return proj_data_map, sign_data


def project_data_ind(status_data_map):
    pca = PCA(n_components=2)
    proj_data_map = {}
    for status in qsurvey.STATUS_LABEL_MAP.keys():
        proj_data_map[status] = pca.fit_transform(np.vstack(status_data_map[status]))

    return proj_data_map


def plot_data(status, data, num_students):
    actual_students = data[:num_students, :]
    synthetic_students = data[num_students:, :]
    plt.scatter(
        synthetic_students[:, 0],
        synthetic_students[:, 1],
        c=status_color_map[status],
        alpha=0.25,
        label="_hidden",
    )
    plt.scatter(
        actual_students[:, 0],
        actual_students[:, 1],
        c=status_color_map[status],
        alpha=0.25,
        s=150,
        label=f"{qsurvey.STATUS_LABEL_MAP[status]}",
    )
    # plt.tick_params(labelbottom=False, labelleft=False)


status_synth_students_map = {}
status_data_map = {}
for status in qsurvey.STATUS_LABEL_MAP.keys():
    synth_students, data = qsurvey.synthesize_students(
        NUM_RAND_SAMP,
        course,
        section,
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
        pref_thresh=pref_thresh,
    )
    status_synth_students_map[status] = synth_students
    status_data_map[status] = data
    synth_students = [LegacyStudent(student, student.preferred_courses, course) for student in synth_students]
    data1=data[-len(synth_students):]
    data_max=[max(val) for val in data1]
    # print(f"max preference: {max(data_max)}")
    for i, response in enumerate(data1):
        student_resp_map[synth_students[i]] = [int(i) for i in response]
    students=[*students, *synth_students]
    for student in synth_students:
        student_status_map[student] = status
    print('synthetic student preferred courses',student.student.preferred_courses)


# proj_data_map, sign_data = project_data(status_data_map, course_map)


X_YS,_,_ = general_yankee_swap_E(students, schedule)
X_RR = round_robin(students, schedule)
X_SD = serial_dictatorship(students, schedule)

print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS Nash welfare: ", nash_welfare(X_YS, students, schedule))
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))
print("RR Nash welfare: ", nash_welfare(X_RR, students, schedule))
print("SD utilitarian welfare: ", utilitarian_welfare(X_SD, students, schedule))
print("SD Nash welfare: ", nash_welfare(X_SD, students, schedule))