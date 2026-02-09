import numpy as np
import pandas as pd
import copy
import random
import time
import os

from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from fair.allocation import (
    general_yankee_swap_E,
    round_robin,
    serial_dictatorship,
    integer_linear_program,
)
from fair.optimization import StudentAllocationProgram
from fair.metrics import utilitarian_welfare, nash_welfare
from fair.envy import EF_violations_reponses
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import qsurvey

NUM_RAND_SAMP = 20
NUM_SUB_KERNELS = 3
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True
seed = 0
RNG = np.random.default_rng(seed)
pref_thresh = 100

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
NUM_STUDENTS_PER_STATUS = {
    1: 239,
    2: 327,
    3: 408,
    4: 573,
    5: 613,
    6: 148,
}

# survey_file = "../resources/random_survey.csv"
survey_file = "../resources/survey_data.csv"
schedule_file = "../resources/anonymized_courses.xlsx"
mapping_file = "../resources/survey_column_mapping.csv"
csv_file_path = "../experiments/reduced_experiment_results.csv"

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
    course_map,
    all_courses,
    features,
    schedule,
    status_max_course_map,
    pref_thresh,
    SPARSE,
)
student_status_map = {students[i]: status for i, status in enumerate(statuses)}
student_resp_map = {students[i]: response for i, response in enumerate(responses)}
course_cap_map = {
    crs: crs_sec_cap_map[course_map[crs]["course num"]][int(course_map[crs]["section"])]
    for crs in all_courses
}
all_students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

n_responses_per_status = np.zeros(6)
for student in all_students:
    student_status = int(student_status_map[student])
    n_responses_per_status[student_status - 1] += 1

rates = [n_responses_per_status[i] / NUM_STUDENTS_PER_STATUS[i + 1] for i in range(6)]
rate = min(rates)

n_per_status = [round(NUM_STUDENTS_PER_STATUS[i + 1] * min(rates)) for i in range(6)]

for sche in schedule:
    sche.capacity = round(sche.capacity * rate)


def add_experiment_result(
    NUM_STUDENTS,
    seed,
    pref_thresh,
    alg,
    runtime,
    X,
    students,
    schedule,
    c,
    csv_file_path,
):
    current_utilities = np.diag(np.dot(c, X))
    USW = sum(current_utilities)
    seats = sum(sum(X)[:NUM_STUDENTS])
    zeros, nash = nash_welfare(X, students, schedule, current_utilities)
    total_envy, status_envy, downward_envy = EF_violations_reponses(
        X, students, schedule, student_status_map, c
    )

    file_exists = os.path.isfile(csv_file_path)

    new_row = pd.DataFrame(
        {
            "NUM_STUDENTS": [NUM_STUDENTS],
            "pref_thresh": [pref_thresh],
            "seed": [seed],
            "alg": [alg],
            "USW": [USW],
            "seats": [seats],
            "zeros": [zeros],
            "nash": [nash],
            "total_envy": [total_envy],
            "status_envy": [status_envy],
            "downward_envy": [downward_envy],
            "runtime": [runtime],
        }
    )

    if file_exists:
        new_row.to_csv(csv_file_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(csv_file_path, mode="w", header=True, index=False)


for seed in range(10,50):
    random.seed(seed)
    reduced_students = []
    for status in range(1, 7):
        students_status = [
            student for student in all_students if student_status_map[student] == status
        ]
        selected_students = random.sample(students_status, n_per_status[status - 1])
        reduced_students = [*reduced_students, *selected_students]

    students = reduced_students
    NUM_STUDENTS = len(students)
    print("Num students,", NUM_STUDENTS)

    students.sort(key=lambda x: student_status_map[x])
    students.reverse()

    c = np.vstack([student_resp_map[student] for student in students]) - 1

    print("run ILP")
    start = time.time()
    X_ILP = integer_linear_program(students, schedule, valuations=c)
    runtime = time.time() - start
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        pref_thresh,
        "ILP",
        runtime,
        X_ILP,
        students,
        schedule,
        c,
        csv_file_path,
    )

    print("run SD")
    start = time.time()
    X_SD = serial_dictatorship(students, schedule, c)
    runtime = time.time() - start
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        pref_thresh,
        "SD",
        runtime,
        X_SD,
        students,
        schedule,
        c,
        csv_file_path,
    )

    print("run RR")
    start = time.time()
    X_RR = round_robin(students, schedule, c)
    runtime = time.time() - start
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        pref_thresh,
        "RR",
        runtime,
        X_RR,
        students,
        schedule,
        c,
        csv_file_path,
    )

    print("run YS")
    start = time.time()
    X_YS_1, _, _ = general_yankee_swap_E(
        students, schedule, valuations=c
    )
    runtime = time.time() - start
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        pref_thresh,
        "YS",
        runtime,
        X_YS_1,
        students,
        schedule,
        c,
        csv_file_path,
    )


