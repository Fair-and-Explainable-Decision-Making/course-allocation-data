import numpy as np
import pandas as pd
import os
import time

from fair.allocation import general_yankee_swap_E, round_robin, serial_dictatorship
from fair.envy import EF_violations, EF1_violations, EFX_violations
from fair.metrics import (
    utilitarian_welfare,
    nash_welfare,
    leximin,
    precompute_bundles_valuations,
    PMMS_violations,
)
from fair.optimization import StudentAllocationProgram
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import random

import qsurvey

NUM_SUB_KERNELS = 1
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True
pref_thresh = 10

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

# survey_file = "./course-allocation-data/resources/survey_data.csv"
# schedule_file = "./course-allocation-data/resources/anonymized_courses.xlsx"
# mapping_file = "./course-allocation-data/resources/survey_column_mapping.csv"
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

for seed in range(10, 100):
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

    print("run ILP")
    start = time.time()
    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    opt_alloc = program.formulateUSW().solve()
    X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()
    time_ILP = time.time() - start

    print("run RR")
    start = time.time()
    X_RR = round_robin(students, schedule)
    time_RR = time.time() - start

    print("run SD")
    start = time.time()
    X_SD = serial_dictatorship(students, schedule)
    time_SD = time.time() - start

    print("run YS")
    start = time.time()
    X_YS, _, _ = general_yankee_swap_E(students, schedule)
    time_YS = time.time() - start

    np.savez(
        f"experiments/leximin_reduced/leximin_reduced_{seed}.npz",
        leximin_RR=leximin(X_RR, students, schedule),
        leximin_SD=leximin(X_SD, students, schedule),
        leximin_YS=leximin(X_YS, students, schedule),
        leximin_ILP=leximin(X_ILP, students, schedule),
    )

    print("Finished allocation algorithms. Now computing metrics")

    csv_file_path = "experiments/reduced_experiment_results.csv"

    runtimes = [time_SD, time_RR, time_ILP, time_YS]
    Xs = [X_SD, X_RR, X_ILP, X_YS]
    algs = ["SD", "RR", "ILP", "YS"]

    def add_experiment_result(
        NUM_STUDENTS,
        seed,
        pref_thresh,
        alg,
        runtime,
        X,
        students,
        schedule,
    ):

        seats = NUM_STUDENTS * utilitarian_welfare(X, students, schedule)
        zeros, nash = nash_welfare(X, students, schedule)
        min_val = min(leximin(X, students, schedule))

        start = time.time()
        bundles, valuations = precompute_bundles_valuations(X, students, schedule)
        print("Precomputing bundles and valuations took: ", time.time() - start)
        PMMS = PMMS_violations(X, students, schedule, bundles, valuations)
        EF = EF_violations(X, students, schedule, valuations)
        EF1 = EF1_violations(X, students, schedule, bundles, valuations)
        EFX = EFX_violations(X, students, schedule, bundles, valuations)

        file_exists = os.path.isfile(csv_file_path)

        new_row = pd.DataFrame(
            {
                "NUM_STUDENTS": [NUM_STUDENTS],
                "pref_thresh": [pref_thresh],
                "seed": [seed],
                "alg": [alg],
                "seats": [seats],
                "nash": [nash],
                "zeros": [zeros],
                "min_val": [min_val],
                "PMMS_violations": [PMMS[0]],
                "PMMS_agents": [PMMS[1]],
                "EF_violations": [EF[0]],
                "EF_agents": [EF[1]],
                "EF1_violations": [EF1[0]],
                "EF1_agents": [EF1[1]],
                "EFX_violations": [EFX[0]],
                "EFX_agents": [EFX[1]],
                "runtime": [runtime],
            }
        )

        if file_exists:
            new_row.to_csv(csv_file_path, mode="a", header=False, index=False)
        else:
            new_row.to_csv(csv_file_path, mode="w", header=True, index=False)

    for i, alg in enumerate(algs):
        print("computing for:   ", alg)
        runtime = runtimes[i]
        X = Xs[i]
        add_experiment_result(
            NUM_STUDENTS,
            seed,
            pref_thresh,
            alg,
            runtime,
            X,
            students,
            schedule,
        )
