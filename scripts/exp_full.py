import numpy as np
import pandas as pd
import time
import os
import random

from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from fair.allocation import (
    general_yankee_swap_E,
    round_robin,
    serial_dictatorship,
    integer_linear_program,
)
from fair.metrics import nash_welfare, first_preference_count
from fair.envy import EF_violations_reponses
import qsurvey

# --- FORCE REPRODUCIBILITY ---
GLOBAL_SEED = 0

os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
# -----------------------------


def add_experiment_result(
    NUM_STUDENTS,
    seed,
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
    first_count = first_preference_count(X, c)
    total_envy, status_envy, downward_envy, _ = EF_violations_reponses(
        X, students, schedule, student_status_map, c
    )

    file_exists = os.path.isfile(csv_file_path)

    new_row = pd.DataFrame(
        {
            "NUM_STUDENTS": [NUM_STUDENTS],
            "seed": [seed],
            "alg": [alg],
            "USW": [USW],
            "seats": [seats],
            "zeros": [zeros],
            "nash": [nash],
            "first": [first_count],
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
csv_file_path = "../experiments/full_experiment.csv"

mp = qsurvey.QMapper(mapping_file)
qd = qsurvey.QSchedule(schedule_file)
crs_sec_cap_map = qd.capacities()
qs = qsurvey.QSurvey(survey_file, mp, list(crs_sec_cap_map.keys()))
course_map = mp.mapping(qs.all_courses)
all_courses = [crs for crs in course_map.keys()]
features = mp.features(course_map)
course, slot, weekday, section = features
schedule = mp.schedule(course_map, crs_sec_cap_map, features)
real_students, responses, statuses = qs.students(
    course_map,
    all_courses,
    features,
    schedule,
    status_max_course_map,
    pref_thresh,
    SPARSE,
)
student_status_map = {real_students[i]: status for i, status in enumerate(statuses)}
student_resp_map = {real_students[i]: response for i, response in enumerate(responses)}
course_cap_map = {
    crs: crs_sec_cap_map[course_map[crs]["course num"]][int(course_map[crs]["section"])]
    for crs in all_courses
}
real_students = [
    student for student in real_students if len(student.student.preferred_courses) > 0
]

student_type_map = {student: "real" for student in real_students}

n_responses_per_status = np.zeros(6)
for student in real_students:
    student_status = int(student_status_map[student])
    n_responses_per_status[student_status - 1] += 1

NUM_RAND_SAMP = {
    i + 1: NUM_STUDENTS_PER_STATUS[i + 1] - int(n_responses_per_status[i])
    for i in range(6)
}


for seed in range(10):
    students = [*real_students]
    RNG = np.random.default_rng(seed)
    status_mbeta_map = {}
    status_surveys_map = {}
    status_students_map = {}
    for status in range(1, 7):
        relevant_idxs = qsurvey.get_status_relevant(
            status, all_courses, course_map, status_crs_prefix_map
        )
        status_students = [
            student
            for student in real_students
            if student_status_map[student] == status
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
        status_mbeta = status_corpus.kde_distribution(
            SAMPLE_PER_STUDENT, NUM_SUB_KERNELS
        )
        status_mbeta_map[status] = status_mbeta
        status_surveys_map[status] = status_surveys

    status_synth_students_map = {}
    status_data_map = {}
    all_synth_students = []
    for status in qsurvey.STATUS_LABEL_MAP.keys():
        synth_students, data = qsurvey.synthesize_students(
            NUM_RAND_SAMP[status],
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
            total_course_list=[
                student.student.total_courses for student in status_students_map[status]
            ],
        )
        status_synth_students_map[status] = synth_students
        status_data_map[status] = data
        synth_students = [
            LegacyStudent(student, student.preferred_courses, course)
            for student in synth_students
        ]
        data1 = data[-len(synth_students) :]
        data_max = [max(val) for val in data1]
        # print(f"max preference: {max(data_max)}")
        for i, response in enumerate(data1):
            student_resp_map[synth_students[i]] = [int(i) for i in response]
            student_type_map[synth_students[i]] = "synth"
        students = [*students, *synth_students]
        for student in synth_students:
            student_status_map[student] = status

    students.sort(
        key=lambda x: (student_status_map[x], str(x))  # deterministic tie-breaker
    )
    students = list(reversed(students))

    c = np.vstack([student_resp_map[student] for student in students]) - 1

    NUM_STUDENTS = len(students)
    print("Num students,", NUM_STUDENTS)

    print("run ILP")
    start = time.time()
    X_ILP = integer_linear_program(students, schedule, valuations=c)
    runtime = time.time() - start
    print(f"ILP runtime = {runtime}. Now computing metrics.")
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        "ILP",
        runtime,
        X_ILP,
        students,
        schedule,
        c,
        csv_file_path,
    )
    print(f"Time to compute metrics: {time.time()-runtime-start}")

    print("run SD")
    start = time.time()
    X_SD = serial_dictatorship(students, schedule, c)
    runtime = time.time() - start
    print(f"SD runtime = {runtime}. Now computing metrics.")
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        "SD",
        runtime,
        X_SD,
        students,
        schedule,
        c,
        csv_file_path,
    )
    print(f"Time to compute metrics: {time.time()-runtime-start}")

    print("run RR")
    start = time.time()
    X_RR = round_robin(students, schedule, c)
    runtime = time.time() - start
    print(f"RR runtime = {runtime}. Now computing metrics.")
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        "RR",
        runtime,
        X_RR,
        students,
        schedule,
        c,
        csv_file_path,
    )
    print(f"Time to compute metrics: {time.time()-runtime-start}")

    print("run YS")
    start = time.time()
    X_YS, _, agents_involved = general_yankee_swap_E(students, schedule, valuations=c)
    runtime = time.time() - start
    print(f"YS runtime = {runtime}. Now computing metrics.")
    add_experiment_result(
        NUM_STUDENTS,
        seed,
        "YS",
        runtime,
        X_YS,
        students,
        schedule,
        c,
        csv_file_path,
    )
    print(f"Time to compute metrics: {time.time()-runtime-start}")

    np.savez(
        f"../experiments/data/data_{seed}.npz",
        c=c,
        X_ILP=X_ILP,
        X_SD=X_SD,
        X_RR=X_RR,
        X_YS=X_YS,
        student_type=[student_type_map[students[i]] for i in range(len(students))],
        agents_involved=agents_involved,
    )
