import numpy as np
import pandas as pd
import os 
import time

from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E, round_robin, serial_dictatorship
from fair.envy import EF_violations, EF1_violations, EFX_violations
from fair.metrics import utilitarian_welfare, nash_welfare, leximin, precompute_bundles_valuations, PMMS_violations
from fair.optimization import StudentAllocationProgram
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import qsurvey

NUM_SUB_KERNELS = 1
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True
seed = 0
RNG = np.random.default_rng(seed)
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

survey_file = "resources/survey_data.csv"
# survey_file = "resources/random_survey.csv"
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
n_responses_per_status = np.zeros(6)
for student in students:
    student_status = int(student_status_map[student])
    n_responses_per_status[student_status - 1] += 1
print("n_responses_per_status", n_responses_per_status)

students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

n_responses_per_status = np.zeros(6)
for student in students:
    student_status = int(student_status_map[student])
    n_responses_per_status[student_status - 1] += 1

NUM_RAND_SAMP = {
    i + 1: NUM_STUDENTS_PER_STATUS[i + 1] - int(n_responses_per_status[i])
    for i in range(6)
}

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
    students = [*students, *synth_students]
    for student in synth_students:
        student_status_map[student] = status





NUM_STUDENTS = len(students)
print("Num students," , NUM_STUDENTS)

students.sort(key=lambda x: student_status_map[x])
students.reverse()

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
# X_YS=X_RR
time_YS = time.time() - start

print("run ILP")
start = time.time()
orig_students = [student.student for student in students]
program = StudentAllocationProgram(orig_students, schedule).compile()
opt_alloc = program.formulateUSW().solve()
X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()
time_ILP = time.time() - start

runtimes = [time_ILP, time_RR, time_SD, time_YS]
Xs = [X_ILP, X_RR, X_SD, X_YS]
algs = ["ILP", "RR", "SD", "YS"]

print("Finished allocation algorithms. Now compute metrics")

csv_file_path = 'experiments/experiment_results.csv'
def add_experiment_result(NUM_SUB_KERNELS,SAMPLE_PER_STUDENT, NUM_STUDENTS, seed, pref_thresh, alg, runtime, X, students, schedule):

    seats= NUM_STUDENTS*utilitarian_welfare(X, students, schedule)
    zeros, nash = nash_welfare(X,students, schedule)
    min_val = min(leximin(X, students, schedule))
    start = time.time()

    bundles, valuations = precompute_bundles_valuations(X, students, schedule)
    PMMS = PMMS_violations(X, students, schedule, bundles, valuations)
    print("PMMS took: ", time.time() - start)
    EF = EF_violations(X,students, schedule,valuations)
    EF1 = EF1_violations(X, students, schedule, bundles, valuations)
    EFX = EFX_violations(X, students, schedule, bundles, valuations)

    file_exists = os.path.isfile(csv_file_path)
    
    new_row = pd.DataFrame({
        'NUM_SUB_KERNELS': [NUM_SUB_KERNELS],
        'SAMPLE_PER_STUDENT': [SAMPLE_PER_STUDENT],
        'NUM_STUDENTS': [NUM_STUDENTS],
        'seed': [seed],
        'pref_thresh': [pref_thresh],
        'alg': [alg],
        'seats': [seats],
        'nash': [nash],
        'zeros': [zeros],
        'min_val': [min_val],
        'PMMS_violations': [PMMS[0]],
        'PMMS_agents': [PMMS[1]],
        'EF_violations': [EF[0]],
        'EF_agents': [EF[1]],
        'EF1_violations': [EF1[0]],
        'EF1_agents': [EF1[1]],
        'EFX_violations': [EFX[0]],
        'EFX_agents': [EFX[1]],
        'runtime': [runtime]
    })
    
    if file_exists:
        new_row.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(csv_file_path, mode='w', header=True, index=False)

print(students[0].preferred_courses)

for i,alg in enumerate(algs):
    print("compute for alg:   ", alg)
    runtime = runtimes[i]
    X= Xs[i]
    add_experiment_result(NUM_SUB_KERNELS,SAMPLE_PER_STUDENT, NUM_STUDENTS, seed, pref_thresh, alg, runtime, X, students, schedule)



print("YS utilitarian welfare: ", utilitarian_welfare(X_YS, students, schedule))
print("YS Nash welfare: ", nash_welfare(X_YS, students, schedule))
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))
print("RR Nash welfare: ", nash_welfare(X_RR, students, schedule))
print("SD utilitarian welfare: ", utilitarian_welfare(X_SD, students, schedule))
print("SD Nash welfare: ", nash_welfare(X_SD, students, schedule))
