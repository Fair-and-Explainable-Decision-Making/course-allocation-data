from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from fair.allocation import general_yankee_swap_E
from fair.metrics import utilitarian_welfare, nash_welfare

import qsurvey

import numpy as np 


NUM_SUB_KERNELS = 3
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True
RNG = np.random.default_rng(None)

survey_file = "resources/survey_data.csv"
schedule_file = "resources/anonymized_courses.xlsx"
mapping_file = "resources/survey_column_mapping.csv"

NUM_STUDENTS_PER_STATUS = {
    1: 239,
    2: 327,
    3: 408,
    4: 573,
    5: 613,
    6: 148,
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
    course_map, all_courses, features, schedule, SPARSE
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

# print(len([student.preferred_courses for student in students if len(student.preferred_courses)==0]))

n_responses_per_status=np.zeros(6)

for student in students:
    student_status=int(student_status_map[student])
    n_responses_per_status[student_status-1]+=1

NUM_RAND_SAMP = {i+1: NUM_STUDENTS_PER_STATUS[i+1]- int(n_responses_per_status[i]) for i in range(6)}

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
synth_students=[]
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
    )
    status_synth_students_map[status] = synth_students_status
    status_data_map[status] = data
    synth_students = [*synth_students, *synth_students_status]

synth_students = [LegacyStudent(student, student.preferred_courses, course) for student in synth_students]

print(len(students), len(synth_students))
print(len([student.preferred_courses for student in synth_students if len(student.preferred_courses)==0]))

X = general_yankee_swap_E([*students, *synth_students], schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
print("YS nash welfare: ", nash_welfare(X[0], students, schedule))

print(len([student.preferred_courses for student in students if len(student.preferred_courses)==0]))
print(len([student.preferred_courses for student in synth_students if len(student.preferred_courses)==0]))