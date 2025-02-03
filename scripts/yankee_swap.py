from fair.allocation import general_yankee_swap_E
from fair.metrics import utilitarian_welfare
from fair.optimization import StudentAllocationProgram

import qsurvey
import time

SPARSE = False
pref_thresh = 10

status_max_course_map = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 4,
    6: 4,
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
students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

students = students[:600]
print(len(students))

start = time.time()
orig_students = [student.student for student in students]
print("1")
program = StudentAllocationProgram(orig_students, schedule).compile()
print("2")
opt_alloc = program.formulateUSW().solve()
print("3")
X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()

print(time.time() - start)

print("ILP utilitarian welfare: ", utilitarian_welfare(X_ILP, students, schedule))

# X = general_yankee_swap_E(students, schedule)
# print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
