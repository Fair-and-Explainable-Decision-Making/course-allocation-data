from fair.allocation import general_yankee_swap_E, round_robin, serial_dictatorship
from fair.metrics import utilitarian_welfare, nash_welfare
from fair.optimization import StudentAllocationProgram

import qsurvey

SPARSE = False

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

print(f"Data type of first preferred_course for the first student: {type(students[0].preferred_courses[0])}")

X = general_yankee_swap_E(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
# print("YS nash welfare: ", nash_welfare(X[0], students, schedule))

X_RR = round_robin(students, schedule)
print("RR utilitarian welfare: ", utilitarian_welfare(X_RR, students, schedule))

X_SD = serial_dictatorship(students, schedule)
print("SD utilitarian welfare: ", utilitarian_welfare(X_SD, students, schedule))

orig_students = [student.student for student in students]
program = StudentAllocationProgram(orig_students, schedule).compile()
opt_alloc = program.formulateUSW().solve()
X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()

print(f"ILP utilitarian welfare: {utilitarian_welfare(X_ILP, students, schedule)}")