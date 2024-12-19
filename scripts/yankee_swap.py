from fair.allocation import general_yankee_swap_E
from fair.metrics import utilitarian_welfare

import qsurvey

SPARSE = False
K = 5

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

X = general_yankee_swap_E(students, schedule)
print("YS utilitarian welfare: ", utilitarian_welfare(X[0], students, schedule))
