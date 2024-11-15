import numpy as np
from fair.stats import GOF
from fair.stats.survey import Corpus, SingleTopicSurvey

import qsurvey

NUM_SUB_KERNELS = 1
SAMPLE_PER_STUDENT = 20
GOF_SAMPLES = 100
SPARSE = False
RNG = np.random.default_rng(None)

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


def test_status(status, students, all_courses, course_map, status_crs_prefix_map):
    m = len(all_courses)
    relevant_idxs = qsurvey.get_status_relevant(
        status, all_courses, course_map, status_crs_prefix_map
    )
    students = [
        student for student in students if student_status_map[student] == status
    ]
    surveys = [
        SingleTopicSurvey(
            [sch for i, sch in enumerate(schedule) if i in relevant_idxs],
            student_resp_map[student][relevant_idxs],
            student.student.total_courses,
            1,
            8,
        )
        for student in students
    ]
    real_mBetas = [
        Corpus([survey], RNG).kde_distribution(SAMPLE_PER_STUDENT, NUM_SUB_KERNELS)
        for survey in surveys
    ]
    real_data_vecs = [
        qsurvey.scale_up_responses(survey.data(), relevant_idxs, len(course_map))
        for survey in surveys
    ]
    point_real_mBetas = [
        qsurvey.mBetaPoint(real_data.reshape((m,)), RNG) for real_data in real_data_vecs
    ]
    point_synth_mBetas = []
    for mBeta in real_mBetas:
        _, data = qsurvey.synthesize_students(
            1,
            course,
            features,
            schedule,
            qs,
            surveys,
            mBeta,
            course_map,
            status_max_course_map[status],
            qsurvey.get_status_relevant(
                status, all_courses, course_map, status_crs_prefix_map
            ),
            rng=RNG,
        )
        point_synth_mBetas.append(qsurvey.mBetaPoint((data[-1, :] - 1) / 7, RNG))

    for i, point_real_mBeta in enumerate(point_real_mBetas):
        for j, point_synth_mBeta in enumerate(point_synth_mBetas):
            gof = GOF(point_real_mBeta, point_synth_mBeta)
            print(i, j, gof.p_value(GOF_SAMPLES, GOF_SAMPLES))


test_status(1, students, all_courses, course_map, status_crs_prefix_map)
