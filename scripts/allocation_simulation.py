import numpy as np
import dill as pickle
from fair.metrics import utilitarian_welfare, nash_welfare, leximin, get_bundle_from_allocation_matrix
from fair.allocation import general_yankee_swap_E, serial_dictatorship, round_robin
from fair.optimization import StudentAllocationProgram
from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from fair.agent import LegacyStudent

import qsurvey
from tabulate import tabulate
import pandas as pd
import os

NUM_SUB_KERNELS = 3
SAMPLE_PER_STUDENT = 10
SPARSE = False
PLOT = True


seed=2
for k in {10,12,15}:
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
        course_map, all_courses, features, schedule, k, SPARSE
    )

    course_cap_map = {
        crs: crs_sec_cap_map[course_map[crs]["course num"]][int(course_map[crs]["section"])]
        for crs in all_courses
    }

    student_status_map = {students[i]: status for i, status in enumerate(statuses)}
    student_resp_map = {students[i]: response for i, response in enumerate(responses)}


    students = [
        student for student in students if len(student.student.preferred_courses) > 0
    ]



    RNG = np.random.default_rng(seed)
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

    n_responses_per_status=np.zeros(6)

    for student in students:
        student_status=int(student_status_map[student])
        n_responses_per_status[student_status-1]+=1

    NUM_RAND_SAMP = {i+1: NUM_STUDENTS_PER_STATUS[i+1]- int(n_responses_per_status[i]) for i in range(6)}

    status_synth_students_map = {}
    status_data_map = {}
    for status in qsurvey.STATUS_LABEL_MAP.keys():
        synth_students, data = qsurvey.synthesize_students(
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
            k=k,
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

    N=len(students)
    print(f"Number of students = {N}")


    X_RR = round_robin(students, schedule)
    X_SD = serial_dictatorship(students, schedule)
    print(N*utilitarian_welfare(X_RR, students, schedule))
    print(N*utilitarian_welfare(X_SD, students, schedule))

    orig_students = [student.student for student in students]
    program = StudentAllocationProgram(orig_students, schedule).compile()
    opt_alloc = program.formulateUSW().solve()

    X_ILP = opt_alloc.reshape(len(students), len(schedule)).transpose()
    X_YS = general_yankee_swap_E(students, schedule)[0]

    seats=[N*utilitarian_welfare(X_YS, students, schedule),N*utilitarian_welfare(X_RR, students, schedule),N*utilitarian_welfare(X_ILP, students, schedule),N*utilitarian_welfare(X_SD, students, schedule)]
    zeros= [nash_welfare(X_YS, students, schedule)[0],nash_welfare(X_RR, students, schedule)[0], nash_welfare(X_ILP, students, schedule)[0], nash_welfare(X_SD, students, schedule)[0]]
    nash= [nash_welfare(X_YS, students, schedule)[1],nash_welfare(X_RR, students, schedule)[1], nash_welfare(X_ILP, students, schedule)[1], nash_welfare(X_SD, students, schedule)[1]]
    lex_min= [min(leximin(X_YS, students, schedule)), min(leximin(X_RR, students, schedule)), min(leximin(X_ILP, students, schedule)), min(leximin(X_SD, students, schedule))]
    table = [['','YS', 'RR', 'ILP', 'SD'], ['Seats',*seats], ['Zeros',*zeros], ['Nash',*nash], ['Min',*lex_min]]

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    np.savez(f"experiments/X_ILP_{k}_{seed}.npz", X_ILP=X_ILP)

    np.savez(f"experiments/X_YS_{k}_{seed}.npz", X_YS=X_YS)

    np.savez(f"experiments/X_SD_{k}_{seed}.npz", X_SD=X_SD)

    np.savez(f"experiments/X_RR_{k}_{seed}.npz", X_RR=X_RR)


    csv_file_path = 'experiments/experiment_results.csv'
    def add_experiment_result(k, seed, alg, seats, zeros, nash, min_val):
        file_exists = os.path.isfile(csv_file_path)
        
        new_row = pd.DataFrame({
            'seed': [seed],
            'k': [k],
            'alg': [alg],
            'seats': [seats],
            'nash': [nash],
            'zeros': [zeros],
            'min': [min_val]
        })
        
        if file_exists:
            new_row.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(csv_file_path, mode='w', header=True, index=False)
        
    add_experiment_result(k, seed, 'YS', N*utilitarian_welfare(X_YS, students, schedule), nash_welfare(X_YS, students, schedule)[0], nash_welfare(X_YS, students, schedule)[1], min(leximin(X_YS, students, schedule)))    
    add_experiment_result(k, seed, 'RR', N*utilitarian_welfare(X_RR, students, schedule), nash_welfare(X_RR, students, schedule)[0], nash_welfare(X_RR, students, schedule)[1], min(leximin(X_RR, students, schedule)))
    add_experiment_result(k, seed, 'ILP', N*utilitarian_welfare(X_ILP, students, schedule), nash_welfare(X_ILP, students, schedule)[0], nash_welfare(X_ILP, students, schedule)[1], min(leximin(X_ILP, students, schedule)))
    add_experiment_result(k, seed, 'SD', N*utilitarian_welfare(X_SD, students, schedule), nash_welfare(X_SD, students, schedule)[0], nash_welfare(X_SD, students, schedule)[1], min(leximin(X_SD, students, schedule)))

    for index,student in enumerate(students):
        bundle=get_bundle_from_allocation_matrix(X_YS, schedule, index)
        if student.valuation(bundle) ==0:
            print(index, student.preferred_courses)

