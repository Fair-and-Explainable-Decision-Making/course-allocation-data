import dill as pickle
from fair.allocation import general_yankee_swap_E, serial_dictatorship, round_robin
from fair.metrics import utilitarian_welfare, nash_welfare, leximin
from fair.optimization import StudentAllocationProgram
from fair.agent import LegacyStudent
import qsurvey
import numpy as np
from tabulate import tabulate
import pandas as pd
import os

SPARSE = False
k=7

NUM_STUDENTS_PER_STATUS = {
    1: 239,
    2: 327,
    3: 408,
    4: 573,
    5: 613,
    6: 148,
}
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
seats=[]
for seed in range(10):
    RNG = np.random.default_rng(seed)
    with open(f"experiments/real_input_{k}.pkl", "rb") as file:
        qs,course,course_cap_map,course_map, all_courses, features, schedule = pickle.load(file)

    students, responses, statuses = qs.students(course_map, all_courses, features, schedule,k, SPARSE)

    student_status_map = {students[i]: status for i, status in enumerate(statuses)}
    student_resp_map = {students[i]: response for i, response in enumerate(responses)}

    students = [
        student for student in students if len(student.student.preferred_courses) > 0
    ]

    n_responses_per_status=np.zeros(6)

    for student in students:
        student_status=int(student_status_map[student])
        n_responses_per_status[student_status-1]+=1

    with open(f"experiments/synthetic_input_{k}_{seed}.pkl", "rb") as file:
        status_surveys_map, status_mbeta_map = pickle.load(file)


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
        # Check if the file exists
        file_exists = os.path.isfile(csv_file_path)
        
        # Create a DataFrame for the new row
        new_row = pd.DataFrame({
            'seed': [seed],
            'k': [k],
            'alg': [alg],
            'seats': [seats],
            'nash': [nash],
            'zeros': [zeros],
            'min': [min_val]
        })
        
        # Append to the CSV file
        if file_exists:
            new_row.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(csv_file_path, mode='w', header=True, index=False)
        
    add_experiment_result(k, seed, 'YS', N*utilitarian_welfare(X_YS, students, schedule), nash_welfare(X_YS, students, schedule)[0], nash_welfare(X_YS, students, schedule)[1], min(leximin(X_YS, students, schedule)))    
    add_experiment_result(k, seed, 'RR', N*utilitarian_welfare(X_RR, students, schedule), nash_welfare(X_RR, students, schedule)[0], nash_welfare(X_RR, students, schedule)[1], min(leximin(X_RR, students, schedule)))
    add_experiment_result(k, seed, 'ILP', N*utilitarian_welfare(X_ILP, students, schedule), nash_welfare(X_ILP, students, schedule)[0], nash_welfare(X_ILP, students, schedule)[1], min(leximin(X_ILP, students, schedule)))
    add_experiment_result(k, seed, 'SD', N*utilitarian_welfare(X_SD, students, schedule), nash_welfare(X_SD, students, schedule)[0], nash_welfare(X_SD, students, schedule)[1], min(leximin(X_SD, students, schedule)))
