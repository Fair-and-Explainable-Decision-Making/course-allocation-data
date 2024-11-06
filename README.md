# course-allocation-data

This repository contains data files and Python scripts related to a course preference survey conducted in the UMass Amherst Computer Science department, for the Fall 2024 semester. The survey was designed to gather preferences from students across the CS department. The repository also includes scripts that process this survey data and simulate course allocation using the Yankee Swap algorithm.

## Survey

This survey was distributed a few weeks before the official enrollment period, allowing students to voluntarily share their anticipated course selections and scheduling preferences. The survey consisted of the following questions:

1. Please indicate your current status at UMass: 1: Freshman; 2: Sophomore, 3: Junior, 4: Senior, 5: MS, 6: PhD.
2. How many CICS classes are you planning to enroll in before the add/drop period of the Fall 2024 semester? (You might want to take more classes than you are planing to keep by the end of the semester)
3. How many CICS classes are you planning to stay enrolled in after the add/drop period of the Fall 2024 semester?
4. Which categories of courses are you interested in taking? Select all that apply: 1: Theory, 2: Systems, 3: AI/ML, 4: Other.
5. Are there particular times of the day you would prefer not to have classes? Unselect all time slots that you would want to avoid having classes on - Day of the Week - 
   5.1 Before 9:00 AM: 1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.2 Between 9:00AM-10:00AM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.3 Between 10:00AM-11:00AM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.4 Between 11:00AM-12:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.5 Between 12:00PM-1:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.6 Between 1:00PM-2:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.7 Between 2:00PM-3:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.8 Between 3:00PM-4:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.9 Between 4:00PM-5:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.10 Between 5:00PM-6:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
   5.11 After 6:00PM: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday.
6. Email Address
7. Please rank each of the following CICS classes on a scale of 1-7, where 1 is 'not interested' and 7 is 'very interested' If you are interested in a course this semester to fill a major requirement, please select 8.

## Contents

The repository has the following structure:
```
.
├── resources
    ├── survey_data.csv   
    ├── random_survey.csv   
    ├── anonymized_courses.xlsx  
    └── survey_column_mapping.csv 
├── scripts
    ├── generate_random_survey.py   
    ├── survey_simulation.py
    └── yankee_swap.py  
├── src
    ├── qsurvey  
        ├── __init__.py
        └──parser.py
├── Readme.md
└── pyproject.toml
```
### Data Files

The `resource/` folder contains files with the data corresponding to anonymized students responses and course schedules

- `survey_data.csv`: This file contains 1,063 student responses to the survey questions outlined above. Each response entry includes:
    - ResponseID: a unique identifier for each response
    - Finished: indicates whether the survey was fully completed by the respondent
    - Progress: The completion percentage for each response; and
    - Duration (in seconds): the time taken by the respondent to complete the survey, measured in seconds
- `random_survey.csv`: This file contains a small instance of randdomly generated preferences with the same structure as the `survey_data.csv' file
- `anonymized_courses.xlsx`: This file provides a list of all courses offered in Fall 2024, with anonymized details to protect specific class and instructor identities. Each course entry includes the following information:
    - Catalog: An anonymized unique identifier for each course (not matching the actual course code)
    - Subject: The department offering the course, anonymized for privacy
    - Categories: Specifies the course level as either "Graduate" or "Undergraduate"
    - Enrl Capacity: The number of seats available in the course
    - Section: Indicates different sections of the same course (Catalog number), which may vary by schedule or instructor
    - zc.days: The days of the week the class is scheduled
    - Mgt Time: The time slot during which the class is held
    - InstructorPrint: Anonymized teaching staff
    - Credits: The number of credits the course carries
- `survey_column_mapping.csv`: A file mapping survey responses to actual course entries in `anonymized_courses.xlsx`

### Scripts

The `scripts/` folder contains Python scripts to work with survey data and model students based on this data. These scripts rely on the Student class from the [yankee-swap-allocation-framework repository](https://github.com/Fair-and-Explainable-Decision-Making/yankee-swap-allocation-framework). 

- `generate_random_survey.py`: This script generates a random instance of survey data.
- `survey_simulation.py`: This script generates `Student` instances based on the survey data. For each real survey response, it creates a corresponding `Student` object modeled on that respondent’s answers. Additionally, the script can generate synthetic students by randomly sampling characteristics and preferences from the collected survey data.
- `yankee_swap.py`: This script runs a course allocation process using the Yankee Swap algorithm implemented in the yankee-swap-allocation-framework repository, leveraging both real and synthetic student data.
The allocation script takes into account course capacities, time slots, and student preferences.

## Getting Started
### Installing Dependencies and Packages
Use these steps for setting up a development environment to install and work with code in this template:

1) Set up a Python 3 virtual environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#) or [Virtualenv](https://virtualenv.pypa.io/en/latest/index.html).

2) Activate your virtual environment by running `conda acivate your_environment` in a new terminal

3) Install the requierements by running `pip install -e .`

4) Run any of the scripts by running `python3 scripts/...`

## License
This project is open-source and available under the MIT License.




