from datetime import datetime
import re


def extract_course_info(html):
    course_match = re.search(r"<strong>Course:.*?</strong>(.*?)\s*&nbsp;", html)
    parts = course_match.group(1).strip().split(" ")
    catalog = parts[0]
    try:
        course, section = parts[1].split("-")
        if "," in section:
            section = section.split(",")[0]
    except ValueError:
        course, section = parts[1], "01"
    description = " ".join(parts[2:])

    return catalog, course, section, description


def extract_instructor_info(html):
    instructor_match = re.search(
        r"<strong>Instructor:.*?</strong>(.*?)(?:\s*&nbsp;|$)", html
    )

    return instructor_match.group(1).strip() if instructor_match else None


def extract_schedule_info(html):
    schedule_match = re.search(r"<strong>Schedule:.*?</strong>(.*?)\s*&nbsp;", html)
    schedule = schedule_match.group(1).strip() if schedule_match else None

    days_str, start_time, end_time = None, None, None
    if schedule:
        day_time_match = re.match(
            r"([A-Za-z]+)\s+(\d{2}:\d{2}\s+[AP]M)\s+-\s+(\d{2}:\d{2}\s+[AP]M)", schedule
        )
        if day_time_match:
            days_str, start_time_str, end_time_str = day_time_match.groups()
            start_time = datetime.strptime(start_time_str, "%I:%M %p").time()
            end_time = datetime.strptime(end_time_str, "%I:%M %p").time()

    return days_str, start_time, end_time
