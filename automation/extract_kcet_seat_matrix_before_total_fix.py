
from pathlib import Path
import csv
import re
import pymupdf

PDF_PATH = Path(
    "automation/source_samples/engineering_seat_matrix_2026.pdf"
)

OUTPUT_PATH = Path(
    "automation/source_samples/kcet_engineering_seat_matrix_2026.csv"
)

# Approximate horizontal boundaries measured from the PDF.
COLLEGE_CODE_X = (40, 78)
COLLEGE_NAME_X = (78, 185)
COURSE_CODE_X = (180, 215)
COURSE_NAME_X = (215, 302)
SEAT_TYPE_X = (295, 335)
COUNT_START_X = 335

CATEGORIES = [
    "C-1", "2A", "2B", "3A", "3B", "ST",
    "SCA", "SCB", "SCC", "SCC (20%)", "GM",
]

GROUPS = ["General", "Kannada", "Rural"]

# The category heading coordinates are approximately centered
# over the seat-count columns. Confirm against the PDF before
# treating the output as final.

CATEGORY_CENTERS = [
    368.9,
    399.7,
    427.7,
    461.3,
    490.4,
    520.7,
    547.3,
    573.2,
    599.2,
    627.1,
    656.1,
    685.9,
    706.6,
    729.4,
    750.5,
    776.1,
    797.4,
    823.3,
    852.6,
    880.7,
    911.8,
    942.1,
    968.0,
    996.6,
    1017.1,
    1038.5,
    1062.1,
    1087.2,
    1115.4,
    1142.0,
    1169.4,
    1198.2,
    1207.7,
]

TOTAL_X = (1230, 1295)


def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def get_words(page):
    """Return words with their coordinates."""
    return page.get_text("words", sort=True)


def group_by_y(words, tolerance=2.5):
    """Group words whose vertical positions are close."""
    rows = []

    for word in sorted(words, key=lambda item: (item[1], item[0])):
        y = word[1]

        if not rows or abs(rows[-1]["y"] - y) > tolerance:
            rows.append({"y": y, "words": [word]})
        else:
            rows[-1]["words"].append(word)

    return rows


def text_in_x_range(words, x_min, x_max):
    return clean(" ".join(
        word[4]
        for word in sorted(words, key=lambda item: item[0])
        if x_min <= word[0] < x_max
    ))


def numeric_value(text):
    text = clean(text)

    if re.fullmatch(r"\d+", text):
        return int(text)

    return None


def main():
    if not PDF_PATH.exists():
        print("ERROR: PDF not found:", PDF_PATH)
        return

    records = []
    pdf = pymupdf.open(PDF_PATH)

    try:
        for page_number, page in enumerate(pdf, start=1):
            words = get_words(page)

            # Ignore the header area. College/course rows begin below it.
            data_words = [w for w in words if w[1] >= 140]

            # College code anchors define the start of each college.
            college_anchors = [
                w for w in data_words
                if COLLEGE_CODE_X[0] <= w[0] < COLLEGE_CODE_X[1]
                and re.fullmatch(r"E\d{3}", w[4])
            ]

            college_anchors.sort(key=lambda w: w[1])

            # Course code anchors identify individual courses.
            course_anchors = [
                w for w in data_words
                if COURSE_CODE_X[0] <= w[0] < COURSE_CODE_X[1]
                and re.fullmatch(r"[A-Z]{2,3}", w[4])
            ]

            # HK/RK anchors identify the two seat-type rows.
            seat_anchors = [
                w for w in data_words
                if SEAT_TYPE_X[0] <= w[0] < SEAT_TYPE_X[1]
                and w[4] in ("HK", "RK")
            ]

            # Link both HK and RK rows to the course whose
            # vertical section contains each seat-type row.
            for course_index, course in enumerate(course_anchors):
                course_y = course[1]

                next_course_y = (
                    course_anchors[course_index + 1][1]
                    if course_index + 1 < len(course_anchors)
                    else float("inf")
                )

                matching_seats = [
                    seat for seat in seat_anchors
                    if course_y - 2.5 <= seat[1] < next_course_y - 2.5
                ]

                matching_types = [seat[4] for seat in matching_seats]

                # Skip incomplete or ambiguous course sections.
                if (
                    matching_types.count("HK") != 1
                    or matching_types.count("RK") != 1
                ):
                    print(
                        f"SKIPPED SEAT MATCH: page={page_number}, "
                        f"course={course[4]}, y={course_y:.1f}, "
                        f"types={matching_types}"
                    )
                    continue

                # Find the college whose vertical section contains
                # this course. College sections can overlap vertically
                # because addresses wrap onto multiple lines.
                previous_colleges = [
                    c for c in college_anchors if c[1] <= course_y
                ]

                if not previous_colleges:
                    print(
                        f"SKIPPED COLLEGE MATCH: page={page_number}, "
                        f"course={course[4]}, y={course_y:.1f}"
                    )
                    continue

                college_anchor = previous_colleges[-1]
                college_code = college_anchor[4]

                # Build college name from the college-name column
                # between this college anchor and the next one.
                next_colleges = [
                    c for c in college_anchors if c[1] > college_anchor[1]
                ]
                next_college_y = (
                    next_colleges[0][1] if next_colleges else float("inf")
                )

                college_name_words = [
                    w for w in data_words
                    if COLLEGE_NAME_X[0] <= w[0] < COLLEGE_NAME_X[1]
                    and college_anchor[1] - 1 <= w[1] < next_college_y
                ]

                college_name = clean(" ".join(
                    w[4] for w in sorted(
                        college_name_words, key=lambda item: (item[1], item[0])
                    )
                ))

                # The course name is the text between this course's
                # code and the next course code, restricted to its column.
                next_courses = [
                    c for c in course_anchors if c[1] > course_y
                ]
                next_course_y = (
                    next_courses[0][1] if next_courses else float("inf")
                )

                course_name_words = [
                    w for w in data_words
                    if COURSE_NAME_X[0] <= w[0] < COURSE_NAME_X[1]
                    and course_y - 1 <= w[1] < next_course_y
                ]

                course_name = clean(" ".join(
                    w[4] for w in sorted(
                        course_name_words, key=lambda item: (item[1], item[0])
                    )
                ))

                # Read the numeric counts at the same y-coordinate
                # as each HK/RK seat-type row.
                for seat in matching_seats:
                    seat_y = seat[1]
                    row_words = [
                        w for w in data_words
                        if abs(w[1] - seat_y) <= 2.5
                    ]

                    counts = []
                    for center in CATEGORY_CENTERS:
                        candidates = [
                            w for w in row_words
                            if abs(((w[0] + w[2]) / 2) - center) <= 10
                            and w[0] >= COUNT_START_X
                        ]

                        # Avoid silently selecting an ambiguous value.
                        if len(candidates) == 1:
                            counts.append(numeric_value(candidates[0][4]))
                        else:
                            counts.append(None)

                    total_candidates = [
                        w for w in row_words
                        if TOTAL_X[0] <= w[0] < TOTAL_X[1]
                        and numeric_value(w[4]) is not None
                    ]

                    total = (
                        numeric_value(total_candidates[-1][4])
                        if total_candidates else None
                    )

                    record = {
                        "Source Year": 2026,
                        "PDF Page": page_number,
                        "College Code": college_code,
                        "College Name": college_name,
                        "Course Code": course[4],
                        "Course Name": course_name,
                        "Seat Type": seat[4],
                    }

                    for group_index, group in enumerate(GROUPS):
                        for category_index, category in enumerate(CATEGORIES):
                            count_index = group_index * len(CATEGORIES) + category_index
                            record[f"{group} - {category}"] = counts[count_index]

                    record["Total HK-RK"] = total
                    records.append(record)

    finally:
        pdf.close()

    if not records:
        print("ERROR: No records extracted.")
        print("No output CSV was written.")
        return

    # Sort and remove exact duplicate records.
    unique_records = []
    seen = set()

    for record in records:
        key = tuple(record.items())
        if key not in seen:
            seen.add(key)
            unique_records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(unique_records[0].keys())

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_records)

    print("Extraction finished; this is a preliminary result.")
    print("Records:", len(unique_records))
    print("Unique colleges:", len({
        r["College Code"] for r in unique_records
    }))
    print("Output:", OUTPUT_PATH)
    print("\nFirst 10 records:")

    for record in unique_records[:10]:
        print(
            record["College Code"],
            "|", record["College Name"][:55],
            "|", record["Course Code"],
            "|", record["Course Name"][:45],
            "|", record["Seat Type"],
            "| Total:", record["Total HK-RK"],
        )

    print("\nIMPORTANT: Validate counts and course names before use.")
    print("The live predictor datasets and ML model were not changed.")


if __name__ == "__main__":
    main()
