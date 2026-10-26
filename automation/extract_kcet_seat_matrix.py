
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

# Approximate centers of the seat-count columns.
# These positions must still be validated against the PDF.

CATEGORY_CENTERS = [
    371.65, 402.45, 430.45, 464.0, 493.2, 523.45,
    550.05, 576.0, 601.9, 629.9, 664.4,
    688.7, 709.4, 732.2, 753.3, 778.9,
    800.2, 826.1, 855.4, 883.5, 914.6,
    944.8, 970.8, 999.4, 1019.9, 1041.3,
    1064.9, 1090.0, 1118.2, 1144.8, 1172.2,
    1201.0,
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

            # Ignore the header area.
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

            course_anchors.sort(key=lambda w: w[1])

            # HK/RK anchors identify seat-type rows.
            seat_anchors = [
                w for w in data_words
                if SEAT_TYPE_X[0] <= w[0] < SEAT_TYPE_X[1]
                and w[4] in ("HK", "RK")
            ]

            seat_anchors.sort(key=lambda w: w[1])

            # Find rows marked TOT. These are summary rows, not
            # additional HK/RK rows belonging to a course.
            total_row_y = [
                w[1] for w in data_words
                if SEAT_TYPE_X[0] <= w[0] < SEAT_TYPE_X[1]
                and w[4] == "TOT"
            ]

            # Match HK/RK rows to their course.
            for course_index, course in enumerate(course_anchors):
                course_y = course[1]

                next_course_y = (
                    course_anchors[course_index + 1][1]
                    if course_index + 1 < len(course_anchors)
                    else float("inf")
                )

                # Stop the last course section before the total rows.
                # For earlier courses, the next course boundary
                # normally comes first.
                section_end_y = next_course_y

                following_totals = [
                    y for y in total_row_y
                    if y >= course_y
                ]

                if following_totals:
                    section_end_y = min(
                        section_end_y,
                        min(following_totals)
                    )

                matching_seats = [
                    seat for seat in seat_anchors
                    if course_y - 2.5 <= seat[1] < section_end_y - 2.5
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

                # Find the college whose section contains this course.
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

                # Build the college name between college anchors.
                next_colleges = [
                    c for c in college_anchors
                    if c[1] > college_anchor[1]
                ]

                next_college_y = (
                    next_colleges[0][1]
                    if next_colleges
                    else float("inf")
                )

                college_name_words = [
                    w for w in data_words
                    if COLLEGE_NAME_X[0] <= w[0] < COLLEGE_NAME_X[1]
                    and college_anchor[1] - 1 <= w[1] < next_college_y
                ]

                college_name = clean(" ".join(
                    w[4] for w in sorted(
                        college_name_words,
                        key=lambda item: (item[1], item[0])
                    )
                ))

                # Build the course name between course-code anchors.
                next_courses = [
                    c for c in course_anchors if c[1] > course_y
                ]

                course_name_end_y = (
                    next_courses[0][1]
                    if next_courses
                    else float("inf")
                )

                course_name_words = [
                    w for w in data_words
                    if COURSE_NAME_X[0] <= w[0] < COURSE_NAME_X[1]
                    and course_y - 1 <= w[1] < course_name_end_y
                ]

                course_name = clean(" ".join(
                    w[4] for w in sorted(
                        course_name_words,
                        key=lambda item: (item[1], item[0])
                    )
                ))

                # Extract category counts from each HK/RK row.
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

                        if len(candidates) == 1:
                            counts.append(
                                numeric_value(candidates[0][4])
                            )
                        else:
                            counts.append(None)

                    total_candidates = [
                        w for w in row_words
                        if TOTAL_X[0] <= w[0] < TOTAL_X[1]
                        and numeric_value(w[4]) is not None
                    ]

                    total = (
                        numeric_value(total_candidates[-1][4])
                        if total_candidates
                        else None
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
                            count_index = (
                                group_index * len(CATEGORIES)
                                + category_index
                            )

                            record[f"{group} - {category}"] = (
                                counts[count_index]
                            )

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
