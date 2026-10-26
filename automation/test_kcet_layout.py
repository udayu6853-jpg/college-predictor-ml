
import pymupdf
import re

PDF_PATH = (
    "automation/source_samples/"
    "engineering_seat_matrix_2026.pdf"
)

# Page index 1 means page 2 of the PDF.
PAGE_INDEX = 1

def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def main():
    pdf = pymupdf.open(PDF_PATH)
    page = pdf[PAGE_INDEX]

    # Extract words with their actual PDF coordinates.
    words = page.get_text("words")

    # Select words belonging to the E004 college entry.
    college_words = [
        w for w in words
        if 145 <= w[1] <= 280 and 40 <= w[0] < 185
    ]

    # Select words belonging to its course-name column.
    course_words = [
        w for w in words
        if 145 <= w[1] <= 280 and 210 <= w[0] < 300
    ]

    # Select words belonging to its course-code column.
    code_words = [
        w for w in words
        if 145 <= w[1] <= 280 and 185 <= w[0] < 210
    ]

    # Print the extracted text for inspection.
    print("COLLEGE COLUMN:")
    print(clean(" ".join(w[4] for w in college_words)))

    print("\nCOURSE CODE COLUMN:")
    print(clean(" ".join(w[4] for w in code_words)))

    print("\nCOURSE NAME COLUMN:")
    print(clean(" ".join(w[4] for w in course_words)))

    print("\nSEAT-TYPE AND COUNT SAMPLE:")
    for w in words:
        if 145 <= w[1] <= 180 and 300 <= w[0] < 340:
            print(
                f"text={w[4]!r}, "
                f"x={w[0]:.1f}, y={w[1]:.1f}"
            )

    pdf.close()
    print("\nRead-only test completed.")


if __name__ == "__main__":
    main()
