from pathlib import Path
import pymupdf
import csv
import re

BASE = Path("automation/source_samples/kea_pdf_inspection")
OUT = Path("automation/output")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "R_E.pdf": "Rest of Karnataka",
    "H_E.pdf": "Kalyana Karnataka 371(j)",
    "SPL_E.pdf": "Special category",
}

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def inspect_pdf(filename, seat_type):
    path = BASE / filename

    with pymupdf.open(path) as doc:
        first_page = doc[0].get_text("text")
        title_lines = [
            clean_text(line)
            for line in first_page.splitlines()
            if "CUT-OFF RANKS" in line.upper()
            or "CUT-OFF RANK" in line.upper()
        ]

        print(f"\nFile: {filename}")
        print(f"Seat type: {seat_type}")
        print(f"Pages: {len(doc)}")
        print(f"Title: {' | '.join(title_lines) or 'Not detected'}")

        college_count = 0
        course_count = 0

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            college_count += len(re.findall(r"\bCollege:\s*[A-Z]\d{3}\b", text))
            course_count += len(re.findall(
                r"(?m)^\s*(?:ARTIFICIAL|CIVIL|COMPUTER|ELECTRICAL|"
                r"ELECTRONICS|INFORMATION|MECHANICAL|ROBOTICS|"
                r"INDUSTRIAL|BIO-TECHNOLOGY|SILK|TEXTILES)",
                text,
                flags=re.IGNORECASE,
            ))

        print(f"College headers detected: {college_count}")
        print(f"Course-name starts detected: {course_count}")

        return {
            "Source PDF": filename,
            "Seat Type": seat_type,
            "Pages": len(doc),
            "Title": " | ".join(title_lines),
            "College Headers Detected": college_count,
            "Course Starts Detected": course_count,
            "Status": "INSPECTED_ONLY_NOT_IMPORTED",
        }

reports = []

for filename, seat_type in SOURCES.items():
    if not (BASE / filename).exists():
        print(f"\nMISSING FILE: {BASE / filename}")
        continue

    reports.append(inspect_pdf(filename, seat_type))

report_path = OUT / "kea_cutoff_pdf_inspection_v1.csv"

with report_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=reports[0].keys())
    writer.writeheader()
    writer.writerows(reports)

print(f"\nInspection report saved to: {report_path}")
print("No college dataset, database, or model was modified.")
