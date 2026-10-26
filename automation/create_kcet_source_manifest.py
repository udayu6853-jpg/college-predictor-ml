
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "automation"
    / "source_samples"
    / "engineering_seat_matrix_2026.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "automation"
    / "source_samples"
    / "kcet_source_manifest.csv"
)

SOURCE_URL = (
    "https://cetonline.karnataka.gov.in/"
    "keawebentry456/ugcet2026/"
    "ENG_SEAT_08072026kannada.pdf"
)


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().upper()


def main():
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Original source PDF not found: {PDF_PATH}"
        )

    request = Request(
        SOURCE_URL,
        method="HEAD",
        headers={"User-Agent": "CollegePredictor-SourceVerifier/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        status_code = response.status
        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length", "")
        last_modified = response.headers.get("Last-Modified", "")
        etag = response.headers.get("ETag", "")

    file_size = PDF_PATH.stat().st_size
    file_hash = calculate_sha256(PDF_PATH)

    source_date = ""
    if last_modified:
        try:
            source_date = parsedate_to_datetime(
                last_modified
            ).isoformat()
        except (TypeError, ValueError):
            source_date = last_modified

    record = {
        "Retrieved At": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "Source URL": SOURCE_URL,
        "HTTP Status": status_code,
        "Content Type": content_type,
        "Official Content Length": content_length,
        "Local File Size": file_size,
        "Official Last Modified": source_date,
        "ETag": etag,
        "Local SHA256": file_hash,
        "Source File": str(PDF_PATH.relative_to(PROJECT_ROOT)),
        "Validation Status": (
            "SOURCE_URL_REACHABLE_HASH_RECORDED"
            if status_code == 200 and "pdf" in content_type.lower()
            else "SOURCE_CHECK_REQUIRES_REVIEW"
        ),
        "Notes": (
            "Records source metadata only. Does not prove that no "
            "newer official seat matrix exists. Does not modify the "
            "live dataset, database, or ML model."
        ),
    }

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=record.keys()
        )
        writer.writeheader()
        writer.writerow(record)

    print("KCET source manifest created.")
    print(f"Source HTTP status: {status_code}")
    print(f"Content type: {content_type}")
    print(f"Official content length: {content_length}")
    print(f"Local file size: {file_size}")
    print(f"Official Last-Modified: {source_date}")
    print(f"SHA256: {file_hash}")
    print(f"Manifest: {OUTPUT_PATH}")
    print("Live CSV, database, and model were not modified.")


if __name__ == "__main__":
    main()