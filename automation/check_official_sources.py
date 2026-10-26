
import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests


# Project paths
BASE_DIR = Path(__file__).resolve().parent
SOURCE_SAMPLES_DIR = BASE_DIR / "source_samples"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "official_source_check_report.csv"

# Official source pages only.
# A reachable page does not guarantee that its data is current.
SOURCES = [
    {
        "exam": "KCET",
        "source_type": "Official counselling portal",
        "url": "https://cetonline.karnataka.gov.in/kea/",
    },
    {
        "exam": "AP EAPCET",
        "source_type": "Official counselling portal",
        "url": (
            "https://cets.apsche.ap.gov.in/"
            "EAPCET/Eapcet/EAPCET_HomePage.aspx"
        ),
    },
]

TIMEZONE = ZoneInfo("Asia/Kolkata")
TIMEOUT_SECONDS = 30


def check_source(source):
    """Check an official source without modifying project datasets."""

    checked_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")
    parsed_url = urlparse(source["url"])

    result = {
        "Checked At": checked_at,
        "Exam": source["exam"],
        "Source Type": source["source_type"],
        "Source URL": source["url"],
        "HTTP Status": "",
        "Content Type": "",
        "Content Length": "",
        "Last Modified": "",
        "ETag": "",
        "SHA256": "",
        "Check Status": "",
        "Notes": "",
    }

    if parsed_url.scheme != "https" or not parsed_url.netloc:
        result["Check Status"] = "INVALID_URL"
        result["Notes"] = "Only valid HTTPS source URLs are allowed."
        return result

    try:
        response = requests.get(
            source["url"],
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
            stream=True,
            headers={
                "User-Agent": "CollegePredictor-SourceChecker/1.0"
            },
        )

        result["HTTP Status"] = response.status_code
        result["Content Type"] = response.headers.get(
            "Content-Type", ""
        )
        result["Content Length"] = response.headers.get(
            "Content-Length", ""
        )
        result["Last Modified"] = response.headers.get(
            "Last-Modified", ""
        )
        result["ETag"] = response.headers.get("ETag", "")

        if response.ok:
            # Read in chunks to calculate a hash without saving a file.
            digest = hashlib.sha256()
            bytes_read = 0

            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    digest.update(chunk)
                    bytes_read += len(chunk)

            result["SHA256"] = digest.hexdigest()
            result["Check Status"] = "REACHABLE_HASH_RECORDED"
            result["Notes"] = (
                f"Read {bytes_read} response bytes. "
                "Hash identifies this response only; it does not "
                "prove the page contains current college data."
            )
        else:
            result["Check Status"] = "HTTP_ERROR"
            result["Notes"] = "Source returned a non-success HTTP status."

        response.close()

    except requests.RequestException as exc:
        result["Check Status"] = "REQUEST_FAILED"
        result["Notes"] = f"{type(exc).__name__}: {exc}"

    return result


def main():
    results = []

    print("Official source check")
    print("Live datasets, database, and model will not be modified.")
    print("-" * 60)

    for source in SOURCES:
        print(f"\nChecking {source['exam']}...")
        result = check_source(source)
        results.append(result)

        print(f"Status: {result['Check Status']}")
        print(f"HTTP: {result['HTTP Status']}")
        print(f"Content type: {result['Content Type']}")
        print(f"Last modified: {result['Last Modified']}")

    fieldnames = list(results[0].keys())

    with REPORT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "-" * 60)
    print(f"Report saved to: {REPORT_FILE}")
    print("No live datasets, database tables, or model files were changed.")


if __name__ == "__main__":
    main()
