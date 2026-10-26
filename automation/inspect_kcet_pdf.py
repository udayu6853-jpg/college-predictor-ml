
from pathlib import Path

import requests

URL = (
    "https://cetonline.karnataka.gov.in/"
    "keawebentry456/ugcet2026/"
    "ENG_SEAT_08072026kannada.pdf"
)

OUTPUT = Path("automation/source_samples/engineering_seat_matrix_2026.pdf")


def main():
    print("Downloading official KEA PDF...")
    response = requests.get(
        URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )

    print("HTTP status:", response.status_code)
    print("Content type:", response.headers.get("Content-Type"))
    response.raise_for_status()

    if not response.content.startswith(b"%PDF-"):
        print("The response is not a valid PDF file.")
        print("First bytes:", response.content[:100])
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(response.content)

    print("PDF saved to:", OUTPUT)
    print("File size:", len(response.content), "bytes")
    print("Live datasets and ML model were not changed.")


if __name__ == "__main__":
    main()
