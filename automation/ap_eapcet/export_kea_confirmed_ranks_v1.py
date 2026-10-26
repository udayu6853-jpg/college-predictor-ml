from pathlib import Path
import pandas as pd
import re
from datetime import date

ROOT = Path.cwd()
WORKSHEET = ROOT / "automation" / "output" / "kea_split_decimal_verification_v1.csv"
EXPORT = ROOT / "automation" / "output" / "kea_split_decimal_confirmed_v1.csv"

REQUIRED = [
    "Source PDF",
    "Source Page",
    "College Code",
    "College Heading",
    "Course Name Normalized",
    "Category",
    "Rank Raw",
    "Proposed Rank",
    "Reviewer Decision",
    "Verified Rank",
    "Evidence Checked",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
    "Reviewer Notes",
]

def valid_date(value):
    try:
        return date.fromisoformat(value).isoformat() == value
    except (ValueError, TypeError):
        return False

def main():
    if not WORKSHEET.is_file():
        raise FileNotFoundError(f"Worksheet not found: {WORKSHEET}")

    if EXPORT.exists():
        raise FileExistsError(
            f"Export already exists; refusing to overwrite: {EXPORT}"
        )

    df = pd.read_csv(WORKSHEET, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    confirmed = df[
        df["Reviewer Decision"].str.strip().str.upper() == "CONFIRMED"
    ].copy()

    accepted = []
    excluded = []

    for _, row in confirmed.iterrows():
        problems = []

        if row["Evidence Checked"].strip().upper() != "YES":
            problems.append("Evidence Checked must be YES")

        rank = row["Verified Rank"].strip()
        if not rank or not re.fullmatch(r"\d+(?:\.\d+)?", rank):
            problems.append("Verified Rank must be a valid numeric rank")

        if not row["Reviewed By"].strip():
            problems.append("Reviewed By is required")

        review_date = row["Review Date (YYYY-MM-DD)"].strip()
        if not valid_date(review_date):
            problems.append("Valid Review Date (YYYY-MM-DD) is required")

        for field in ["Source PDF", "College Code", "Category",
                      "Course Name Normalized", "Rank Raw"]:
            if not row[field].strip():
                problems.append(f"{field} is required")

        if problems:
            excluded.append({
                "Source PDF": row["Source PDF"],
                "Source Page": row["Source Page"],
                "College Code": row["College Code"],
                "Category": row["Category"],
                "Problems": "; ".join(problems),
            })
        else:
            accepted.append(row.to_dict())

    export_columns = [
        "Source PDF", "Source Page", "College Code", "College Heading",
        "Course Name Normalized", "Category", "Rank Raw", "Proposed Rank",
        "Verified Rank", "Evidence Checked", "Reviewed By",
        "Review Date (YYYY-MM-DD)", "Reviewer Notes",
    ]

    pd.DataFrame(accepted, columns=REQUIRED).reindex(
        columns=export_columns
    ).to_csv(EXPORT, index=False, encoding="utf-8-sig")

    print("Worksheet records:", len(df))
    print("Marked CONFIRMED:", len(confirmed))
    print("Eligible confirmed records:", len(accepted))
    print("Excluded confirmed records:", len(excluded))
    print("Export:", EXPORT)
    print("Live datasets, database, and model were not modified.")

if __name__ == "__main__":
    main()
