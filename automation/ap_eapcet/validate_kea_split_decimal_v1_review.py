from pathlib import Path
from datetime import date
import pandas as pd
import re

ROOT = Path.cwd()
WORKSHEET = ROOT / "automation" / "output" / "kea_split_decimal_verification_v1.csv"
REPORT = ROOT / "automation" / "output" / "kea_split_decimal_validation_v1_review.csv"

if not WORKSHEET.is_file():
    raise FileNotFoundError(f"Worksheet not found: {WORKSHEET}")

if REPORT.exists():
    raise FileExistsError(
        f"Validation report already exists; refusing to overwrite: {REPORT}"
    )

df = pd.read_csv(WORKSHEET, dtype=str, keep_default_na=False)

required = [
    "Source PDF",
    "Source Page",
    "College Code",
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

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

issues = []

for index, row in df.iterrows():
    row_number = index + 2
    decision = row["Reviewer Decision"].strip().upper()
    evidence = row["Evidence Checked"].strip().upper()
    verified_rank = row["Verified Rank"].strip()
    reviewer = row["Reviewed By"].strip()
    review_date = row["Review Date (YYYY-MM-DD)"].strip()

    if decision not in {"PENDING", "CONFIRMED", "REJECTED"}:
        issues.append((row_number, "INVALID_DECISION", decision))

    if decision == "CONFIRMED":
        if evidence != "YES":
            issues.append((row_number, "CONFIRMED_WITHOUT_EVIDENCE", evidence))

        if not verified_rank:
            issues.append((row_number, "CONFIRMED_WITHOUT_VERIFIED_RANK", ""))

        elif not re.fullmatch(r"\d+(?:\.\d+)?", verified_rank):
            issues.append((row_number, "INVALID_VERIFIED_RANK_FORMAT", verified_rank))

        if not reviewer:
            issues.append((row_number, "MISSING_REVIEWER", ""))

        if not review_date:
            issues.append((row_number, "MISSING_REVIEW_DATE", ""))
        else:
            try:
                parsed_date = date.fromisoformat(review_date)
                if parsed_date.isoformat() != review_date:
                    raise ValueError
            except ValueError:
                issues.append((row_number, "INVALID_REVIEW_DATE", review_date))

    if decision == "REJECTED" and not row["Reviewer Notes"].strip():
        issues.append((row_number, "REJECTION_NEEDS_NOTES", ""))

    if evidence not in {"YES", "NO"}:
        issues.append((row_number, "INVALID_EVIDENCE_FLAG", evidence))

issue_df = pd.DataFrame(
    issues,
    columns=["Worksheet Row", "Issue", "Value"]
)

issue_df.to_csv(REPORT, index=False, encoding="utf-8-sig")

print("Worksheet records:", len(df))
print("Pending:", (df["Reviewer Decision"].str.upper() == "PENDING").sum())
print("Confirmed:", (df["Reviewer Decision"].str.upper() == "CONFIRMED").sum())
print("Rejected:", (df["Reviewer Decision"].str.upper() == "REJECTED").sum())
print("Validation issues:", len(issue_df))
print("Validation report:", REPORT)
print("Read-only validation complete; worksheet and live data were not modified.")

