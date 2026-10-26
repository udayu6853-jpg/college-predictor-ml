"""
AP EAPCET College Mapping — Human Verification Workflow v5.10

Reads the V5.9 review CSV and creates:
1. automation/output/ap_eapcet_human_review_v5_10.csv
   A spreadsheet for a person to verify every current college record.
2. automation/output/ap_eapcet_confirmed_mapping_v5_10.csv
   Contains ONLY rows explicitly marked CONFIRMED with an official code,
   verified official name, verified location, and evidence URL.

Safety:
- Does not modify the source V5.9 CSV.
- Does not modify datasets, model files, database, or live app.
- Does not auto-confirm any match, including V5.9 SAFE rows.
- Preserves human review fields if you rerun the script.
"""

from pathlib import Path
import sys
import pandas as pd


SCRIPT_VERSION = "5.10"

INPUT_RELATIVE = Path(
    "automation/output/ap_eapcet_unique_college_mapping_review_v5_9.csv"
)
REVIEW_RELATIVE = Path(
    "automation/output/ap_eapcet_human_review_v5_10.csv"
)
CONFIRMED_RELATIVE = Path(
    "automation/output/ap_eapcet_confirmed_mapping_v5_10.csv"
)

REVIEW_FIELDS = [
    "Reviewer Decision",
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewer Notes",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
]

ALLOWED_DECISIONS = {
    "",
    "PENDING",
    "CONFIRMED",
    "REJECTED",
    "NEEDS_MORE_EVIDENCE",
    "NO_MATCH",
}

BASE_COLUMNS = [
    "Status",
    "Decision Reason",
    "Review Flags",
    "Current College Name",
    "Current Location",
    "Current Canonical Name",
    "Current Canonical Location",
    "Current Acronym",
    "Valid Candidate Count",
    "Relevant Type Conflict Count",
    "Best Candidate Name Score",
    "Best Candidate Total Score",
    "Best Candidate Same Location",
    "Best Candidate Exact Canonical Name",
    "Candidate Ranking Note",
]

CANDIDATE_COLUMNS = []
for number in range(1, 6):
    CANDIDATE_COLUMNS.extend(
        [
            f"Candidate {number} Name",
            f"Candidate {number} Location",
            f"Candidate {number} Code",
            f"Candidate {number} Branches",
            f"Candidate {number} Year",
            f"Candidate {number} Name Similarity",
            f"Candidate {number} Total Score",
            f"Candidate {number} Same Location",
            f"Candidate {number} Code Record Count",
            f"Candidate {number} Exact Canonical Name",
            f"Candidate {number} Token Jaccard",
            f"Candidate {number} Token Containment",
            f"Candidate {number} Acronym Match",
        ]
    )

TYPE_CONFLICT_COLUMNS = []
for number in range(1, 4):
    TYPE_CONFLICT_COLUMNS.extend(
        [
            f"Type Conflict {number} Name",
            f"Type Conflict {number} Location",
            f"Type Conflict {number} Code",
            f"Type Conflict {number} Name Score",
        ]
    )

OUTPUT_COLUMNS = (
    BASE_COLUMNS
    + CANDIDATE_COLUMNS
    + TYPE_CONFLICT_COLUMNS
    + REVIEW_FIELDS
)


def clean_text(value):
    """Convert a spreadsheet cell to clean text without 'nan' strings."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_record_key(row):
    """
    Stable key used to preserve reviewer entries when rerunning.
    Include the current college name and canonical location as extra protection.
    """
    return " | ".join(
        clean_text(row.get(column, ""))
        for column in [
            "Current Canonical Name",
            "Current Canonical Location",
            "Current College Name",
            "Current Location",
        ]
    ).casefold()


def find_input(project_root):
    """Find the expected V5.9 CSV; also try case-insensitive filename search."""
    expected = project_root / INPUT_RELATIVE
    if expected.exists():
        return expected

    output_dir = project_root / "automation" / "output"
    if output_dir.exists():
        for file_path in output_dir.glob("*.csv"):
            if file_path.name.casefold() == INPUT_RELATIVE.name.casefold():
                return file_path

    return None


def load_existing_reviews(review_path):
    """Load prior human decisions so rerunning does not erase them."""
    if not review_path.exists():
        return {}

    try:
        previous = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        print(
            f"WARNING: Could not read existing review file: {exc}\n"
            "The script will continue without importing prior decisions."
        )
        return {}

    required_key_columns = [
        "Current Canonical Name",
        "Current Canonical Location",
        "Current College Name",
        "Current Location",
    ]
    if not all(column in previous.columns for column in required_key_columns):
        print(
            "WARNING: Existing review file does not have the expected key "
            "columns. Prior decisions were not imported."
        )
        return {}

    preserved = {}
    for _, row in previous.iterrows():
        key = make_record_key(row)
        preserved[key] = {
            field: clean_text(row.get(field, ""))
            for field in REVIEW_FIELDS
            if field in previous.columns
        }
    return preserved


def build_review_file(source_df, previous_reviews):
    """Build the reviewer worksheet without accepting any machine suggestion."""
    result = pd.DataFrame(index=source_df.index)

    for column in OUTPUT_COLUMNS:
        if column in source_df.columns:
            result[column] = source_df[column].fillna("")
        elif column in REVIEW_FIELDS:
            result[column] = ""
        else:
            # Keep the output shape predictable if a source column is absent.
            result[column] = ""

    # Initialize the decision field explicitly; never auto-confirm SAFE records.
    for idx, row in result.iterrows():
        key = make_record_key(row)
        old_values = previous_reviews.get(key, {})
        for field in REVIEW_FIELDS:
            if field in old_values:
                result.at[idx, field] = old_values[field]

        if not clean_text(result.at[idx, "Reviewer Decision"]):
            result.at[idx, "Reviewer Decision"] = "PENDING"

    # Make the most important fields easy to review first.
    first_columns = [
        "Reviewer Decision",
        "Status",
        "Current College Name",
        "Current Location",
        "Candidate 1 Name",
        "Candidate 1 Location",
        "Candidate 1 Code",
        "Candidate 1 Name Similarity",
        "Candidate 1 Total Score",
        "Candidate 1 Same Location",
        "Candidate 2 Name",
        "Candidate 2 Location",
        "Candidate 2 Code",
        "Candidate 2 Name Similarity",
        "Candidate 2 Total Score",
        "Candidate 3 Name",
        "Candidate 3 Location",
        "Candidate 3 Code",
        "Candidate 3 Name Similarity",
        "Candidate 3 Total Score",
        "Verified Official Name",
        "Verified Location",
        "Verified Official Code",
        "Evidence URL",
        "Reviewer Notes",
        "Reviewed By",
        "Review Date (YYYY-MM-DD)",
    ]
    ordered = [column for column in first_columns if column in result.columns]
    ordered += [column for column in result.columns if column not in ordered]
    return result[ordered]


def validate_review_decisions(review_df):
    """Warn about unsupported decisions and incomplete confirmations."""
    errors = []
    warnings = []

    for index, row in review_df.iterrows():
        decision = clean_text(row.get("Reviewer Decision", "")).upper()
        college = clean_text(row.get("Current College Name", ""))

        if decision not in ALLOWED_DECISIONS:
            errors.append(
                f"Row {index + 2} ({college}): invalid decision '{decision}'."
            )
            continue

        if decision == "CONFIRMED":
            required = [
                "Verified Official Name",
                "Verified Location",
                "Verified Official Code",
                "Evidence URL",
            ]
            missing = [
                field for field in required
                if not clean_text(row.get(field, ""))
            ]
            if missing:
                errors.append(
                    f"Row {index + 2} ({college}): marked CONFIRMED but "
                    f"missing: {', '.join(missing)}."
                )

            evidence = clean_text(row.get("Evidence URL", ""))
            if evidence and not (
                evidence.startswith("https://") or evidence.startswith("http://")
            ):
                warnings.append(
                    f"Row {index + 2} ({college}): Evidence URL should start "
                    "with http:// or https://."
                )

    return errors, warnings


def make_confirmed_mapping(review_df):
    """Extract only fully completed, explicitly human-confirmed mappings."""
    rows = []
    for _, row in review_df.iterrows():
        decision = clean_text(row.get("Reviewer Decision", "")).upper()
        if decision != "CONFIRMED":
            continue

        official_name = clean_text(row.get("Verified Official Name", ""))
        location = clean_text(row.get("Verified Location", ""))
        code = clean_text(row.get("Verified Official Code", ""))
        evidence = clean_text(row.get("Evidence URL", ""))

        if not all([official_name, location, code, evidence]):
            continue

        rows.append(
            {
                "Current College Name": clean_text(
                    row.get("Current College Name", "")
                ),
                "Current Location": clean_text(row.get("Current Location", "")),
                "Verified Official Name": official_name,
                "Verified Location": location,
                "Verified Official Code": code,
                "Evidence URL": evidence,
                "Reviewer Decision": "CONFIRMED",
                "Reviewed By": clean_text(row.get("Reviewed By", "")),
                "Review Date (YYYY-MM-DD)": clean_text(
                    row.get("Review Date (YYYY-MM-DD)", "")
                ),
                "Reviewer Notes": clean_text(row.get("Reviewer Notes", "")),
                "Original V5.9 Status": clean_text(row.get("Status", "")),
                "Original Decision Reason": clean_text(
                    row.get("Decision Reason", "")
                ),
                "Candidate 1 Suggested Name": clean_text(
                    row.get("Candidate 1 Name", "")
                ),
                "Candidate 1 Suggested Location": clean_text(
                    row.get("Candidate 1 Location", "")
                ),
                "Candidate 1 Suggested Code": clean_text(
                    row.get("Candidate 1 Code", "")
                ),
                "Candidate 1 Name Similarity": clean_text(
                    row.get("Candidate 1 Name Similarity", "")
                ),
                "Candidate 1 Total Score": clean_text(
                    row.get("Candidate 1 Total Score", "")
                ),
            }
        )

    columns = [
        "Current College Name",
        "Current Location",
        "Verified Official Name",
        "Verified Location",
        "Verified Official Code",
        "Evidence URL",
        "Reviewer Decision",
        "Reviewed By",
        "Review Date (YYYY-MM-DD)",
        "Reviewer Notes",
        "Original V5.9 Status",
        "Original Decision Reason",
        "Candidate 1 Suggested Name",
        "Candidate 1 Suggested Location",
        "Candidate 1 Suggested Code",
        "Candidate 1 Name Similarity",
        "Candidate 1 Total Score",
    ]
    return pd.DataFrame(rows, columns=columns)


def main():
    # This script is located in project_root/automation/ap_eapcet/.
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "automation" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = find_input(project_root)
    if input_path is None:
        print("ERROR: Could not find the V5.9 input CSV.")
        print(f"Expected file: {project_root / INPUT_RELATIVE}")
        print(
            "\nCopy your existing V5.9 report to that exact path, then run "
            "this script again."
        )
        sys.exit(1)

    review_path = project_root / REVIEW_RELATIVE
    confirmed_path = project_root / CONFIRMED_RELATIVE

    try:
        source_df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        print(f"ERROR: Could not read {input_path}\n{exc}")
        sys.exit(1)

    if "Current College Name" not in source_df.columns:
        print(
            "ERROR: The input CSV does not contain 'Current College Name'. "
            "Make sure you selected the V5.9 mapping review CSV."
        )
        sys.exit(1)

    previous_reviews = load_existing_reviews(review_path)
    review_df = build_review_file(source_df, previous_reviews)

    errors, warnings = validate_review_decisions(review_df)

    # Write the review worksheet even when a previous confirmation is incomplete;
    # the validation output tells the reviewer what needs fixing.
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")

    confirmed_df = make_confirmed_mapping(review_df)
    confirmed_df.to_csv(confirmed_path, index=False, encoding="utf-8-sig")

    print("\nAP EAPCET College Mapping — Human Review v" + SCRIPT_VERSION)
    print("-" * 58)
    print(f"Input rows:              {len(source_df)}")
    print(f"Review worksheet rows:   {len(review_df)}")
    print(f"Confirmed mapping rows:  {len(confirmed_df)}")
    print(f"\nInput:     {input_path}")
    print(f"Review:    {review_path}")
    print(f"Confirmed: {confirmed_path}")

    if previous_reviews:
        print(
            f"\nPreserved prior reviewer decisions for "
            f"{len(previous_reviews)} review key(s), where keys matched."
        )

    if errors:
        print(f"\nVALIDATION ERRORS ({len(errors)}):")
        for message in errors:
            print(" - " + message)
        print(
            "\nFix these rows in the review CSV. Only complete CONFIRMED rows "
            "are exported to the confirmed mapping file."
        )
    else:
        print("\nNo incomplete CONFIRMED rows found.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for message in warnings:
            print(" - " + message)

    print(
        "\nIMPORTANT: The script does not verify the truth of a college match. "
        "A human reviewer must check official evidence before marking a row "
        "CONFIRMED."
    )
    print(
        "Allowed decisions: PENDING, CONFIRMED, REJECTED, "
        "NEEDS_MORE_EVIDENCE, NO_MATCH."
    )


if __name__ == "__main__":
    main()
