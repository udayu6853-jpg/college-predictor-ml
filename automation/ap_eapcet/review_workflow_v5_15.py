
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd


# ============================================================
# V5.15 CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("automation/output")

INPUT_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_14.csv"

REVIEW_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_15.csv"

CONFIRMED_FILE = OUTPUT_DIR / "ap_eapcet_confirmed_mapping_v5_15.csv"

ISSUES_FILE = OUTPUT_DIR / "ap_eapcet_review_validation_issues_v5_15.csv"

SUMMARY_FILE = OUTPUT_DIR / "ap_eapcet_review_validation_summary_v5_15.txt"


IDENTITY_FIELDS = [
    "Current Canonical Name",
    "Current Canonical Location",
    "Current College Name",
    "Current Location",
]


REVIEW_FIELDS = [
    "Reviewer Decision",
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewer Notes",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
    "Decision Reason",
]


REQUIRED_CONFIRMATION_FIELDS = [
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
    "Decision Reason",
]


ALLOWED_DECISIONS = {
    "PENDING",
    "CONFIRMED",
    "REJECTED",
    "NEEDS_MORE_EVIDENCE",
    "NO_MATCH",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """Convert a value into trimmed text."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def identity_key(row):
    """Create a consistent identity key for a college record."""
    return tuple(
        clean_text(row.get(column, "")).casefold()
        for column in IDENTITY_FIELDS
    )


def is_valid_evidence_url(value):
    """Accept only HTTP or HTTPS URLs with a hostname."""
    value = clean_text(value)

    if not value:
        return False

    try:
        parsed = urlparse(value)

        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
            and "." in parsed.hostname
            and not any(char.isspace() for char in value)
        )

    except (ValueError, AttributeError):
        return False


def is_valid_review_date(value):
    """Require a real date formatted as YYYY-MM-DD."""
    value = clean_text(value)

    if not value:
        return False

    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")

        # Reject non-zero-padded dates such as 2026-1-4.
        return parsed.strftime("%Y-%m-%d") == value

    except ValueError:
        return False


def add_issue(issues, row, issue):
    """Record a validation issue without changing the decision."""
    issues.append({
        "Current College Name": clean_text(
            row.get("Current College Name", "")
        ),
        "Current Location": clean_text(
            row.get("Current Location", "")
        ),
        "Reviewer Decision": clean_text(
            row.get("Reviewer Decision", "")
        ),
        "Review Priority": clean_text(
            row.get("Review Priority", "")
        ),
        "Issue": issue,
    })


# ============================================================
# PRESERVE EXISTING REVIEW WORK
# ============================================================

def preserve_existing_reviews(df):
    """
    Start with V5.14 data and overlay non-empty review fields
    from an existing V5.15 file.

    Existing non-PENDING decisions are retained.
    Empty V5.15 fields do not erase V5.14 review information.
    """
    if not REVIEW_FILE.exists():
        print("No previous V5.15 file found.")
        print("Starting with the V5.14 review queue.")
        return df

    previous = pd.read_csv(
        REVIEW_FILE,
        dtype=str,
        keep_default_na=False
    )

    if not all(
        column in previous.columns
        for column in IDENTITY_FIELDS
    ):
        print(
            "WARNING: Existing V5.15 file lacks identity fields. "
            "Its review edits cannot be safely matched."
        )
        return df

    previous_by_key = {}

    for _, old_row in previous.iterrows():
        key = identity_key(old_row)

        if key in previous_by_key:
            print(
                "WARNING: Duplicate identity in the previous "
                "V5.15 file. The last occurrence will be used."
            )

        previous_by_key[key] = old_row

    matched = 0
    preserved_fields = 0

    for index, row in df.iterrows():
        old_row = previous_by_key.get(identity_key(row))

        if old_row is None:
            continue

        matched += 1

        for field in REVIEW_FIELDS:
            if field not in previous.columns:
                continue

            old_value = clean_text(old_row.get(field, ""))
            current_value = clean_text(row.get(field, ""))

            if field == "Reviewer Decision":
                old_decision = old_value.upper()
                current_decision = current_value.upper()

                # A previous non-PENDING decision is a review edit.
                # A previous PENDING value must not erase a V5.14
                # decision that has already been entered.
                if (
                    old_decision
                    and old_decision != "PENDING"
                ):
                    df.at[index, field] = old_decision
                    preserved_fields += 1

                elif not current_decision and old_decision:
                    df.at[index, field] = old_decision
                    preserved_fields += 1

            else:
                # Preserve a non-empty previous V5.15 field.
                # Otherwise keep the existing V5.14 value.
                if old_value:
                    df.at[index, field] = old_value
                    preserved_fields += 1

    print("Previous V5.15 records matched:", matched)
    print("Review field values preserved:", preserved_fields)

    return df


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Read and validate the input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"V5.14 input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if df.empty:
        raise ValueError("The V5.14 review queue is empty.")

    if df.columns.duplicated().any():
        raise ValueError(
            "The V5.14 input contains duplicate column names."
        )

    required_source_columns = IDENTITY_FIELDS + [
        "Reviewer Decision",
    ]

    missing = [
        column
        for column in required_source_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required source columns: "
            + ", ".join(missing)
        )

    print("V5.14 input records:", len(df))

    # --------------------------------------------------------
    # 2. Preserve existing V5.15 edits
    # --------------------------------------------------------

    df = preserve_existing_reviews(df)

    # Ensure all review fields are present.
    for field in REVIEW_FIELDS:
        if field not in df.columns:
            df[field] = ""

    # --------------------------------------------------------
    # 3. Normalize reviewer decisions
    # --------------------------------------------------------

    df["Reviewer Decision"] = (
        df["Reviewer Decision"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df.loc[
        df["Reviewer Decision"].eq(""),
        "Reviewer Decision"
    ] = "PENDING"

    # --------------------------------------------------------
    # 4. Validate records and evidence
    # --------------------------------------------------------

    issues = []
    valid_confirmed_indices = []

    for index, row in df.iterrows():

        decision = clean_text(
            row.get("Reviewer Decision", "")
        ).upper()

        # Invalid decisions are never exported.
        if decision not in ALLOWED_DECISIONS:
            add_issue(
                issues,
                row,
                f"Invalid reviewer decision: {decision}"
            )
            continue

        # Non-confirmed records remain in the review queue.
        if decision != "CONFIRMED":
            continue

        row_issues = []

        # All required confirmation fields must be populated.
        for field in REQUIRED_CONFIRMATION_FIELDS:
            value = clean_text(row.get(field, ""))

            if not value:
                row_issues.append(
                    f"Missing required field: {field}"
                )

        # Evidence URL must have HTTP/HTTPS scheme and hostname.
        evidence_url = clean_text(
            row.get("Evidence URL", "")
        )

        if evidence_url and not is_valid_evidence_url(
            evidence_url
        ):
            row_issues.append(
                "Evidence URL must be a valid HTTP/HTTPS URL "
                "with a hostname."
            )

        # Reviewer date must be a real YYYY-MM-DD date.
        review_date = clean_text(
            row.get("Review Date (YYYY-MM-DD)", "")
        )

        if review_date and not is_valid_review_date(review_date):
            row_issues.append(
                "Review Date must be a valid date in YYYY-MM-DD format."
            )

        if row_issues:
            for issue in row_issues:
                add_issue(issues, row, issue)

        else:
            # Only complete, explicitly confirmed rows are eligible.
            valid_confirmed_indices.append(index)

    # --------------------------------------------------------
    # 5. Build the confirmed export
    # --------------------------------------------------------

    confirmed = df.loc[valid_confirmed_indices].copy()

    # Recheck confirmation eligibility before export.
    if not confirmed.empty:

        if not confirmed["Reviewer Decision"].eq("CONFIRMED").all():
            raise RuntimeError(
                "Safety check failed: export contains a "
                "non-CONFIRMED decision."
            )

        for field in REQUIRED_CONFIRMATION_FIELDS:
            complete = confirmed[field].map(
                lambda value: bool(clean_text(value))
            )

            if not complete.all():
                raise RuntimeError(
                    f"Safety check failed: incomplete field {field}."
                )

        if not confirmed["Evidence URL"].map(
            is_valid_evidence_url
        ).all():
            raise RuntimeError(
                "Safety check failed: invalid evidence URL in export."
            )

        if not confirmed["Review Date (YYYY-MM-DD)"].map(
            is_valid_review_date
        ).all():
            raise RuntimeError(
                "Safety check failed: invalid review date in export."
            )

    # --------------------------------------------------------
    # 6. Save the review queue, confirmed export and issue report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove temporary columns if any were added in future versions.
    df.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    confirmed.to_csv(
        CONFIRMED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    issue_columns = [
        "Current College Name",
        "Current Location",
        "Reviewer Decision",
        "Review Priority",
        "Issue",
    ]

    issues_df = pd.DataFrame(
        issues,
        columns=issue_columns
    )

    issues_df.to_csv(
        ISSUES_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 7. Create the audit summary
    # --------------------------------------------------------

    decision_counts = (
        df["Reviewer Decision"].value_counts(dropna=False)
    )

    pending_count = int(
        df["Reviewer Decision"].eq("PENDING").sum()
    )

    confirmed_decision_count = int(
        df["Reviewer Decision"].eq("CONFIRMED").sum()
    )

    rejected_count = int(
        df["Reviewer Decision"].eq("REJECTED").sum()
    )

    invalid_confirmed_count = (
        confirmed_decision_count - len(confirmed)
    )

    summary_lines = [
        "AP EAPCET V5.15 Review Validation Summary",
        "=" * 48,
        f"V5.14 input records: {len(df)}",
        f"V5.15 review records: {len(df)}",
        f"Pending decisions: {pending_count}",
        f"Reviewer-marked CONFIRMED: {confirmed_decision_count}",
        f"Valid confirmed mappings exported: {len(confirmed)}",
        f"Confirmed records excluded for validation errors: "
        f"{invalid_confirmed_count}",
        f"Rejected decisions: {rejected_count}",
        f"Validation issues: {len(issues_df)}",
        "",
        "Decision counts:",
    ]

    for decision, count in decision_counts.items():
        summary_lines.append(
            f"  {decision}: {int(count)}"
        )

    summary_lines.extend([
        "",
        "Export rules:",
        "- Only explicitly CONFIRMED records can be exported.",
        "- Required official identity and evidence fields must be complete.",
        "- Evidence URL must be a valid HTTP/HTTPS URL.",
        "- Reviewer and valid review date are required.",
        "- Decision Reason is required.",
        "- Invalid or incomplete confirmations are excluded.",
        "- No college was automatically confirmed.",
        "",
        f"Review file: {REVIEW_FILE}",
        f"Confirmed export: {CONFIRMED_FILE}",
        f"Validation issues: {ISSUES_FILE}",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 8. Display final status
    # --------------------------------------------------------

    print("\nV5.15 workflow completed.")
    print("Review records:", len(df))
    print("Pending decisions:", pending_count)
    print("Reviewer-marked CONFIRMED:", confirmed_decision_count)
    print("Valid confirmed mappings exported:", len(confirmed))
    print(
        "Confirmed records excluded for validation errors:",
        invalid_confirmed_count
    )
    print("Validation issues:", len(issues_df))

    print("\nDecision counts:")
    print(decision_counts)

    print("\nCreated files:")
    print("Review queue:", REVIEW_FILE)
    print("Confirmed mappings:", CONFIRMED_FILE)
    print("Validation issues:", ISSUES_FILE)
    print("Summary:", SUMMARY_FILE)

    print("\nNo mappings were automatically confirmed.")


if __name__ == "__main__":
    main()
