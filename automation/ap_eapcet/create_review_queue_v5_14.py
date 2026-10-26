
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("automation/output")

INPUT_FILE = OUTPUT_DIR / "ap_eapcet_human_review_v5_13.csv"
OUTPUT_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_14.csv"
SUMMARY_FILE = OUTPUT_DIR / "ap_eapcet_review_queue_summary_v5_14.txt"

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

IDENTITY_FIELDS = [
    "Current Canonical Name",
    "Current Canonical Location",
    "Current College Name",
    "Current Location",
]

FRONT_COLUMNS = [
    "Reviewer Decision",
    "Review Priority",
    "Review Priority Score",
    "Priority Reason",
    "Status",
    "Current College Name",
    "Current Location",
    "Current Canonical Name",
    "Current Canonical Location",
    "Current Acronym",
    "Review Flags",
    "Candidate Ranking Note",
    "Valid Candidate Count",
    "Relevant Type Conflict Count",
]

# Add details for the five suggested candidates.
for number in range(1, 6):
    FRONT_COLUMNS.extend([
        f"Candidate {number} Name",
        f"Candidate {number} Location",
        f"Candidate {number} Code",
        f"Candidate {number} Name Similarity",
        f"Candidate {number} Total Score",
        f"Candidate {number} Same Location",
        f"Candidate {number} Branches",
        f"Candidate {number} Year",
        f"Candidate {number} Exact Canonical Name",
    ])

# Add competing institution-type candidates.
for number in range(1, 4):
    FRONT_COLUMNS.extend([
        f"Type Conflict {number} Name",
        f"Type Conflict {number} Location",
        f"Type Conflict {number} Code",
        f"Type Conflict {number} Name Score",
    ])

FRONT_COLUMNS.extend(REVIEW_FIELDS)


def clean_text(value):
    """Convert a value to trimmed text safely."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def identity_key(row):
    """Build a consistent key for matching college review records."""
    return tuple(
        clean_text(row.get(column, "")).casefold()
        for column in IDENTITY_FIELDS
    )


def preserve_existing_reviews(df):
    """
    Preserve reviewer edits from an existing V5.14 queue.

    The V5.13 input is never modified.
    """
    if not OUTPUT_FILE.exists():
        print("No existing V5.14 queue found. Creating a new queue.")
        return df

    previous = pd.read_csv(
        OUTPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if not all(column in previous.columns for column in IDENTITY_FIELDS):
        print(
            "WARNING: Existing V5.14 queue is missing identity columns. "
            "Existing reviewer edits were not imported."
        )
        return df

    # Map previous rows by identity.
    previous_by_key = {}

    for _, old_row in previous.iterrows():
        key = identity_key(old_row)

        if key in previous_by_key:
            print(
                "WARNING: Duplicate identity found in the previous "
                "V5.14 queue. The last occurrence will be used."
            )

        previous_by_key[key] = old_row

    preserved = 0

    for index, row in df.iterrows():
        old_row = previous_by_key.get(identity_key(row))

        if old_row is None:
            continue

        for field in REVIEW_FIELDS:
            if field in previous.columns:
                df.at[index, field] = clean_text(
                    old_row.get(field, "")
                )

        preserved += 1

    print("Existing V5.14 records matched:", preserved)

    return df


def main():
    # 1. Validate input.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    if df.empty:
        raise ValueError(
            "The V5.13 input review file contains no records."
        )

    print("V5.13 input records:", len(df))

    # 2. Check required source columns.
    required_columns = IDENTITY_FIELDS + [
        "Reviewer Decision",
        "Review Priority",
        "Review Priority Score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing)
        )

    # 3. Import existing V5.14 reviewer edits, if available.
    df = preserve_existing_reviews(df)

    # 4. Ensure all review fields exist.
    for field in REVIEW_FIELDS:
        if field not in df.columns:
            df[field] = ""

    # 5. Normalize and validate reviewer decisions.
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

    allowed_decisions = {
        "PENDING",
        "CONFIRMED",
        "REJECTED",
        "NEEDS_MORE_EVIDENCE",
        "NO_MATCH",
    }

    invalid = sorted(
        set(df["Reviewer Decision"]) - allowed_decisions
    )

    if invalid:
        raise ValueError(
            "Invalid reviewer decisions found: "
            + ", ".join(invalid)
        )

    # 6. Sort HIGH, MEDIUM, LOW priorities.
    priority_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    df["_priority_order"] = (
        df["Review Priority"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(priority_order)
        .fillna(3)
    )

    df["_score_numeric"] = pd.to_numeric(
        df["Review Priority Score"],
        errors="coerce"
    ).fillna(-1)

    df = df.sort_values(
        by=[
            "_priority_order",
            "_score_numeric",
            "Current College Name",
        ],
        ascending=[True, False, True],
        kind="stable"
    )

    # 7. Arrange columns.
    # IMPORTANT: dict.fromkeys removes duplicate column names
    # while preserving the original column order.
    ordered = list(dict.fromkeys(
        column
        for column in FRONT_COLUMNS
        if column in df.columns
    ))

    remaining = [
        column
        for column in df.columns
        if column not in ordered
        and column not in {
            "_priority_order",
            "_score_numeric",
        }
    ]

    result = df[ordered + remaining].copy()

    # Final safeguard against duplicate output column names.
    if result.columns.duplicated().any():
        duplicates = result.columns[
            result.columns.duplicated()
        ].tolist()

        raise ValueError(
            "Duplicate output columns detected: "
            + ", ".join(duplicates)
        )

    # 8. Calculate summary statistics.
    priority_series = (
        result["Review Priority"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    decision_series = (
        result["Reviewer Decision"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    priority_counts = priority_series.value_counts()
    decision_counts = decision_series.value_counts()

    pending_high = int((
        priority_series.eq("HIGH")
        & decision_series.eq("PENDING")
    ).sum())

    # 9. Write the queue and summary.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary_lines = [
        "AP EAPCET V5.14 Manual Review Queue",
        "=" * 40,
        f"Input records: {len(df)}",
        f"Output records: {len(result)}",
        f"Pending HIGH-priority records: {pending_high}",
        "",
        "Priority counts:",
    ]

    for priority in ["HIGH", "MEDIUM", "LOW"]:
        summary_lines.append(
            f"  {priority}: "
            f"{int(priority_counts.get(priority, 0))}"
        )

    summary_lines.extend([
        "",
        "Reviewer decision counts:",
    ])

    for decision, count in decision_counts.items():
        summary_lines.append(
            f"  {decision}: {int(count)}"
        )

    summary_lines.extend([
        "",
        "No college mapping was automatically confirmed.",
        "Review priorities are not match-accuracy probabilities.",
        f"Review queue: {OUTPUT_FILE}",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8"
    )

    # 10. Display final status.
    print("\nV5.14 review queue created successfully.")
    print("Input records:", len(df))
    print("Output records:", len(result))
    print("Duplicate output columns:", int(
        result.columns.duplicated().sum()
    ))
    print("Pending HIGH-priority records:", pending_high)

    print("\nPriority counts:")
    print(priority_series.value_counts(dropna=False))

    print("\nDecision counts:")
    print(decision_series.value_counts(dropna=False))

    print("\nReview queue:", OUTPUT_FILE)
    print("Summary:", SUMMARY_FILE)
    print("\nNo mappings were automatically confirmed.")


if __name__ == "__main__":
    main()
