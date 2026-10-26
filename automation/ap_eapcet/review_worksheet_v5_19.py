
from pathlib import Path
import pandas as pd

# AP EAPCET V5.19 - Manual Review Worksheet Generator
# This script only reads the current queue and creates new reports.
# It does not edit the queue, V5.17 baseline, or V5.18 snapshot.

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

QUEUE_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_16.csv"
WORKSHEET_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_worksheet_v5_19.csv"
SUMMARY_FILE = OUTPUT_DIR / "ap_eapcet_review_worksheet_summary_v5_19.txt"

REQUIRED_COLUMNS = [
    "Stable Review ID",
    "Current College Name",
    "Current Location",
    "Current Canonical Name",
    "Current Canonical Location",
    "Reviewer Decision",
    "Review Priority",
    "Review Priority Score",
    "Priority Reason",
    "Review Flags",
    "Valid Candidate Count",
    "Candidate Ranking Note",
    "Candidate 1 Name",
    "Candidate 1 Location",
    "Candidate 1 Code",
    "Candidate 1 Name Similarity",
    "Candidate 1 Total Score",
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
    "Decision Reason",
]

PRIORITY_ORDER = {
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


def main():
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(
            f"Review queue not found: {QUEUE_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        QUEUE_FILE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n- "
            + "\n- ".join(missing)
        )

    # Select only the fields needed for manual review.
    worksheet = df[REQUIRED_COLUMNS].copy()

    # Numeric helper columns are used only for sorting.
    priority_scores = pd.to_numeric(
        worksheet["Review Priority Score"],
        errors="coerce",
    )

    candidate_counts = pd.to_numeric(
        worksheet["Valid Candidate Count"],
        errors="coerce",
    ).fillna(0)

    flags = worksheet["Review Flags"].str.upper()
    decisions = worksheet["Reviewer Decision"].str.upper()

    no_candidate = (
        (candidate_counts == 0)
        | flags.str.contains("NO_VALID_CANDIDATE", regex=False)
    )

    ambiguity = (
        flags.str.contains("CLOSE_CANDIDATES", regex=False)
        | flags.str.contains("LOCATION_DIFFERENCE", regex=False)
        | flags.str.contains("NO_STRONG_NAME_MATCH", regex=False)
    )

    # A sorting aid, not a prediction of official correctness.
    worksheet["Manual Review Category"] = "STANDARD_REVIEW"
    worksheet.loc[ambiguity, "Manual Review Category"] = (
        "CHECK_AMBIGUITY"
    )
    worksheet.loc[no_candidate, "Manual Review Category"] = (
        "INVESTIGATE_NO_CANDIDATE"
    )

    worksheet["Manual Review Order"] = 3
    worksheet.loc[ambiguity, "Manual Review Order"] = 2
    worksheet.loc[no_candidate, "Manual Review Order"] = 1

    worksheet["_priority_order"] = (
        worksheet["Review Priority"]
        .str.upper()
        .map(PRIORITY_ORDER)
        .fillna(4)
    )
    worksheet["_priority_score"] = priority_scores.fillna(-1)
    worksheet["_candidate_count"] = candidate_counts

    worksheet = worksheet.sort_values(
        by=[
            "Manual Review Order",
            "_priority_order",
            "_priority_score",
            "_candidate_count",
            "Stable Review ID",
        ],
        ascending=[True, True, False, True, True],
        kind="stable",
    )

    worksheet = worksheet.drop(
        columns=[
            "_priority_order",
            "_priority_score",
            "_candidate_count",
        ]
    )

    # Explicitly keep all rows and never auto-confirm.
    worksheet["Reviewer Decision"] = (
        worksheet["Reviewer Decision"].str.upper()
    )

    worksheet.to_csv(
        WORKSHEET_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    category_counts = (
        worksheet["Manual Review Category"]
        .value_counts()
        .to_dict()
    )

    decision_counts = (
        worksheet["Reviewer Decision"]
        .value_counts()
        .to_dict()
    )

    summary = [
        "AP EAPCET V5.19 Manual Review Worksheet",
        "=" * 48,
        f"Source records: {len(df)}",
        f"Worksheet records: {len(worksheet)}",
        f"Unique stable IDs: {worksheet['Stable Review ID'].nunique()}",
        "",
        "Manual review categories:",
    ]

    for category in [
        "INVESTIGATE_NO_CANDIDATE",
        "CHECK_AMBIGUITY",
        "STANDARD_REVIEW",
    ]:
        summary.append(
            f"  {category}: {category_counts.get(category, 0)}"
        )

    summary.extend(["", "Current reviewer decisions:"])

    for decision, count in sorted(decision_counts.items()):
        summary.append(f"  {decision}: {count}")

    summary.extend([
        "",
        "Safety notes:",
        "- This worksheet is a separate output file.",
        "- No college mapping is automatically confirmed.",
        "- Candidate scores are not official verification.",
        "- Official evidence must be checked by a reviewer.",
        "- The source queue is not modified.",
        "- The V5.17 baseline is not modified.",
        "- The V5.18 snapshot is not modified.",
        "",
        f"Source queue: {QUEUE_FILE}",
        f"Worksheet: {WORKSHEET_FILE}",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("V5.19 worksheet created successfully.")
    print(f"Source records: {len(df)}")
    print(f"Worksheet records: {len(worksheet)}")
    print(
        "Unique stable IDs:",
        worksheet["Stable Review ID"].nunique(),
    )
    print("No source queue or baseline was modified.")
    print(f"Worksheet: {WORKSHEET_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
