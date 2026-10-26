
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# V5.12: Manual Review Prioritization
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

INPUT_FILE = (
    OUTPUT_DIR / "ap_eapcet_human_review_v5_10.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "ap_eapcet_manual_review_priority_v5_12.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "ap_eapcet_manual_review_priority_summary_v5_12.txt"
)


def get_value(row, column, default=""):
    """Return a safe value from a DataFrame row."""
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def as_number(value):
    """Convert a value to a number when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value):
    """Interpret common CSV boolean representations."""
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def normalize_similarity(value):
    """
    Support similarity values represented as either
    fractions (0 to 1) or percentages (0 to 100).
    """
    number = as_number(value)

    if number is None:
        return 0.0

    if 0 <= number <= 1:
        return number * 100

    return max(0.0, min(number, 100.0))


def prioritize_row(row):
    """
    Create a review priority score.

    This score only helps organize human review.
    It does not verify or confirm a college mapping.
    """
    score = 0
    reasons = []

    status = str(get_value(row, "Status", "")).strip()
    decision = str(
        get_value(row, "Reviewer Decision", "PENDING")
    ).strip().upper()

    if status == "UNMATCHED":
        reasons.append("No reliable candidate identified")

    same_location = as_bool(
        get_value(row, "Candidate 1 Same Location", False)
    )

    exact_name = as_bool(
        get_value(row, "Candidate 1 Exact Canonical Name", False)
    )

    acronym_match = as_bool(
        get_value(row, "Candidate 1 Acronym Match", False)
    )

    token_containment = as_bool(
        get_value(row, "Candidate 1 Token Containment", False)
    )

    similarity = normalize_similarity(
        get_value(row, "Candidate 1 Name Similarity", 0)
    )

    total_score = as_number(
        get_value(row, "Candidate 1 Total Score", None)
    )

    conflicts = as_number(
        get_value(row, "Relevant Type Conflict Count", 0)
    ) or 0

    candidate_count = as_number(
        get_value(row, "Valid Candidate Count", 0)
    ) or 0

    if exact_name:
        score += 40
        reasons.append("Exact canonical name signal")

    if same_location:
        score += 25
        reasons.append("Same-location signal")

    if token_containment:
        score += 10
        reasons.append("Name-token containment signal")

    if acronym_match:
        score += 10
        reasons.append("Acronym match signal")

    if similarity >= 90:
        score += 20
        reasons.append("Very high name similarity")
    elif similarity >= 75:
        score += 12
        reasons.append("High name similarity")
    elif similarity >= 60:
        score += 5
        reasons.append("Moderate name similarity")

    if conflicts > 0:
        score -= 35
        reasons.append("Relevant college-type conflicts exist")

    if candidate_count > 1:
        reasons.append("Multiple candidates require comparison")

    if not same_location:
        reasons.append("Location needs verification")

    if not reasons:
        reasons.append("Insufficient matching signals")

    # A recorded total score is retained for reviewer reference.
    # It is not assumed to use the same scale as similarity.
    return {
        "Review Priority Score": score,
        "Review Priority": (
            "HIGH"
            if score >= 55
            else "MEDIUM"
            if score >= 25
            else "LOW"
        ),
        "Priority Reason": "; ".join(reasons),
        "Candidate 1 Similarity Normalized (%)": round(
            similarity, 2
        ),
        "Candidate 1 Total Score Numeric": total_score,
        "Review Queue": (
            "ALREADY_REVIEWED"
            if decision not in {"", "PENDING", "NOT_REVIEWED"}
            else "PENDING_HUMAN_REVIEW"
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    if df.empty:
        raise ValueError("The human-review CSV has no rows.")

    required_columns = [
        "Current College Name",
        "Candidate 1 Name",
        "Reviewer Decision",
        "Status",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(missing)
        )

    # Preserve every original column and add priority fields.
    priority_data = df.apply(
        lambda row: prioritize_row(row),
        axis=1,
        result_type="expand",
    )

    result = pd.concat(
        [df.reset_index(drop=True), priority_data],
        axis=1,
    )

    # Pending items appear first, followed by priority score.
    queue_order = {
        "PENDING_HUMAN_REVIEW": 0,
        "ALREADY_REVIEWED": 1,
    }

    result["_queue_order"] = (
        result["Review Queue"].map(queue_order).fillna(2)
    )

    result = result.sort_values(
        by=[
            "_queue_order",
            "Review Priority Score",
            "Candidate 1 Similarity Normalized (%)",
        ],
        ascending=[True, False, False],
        kind="stable",
    ).drop(columns=["_queue_order"])

    # Only create new files; do not modify the original inputs.
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    pending = result[
        result["Review Queue"] == "PENDING_HUMAN_REVIEW"
    ]

    summary = [
        "AP EAPCET COLLEGE MAPPING",
        "V5.12 MANUAL REVIEW PRIORITIZATION SUMMARY",
        "=" * 55,
        f"Input rows: {len(df)}",
        f"Output rows: {len(result)}",
        f"Pending human review: {len(pending)}",
        "",
        "Priority counts among pending records:",
    ]

    for priority in ["HIGH", "MEDIUM", "LOW"]:
        count = int(
            (pending["Review Priority"] == priority).sum()
        )
        summary.append(f"{priority}: {count}")

    summary.extend([
        "",
        "Important safeguards:",
        "- No college mapping was automatically confirmed.",
        "- Original input files were not modified.",
        "- Priority scores organize human review only.",
        "- Verify official name, location, code, and evidence",
        "  before changing any reviewer decision.",
        "",
        f"Output CSV: {OUTPUT_FILE}",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("\nV5.12 prioritization completed.")
    print(f"Input records: {len(df)}")
    print(f"Pending review: {len(pending)}")
    print(
        "HIGH priority:",
        int((pending["Review Priority"] == "HIGH").sum()),
    )
    print(
        "MEDIUM priority:",
        int((pending["Review Priority"] == "MEDIUM").sum()),
    )
    print(
        "LOW priority:",
        int((pending["Review Priority"] == "LOW").sum()),
    )
    print(f"\nCreated: {OUTPUT_FILE}")
    print(f"Created: {SUMMARY_FILE}")
    print("\nNo mappings were automatically confirmed.")


if __name__ == "__main__":
    main()
