
from pathlib import Path
import pandas as pd

from create_review_report_v5_2 import (
    load_data,
    find_candidates,
    classify_match,
    normalize_name,
    compact_name,
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

OUTPUT_FILE = (
    OUTPUT_DIR /
    "ap_eapcet_unique_college_mapping_review_v5_3.csv"
)


# =========================================================
# REVIEW SETTINGS
# =========================================================

TOP_CANDIDATES = 3

LOCATION_MISMATCH_THRESHOLD = 0.80
CLOSE_CANDIDATE_MARGIN = 0.08


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_branches(group):
    """Return distinct branches for a current college."""
    if "Branch" not in group.columns:
        return ""

    branches = (
        group["Branch"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    branches = sorted({
        branch for branch in branches
        if branch and branch.lower() != "nan"
    })

    return ", ".join(branches)


def get_exam_names(group):
    """Return exam names when an Exam column exists."""
    if "Exam" not in group.columns:
        return ""

    exams = sorted({
        str(value).strip().upper()
        for value in group["Exam"].dropna()
        if str(value).strip()
    })

    return ", ".join(exams)


def get_suspicious_flags(
    current_name,
    current_location,
    candidates,
):
    """Flag cases requiring careful human verification."""

    flags = []

    if not candidates:
        return "NO_CANDIDATE"

    best = candidates[0]

    # Name/type conflicts must not be automatically accepted.
    if best.get("Identity Conflict", False):
        flags.append("INSTITUTION_TYPE_CONFLICT")

    # Location can differ because of naming conventions or
    # broad city labels, so this is a review flag, not proof.
    location_score = best.get("Location Score", 0.0)

    if location_score < LOCATION_MISMATCH_THRESHOLD:
        flags.append("LOCATION_DIFFERENCE")

    # Detect near-tied candidates.
    if len(candidates) >= 2:
        margin = (
            candidates[0]["Final Score"]
            - candidates[1]["Final Score"]
        )

        if margin < CLOSE_CANDIDATE_MARGIN:
            flags.append("CLOSE_CANDIDATES")

    # Flag repeated normalized official names with different codes.
    best_name = normalize_name(best["Official College Name"])

    matching_name_candidates = [
        candidate
        for candidate in candidates
        if normalize_name(candidate["Official College Name"]) == best_name
    ]

    matching_codes = {
        candidate["Official College Code"]
        for candidate in matching_name_candidates
        if candidate["Official College Code"]
    }

    if len(matching_codes) > 1:
        flags.append("DUPLICATE_OFFICIAL_NAME")

    # Exact spelling and compact-name differences are useful
    # for reviewing abbreviations and spelling variants.
    if (
        normalize_name(current_name)
        != normalize_name(best["Official College Name"])
        and compact_name(current_name)
        == compact_name(best["Official College Name"])
    ):
        flags.append("PUNCTUATION_OR_FORMAT_VARIANT")

    if not flags:
        flags.append("CHECK_BEFORE_FINAL_MAPPING")

    return "; ".join(flags)


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current, official, official_colleges = load_data()

    print("\nAP EAPCET UNIQUE COLLEGE REVIEW - VERSION 5.3")
    print("=" * 65)

    print(f"Current dataset rows: {len(current)}")
    print(
        "Current unique college names: "
        f"{current['College Name'].nunique()}"
    )
    print(f"Official normalized rows: {len(official)}")
    print(f"Unique official colleges: {len(official_colleges)}")

    # One record per current college name and location.
    # Keeping location avoids merging distinct institutions
    # that happen to share a college name.
    unique_current = (
        current
        .drop_duplicates(
            subset=["College Name", "Location"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "\nUnique college/location combinations to review: "
        f"{len(unique_current)}"
    )
    print("\nFinding the top three candidates for each college...\n")

    report_rows = []

    for index, row in unique_current.iterrows():
        candidates = find_candidates(row, official_colleges)

        status, reason = classify_match(row, candidates)

        best = candidates[0] if candidates else {}
        second = candidates[1] if len(candidates) > 1 else {}
        third = candidates[2] if len(candidates) > 2 else {}

        flags = get_suspicious_flags(
            row["College Name"],
            row["Location"],
            candidates,
        )

        source_group = current[
            (current["College Name"] == row["College Name"])
            & (current["Location"] == row["Location"])
        ]

        result = {
            "Current College Name": row["College Name"],
            "Current Location": row["Location"],
            "Current Branches": get_branches(source_group),
            "Current Branch Count": (
                len(get_branches(source_group).split(", "))
                if get_branches(source_group)
                else 0
            ),
            "Current Exams": get_exam_names(source_group),
            "Current Dataset Row Count": len(source_group),

            "Review Status": status,
            "Review Reason": reason,
            "Suspicious Flags": flags,

            "Best Official College Name":
                best.get("Official College Name", ""),
            "Best Official College Code":
                best.get("Official College Code", ""),
            "Best Official Location":
                best.get("Official Location", ""),
            "Best Final Score":
                best.get("Final Score", 0.0),
            "Best Name Score":
                best.get("Name Score", 0.0),
            "Best Location Score":
                best.get("Location Score", 0.0),
            "Best Identity Conflict":
                best.get("Identity Conflict", False),

            "Second Official College Name":
                second.get("Official College Name", ""),
            "Second Official College Code":
                second.get("Official College Code", ""),
            "Second Official Location":
                second.get("Official Location", ""),
            "Second Final Score":
                second.get("Final Score", 0.0),

            "Third Official College Name":
                third.get("Official College Name", ""),
            "Third Official College Code":
                third.get("Official College Code", ""),
            "Third Official Location":
                third.get("Official Location", ""),
            "Third Final Score":
                third.get("Final Score", 0.0),
        }

        report_rows.append(result)

        if (index + 1) % 25 == 0 or index + 1 == len(unique_current):
            print(f"Processed {index + 1}/{len(unique_current)}")

    report = pd.DataFrame(report_rows)

    # Sort review cases to put higher-risk matches first.
    report["_priority"] = report["Suspicious Flags"].apply(
        lambda value: (
            0 if "INSTITUTION_TYPE_CONFLICT" in value
            else 1 if "CLOSE_CANDIDATES" in value
            else 2 if "LOCATION_DIFFERENCE" in value
            else 3
        )
    )

    report = (
        report
        .sort_values(
            by=["_priority", "Review Status", "Current College Name"],
            ascending=[True, True, True],
        )
        .drop(columns=["_priority"])
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("V5.3 REVIEW SUMMARY")
    print("=" * 65)

    print(f"Unique colleges reviewed: {len(report)}")

    for status in ["SAFE", "MANUAL_REVIEW", "UNMATCHED"]:
        count = int((report["Review Status"] == status).sum())
        print(f"{status:18} {count}")

    print("\nSuspicious flag counts:")

    flag_counts = {}

    for value in report["Suspicious Flags"]:
        for flag in value.split("; "):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    for flag, count in sorted(
        flag_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{flag:35} {count}")

    print(f"\nReport saved to:\n{OUTPUT_FILE}")

    print("\nFIRST 15 COLLEGES FOR REVIEW")
    print("-" * 65)

    for _, row in report.head(15).iterrows():
        print(f"\nCurrent: {row['Current College Name']}")
        print(f"Location: {row['Current Location']}")
        print(f"Branches: {row['Current Branches']}")
        print(f"Status: {row['Review Status']}")
        print(f"Flags: {row['Suspicious Flags']}")

        print(
            f"1. {row['Best Official College Name']} "
            f"| Code: {row['Best Official College Code']} "
            f"| Score: {row['Best Final Score']}"
        )

        print(
            f"2. {row['Second Official College Name']} "
            f"| Code: {row['Second Official College Code']} "
            f"| Score: {row['Second Final Score']}"
        )

        print(
            f"3. {row['Third Official College Name']} "
            f"| Code: {row['Third Official College Code']} "
            f"| Score: {row['Third Final Score']}"
        )

    print("\nIMPORTANT")
    print("- Scores are similarity estimates, not probabilities.")
    print("- SAFE is a candidate status, not official certification.")
    print("- Check identity, campus and location before final mapping.")
    print("- No datasets, PostgreSQL tables or ML models were modified.")


if __name__ == "__main__":
    main()