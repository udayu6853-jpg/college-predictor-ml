
from pathlib import Path
import re
import pandas as pd

from create_review_report_v5_2 import (
    load_data,
    find_candidates,
    normalize_name,
    compact_name,
    meaningful_tokens,
    similarity,
    token_similarity,
    location_similarity,
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

OUTPUT_FILE = (
    OUTPUT_DIR /
    "ap_eapcet_unique_college_mapping_review_v5_4.csv"
)


# =========================================================
# SETTINGS
# =========================================================

TOP_CANDIDATES = 3
MIN_CANDIDATE_SCORE = 0.48
CLOSE_SCORE_MARGIN = 0.08

# These terms distinguish institution types.
TYPE_GROUPS = {
    "UNIVERSITY": {"UNIVERSITY"},
    "WOMENS": {"WOMEN", "WOMENS"},
    "PHARMACY": {"PHARMACY"},
    "POLYTECHNIC": {"POLYTECHNIC"},
    "MEDICAL": {"MEDICAL"},
    "DENTAL": {"DENTAL"},
    "LAW": {"LAW"},
    "ARTS_SCIENCE": {"ARTS", "SCIENCE"},
    "AGRICULTURE": {"AGRICULTURAL", "AGRICULTURE"},
    "ARCHITECTURE": {"ARCHITECTURE"},
}

GENERIC_TOKENS = {
    "AND", "OF", "THE", "FOR", "IN", "AT", "TO",
    "A", "AN", "COLLEGE", "INSTITUTE", "ENGINEERING",
    "TECHNOLOGY", "SCIENCE", "AUTONOMOUS",
}


# =========================================================
# NAME AND TYPE HELPERS
# =========================================================

def words(value):
    return set(normalize_name(value).split())


def type_labels(value):
    tokens = words(value)
    labels = set()

    for label, alternatives in TYPE_GROUPS.items():
        if tokens & alternatives:
            labels.add(label)

    # Distinguish engineering colleges from unrelated institutions.
    if "ENGINEERING" in tokens or "ENGG" in tokens:
        labels.add("ENGINEERING")

    return labels


def has_type_conflict(current_name, official_name):
    current_types = type_labels(current_name)
    official_types = type_labels(official_name)

    # If only one name explicitly identifies itself as a university,
    # pharmacy college, women's college, etc., do not treat it as a
    # confirmed match.
    important = {
        "UNIVERSITY", "WOMENS", "PHARMACY", "POLYTECHNIC",
        "MEDICAL", "DENTAL", "LAW", "AGRICULTURE", "ARCHITECTURE",
    }

    for label in important:
        if (label in current_types) != (label in official_types):
            return True

    # Arts-and-science colleges should not be matched to engineering
    # colleges merely because they share words such as "college".
    current_arts = "ARTS" in words(current_name)
    official_arts = "ARTS" in words(official_name)

    current_engineering = "ENGINEERING" in words(current_name)
    official_engineering = "ENGINEERING" in words(official_name)

    if current_arts != official_arts and (
        current_engineering or official_engineering
    ):
        return True

    return False


def distinctive_tokens(value):
    tokens = meaningful_tokens(value)
    return {
        token for token in tokens
        if token not in GENERIC_TOKENS
    }


def distinctive_similarity(a, b):
    ta = distinctive_tokens(a)
    tb = distinctive_tokens(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def score_candidate(current_row, official_row):
    current_name = current_row["College Name"]
    official_name = official_row["College Name"]

    name_score = similarity(
        normalize_name(current_name),
        normalize_name(official_name),
    )

    compact_score = similarity(
        compact_name(current_name),
        compact_name(official_name),
    )

    token_score = token_similarity(current_name, official_name)
    distinctive_score = distinctive_similarity(
        current_name,
        official_name,
    )

    loc_score = location_similarity(
        current_row["Location"],
        official_row["Location"],
    )

    type_conflict = has_type_conflict(
        current_name,
        official_name,
    )

    # Name identity has more weight than location.
    # Similar location alone must not make unrelated names a match.
    final_score = (
        0.25 * name_score
        + 0.15 * compact_score
        + 0.20 * token_score
        + 0.25 * distinctive_score
        + 0.15 * loc_score
    )

    if type_conflict:
        final_score = min(final_score, 0.30)

    return {
        "Name Score": round(name_score, 4),
        "Compact Score": round(compact_score, 4),
        "Token Score": round(token_score, 4),
        "Distinctive Token Score": round(distinctive_score, 4),
        "Location Score": round(loc_score, 4),
        "Identity Conflict": type_conflict,
        "Final Score": round(final_score, 4),
    }


# =========================================================
# CANDIDATE SEARCH
# =========================================================

def find_v54_candidates(current_row, official_colleges):
    candidates = []

    for _, official_row in official_colleges.iterrows():
        scores = score_candidate(current_row, official_row)

        candidates.append({
            "Official College Name": official_row["College Name"],
            "Official College Code": str(
                official_row["College Code"]
            ).strip(),
            "Official Location": official_row["Location"],
            **scores,
        })

    # Exclude institution-type conflicts from the valid candidate
    # shortlist. Keep their existence visible through a separate flag.
    valid_candidates = [
        candidate for candidate in candidates
        if not candidate["Identity Conflict"]
        and candidate["Final Score"] >= MIN_CANDIDATE_SCORE
    ]

    valid_candidates.sort(
        key=lambda candidate: (
            candidate["Final Score"],
            candidate["Location Score"],
            candidate["Distinctive Token Score"],
        ),
        reverse=True,
    )

    return valid_candidates, candidates


# =========================================================
# REVIEW CLASSIFICATION
# =========================================================

def classify(current_row, candidates, all_candidates):
    if not candidates:
        return "UNMATCHED", "No sufficiently similar candidate without type conflict"

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    current_name = current_row["College Name"]
    current_location = current_row["Location"]

    exact_name = (
        normalize_name(current_name)
        == normalize_name(best["Official College Name"])
    )

    exact_location = (
        bool(normalize_name(current_location))
        and normalize_name(current_location)
        == normalize_name(best["Official Location"])
    )

    margin = (
        best["Final Score"] - second["Final Score"]
        if second else best["Final Score"]
    )

    # Exact name/location is a candidate for SAFE, but also check whether
    # multiple official records share that name and location.
    competing_exact = [
        candidate for candidate in all_candidates
        if normalize_name(candidate["Official College Name"])
        == normalize_name(current_name)
        and normalize_name(candidate["Official Location"])
        == normalize_name(current_location)
        and not candidate["Identity Conflict"]
    ]

    exact_codes = {
        candidate["Official College Code"]
        for candidate in competing_exact
        if candidate["Official College Code"]
    }

    if exact_name and exact_location and len(exact_codes) == 1:
        return "SAFE", "Exact normalized name and location; one official code"

    if exact_name and exact_location and len(exact_codes) > 1:
        return "MANUAL_REVIEW", "Duplicate official codes for exact name/location"

    if margin < CLOSE_SCORE_MARGIN:
        return "MANUAL_REVIEW", "Top candidates have similar scores"

    if best["Final Score"] >= 0.82 and best["Location Score"] >= 0.75:
        return "MANUAL_REVIEW", "Strong similarity; verify campus and official code"

    if best["Final Score"] >= 0.60:
        return "MANUAL_REVIEW", "Possible match; verify manually"

    return "UNMATCHED", "Candidate score is too low for a reliable match"


# =========================================================
# REPORT GENERATION
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current, official, official_colleges = load_data()

    # One record per college name/location combination.
    unique_current = (
        current
        .drop_duplicates(
            subset=["College Name", "Location"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print("\nAP EAPCET COLLEGE REVIEW - VERSION 5.4")
    print("=" * 65)
    print(f"Current dataset rows: {len(current)}")
    print(f"Unique current college names: {current['College Name'].nunique()}")
    print(f"Unique college/location combinations: {len(unique_current)}")
    print(f"Unique official colleges: {len(official_colleges)}")
    print("\nMatching with type-conflict safeguards...\n")

    report_rows = []

    for index, row in unique_current.iterrows():
        candidates, all_candidates = find_v54_candidates(
            row,
            official_colleges,
        )

        status, reason = classify(
            row,
            candidates,
            all_candidates,
        )

        best = candidates[0] if candidates else {}
        second = candidates[1] if len(candidates) > 1 else {}
        third = candidates[2] if len(candidates) > 2 else {}

        # Report if the name has any conflicting candidates that were
        # excluded from the valid shortlist.
        type_conflicts = [
            candidate for candidate in all_candidates
            if candidate["Identity Conflict"]
        ]

        flags = []

        if type_conflicts:
            flags.append("TYPE_CONFLICT_CANDIDATES_EXCLUDED")

        if len({
            candidate["Official College Code"]
            for candidate in candidates
            if candidate["Official College Code"]
        }) < len([
            candidate for candidate in candidates
            if candidate["Official College Code"]
        ]):
            flags.append("REPEATED_CODES_IN_CANDIDATES")

        if len(candidates) >= 2:
            if (
                candidates[0]["Final Score"]
                - candidates[1]["Final Score"]
                < CLOSE_SCORE_MARGIN
            ):
                flags.append("CLOSE_CANDIDATES")

        if best:
            if best["Location Score"] < 0.75:
                flags.append("LOCATION_DIFFERENCE")
        else:
            flags.append("NO_VALID_CANDIDATE")

        if not flags:
            flags.append("CHECK_BEFORE_FINAL_MAPPING")

        report_rows.append({
            "Current College Name": row["College Name"],
            "Current Location": row["Location"],
            "Review Status": status,
            "Review Reason": reason,
            "Review Flags": "; ".join(flags),

            "Candidate 1 Name": best.get("Official College Name", ""),
            "Candidate 1 Code": best.get("Official College Code", ""),
            "Candidate 1 Location": best.get("Official Location", ""),
            "Candidate 1 Score": best.get("Final Score", 0.0),
            "Candidate 1 Location Score": best.get("Location Score", 0.0),

            "Candidate 2 Name": second.get("Official College Name", ""),
            "Candidate 2 Code": second.get("Official College Code", ""),
            "Candidate 2 Location": second.get("Official Location", ""),
            "Candidate 2 Score": second.get("Final Score", 0.0),

            "Candidate 3 Name": third.get("Official College Name", ""),
            "Candidate 3 Code": third.get("Official College Code", ""),
            "Candidate 3 Location": third.get("Official Location", ""),
            "Candidate 3 Score": third.get("Final Score", 0.0),

            "Excluded Type-Conflict Candidate Count": len(type_conflicts),
        })

        if (index + 1) % 25 == 0 or index + 1 == len(unique_current):
            print(f"Processed {index + 1}/{len(unique_current)}")

    report = pd.DataFrame(report_rows)

    report.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("V5.4 SUMMARY")
    print("=" * 65)
    print(f"Unique college/location combinations: {len(report)}")

    for status in ["SAFE", "MANUAL_REVIEW", "UNMATCHED"]:
        count = int((report["Review Status"] == status).sum())
        print(f"{status:16} {count}")

    print("\nReview flags:")
    for flag in [
        "TYPE_CONFLICT_CANDIDATES_EXCLUDED",
        "CLOSE_CANDIDATES",
        "LOCATION_DIFFERENCE",
        "NO_VALID_CANDIDATE",
    ]:
        count = report["Review Flags"].str.contains(
            flag,
            regex=False,
        ).sum()
        print(f"{flag:38} {count}")

    print(f"\nReport saved to:\n{OUTPUT_FILE}")

    print("\nFIRST 12 COLLEGES")
    print("-" * 65)

    for _, row in report.head(12).iterrows():
        print(f"\nCurrent: {row['Current College Name']}")
        print(f"Location: {row['Current Location']}")
        print(f"Status: {row['Review Status']}")
        print(f"Flags: {row['Review Flags']}")

        for number in [1, 2, 3]:
            print(
                f"{number}. {row[f'Candidate {number} Name']} "
                f"| {row[f'Candidate {number} Code']} "
                f"| {row[f'Candidate {number} Location']} "
                f"| Score: {row[f'Candidate {number} Score']}"
            )

    print("\nIMPORTANT")
    print("- Similarity scores are not probabilities.")
    print("- Type-conflicting candidates are excluded from the shortlist.")
    print("- SAFE is a provisional candidate status, not official certification.")
    print("- No dataset, PostgreSQL table, or ML model was modified.")


if __name__ == "__main__":
    main()