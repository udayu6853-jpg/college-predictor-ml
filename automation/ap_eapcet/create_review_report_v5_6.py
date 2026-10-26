
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
import re

import pandas as pd


# ============================================================
# V5.6 COLLEGE MAPPING REVIEW
# Read-only review: does not update datasets, database or model
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CURRENT_FILE = ROOT / "dataset" / "eamcet_colleges.csv"

OFFICIAL_FILE = (
    ROOT / "automation" / "output" / "ap_eapcet_normalized.csv"
)

OUTPUT_FILE = (
    ROOT
    / "automation"
    / "output"
    / "ap_eapcet_unique_college_mapping_review_v5_6.csv"
)


# ------------------------------------------------------------
# 1. Read CSV files safely
# ------------------------------------------------------------

def read_csv_safely(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding).fillna("")
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read CSV encoding: {path}")


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def find_column(df, possible_names, required=True):
    """
    Find a column without depending on capitalization,
    spaces, underscores or punctuation.
    """
    def key(value):
        return re.sub(r"[^A-Z0-9]", "", str(value).upper())

    columns = {key(column): column for column in df.columns}

    for name in possible_names:
        found = columns.get(key(name))
        if found is not None:
            return found

    if required:
        raise ValueError(
            f"Could not find any of these columns: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# ------------------------------------------------------------
# 2. Normalize names and locations
# ------------------------------------------------------------

def normalize_name(value):
    text = clean_text(value).upper()

    replacements = {
        "&": " AND ",
        "INSTITUTION": " INSTITUTION ",
        "INSTITUTE": " INSTITUTE ",
        "INSTT": " INSTITUTE ",
        "INST": " INSTITUTE ",
        "TECHNOLOGY": " TECHNOLOGY ",
        "TECH": " TECHNOLOGY ",
        "COLL": " COLLEGE ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation but retain letters and numbers.
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize a few common spelling variations.
    text = re.sub(r"\bENGG\b", "ENGINEERING", text)
    text = re.sub(r"\bENG\b", "ENGINEERING", text)
    text = re.sub(r"\bSCI\b", "SCIENCE", text)

    return text


def normalize_location(value):
    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    aliases = {
        "TIRUPATHI": "TIRUPATI",
        "TIRUPATHY": "TIRUPATI",
        "TIRUPATI": "TIRUPATI",
        "RAYCHOTI": "RAYACHOTY",
        "RAYACHOTI": "RAYACHOTY",
        "RAYACHOTY": "RAYACHOTY",
    }

    return aliases.get(text, text)


def compact_name(value):
    return normalize_name(value).replace(" ", "")


def tokens(value):
    # Generic words contribute little to college identity.
    ignored = {
        "THE", "OF", "AND", "FOR", "AT", "IN",
        "A", "AN", "TO", "BY"
    }
    return {
        token
        for token in normalize_name(value).split()
        if token not in ignored
    }


# ------------------------------------------------------------
# 3. Compare names
# ------------------------------------------------------------

def name_similarity(name_a, name_b):
    a = normalize_name(name_a)
    b = normalize_name(name_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    compact_a = compact_name(name_a)
    compact_b = compact_name(name_b)
    compact_score = SequenceMatcher(
        None, compact_a, compact_b
    ).ratio()

    set_a = tokens(name_a)
    set_b = tokens(name_b)

    if set_a or set_b:
        jaccard_score = len(set_a & set_b) / len(set_a | set_b)
    else:
        jaccard_score = 0.0

    # Reward one name being almost fully contained in the other,
    # but do not treat this alone as proof of a correct match.
    if set_a and set_b:
        containment_score = (
            len(set_a & set_b) / min(len(set_a), len(set_b))
        )
    else:
        containment_score = 0.0

    score = (
        0.35 * sequence_score
        + 0.20 * compact_score
        + 0.30 * jaccard_score
        + 0.15 * containment_score
    )

    return round(min(score, 1.0), 4)


# ------------------------------------------------------------
# 4. Detect clear institution-type conflicts
# ------------------------------------------------------------

TYPE_PATTERNS = {
    "PHARMACY": [
        r"\bPHARMACY\b",
        r"\bPHARMACEUTICAL\b",
        r"\bPHARM D\b",
    ],
    "MEDICAL": [
        r"\bMEDICAL COLLEGE\b",
        r"\bMEDICAL INSTITUTE\b",
        r"\bMBBS\b",
    ],
    "DENTAL": [
        r"\bDENTAL COLLEGE\b",
        r"\bDENTAL SCIENCES\b",
    ],
    "LAW": [
        r"\bLAW COLLEGE\b",
        r"\bCOLLEGE OF LAW\b",
    ],
    "POLYTECHNIC": [
        r"\bPOLYTECHNIC\b",
    ],
    "AGRICULTURE": [
        r"\bAGRICULTURAL COLLEGE\b",
        r"\bCOLLEGE OF AGRICULTURE\b",
        r"\bAGRICULTURE UNIVERSITY\b",
    ],
    "ARTS_SCIENCE": [
        r"\bARTS AND SCIENCE COLLEGE\b",
        r"\bARTS SCIENCE COLLEGE\b",
        r"\bARTS AND SCIENCES COLLEGE\b",
        r"\bDEGREE COLLEGE\b",
    ],
    "ENGINEERING": [
        r"\bENGINEERING COLLEGE\b",
        r"\bENGINEERING INSTITUTE\b",
        r"\bCOLLEGE OF ENGINEERING\b",
        r"\bTECHNOLOGY AND ENGINEERING\b",
    ],
}


def institution_types(name):
    text = normalize_name(name)
    found = set()

    for category, patterns in TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                found.add(category)
                break

    return found


def type_conflict(name_a, name_b):
    """
    Only exclude candidates when both names explicitly identify
    incompatible institution types.

    Generic words such as SCIENCE or UNIVERSITY alone do not
    automatically establish a conflict.
    """
    types_a = institution_types(name_a)
    types_b = institution_types(name_b)

    incompatible_pairs = [
        ("ENGINEERING", "PHARMACY"),
        ("ENGINEERING", "MEDICAL"),
        ("ENGINEERING", "DENTAL"),
        ("ENGINEERING", "LAW"),
        ("ENGINEERING", "POLYTECHNIC"),
        ("ENGINEERING", "ARTS_SCIENCE"),
        ("ENGINEERING", "AGRICULTURE"),
        ("PHARMACY", "ARTS_SCIENCE"),
        ("PHARMACY", "LAW"),
        ("MEDICAL", "ARTS_SCIENCE"),
        ("DENTAL", "ARTS_SCIENCE"),
        ("LAW", "ENGINEERING"),
    ]

    for left, right in incompatible_pairs:
        if (
            (left in types_a and right in types_b)
            or (right in types_a and left in types_b)
        ):
            return True

    return False


# ------------------------------------------------------------
# 5. Prepare current and official records
# ------------------------------------------------------------

def prepare_data():
    current_df = read_csv_safely(CURRENT_FILE)
    official_df = read_csv_safely(OFFICIAL_FILE)

    current_name_col = find_column(
        current_df, ["College Name", "College", "Institution Name"]
    )
    current_location_col = find_column(
        current_df, ["Location", "City", "District"], required=False
    )

    official_name_col = find_column(
        official_df,
        ["College Name", "NAME OF THE INSTITUTION", "Institution Name"]
    )
    official_location_col = find_column(
        official_df, ["Location", "City", "District"], required=False
    )
    official_code_col = find_column(
        official_df, ["College Code", "INSTCODE", "Institution Code"],
        required=False
    )
    official_branch_col = find_column(
        official_df, ["Branch", "branch_code", "Course"],
        required=False
    )
    official_year_col = find_column(
        official_df, ["Year", "Academic Year"], required=False
    )

    # Keep raw source values for auditability.
    current = pd.DataFrame({
        "Current College Name": current_df[current_name_col].map(clean_text),
        "Current Location": (
            current_df[current_location_col].map(clean_text)
            if current_location_col
            else ""
        ),
    })

    current["Normalized Name"] = current["Current College Name"].map(
        normalize_name
    )
    current["Normalized Location"] = current["Current Location"].map(
        normalize_location
    )

    current = current[
        current["Normalized Name"] != ""
    ].copy()

    # Group only after normalization, and preserve original variants.
    grouped_current = []

    for (norm_name, norm_location), group in current.groupby(
        ["Normalized Name", "Normalized Location"],
        dropna=False,
        sort=True
    ):
        names = sorted({
            name for name in group["Current College Name"] if name
        })
        locations = sorted({
            loc for loc in group["Current Location"] if loc
        })

        grouped_current.append({
            "Current College Name": names[0] if names else "",
            "Current Name Variants": " || ".join(names),
            "Current Location": locations[0] if locations else "",
            "Current Location Variants": " || ".join(locations),
            "Normalized Name": norm_name,
            "Normalized Location": norm_location,
            "Current Source Row Count": len(group),
        })

    current_unique = pd.DataFrame(grouped_current)

    official = pd.DataFrame({
        "Official College Name": official_df[official_name_col].map(clean_text),
        "Official Location": (
            official_df[official_location_col].map(clean_text)
            if official_location_col
            else ""
        ),
        "Official College Code": (
            official_df[official_code_col].map(clean_text)
            if official_code_col
            else ""
        ),
        "Official Branch": (
            official_df[official_branch_col].map(clean_text)
            if official_branch_col
            else ""
        ),
        "Official Year": (
            official_df[official_year_col].map(clean_text)
            if official_year_col
            else ""
        ),
    })

    official["Normalized Name"] = official["Official College Name"].map(
        normalize_name
    )
    official["Normalized Location"] = official["Official Location"].map(
        normalize_location
    )

    official = official[
        official["Normalized Name"] != ""
    ].copy()

    # Remove only exact duplicate official records. Different codes
    # remain separate candidates for review.
    dedup_columns = [
        "Normalized Name",
        "Normalized Location",
        "Official College Code",
        "Official Branch",
        "Official Year",
    ]
    official = official.drop_duplicates(
        subset=dedup_columns,
        keep="first"
    ).reset_index(drop=True)

    # A code is uniquely identifying here only if it is non-empty
    # and appears on exactly one retained official record.
    code_counts = Counter(
        code for code in official["Official College Code"]
        if code
    )
    official["Official Code Record Count"] = official[
        "Official College Code"
    ].map(lambda code: code_counts.get(code, 0) if code else 0)

    return current_unique, official


# ------------------------------------------------------------
# 6. Build candidates and score them
# ------------------------------------------------------------

def evaluate_candidates(current_row, official):
    current_name = current_row["Current College Name"]
    current_location = current_row["Normalized Location"]

    candidates = []

    for _, official_row in official.iterrows():
        official_name = official_row["Official College Name"]
        official_location = official_row["Normalized Location"]

        similarity = name_similarity(current_name, official_name)

        same_location = (
            bool(current_location)
            and bool(official_location)
            and current_location == official_location
        )

        conflict = type_conflict(current_name, official_name)

        # Location contributes to ranking, but cannot rescue a
        # weak name match.
        location_score = 1.0 if same_location else 0.0
        total_score = 0.82 * similarity + 0.18 * location_score

        candidates.append({
            "name_similarity": similarity,
            "location_score": location_score,
            "total_score": round(total_score, 4),
            "same_location": same_location,
            "type_conflict": conflict,
            "row": official_row,
        })

    candidates.sort(
        key=lambda item: (
            item["total_score"],
            item["name_similarity"],
            item["same_location"],
        ),
        reverse=True,
    )

    valid_candidates = [
        item for item in candidates
        if not item["type_conflict"]
        and item["name_similarity"] >= 0.55
    ]

    excluded_type_candidates = [
        item for item in candidates
        if item["type_conflict"]
        and item["name_similarity"] >= 0.55
    ]

    return candidates, valid_candidates, excluded_type_candidates


# ------------------------------------------------------------
# 7. Classify each college conservatively
# ------------------------------------------------------------

def classify_match(current_row, valid_candidates):
    if not valid_candidates:
        return "UNMATCHED", ["NO_VALID_CANDIDATE"]

    best = valid_candidates[0]
    best_row = best["row"]
    flags = []

    if best["name_similarity"] < 0.70:
        return "UNMATCHED", ["NO_STRONG_NAME_MATCH"]

    exact_name = (
        current_row["Normalized Name"]
        == best_row["Normalized Name"]
    )
    same_location = best["same_location"]

    # A competitor is close when its name score is within 0.04
    # of the leading candidate. Such matches need human review.
    close_competitors = [
        item for item in valid_candidates[1:]
        if best["name_similarity"] - item["name_similarity"] <= 0.04
    ]

    code = clean_text(best_row["Official College Code"])
    code_count = int(best_row["Official Code Record Count"])

    if not exact_name:
        flags.append("NAME_VARIANT")

    if not same_location:
        flags.append("LOCATION_DIFFERENCE")

    if close_competitors:
        flags.append("CLOSE_CANDIDATES")

    if not code:
        flags.append("MISSING_OFFICIAL_CODE")
    elif code_count > 1:
        flags.append("DUPLICATE_OFFICIAL_CODE")

    if best["name_similarity"] < 0.90:
        flags.append("NAME_SIMILARITY_NOT_EXACT")

    if exact_name and same_location and code and code_count == 1:
        if not close_competitors:
            return "SAFE", ["NONE"]

    return "MANUAL_REVIEW", flags or ["CHECK_BEFORE_APPROVAL"]


# ------------------------------------------------------------
# 8. Create report
# ------------------------------------------------------------

def build_report(current_unique, official):
    report_rows = []

    for _, current_row in current_unique.iterrows():
        all_candidates, valid, excluded_types = evaluate_candidates(
            current_row, official
        )

        status, flags = classify_match(current_row, valid)

        best = valid[0] if valid else None
        second = valid[1] if len(valid) > 1 else None
        third = valid[2] if len(valid) > 2 else None

        row = {
            "Status": status,
            "Review Flags": ";".join(flags),
            "Current College Name": current_row["Current College Name"],
            "Current Name Variants": current_row["Current Name Variants"],
            "Current Location": current_row["Current Location"],
            "Current Location Variants": current_row[
                "Current Location Variants"
            ],
            "Current Source Row Count": current_row["Current Source Row Count"],
            "Candidate Count": len(valid),
            "Relevant Type Conflict Count": len(excluded_types),
        }

        for position, candidate in enumerate(
            [best, second, third], start=1
        ):
            prefix = f"Candidate {position}"

            if candidate is None:
                row[f"{prefix} Name"] = ""
                row[f"{prefix} Location"] = ""
                row[f"{prefix} Code"] = ""
                row[f"{prefix} Branch"] = ""
                row[f"{prefix} Year"] = ""
                row[f"{prefix} Name Similarity"] = ""
                row[f"{prefix} Total Score"] = ""
                row[f"{prefix} Same Location"] = ""
                row[f"{prefix} Code Record Count"] = ""
                continue

            official_row = candidate["row"]

            row[f"{prefix} Name"] = official_row["Official College Name"]
            row[f"{prefix} Location"] = official_row["Official Location"]
            row[f"{prefix} Code"] = official_row["Official College Code"]
            row[f"{prefix} Branch"] = official_row["Official Branch"]
            row[f"{prefix} Year"] = official_row["Official Year"]
            row[f"{prefix} Name Similarity"] = candidate["name_similarity"]
            row[f"{prefix} Total Score"] = candidate["total_score"]
            row[f"{prefix} Same Location"] = candidate["same_location"]
            row[f"{prefix} Code Record Count"] = official_row[
                "Official Code Record Count"
            ]

        # Show up to two relevant incompatible-type candidates for
        # transparency. They are not eligible for automatic mapping.
        for position in range(1, 3):
            row[f"Type Conflict {position} Name"] = ""
            row[f"Type Conflict {position} Location"] = ""
            row[f"Type Conflict {position} Name Similarity"] = ""

        for position, candidate in enumerate(excluded_types[:2], start=1):
            official_row = candidate["row"]
            row[f"Type Conflict {position} Name"] = official_row[
                "Official College Name"
            ]
            row[f"Type Conflict {position} Location"] = official_row[
                "Official Location"
            ]
            row[f"Type Conflict {position} Name Similarity"] = candidate[
                "name_similarity"
            ]

        report_rows.append(row)

    report = pd.DataFrame(report_rows)

    status_order = {
        "MANUAL_REVIEW": 0,
        "UNMATCHED": 1,
        "SAFE": 2,
    }

    report["_sort"] = report["Status"].map(status_order)
    report = report.sort_values(
        ["_sort", "Current College Name"],
        kind="stable"
    ).drop(columns="_sort")

    return report


def main():
    print("\nV5.6 College Mapping Review")
    print("-" * 55)

    current_unique, official = prepare_data()

    print(f"Current unique college/location combinations: {len(current_unique)}")
    print(f"Official candidate records: {len(official)}")
    print("Building review report...")

    report = build_report(current_unique, official)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    counts = report["Status"].value_counts()

    flag_counts = Counter()
    for value in report["Review Flags"]:
        for flag in str(value).split(";"):
            if flag and flag != "NONE":
                flag_counts[flag] += 1

    print("\nReport created successfully.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Unique rows reviewed: {len(report)}")

    print("\nClassification summary:")
    for status in ["SAFE", "MANUAL_REVIEW", "UNMATCHED"]:
        print(f"  {status}: {int(counts.get(status, 0))}")

    print("\nMost common review flags:")
    for flag, count in flag_counts.most_common(10):
        print(f"  {flag}: {count}")

    print("\nIMPORTANT:")
    print("This script creates a review CSV only.")
    print("It does NOT update the live application, database,")
    print("source datasets, or trained machine-learning model.")


if __name__ == "__main__":
    main()