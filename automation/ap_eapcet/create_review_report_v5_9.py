from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
import re
import pandas as pd

# ============================================================
# V5.9 COLLEGE MAPPING REVIEW — READ ONLY
# Creates a new review CSV only. It never edits source data,
# the database, the trained model, or the live application.
#
# Main changes from V5.8:
# - safer word-boundary abbreviation normalization
# - additional acronym / token / containment diagnostics
# - explicit location and candidate-gap diagnostics
# - more candidate slots for investigating plausible unmatched
#   names (suggestions are NOT automatic mappings)
# - duplicate-code reporting across institution candidates
# - conservative SAFE classification
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
CURRENT_FILE = ROOT / "dataset" / "eamcet_colleges.csv"
OFFICIAL_FILE = ROOT / "automation" / "output" / "ap_eapcet_normalized.csv"
OUTPUT_FILE = ROOT / "automation" / "output" / "ap_eapcet_unique_college_mapping_review_v5_9.csv"

MIN_CANDIDATE_SIMILARITY = 0.55
MIN_REVIEW_SIMILARITY = 0.70
HIGH_SIMILARITY = 0.90
CLOSE_CANDIDATE_GAP = 0.04
TOP_CANDIDATES = 5


# ------------------------------------------------------------
# 1. Read and clean source data
# ------------------------------------------------------------

def read_csv_safely(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding).fillna("")
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV: {path}") from last_error


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def column_key(value):
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def find_column(df, possible_names, required=True):
    lookup = {column_key(column): column for column in df.columns}
    for name in possible_names:
        result = lookup.get(column_key(name))
        if result is not None:
            return result
    if required:
        raise ValueError(
            f"Could not find required column. Expected one of: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


# ------------------------------------------------------------
# 2. Canonical name and location normalization
# ------------------------------------------------------------

# Apply replacements as whole words, not arbitrary substrings.
# This avoids corrupting words such as INSTITUTE by replacing INST.
NAME_REPLACEMENTS = [
    (r"\bINSTITUTION\b", "INSTITUTION"),
    (r"\bINSTT?\b", "INSTITUTE"),
    (r"\bINST\b", "INSTITUTE"),
    (r"\bINSTITUTE\b", "INSTITUTE"),
    (r"\bCOLL(?:EGE)?\b", "COLLEGE"),
    (r"\bENGG?\b", "ENGINEERING"),
    (r"\bTECH\b", "TECHNOLOGY"),
    (r"\bSCI\b", "SCIENCE"),
    (r"\bUNIV\b", "UNIVERSITY"),
    (r"\bWOMENS\b", "WOMEN"),
    (r"\bWOMAN\b", "WOMEN"),
    (r"\b&\b", "AND"),
]

LOCATION_ALIASES = {
    "TIRUPATHI": "TIRUPATI",
    "TIRUPATHY": "TIRUPATI",
    "TIRUPATI": "TIRUPATI",
    "RAYCHOTI": "RAYACHOTY",
    "RAYACHOTI": "RAYACHOTY",
    "RAYACHOTY": "RAYACHOTY",
    "ANANTAPURAMU": "ANANTAPUR",
    "ANANTAPURAM": "ANANTAPUR",
}


def normalize_name(value):
    text = clean_text(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for pattern, replacement in NAME_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_location(value):
    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return LOCATION_ALIASES.get(text, text)


def compact_name(value):
    return normalize_name(value).replace(" ", "")


def meaningful_tokens(value):
    ignored = {
        "THE", "OF", "AND", "FOR", "AT", "IN", "A", "AN", "TO", "BY",
        "COLLEGE", "INSTITUTE", "INSTITUTION", "ENGINEERING",
        "TECHNOLOGY", "SCIENCE", "UNIVERSITY",
    }
    return {token for token in normalize_name(value).split() if token not in ignored}


def acronym(value):
    tokens = normalize_name(value).split()
    ignored = {"THE", "OF", "AND", "FOR", "AT", "IN", "A", "AN", "TO", "BY"}
    tokens = [token for token in tokens if token not in ignored]
    return "".join(token[0] for token in tokens if token)


def token_scores(name_a, name_b):
    a = set(normalize_name(name_a).split())
    b = set(normalize_name(name_b).split())
    if not a or not b:
        return 0.0, 0.0
    union = a | b
    intersection = a & b
    jaccard = len(intersection) / len(union) if union else 0.0
    containment = len(intersection) / min(len(a), len(b))
    return round(jaccard, 4), round(containment, 4)


# ------------------------------------------------------------
# 3. Name similarity and diagnostic signals
# ------------------------------------------------------------

def name_similarity(name_a, name_b):
    a = normalize_name(name_a)
    b = normalize_name(name_b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    sequence_score = SequenceMatcher(None, a, b).ratio()
    compact_score = SequenceMatcher(None, compact_name(name_a), compact_name(name_b)).ratio()
    tokens_a = meaningful_tokens(name_a)
    tokens_b = meaningful_tokens(name_b)
    union = tokens_a | tokens_b
    intersection = tokens_a & tokens_b
    jaccard_score = len(intersection) / len(union) if union else 0.0
    containment_score = (
        len(intersection) / min(len(tokens_a), len(tokens_b))
        if tokens_a and tokens_b else 0.0
    )

    # Keep the V5.8 score definition for comparability.
    score = (
        0.35 * sequence_score
        + 0.20 * compact_score
        + 0.30 * jaccard_score
        + 0.15 * containment_score
    )
    return round(min(score, 1.0), 4)


def name_diagnostics(name_a, name_b):
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)
    jaccard, containment = token_scores(name_a, name_b)
    acronym_a = acronym(name_a)
    acronym_b = acronym(name_b)
    acronym_match = bool(acronym_a and acronym_b and acronym_a == acronym_b)
    return {
        "exact_normalized_name": bool(norm_a and norm_a == norm_b),
        "sequence_score": round(SequenceMatcher(None, norm_a, norm_b).ratio(), 4) if norm_a and norm_b else 0.0,
        "token_jaccard": jaccard,
        "token_containment": containment,
        "current_acronym": acronym_a,
        "official_acronym": acronym_b,
        "acronym_match": acronym_match,
    }


# ------------------------------------------------------------
# 4. Institution-type conflict checks
# ------------------------------------------------------------

TYPE_PATTERNS = {
    "PHARMACY": [r"\bPHARMACY\b", r"\bPHARMACEUTICAL\b", r"\bPHARM D\b"],
    "MEDICAL": [r"\bMEDICAL COLLEGE\b", r"\bMEDICAL INSTITUTE\b", r"\bMBBS\b"],
    "DENTAL": [r"\bDENTAL COLLEGE\b", r"\bDENTAL SCIENCES\b"],
    "LAW": [r"\bLAW COLLEGE\b", r"\bCOLLEGE OF LAW\b"],
    "POLYTECHNIC": [r"\bPOLYTECHNIC\b"],
    "AGRICULTURE": [r"\bAGRICULTURAL COLLEGE\b", r"\bCOLLEGE OF AGRICULTURE\b", r"\bAGRICULTURE UNIVERSITY\b"],
    "ARTS_SCIENCE": [
        r"\bARTS AND SCIENCE COLLEGE\b", r"\bARTS SCIENCE COLLEGE\b",
        r"\bARTS AND SCIENCES COLLEGE\b", r"\bDEGREE COLLEGE\b",
    ],
    "ENGINEERING": [
        r"\bENGINEERING COLLEGE\b", r"\bENGINEERING INSTITUTE\b",
        r"\bCOLLEGE OF ENGINEERING\b", r"\bTECHNOLOGY AND ENGINEERING\b",
    ],
}


def institution_types(name):
    text = normalize_name(name)
    return {
        category for category, patterns in TYPE_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def type_conflict(name_a, name_b):
    types_a = institution_types(name_a)
    types_b = institution_types(name_b)
    incompatible_pairs = [
        ("ENGINEERING", "PHARMACY"), ("ENGINEERING", "MEDICAL"),
        ("ENGINEERING", "DENTAL"), ("ENGINEERING", "LAW"),
        ("ENGINEERING", "POLYTECHNIC"), ("ENGINEERING", "ARTS_SCIENCE"),
        ("ENGINEERING", "AGRICULTURE"), ("PHARMACY", "ARTS_SCIENCE"),
        ("PHARMACY", "LAW"), ("MEDICAL", "ARTS_SCIENCE"),
        ("DENTAL", "ARTS_SCIENCE"),
    ]
    return any(
        (left in types_a and right in types_b) or (right in types_a and left in types_b)
        for left, right in incompatible_pairs
    )


# ------------------------------------------------------------
# 5. Load current and official data; group official branch rows
# ------------------------------------------------------------

def prepare_data():
    current_df = read_csv_safely(CURRENT_FILE)
    official_df = read_csv_safely(OFFICIAL_FILE)

    current_name_col = find_column(current_df, ["College Name", "College", "Institution Name"])
    current_location_col = find_column(current_df, ["Location", "City", "District"], required=False)
    official_name_col = find_column(official_df, ["College Name", "NAME OF THE INSTITUTION", "Institution Name"])
    official_location_col = find_column(official_df, ["Location", "City", "District"], required=False)
    official_code_col = find_column(official_df, ["College Code", "INSTCODE", "Institution Code"], required=False)
    official_branch_col = find_column(official_df, ["Branch", "branch_code", "Course"], required=False)
    official_year_col = find_column(official_df, ["Year", "Academic Year"], required=False)

    current = pd.DataFrame({
        "Current College Name": current_df[current_name_col].map(clean_text),
        "Current Location": current_df[current_location_col].map(clean_text) if current_location_col else "",
    })
    current["Normalized Name"] = current["Current College Name"].map(normalize_name)
    current["Normalized Location"] = current["Current Location"].map(normalize_location)
    current = current[current["Normalized Name"] != ""].copy()

    current_groups = []
    for (norm_name, norm_location), group in current.groupby(
        ["Normalized Name", "Normalized Location"], dropna=False, sort=True
    ):
        names = sorted({value for value in group["Current College Name"] if value})
        locations = sorted({value for value in group["Current Location"] if value})
        current_groups.append({
            "Current College Name": names[0] if names else "",
            "Current Name Variants": " || ".join(names),
            "Current Location": locations[0] if locations else "",
            "Current Location Variants": " || ".join(locations),
            "Normalized Name": norm_name,
            "Normalized Location": norm_location,
            "Current Source Row Count": len(group),
        })
    current_unique = pd.DataFrame(current_groups)

    official = pd.DataFrame({
        "Official College Name": official_df[official_name_col].map(clean_text),
        "Official Location": official_df[official_location_col].map(clean_text) if official_location_col else "",
        "Official College Code": official_df[official_code_col].map(clean_text) if official_code_col else "",
        "Official Branch": official_df[official_branch_col].map(clean_text) if official_branch_col else "",
        "Official Year": official_df[official_year_col].map(clean_text) if official_year_col else "",
    })
    official["Normalized Name"] = official["Official College Name"].map(normalize_name)
    official["Normalized Location"] = official["Official Location"].map(normalize_location)
    official = official[official["Normalized Name"] != ""].copy()

    institution_keys = ["Normalized Name", "Normalized Location", "Official College Code"]
    official = official.groupby(institution_keys, dropna=False, sort=False).agg({
        "Official College Name": "first",
        "Official Location": "first",
        "Official Branch": lambda values: " | ".join(sorted({clean_text(v) for v in values if clean_text(v)})),
        "Official Year": lambda values: " | ".join(sorted({clean_text(v) for v in values if clean_text(v)})),
    }).reset_index()

    code_counts = Counter(clean_text(code) for code in official["Official College Code"] if clean_text(code))
    official["Official Code Record Count"] = official["Official College Code"].map(
        lambda code: code_counts.get(clean_text(code), 0) if clean_text(code) else 0
    )
    return current_unique, official


# ------------------------------------------------------------
# 6. Rank candidates; keep type-conflict candidates separate
# ------------------------------------------------------------

def evaluate_candidates(current_row, official):
    current_name = current_row["Current College Name"]
    current_location = current_row["Normalized Location"]
    valid_candidates, excluded_type_candidates = [], []

    for _, official_row in official.iterrows():
        official_name = official_row["Official College Name"]
        official_location = official_row["Normalized Location"]
        similarity = name_similarity(current_name, official_name)
        diagnostics = name_diagnostics(current_name, official_name)

        same_location = bool(current_location and official_location and current_location == official_location)
        conflict = type_conflict(current_name, official_name)
        location_score = 1.0 if same_location else 0.0
        total_score = round(0.82 * similarity + 0.18 * location_score, 4)

        candidate = {
            "name_similarity": similarity,
            "location_score": location_score,
            "total_score": total_score,
            "same_location": same_location,
            "type_conflict": conflict,
            "diagnostics": diagnostics,
            "row": official_row,
        }

        if conflict:
            if similarity >= MIN_CANDIDATE_SIMILARITY:
                excluded_type_candidates.append(candidate)
        elif similarity >= MIN_CANDIDATE_SIMILARITY:
            valid_candidates.append(candidate)

    sort_key = lambda item: (
        item["total_score"], item["name_similarity"], item["same_location"]
    )
    valid_candidates.sort(key=sort_key, reverse=True)
    excluded_type_candidates.sort(key=sort_key, reverse=True)
    return valid_candidates, excluded_type_candidates


# ------------------------------------------------------------
# 7. Classification: conservative by design
# ------------------------------------------------------------

def classify_match(current_row, candidates):
    if not candidates:
        return (
            "UNMATCHED",
            "No candidate passed the name-similarity and institution-type checks.",
            ["NO_VALID_CANDIDATE"], None, None,
        )

    best = candidates[0]
    best_row = best["row"]

    if best["name_similarity"] < MIN_REVIEW_SIMILARITY:
        return (
            "UNMATCHED",
            "Best candidate is retained as a suggestion, but its name score is below the review threshold.",
            ["NO_STRONG_NAME_MATCH"], best, candidates[1] if len(candidates) > 1 else None,
        )

    exact_name = best["diagnostics"]["exact_normalized_name"]
    same_location = best["same_location"]
    code = clean_text(best_row["Official College Code"])
    code_count = int(best_row["Official Code Record Count"])

    close_candidates = [
        candidate for candidate in candidates[1:]
        if (best["name_similarity"] - candidate["name_similarity"]) <= CLOSE_CANDIDATE_GAP
    ]

    flags, reasons = [], []
    if not exact_name:
        flags.append("NAME_VARIANT")
        reasons.append("Names differ after canonical normalization.")
    if not same_location:
        flags.append("LOCATION_DIFFERENCE")
        reasons.append("Locations differ after location-alias normalization.")
    if close_candidates:
        flags.append("CLOSE_CANDIDATES")
        reasons.append("At least one competing candidate has a similar name score.")
    if not code:
        flags.append("MISSING_OFFICIAL_CODE")
        reasons.append("The leading official candidate has no college code.")
    elif code_count > 1:
        flags.append("DUPLICATE_OFFICIAL_CODE")
        reasons.append("The official code appears on multiple institution-level candidates.")
    if best["name_similarity"] < HIGH_SIMILARITY:
        flags.append("NAME_SIMILARITY_NOT_HIGH")
        reasons.append("The name similarity score is below the high-confidence threshold.")
    if best["diagnostics"]["acronym_match"] and not exact_name:
        flags.append("ACRONYM_SIGNAL")
        reasons.append("The names share an acronym signal; verify the full institution name.")
    if best["diagnostics"]["token_containment"] >= 0.8 and not exact_name:
        flags.append("TOKEN_CONTAINMENT_SIGNAL")
        reasons.append("Most meaningful name tokens are contained in both names.")

    # Never auto-approve based on fuzzy scores, acronym signals, or location alone.
    if exact_name and same_location and bool(code) and code_count == 1 and not close_candidates:
        return (
            "SAFE",
            "Exact canonical name and normalized location; official code is unique and no close competitor was found.",
            ["NONE"], best, None,
        )

    return (
        "MANUAL_REVIEW",
        " ".join(reasons) if reasons else "The mapping needs manual confirmation.",
        flags or ["CHECK_BEFORE_APPROVAL"],
        best,
        close_candidates[0] if close_candidates else (candidates[1] if len(candidates) > 1 else None),
    )


# ------------------------------------------------------------
# 8. Output helpers and review report
# ------------------------------------------------------------

CANDIDATE_FIELDS = [
    "Name", "Location", "Code", "Branches", "Year",
    "Name Similarity", "Total Score", "Same Location",
    "Code Record Count", "Exact Canonical Name",
    "Token Jaccard", "Token Containment", "Acronym Match",
]


def empty_candidate_fields(row, prefix):
    for suffix in CANDIDATE_FIELDS:
        row[f"{prefix} {suffix}"] = ""


def fill_candidate_fields(row, prefix, candidate):
    if candidate is None:
        empty_candidate_fields(row, prefix)
        return
    official_row = candidate["row"]
    diag = candidate["diagnostics"]
    row[f"{prefix} Name"] = official_row["Official College Name"]
    row[f"{prefix} Location"] = official_row["Official Location"]
    row[f"{prefix} Code"] = official_row["Official College Code"]
    row[f"{prefix} Branches"] = official_row["Official Branch"]
    row[f"{prefix} Year"] = official_row["Official Year"]
    row[f"{prefix} Name Similarity"] = candidate["name_similarity"]
    row[f"{prefix} Total Score"] = candidate["total_score"]
    row[f"{prefix} Same Location"] = candidate["same_location"]
    row[f"{prefix} Code Record Count"] = official_row["Official Code Record Count"]
    row[f"{prefix} Exact Canonical Name"] = diag["exact_normalized_name"]
    row[f"{prefix} Token Jaccard"] = diag["token_jaccard"]
    row[f"{prefix} Token Containment"] = diag["token_containment"]
    row[f"{prefix} Acronym Match"] = diag["acronym_match"]


def build_report(current_unique, official):
    report_rows = []
    for _, current_row in current_unique.iterrows():
        candidates, excluded_types = evaluate_candidates(current_row, official)
        status, reason, flags, best, closest_competitor = classify_match(current_row, candidates)
        candidate_list = candidates[:TOP_CANDIDATES]
        best_diag = best["diagnostics"] if best else {}

        row = {
            "Status": status,
            "Decision Reason": reason,
            "Review Flags": ";".join(flags),
            "Current College Name": current_row["Current College Name"],
            "Current Name Variants": current_row["Current Name Variants"],
            "Current Location": current_row["Current Location"],
            "Current Location Variants": current_row["Current Location Variants"],
            "Current Source Row Count": current_row["Current Source Row Count"],
            "Current Canonical Name": current_row["Normalized Name"],
            "Current Canonical Location": current_row["Normalized Location"],
            "Current Acronym": acronym(current_row["Current College Name"]),
            "Valid Candidate Count": len(candidates),
            "Relevant Type Conflict Count": len(excluded_types),
            "Best Candidate Name Score": best["name_similarity"] if best else "",
            "Best Candidate Total Score": best["total_score"] if best else "",
            "Best Candidate Same Location": best["same_location"] if best else "",
            "Best Candidate Exact Canonical Name": best_diag.get("exact_normalized_name", ""),
            "Best Candidate Token Jaccard": best_diag.get("token_jaccard", ""),
            "Best Candidate Token Containment": best_diag.get("token_containment", ""),
            "Best Candidate Acronym": best_diag.get("official_acronym", ""),
            "Best Candidate Acronym Match": best_diag.get("acronym_match", ""),
            "Closest Competitor Name Score": closest_competitor["name_similarity"] if closest_competitor else "",
            "Name Score Gap To Competitor": (
                round(best["name_similarity"] - closest_competitor["name_similarity"], 4)
                if best and closest_competitor else ""
            ),
            "Candidate Ranking Note": (
                "Candidates are fuzzy suggestions only; manually verify names, location, code and institution type."
            ),
        }

        for position in range(1, TOP_CANDIDATES + 1):
            candidate = candidate_list[position - 1] if position <= len(candidate_list) else None
            fill_candidate_fields(row, f"Candidate {position}", candidate)

        # Show up to three incompatible-type candidates for audit; they are never valid matches.
        for position in range(1, 4):
            for suffix in ["Name", "Location", "Code", "Name Score"]:
                row[f"Type Conflict {position} {suffix}"] = ""
        for position, candidate in enumerate(excluded_types[:3], start=1):
            official_row = candidate["row"]
            row[f"Type Conflict {position} Name"] = official_row["Official College Name"]
            row[f"Type Conflict {position} Location"] = official_row["Official Location"]
            row[f"Type Conflict {position} Code"] = official_row["Official College Code"]
            row[f"Type Conflict {position} Name Score"] = candidate["name_similarity"]

        report_rows.append(row)

    report = pd.DataFrame(report_rows)
    status_order = {"MANUAL_REVIEW": 0, "UNMATCHED": 1, "SAFE": 2}
    report["_sort"] = report["Status"].map(status_order)
    return report.sort_values(["_sort", "Current College Name"], kind="stable").drop(columns="_sort")


# ------------------------------------------------------------
# 9. Print report summary
# ------------------------------------------------------------

def main():
    print("\nV5.9 College Mapping Review")
    print("=" * 55)
    current_unique, official = prepare_data()
    print("Current unique college/location combinations:", len(current_unique))
    print("Official institution-level candidates:", len(official))
    print("\nBuilding review report...")
    report = build_report(current_unique, official)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    status_counts = report["Status"].value_counts()
    flag_counts = Counter()
    for value in report["Review Flags"]:
        for flag in str(value).split(";"):
            if flag and flag != "NONE":
                flag_counts[flag] += 1

    print("\nReport created successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records reviewed: {len(report)}")
    print("\nClassification summary:")
    for status in ["SAFE", "MANUAL_REVIEW", "UNMATCHED"]:
        print(f"  {status}: {int(status_counts.get(status, 0))}")
    print("\nMost common review flags:")
    for flag, count in flag_counts.most_common(15):
        print(f"  {flag}: {count}")

    print("\nDiagnostics:")
    print("  Exact canonical name and same location:",
          int(((report["Candidate 1 Exact Canonical Name"] == True) &
               (report["Candidate 1 Same Location"] == True)).sum()))
    print("  Rows with acronym signal:",
          int((report["Best Candidate Acronym Match"] == True).sum()))
    print("  Rows with at least one type-conflict candidate:",
          int((report["Relevant Type Conflict Count"] > 0).sum()))

    print("\nIMPORTANT:")
    print("This script only creates a new V5.9 review CSV.")
    print("It does not update the live website, database, source datasets, or trained ML model.")
    print("SAFE means the script's conservative checks passed; manually validate before production use.")


if __name__ == "__main__":
    main()
