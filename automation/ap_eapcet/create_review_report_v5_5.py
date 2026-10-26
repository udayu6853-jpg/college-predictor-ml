from pathlib import Path
from difflib import SequenceMatcher
import re
import pandas as pd


# ============================================================
# V5.5 — AP EAPCET COLLEGE MATCHING REVIEW
#
# Review-only script.
# Does NOT modify datasets, PostgreSQL, or the ML model.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CURRENT_CSV = PROJECT_ROOT / "dataset" / "eamcet_colleges.csv"

OFFICIAL_CSV = (
    PROJECT_ROOT
    / "automation"
    / "output"
    / "ap_eapcet_normalized.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "automation"
    / "output"
    / "ap_eapcet_unique_college_mapping_review_v5_5.csv"
)

TOP_CANDIDATES = 3

# Classification thresholds
MIN_NAME_SIMILARITY = 0.55
STRONG_NAME_SIMILARITY = 0.85
CLOSE_CANDIDATE_GAP = 0.05

# ------------------------------------------------------------
# Known location spelling variants.
# These aliases are used for comparison only.
# They do not change either source dataset.
# ------------------------------------------------------------

LOCATION_ALIASES = {
    "TIRUPATHI": "TIRUPATI",
    "TIRUPATHY": "TIRUPATI",
    "TIRUPATI": "TIRUPATI",
    "RAYCHOTI": "RAYACHOTY",
    "RAYACHOTI": "RAYACHOTY",
    "RAYACHOTY": "RAYACHOTY",
}


# ------------------------------------------------------------
# Institution types.
#
# UNIVERSITY is deliberately not treated as a conflicting
# type on its own: a university can also have engineering
# programs.
# ------------------------------------------------------------

TYPE_PATTERNS = {
    "ENGINEERING": [
        r"\bENGINEERING\b",
        r"\bTECHNOLOGY\b",
        r"\bTECHNICAL\b",
    ],
    "PHARMACY": [
        r"\bPHARMACY\b",
        r"\bPHARMACEUTICAL\b",
    ],
    "ARTS_SCIENCE": [
        r"\bARTS\b",
        r"\bSCIENCE\b",
    ],
    "MEDICAL": [
        r"\bMEDICAL\b",
        r"\bMEDICINE\b",
    ],
    "DENTAL": [
        r"\bDENTAL\b",
    ],
    "LAW": [
        r"\bLAW\b",
    ],
    "POLYTECHNIC": [
        r"\bPOLYTECHNIC\b",
    ],
    "AGRICULTURE": [
        r"\bAGRICULTURE\b",
        r"\bAGRICULTURAL\b",
    ],
}

# These types are strongly incompatible for automated matching.
INCOMPATIBLE_TYPE_PAIRS = {
    frozenset(("ENGINEERING", "PHARMACY")),
    frozenset(("ENGINEERING", "ARTS_SCIENCE")),
    frozenset(("ENGINEERING", "MEDICAL")),
    frozenset(("ENGINEERING", "DENTAL")),
    frozenset(("ENGINEERING", "LAW")),
    frozenset(("ENGINEERING", "AGRICULTURE")),
    frozenset(("PHARMACY", "ARTS_SCIENCE")),
    frozenset(("PHARMACY", "MEDICAL")),
    frozenset(("PHARMACY", "DENTAL")),
    frozenset(("PHARMACY", "LAW")),
    frozenset(("ARTS_SCIENCE", "MEDICAL")),
    frozenset(("ARTS_SCIENCE", "DENTAL")),
    frozenset(("ARTS_SCIENCE", "LAW")),
    frozenset(("ARTS_SCIENCE", "AGRICULTURE")),
    frozenset(("MEDICAL", "LAW")),
    frozenset(("DENTAL", "LAW")),
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(value):
    """Convert missing values to an empty string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_name(value):
    """
    Normalize college names for comparison.

    Original CSV values are not changed.
    """
    text = clean_text(value).upper()

    replacements = {
        "&": " AND ",
        "@": " AT ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Conservative common wording normalization
    text = re.sub(r"\bINST\b", "INSTITUTE", text)
    text = re.sub(r"\bINSTT\b", "INSTITUTE", text)
    text = re.sub(r"\bTECH\b", "TECHNOLOGY", text)
    text = re.sub(r"\bCOLL\b", "COLLEGE", text)

    return text


def normalize_location(value):
    """Normalize location spelling without changing source data."""
    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    return LOCATION_ALIASES.get(text, text)


def name_similarity(name_a, name_b):
    """Calculate a name similarity score between 0 and 1."""
    a = normalize_name(name_a)
    b = normalize_name(name_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if tokens_a and tokens_b:
        token_score = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        token_score = 0.0

    # Character similarity helps with spelling differences.
    # Token similarity helps with reordered or shortened names.
    return round((0.65 * sequence_score) + (0.35 * token_score), 4)


def location_similarity(location_a, location_b):
    """Compare locations using normalized spellings."""
    a = normalize_location(location_a)
    b = normalize_location(location_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return round(SequenceMatcher(None, a, b).ratio(), 4)


# ============================================================
# INSTITUTION TYPE CHECKING
# ============================================================

def institution_types(college_name):
    """Return the institution types explicitly indicated by a name."""
    name = normalize_name(college_name)
    found = set()

    for institution_type, patterns in TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name):
                found.add(institution_type)
                break

    return found


def has_type_conflict(current_name, official_name):
    """
    Return True only for a clearly incompatible institution type.

    Generic words such as UNIVERSITY or INSTITUTE are not
    considered type conflicts.
    """
    current_types = institution_types(current_name)
    official_types = institution_types(official_name)

    for current_type in current_types:
        for official_type in official_types:
            if current_type == official_type:
                continue

            pair = frozenset((current_type, official_type))

            if pair in INCOMPATIBLE_TYPE_PAIRS:
                return True

    return False


# ============================================================
# DATA LOADING
# ============================================================

def find_column(df, possible_names, required=True):
    """
    Find a column without depending on capitalization or
    leading/trailing spaces.
    """
    lookup = {
        str(column).strip().casefold(): column
        for column in df.columns
    }

    for name in possible_names:
        key = name.strip().casefold()

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {possible_names}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_data():
    """Load current and official data without modifying either file."""
    if not CURRENT_CSV.exists():
        raise FileNotFoundError(
            f"Current EAMCET CSV not found:\n{CURRENT_CSV}"
        )

    if not OFFICIAL_CSV.exists():
        raise FileNotFoundError(
            f"Normalized official CSV not found:\n{OFFICIAL_CSV}"
        )

    current = pd.read_csv(CURRENT_CSV)
    official = pd.read_csv(OFFICIAL_CSV)

    current_name_col = find_column(
        current,
        ["College Name", "NAME OF THE INSTITUTION", "Name"],
    )

    current_location_col = find_column(
        current,
        ["Location", "College Location", "City"],
        required=False,
    )

    official_name_col = find_column(
        official,
        ["College Name", "NAME OF THE INSTITUTION", "Name"],
    )

    official_location_col = find_column(
        official,
        ["Location", "College Location", "City"],
        required=False,
    )

    official_code_col = find_column(
        official,
        ["College Code", "INSTCODE", "Institution Code", "Code"],
        required=False,
    )

    # Build standardized working copies. Source files remain untouched.
    current_work = pd.DataFrame({
        "Current College Name": current[current_name_col].map(clean_text),
        "Current Location": (
            current[current_location_col].map(clean_text)
            if current_location_col
            else ""
        ),
    })

    official_work = pd.DataFrame({
        "Official College Name": official[official_name_col].map(clean_text),
        "Official Location": (
            official[official_location_col].map(clean_text)
            if official_location_col
            else ""
        ),
        "Official College Code": (
            official[official_code_col].map(clean_text)
            if official_code_col
            else ""
        ),
    })

    current_work = current_work[
        current_work["Current College Name"].ne("")
    ].copy()

    official_work = official_work[
        official_work["Official College Name"].ne("")
    ].copy()

    # A college can have several branch rows in the current dataset.
    # Review each unique college/location combination only once.
    current_work["_name_key"] = current_work[
        "Current College Name"
    ].map(normalize_name)

    current_work["_location_key"] = current_work[
        "Current Location"
    ].map(normalize_location)

    current_work = current_work.drop_duplicates(
        subset=["_name_key", "_location_key"]
    ).reset_index(drop=True)

    official_work["_name_key"] = official_work[
        "Official College Name"
    ].map(normalize_name)

    official_work["_location_key"] = official_work[
        "Official Location"
    ].map(normalize_location)

    # Keep different official codes/campuses separate.
    official_work = official_work.drop_duplicates(
        subset=[
            "_name_key",
            "_location_key",
            "Official College Code",
        ]
    ).reset_index(drop=True)

    return current_work, official_work


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(current_row, official_row):
    current_name = current_row["Current College Name"]
    current_location = current_row["Current Location"]

    official_name = official_row["Official College Name"]
    official_location = official_row["Official Location"]

    name_score = name_similarity(current_name, official_name)
    location_score = location_similarity(
        current_location,
        official_location,
    )

    # The name carries more weight than location.
    # Location is still important for separating campuses.
    total_score = (0.80 * name_score) + (0.20 * location_score)

    exact_name = (
        normalize_name(current_name)
        == normalize_name(official_name)
    )

    exact_location = (
        normalize_location(current_location)
        and normalize_location(current_location)
        == normalize_location(official_location)
    )

    location_missing = (
        not normalize_location(current_location)
        or not normalize_location(official_location)
    )

    conflict = has_type_conflict(current_name, official_name)

    return {
        "name_score": round(name_score, 4),
        "location_score": round(location_score, 4),
        "total_score": round(total_score, 4),
        "exact_name": bool(exact_name),
        "exact_location": bool(exact_location),
        "location_missing": bool(location_missing),
        "type_conflict": bool(conflict),
    }


def build_candidates(current_row, official_work):
    """Score official candidates and keep valid candidates separate."""
    scored = []
    excluded_conflicts = []

    for _, official_row in official_work.iterrows():
        metrics = score_candidate(current_row, official_row)

        item = {
            "official": official_row,
            **metrics,
        }

        if metrics["type_conflict"]:
            excluded_conflicts.append(item)
        else:
            scored.append(item)

    scored.sort(
        key=lambda item: (
            item["total_score"],
            item["name_score"],
            item["location_score"],
        ),
        reverse=True,
    )

    excluded_conflicts.sort(
        key=lambda item: (
            item["name_score"],
            item["total_score"],
        ),
        reverse=True,
    )

    return scored, excluded_conflicts


# ============================================================
# CLASSIFICATION
# ============================================================

def unique_official_code(candidate):
    """
    Check whether the selected official name/location points
    to one non-empty official code in the official dataset.
    """
    official = candidate["official"]
    code = clean_text(official["Official College Code"])

    return bool(code)


def classify_match(current_row, candidates, excluded_conflicts):
    """
    Conservative classification:

    SAFE:
      - exact normalized name
      - exact normalized/aliased location
      - unique official code
      - no close competing candidate

    MANUAL_REVIEW:
      - strong but not exact match
      - location difference
      - close competing candidates
      - ambiguous or missing official code

    UNMATCHED:
      - no valid candidate reaches the minimum threshold
    """
    flags = []
    reasons = []

    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None

    # A conflict flag is relevant only if a conflicting institution
    # is a plausible competitor by name similarity. Unrelated
    # engineering/pharmacy/arts colleges do not trigger the flag.
    relevant_conflicts = []

    if best:
        for item in excluded_conflicts:
            if item["name_score"] < MIN_NAME_SIMILARITY:
                continue

            # Only flag a conflict if it is reasonably close to
            # the best valid candidate's name score.
            if item["name_score"] >= best["name_score"] - 0.10:
                relevant_conflicts.append(item)

    if relevant_conflicts:
        flags.append("RELEVANT_TYPE_CONFLICT_CANDIDATE")
        reasons.append(
            "A similarly named official institution has an "
            "incompatible institution type."
        )

    if not best or best["name_score"] < MIN_NAME_SIMILARITY:
        return {
            "Status": "UNMATCHED",
            "Reason": "No sufficiently similar valid official candidate.",
            "Flags": "; ".join(flags) if flags else "NO_VALID_CANDIDATE",
            "Best": best,
            "Second": second,
            "RelevantConflicts": relevant_conflicts,
        }

    if best["exact_name"] and best["exact_location"]:
        if not unique_official_code(best):
            flags.append("MISSING_OFFICIAL_CODE")
            reasons.append("The official candidate has no college code.")

        # A close candidate may be a separate campus or duplicate code.
        close_competitor = False

        if second:
            gap = best["total_score"] - second["total_score"]

            if (
                gap <= CLOSE_CANDIDATE_GAP
                and second["name_score"] >= STRONG_NAME_SIMILARITY
            ):
                close_competitor = True

        if close_competitor:
            flags.append("CLOSE_CANDIDATES")
            reasons.append(
                "Another valid official candidate has a similar score."
            )

        # Exact normalized name/location can be SAFE only if the
        # best match is not ambiguous and has an official code.
        if unique_official_code(best) and not close_competitor:
            status = "SAFE"
            reasons.append(
                "Exact normalized college name and location; "
                "no close competing candidate."
            )
        else:
            status = "MANUAL_REVIEW"
            reasons.append(
                "Exact name/location match requires code or "
                "candidate ambiguity review."
            )

    else:
        status = "MANUAL_REVIEW"

        if not best["exact_name"]:
            flags.append("NAME_VARIANT")
            reasons.append(
                "The candidate name is similar but not an exact "
                "normalized match."
            )

        if not best["exact_location"]:
            flags.append("LOCATION_DIFFERENCE")
            reasons.append(
                "The candidate location differs or cannot be "
                "confirmed exactly."
            )

        if second:
            gap = best["total_score"] - second["total_score"]

            if (
                gap <= CLOSE_CANDIDATE_GAP
                and second["name_score"] >= STRONG_NAME_SIMILARITY
            ):
                flags.append("CLOSE_CANDIDATES")
                reasons.append(
                    "Another valid official candidate has a similar score."
                )

    return {
        "Status": status,
        "Reason": " ".join(reasons),
        "Flags": "; ".join(dict.fromkeys(flags)) if flags else "NONE",
        "Best": best,
        "Second": second,
        "RelevantConflicts": relevant_conflicts,
    }


# ============================================================
# REPORT CREATION
# ============================================================

def candidate_value(candidate, key):
    if not candidate:
        return ""

    return candidate["official"].get(key, "")


def format_candidate(candidate):
    if not candidate:
        return ""

    official = candidate["official"]

    return (
        f"{clean_text(official['Official College Name'])} "
        f"| Code: {clean_text(official['Official College Code'])} "
        f"| Location: {clean_text(official['Official Location'])} "
        f"| Name score: {candidate['name_score']:.4f} "
        f"| Total score: {candidate['total_score']:.4f}"
    )


def create_report():
    current_work, official_work = load_data()

    rows = []

    print("\nV5.5 college mapping review")
    print("-" * 55)
    print(f"Current unique college/location combinations: {len(current_work)}")
    print(f"Official candidate rows: {len(official_work)}")
    print("Building review report...\n")

    for _, current_row in current_work.iterrows():
        candidates, excluded_conflicts = build_candidates(
            current_row,
            official_work,
        )

        result = classify_match(
            current_row,
            candidates,
            excluded_conflicts,
        )

        best = result["Best"]
        second = result["Second"]

        row = {
            "Current College Name": current_row["Current College Name"],
            "Current Location": current_row["Current Location"],
            "Status": result["Status"],
            "Reason": result["Reason"],
            "Flags": result["Flags"],
            "Best Candidate Name": candidate_value(
                best, "Official College Name"
            ),
            "Best Candidate Code": candidate_value(
                best, "Official College Code"
            ),
            "Best Candidate Location": candidate_value(
                best, "Official Location"
            ),
            "Best Candidate Name Score": (
                best["name_score"] if best else ""
            ),
            "Best Candidate Location Score": (
                best["location_score"] if best else ""
            ),
            "Best Candidate Total Score": (
                best["total_score"] if best else ""
            ),
            "Second Candidate Name": candidate_value(
                second, "Official College Name"
            ),
            "Second Candidate Code": candidate_value(
                second, "Official College Code"
            ),
            "Second Candidate Location": candidate_value(
                second, "Official Location"
            ),
            "Second Candidate Total Score": (
                second["total_score"] if second else ""
            ),
            "Relevant Type Conflict Candidates": " || ".join(
                format_candidate(item)
                for item in result["RelevantConflicts"][:TOP_CANDIDATES]
            ),
        }

        # Include the top three valid candidates for manual checking.
        for index in range(TOP_CANDIDATES):
            candidate = (
                candidates[index] if index < len(candidates) else None
            )

            number = index + 1

            row[f"Candidate {number} Name"] = candidate_value(
                candidate, "Official College Name"
            )
            row[f"Candidate {number} Code"] = candidate_value(
                candidate, "Official College Code"
            )
            row[f"Candidate {number} Location"] = candidate_value(
                candidate, "Official Location"
            )
            row[f"Candidate {number} Name Score"] = (
                candidate["name_score"] if candidate else ""
            )
            row[f"Candidate {number} Total Score"] = (
                candidate["total_score"] if candidate else ""
            )

        rows.append(row)

    report = pd.DataFrame(rows)

    # Sort for convenient manual review.
    status_order = {
        "MANUAL_REVIEW": 0,
        "UNMATCHED": 1,
        "SAFE": 2,
    }

    report["_status_order"] = report["Status"].map(status_order)
    report = report.sort_values(
        by=["_status_order", "Current College Name"],
        ascending=[True, True],
    ).drop(columns=["_status_order"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Report created successfully.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Unique rows reviewed: {len(report)}")
    print("\nClassification summary:")

    summary = (
        report["Status"]
        .value_counts()
        .reindex(["SAFE", "MANUAL_REVIEW", "UNMATCHED"], fill_value=0)
    )

    for status, count in summary.items():
        print(f"  {status}: {count}")

    print("\nMost common flags:")
    flag_counts = (
        report["Flags"]
        .str.split("; ")
        .explode()
        .value_counts()
    )

    for flag, count in flag_counts.head(10).items():
        print(f"  {flag}: {count}")

    print("\nImportant: this script only creates a CSV review report.")
    print("It does not update the live application, database, datasets,")
    print("or trained machine-learning model.")


if __name__ == "__main__":
    try:
        create_report()
    except Exception as error:
        print("\nERROR while creating the V5.5 review report:")
        print(error)
        raise