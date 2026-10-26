
from pathlib import Path
from difflib import SequenceMatcher
import re
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

CURRENT_FILE = BASE_DIR / "dataset" / "eamcet_colleges.csv"
OFFICIAL_FILE = OUTPUT_DIR / "ap_eapcet_normalized.csv"
OUTPUT_FILE = OUTPUT_DIR / "ap_eapcet_college_mapping_review_v5_2.csv"


# =========================================================
# CONFIGURATION
# =========================================================

GENERIC_WORDS = {
    "AND", "OF", "THE", "FOR", "IN", "AT", "TO",
    "A", "AN", "COLLEGE", "INSTITUTE", "ENGINEERING",
    "TECHNOLOGY", "SCIENCE", "SCiences"
}

IDENTITY_WORDS = {
    "UNIVERSITY", "WOMEN", "WOMENS", "PHARMACY",
    "POLYTECHNIC", "MEDICAL", "DENTAL", "LAW",
    "ARCHITECTURE", "AGRICULTURAL", "AGRICULTURE",
    "ARTS", "DEGREE", "JUNIOR", "AUTONOMOUS"
}

WORD_VARIANTS = {
    "INST": "INSTITUTE",
    "INSTT": "INSTITUTE",
    "INSTTUTE": "INSTITUTE",
    "COLL": "COLLEGE",
    "COLLGE": "COLLEGE",
    "ENGG": "ENGINEERING",
    "ENG": "ENGINEERING",
    "TECH": "TECHNOLOGY",
    "SCI": "SCIENCE",
    "CHALAMAYYA": "CHALAMAIAH",
    "MADANAPALLI": "MADANAPALLY",
    "WELFARE": "WELLFARE",
}

# These are common abbreviations for matching names.
# They do not by themselves establish that two colleges are identical.
SHORT_FORMS = {
    "RVR": "R V R",
    "J C": "J C",
    "JNTUK": "JNTUK",
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_name(value):
    """Normalize common spelling, punctuation and abbreviation differences."""
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()
    text = text.replace("&", " AND ")
    text = text.replace("’", "'")
    text = text.replace("'", "")
    text = re.sub(r"[^A-Z0-9]+", " ", text)

    words = []

    for word in text.split():
        # Correct common misspellings first.
        word = WORD_VARIANTS.get(word, word)
        words.append(word)

    return " ".join(words)


def compact_name(value):
    """Return a punctuation-free name for compact comparison."""
    return re.sub(r"[^A-Z0-9]", "", normalize_name(value))


def meaningful_tokens(value):
    """Return name tokens, excluding generic institution words."""
    tokens = normalize_name(value).split()
    return [t for t in tokens if t not in GENERIC_WORDS]


def identity_tokens(value):
    """Words that can distinguish different types of institutions."""
    return {
        token
        for token in normalize_name(value).split()
        if token in IDENTITY_WORDS
    }


def acronym(value):
    """Build initials from meaningful tokens."""
    tokens = meaningful_tokens(value)

    if not tokens:
        return ""

    return "".join(token[0] for token in tokens if token)


def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_similarity(a, b):
    ta = set(meaningful_tokens(a))
    tb = set(meaningful_tokens(b))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def location_similarity(a, b):
    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return similarity(a, b)


def has_identity_conflict(current_name, official_name):
    """
    Prevent obvious mismatches such as an engineering college being
    matched to a university or a women's college.
    """
    current = identity_tokens(current_name)
    official = identity_tokens(official_name)

    # If one name explicitly says "WOMEN" and the other does not,
    # require manual review rather than treating them as identical.
    for word in {"UNIVERSITY", "WOMEN", "WOMENS", "PHARMACY",
                 "POLYTECHNIC", "MEDICAL", "DENTAL", "LAW",
                 "ARCHITECTURE", "AGRICULTURAL", "AGRICULTURE",
                 "ARTS", "DEGREE", "JUNIOR"}:
        if (word in current) != (word in official):
            return True

    return False


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            f"Current dataset not found: {CURRENT_FILE}"
        )

    if not OFFICIAL_FILE.exists():
        raise FileNotFoundError(
            f"Official normalized file not found: {OFFICIAL_FILE}"
        )

    current = pd.read_csv(CURRENT_FILE)
    official = pd.read_csv(OFFICIAL_FILE)

    required_current = {"College Name", "Location"}
    required_official = {
        "College Name", "Location", "College Code", "Branch"
    }

    missing_current = required_current - set(current.columns)
    missing_official = required_official - set(official.columns)

    if missing_current:
        raise ValueError(
            f"Current dataset is missing columns: {sorted(missing_current)}"
        )

    if missing_official:
        raise ValueError(
            f"Official dataset is missing columns: {sorted(missing_official)}"
        )

    current = current.copy()
    official = official.copy()

    current["College Name"] = current["College Name"].fillna("").astype(str)
    current["Location"] = current["Location"].fillna("").astype(str)

    official["College Name"] = official["College Name"].fillna("").astype(str)
    official["Location"] = official["Location"].fillna("").astype(str)
    official["College Code"] = official["College Code"].fillna("").astype(str)
    official["Branch"] = official["Branch"].fillna("").astype(str)

    current["Normalized Name"] = current["College Name"].apply(normalize_name)
    current["Compact Name"] = current["College Name"].apply(compact_name)

    official["Normalized Name"] = official["College Name"].apply(normalize_name)
    official["Compact Name"] = official["College Name"].apply(compact_name)

    # Score each distinct official college once, but preserve its branches.
    official_colleges = (
        official[
            [
                "College Name", "Location", "College Code",
                "Normalized Name", "Compact Name"
            ]
        ]
        .drop_duplicates(subset=["College Code", "College Name", "Location"])
        .reset_index(drop=True)
    )

    return current, official, official_colleges


# =========================================================
# MATCHING
# =========================================================

def score_candidate(current_row, official_row):
    current_name = current_row["College Name"]
    official_name = official_row["College Name"]

    current_norm = current_row["Normalized Name"]
    official_norm = official_row["Normalized Name"]

    current_compact = current_row["Compact Name"]
    official_compact = official_row["Compact Name"]

    loc_score = location_similarity(
        current_row["Location"],
        official_row["Location"]
    )

    name_score = similarity(current_norm, official_norm)
    compact_score = similarity(current_compact, official_compact)
    tokens_score = token_similarity(current_name, official_name)

    current_acronym = acronym(current_name)
    official_acronym = acronym(official_name)

    acronym_score = similarity(current_acronym, official_acronym)

    conflict = has_identity_conflict(current_name, official_name)

    # Name is the main signal. Location helps distinguish colleges
    # with similar names, but cannot override a clear identity conflict.
    final_score = (
        0.40 * name_score
        + 0.25 * compact_score
        + 0.20 * tokens_score
        + 0.10 * loc_score
        + 0.05 * acronym_score
    )

    if conflict:
        final_score = min(final_score, 0.49)

    return {
        "Name Score": round(name_score, 4),
        "Compact Score": round(compact_score, 4),
        "Token Score": round(tokens_score, 4),
        "Location Score": round(loc_score, 4),
        "Acronym Score": round(acronym_score, 4),
        "Final Score": round(final_score, 4),
        "Identity Conflict": conflict,
    }


def find_candidates(current_row, official_colleges):
    candidates = []

    for _, official_row in official_colleges.iterrows():
        scores = score_candidate(current_row, official_row)

        candidates.append({
            "Official College Name": official_row["College Name"],
            "Official College Code": official_row["College Code"],
            "Official Location": official_row["Location"],
            **scores,
        })

    candidates.sort(
        key=lambda x: (
            x["Final Score"],
            x["Location Score"],
            x["Compact Score"]
        ),
        reverse=True
    )

    return candidates


def classify_match(current_row, candidates):
    if not candidates:
        return "UNMATCHED", "No official candidates found"

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    current_name = current_row["College Name"]
    current_location = normalize_name(current_row["Location"])

    best_name = best["Official College Name"]
    best_location = normalize_name(best["Official Location"])

    # SAFE only for exact normalized name + exact location,
    # with no identity conflict and no competing exact match.
    exact_name = normalize_name(current_name) == normalize_name(best_name)
    exact_location = bool(current_location) and current_location == best_location

    competing_exact = False

    if exact_name and exact_location:
        for candidate in candidates[1:]:
            if (
                normalize_name(candidate["Official College Name"])
                == normalize_name(current_name)
                and normalize_name(candidate["Official Location"])
                == current_location
            ):
                competing_exact = True
                break

    if (
        exact_name
        and exact_location
        and not best["Identity Conflict"]
        and not competing_exact
    ):
        return "SAFE", "Exact normalized name and exact location"

    # Exact compact name + location can be strong, but it is still
    # manual review until checked by a person.
    exact_compact = (
        compact_name(current_name) == compact_name(best_name)
    )

    if (
        exact_compact
        and exact_location
        and not best["Identity Conflict"]
    ):
        return (
            "MANUAL_REVIEW",
            "Exact compact name and location; verify official identity"
        )

    margin = (
        best["Final Score"] - second["Final Score"]
        if second else best["Final Score"]
    )

    if best["Identity Conflict"]:
        return "MANUAL_REVIEW", "Institution type/name conflict"

    if best["Final Score"] >= 0.78 and margin >= 0.08:
        return (
            "MANUAL_REVIEW",
            "Strong candidate; confirm identity before mapping"
        )

    if best["Final Score"] >= 0.55:
        return "MANUAL_REVIEW", "Possible match; human verification required"

    return "UNMATCHED", "No sufficiently reliable candidate"


# =========================================================
# MAIN REPORT
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current, official, official_colleges = load_data()

    print("\nAP EAPCET COLLEGE MAPPING REVIEW - VERSION 5.2")
    print("-" * 58)
    print(f"Current dataset rows: {len(current)}")
    print(f"Current unique colleges: {current['College Name'].nunique()}")
    print(f"Official normalized rows: {len(official)}")
    print(f"Unique official colleges: {len(official_colleges)}")
    print("\nMatching colleges. This does not change the dataset or database.\n")

    results = []

    for index, row in current.iterrows():
        candidates = find_candidates(row, official_colleges)
        status, reason = classify_match(row, candidates)

        best = candidates[0] if candidates else {}
        second = candidates[1] if len(candidates) > 1 else {}

        result = {
            "Current College Name": row["College Name"],
            "Current Location": row["Location"],
            "Status": status,
            "Reason": reason,
            "Best Official College Name": best.get("Official College Name", ""),
            "Best Official College Code": best.get("Official College Code", ""),
            "Best Official Location": best.get("Official Location", ""),
            "Final Score": best.get("Final Score", 0.0),
            "Name Score": best.get("Name Score", 0.0),
            "Compact Score": best.get("Compact Score", 0.0),
            "Token Score": best.get("Token Score", 0.0),
            "Location Score": best.get("Location Score", 0.0),
            "Acronym Score": best.get("Acronym Score", 0.0),
            "Second Candidate Name": second.get("Official College Name", ""),
            "Second Candidate Code": second.get("Official College Code", ""),
            "Second Candidate Location": second.get("Official Location", ""),
            "Second Candidate Score": second.get("Final Score", 0.0),
        }

        results.append(result)

        if (index + 1) % 25 == 0 or index + 1 == len(current):
            print(f"Processed {index + 1}/{len(current)}")

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 58)
    print("V5.2 SUMMARY")
    print("=" * 58)

    counts = report["Status"].value_counts()

    for status in ["SAFE", "MANUAL_REVIEW", "UNMATCHED"]:
        print(f"{status:16} {int(counts.get(status, 0))}")

    print(f"{'TOTAL':16} {len(report)}")
    print(f"\nReport saved to:\n{OUTPUT_FILE}")

    print("\nExamples requiring manual review:")
    manual = report[report["Status"] == "MANUAL_REVIEW"]

    if manual.empty:
        print("None")
    else:
        for _, row in manual.head(12).iterrows():
            print(f"\nCurrent:  {row['Current College Name']}")
            print(f"Location: {row['Current Location']}")
            print(
                f"Candidate: {row['Best Official College Name']} "
                f"({row['Best Official College Code']})"
            )
            print(f"Official location: {row['Best Official Location']}")
            print(f"Score: {row['Final Score']}")
            print(f"Reason: {row['Reason']}")

    print("\nExamples marked unmatched:")
    unmatched = report[report["Status"] == "UNMATCHED"]

    if unmatched.empty:
        print("None")
    else:
        for _, row in unmatched.head(12).iterrows():
            print(f"\nCurrent:  {row['Current College Name']}")
            print(f"Location: {row['Current Location']}")
            print(
                f"Best candidate: {row['Best Official College Name']} "
                f"({row['Best Official College Code']})"
            )
            print(f"Score: {row['Final Score']}")

    print("\nIMPORTANT:")
    print("- SAFE means exact normalized name and location, not government-certified.")
    print("- MANUAL_REVIEW must be checked before mapping.")
    print("- UNMATCHED must not be forced to a candidate.")
    print("- PostgreSQL, datasets and the ML model were not modified.")


if __name__ == "__main__":
    main()