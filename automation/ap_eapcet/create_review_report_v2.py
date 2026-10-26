import os
import re
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

OFFICIAL_FILE = (
    "automation/output/ap_eapcet_normalized.csv"
)

CURRENT_FILE = (
    "dataset/eamcet_colleges.csv"
)

OUTPUT_FILE = (
    "automation/output/ap_eapcet_review_report_v2.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_CANDIDATES = 3


# Generic words that occur in many college names.
# These are NOT used when calculating distinctive-token
# similarity.
GENERIC_WORDS = {
    "THE",
    "AND",
    "OF",
    "INSTITUTE",
    "INSTITUTES",
    "COLLEGE",
    "COLLEGES",
    "ENGINEERING",
    "ENGINEERINGS",
    "TECHNOLOGY",
    "TECHNOLOGIES",
    "SCIENCE",
    "SCIENCES",
    "STUDIES",
    "EDUCATION",
    "EDUCATIONAL",
    "UNIVERSITY",
    "UNIVERSITIES",
    "AUTONOMOUS",
    "CAMPUS",
}


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(name):

    if pd.isna(name):
        return ""

    name = str(name).upper().strip()

    replacements = {
        "&": " AND ",

        "INST.": " INSTITUTE ",
        "INST": " INSTITUTE ",

        "COLL.": " COLLEGE ",
        "COLL": " COLLEGE ",

        "ENGG.": " ENGINEERING ",
        "ENGG": " ENGINEERING ",

        "TECH.": " TECHNOLOGY ",
        "TECH": " TECHNOLOGY ",

        "UNIV.": " UNIVERSITY ",
        "UNIV": " UNIVERSITY ",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Replace punctuation with spaces.
    name = re.sub(
        r"[^A-Z0-9 ]+",
        " ",
        name
    )

    # Collapse spaces.
    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


# ============================================================
# TOKEN EXTRACTION
# ============================================================

def get_tokens(normalized_name):

    if not normalized_name:
        return set()

    tokens = normalized_name.split()

    return {
        token
        for token in tokens
        if token not in GENERIC_WORDS
        and len(token) >= 2
    }


# ============================================================
# DISTINCTIVE TOKEN SCORE
# ============================================================

def distinctive_token_score(
    current_tokens,
    official_tokens
):

    if not current_tokens or not official_tokens:
        return 0.0

    intersection = (
        current_tokens
        & official_tokens
    )

    if not intersection:
        return 0.0

    # Jaccard similarity
    union = (
        current_tokens
        | official_tokens
    )

    jaccard = (
        len(intersection)
        / len(union)
    )

    # Coverage from the current college name.
    current_coverage = (
        len(intersection)
        / len(current_tokens)
    )

    # Coverage from the official college name.
    official_coverage = (
        len(intersection)
        / len(official_tokens)
    )

    # Give more importance to matching
    # distinctive words than generic words.
    score = (
        0.45 * jaccard
        + 0.35 * current_coverage
        + 0.20 * official_coverage
    )

    return score


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def character_similarity(
    current_name,
    official_name
):

    return SequenceMatcher(
        None,
        current_name,
        official_name
    ).ratio()


# ============================================================
# COMBINED SCORE
# ============================================================

def calculate_score(
    current_normalized,
    official_normalized,
    current_tokens,
    official_tokens
):

    token_score = distinctive_token_score(
        current_tokens,
        official_tokens
    )

    char_score = character_similarity(
        current_normalized,
        official_normalized
    )

    # Distinctive tokens are more important.
    #
    # Token score: 70%
    # Character score: 30%
    combined = (
        0.70 * token_score
        + 0.30 * char_score
    )

    return (
        token_score,
        char_score,
        combined
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AP EAPCET COLLEGE REVIEW REPORT - VERSION 2")
print("=" * 70)

print()
print("Loading official AP EAPCET data...")

official = pd.read_csv(
    OFFICIAL_FILE
)

print(
    f"Official records loaded: {len(official)}"
)


print()
print("Loading current EAMCET data...")

current = pd.read_csv(
    CURRENT_FILE
)

print(
    f"Current records loaded: {len(current)}"
)


# ============================================================
# UNIQUE COLLEGES
# ============================================================

official_colleges = (
    official[
        [
            "College Name",
            "College Code"
        ]
    ]
    .dropna(
        subset=["College Name"]
    )
    .drop_duplicates(
        subset=["College Code"]
    )
    .copy()
)


current_colleges = (
    current[
        [
            "College Name"
        ]
    ]
    .dropna(
        subset=["College Name"]
    )
    .drop_duplicates()
    .copy()
)


print()
print(
    f"Unique official colleges: "
    f"{len(official_colleges)}"
)

print(
    f"Unique current colleges : "
    f"{len(current_colleges)}"
)


# ============================================================
# NORMALIZATION + TOKENS
# ============================================================

official_colleges[
    "Normalized Name"
] = official_colleges[
    "College Name"
].apply(
    normalize_name
)

official_colleges[
    "Tokens"
] = official_colleges[
    "Normalized Name"
].apply(
    get_tokens
)


current_colleges[
    "Normalized Name"
] = current_colleges[
    "College Name"
].apply(
    normalize_name
)

current_colleges[
    "Tokens"
] = current_colleges[
    "Normalized Name"
].apply(
    get_tokens
)


# ============================================================
# EXACT MATCH LOOKUP
# ============================================================

official_lookup = {}

for _, row in official_colleges.iterrows():

    normalized = row[
        "Normalized Name"
    ]

    if not normalized:
        continue

    official_lookup.setdefault(
        normalized,
        []
    ).append(row)


# ============================================================
# CREATE REPORT
# ============================================================

results = []


for number, (_, current_row) in enumerate(
    current_colleges.iterrows(),
    start=1
):

    current_name = current_row[
        "College Name"
    ]

    current_normalized = current_row[
        "Normalized Name"
    ]

    current_tokens = current_row[
        "Tokens"
    ]


    print(
        f"\rProcessing "
        f"{number}/{len(current_colleges)}: "
        f"{str(current_name)[:55]}",
        end=""
    )


    # ========================================================
    # EXACT MATCH
    # ========================================================

    exact_matches = official_lookup.get(
        current_normalized,
        []
    )


    if len(exact_matches) == 1:

        official_row = exact_matches[0]

        results.append(
            {
                "Current College": current_name,

                "Match Category": "EXACT",

                "Candidate 1": official_row[
                    "College Name"
                ],

                "Code 1": official_row[
                    "College Code"
                ],

                "Token Score 1": 1.0000,

                "Character Score 1": 1.0000,

                "Combined Score 1": 1.0000,

                "Candidate 2": "",
                "Code 2": "",
                "Token Score 2": "",
                "Character Score 2": "",
                "Combined Score 2": "",

                "Candidate 3": "",
                "Code 3": "",
                "Token Score 3": "",
                "Character Score 3": "",
                "Combined Score 3": "",

                "Score Margin": "",

                "Review Status": "SAFE_EXACT",

                "Suggested Action": "ACCEPT"
            }
        )

        continue


    # ========================================================
    # FUZZY CANDIDATES
    # ========================================================

    candidates = []


    for _, official_row in (
        official_colleges.iterrows()
    ):

        official_normalized = (
            official_row[
                "Normalized Name"
            ]
        )

        official_tokens = (
            official_row[
                "Tokens"
            ]
        )

        if not official_normalized:
            continue


        (
            token_score,
            char_score,
            combined_score
        ) = calculate_score(
            current_normalized,
            official_normalized,
            current_tokens,
            official_tokens
        )


        candidates.append(
            {
                "name": official_row[
                    "College Name"
                ],

                "code": official_row[
                    "College Code"
                ],

                "token_score": token_score,

                "char_score": char_score,

                "combined_score": combined_score
            }
        )


    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda x: (
            x["combined_score"],
            x["token_score"],
            x["char_score"]
        ),
        reverse=True
    )


    top = candidates[
        :TOP_CANDIDATES
    ]


    # ========================================================
    # BEST / SECOND SCORE
    # ========================================================

    if top:

        best_score = top[0][
            "combined_score"
        ]

    else:

        best_score = 0.0


    if len(top) >= 2:

        second_score = top[1][
            "combined_score"
        ]

    else:

        second_score = 0.0


    margin = (
        best_score
        - second_score
    )


    # ========================================================
    # CANDIDATE VALUES
    # ========================================================

    candidate_values = []


    for candidate in top:

        candidate_values.extend(
            [
                candidate["name"],
                candidate["code"],
                round(
                    candidate["token_score"],
                    4
                ),
                round(
                    candidate["char_score"],
                    4
                ),
                round(
                    candidate["combined_score"],
                    4
                )
            ]
        )


    while len(candidate_values) < 15:

        candidate_values.extend(
            [
                "",
                "",
                "",
                "",
                ""
            ]
        )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if best_score >= 0.80:

        review_status = (
            "STRONG_FUZZY_REVIEW"
        )

    elif best_score >= 0.65:

        review_status = (
            "MEDIUM_FUZZY_REVIEW"
        )

    elif best_score >= 0.50:

        review_status = (
            "WEAK_FUZZY_REVIEW"
        )

    else:

        review_status = (
            "LIKELY_NO_MATCH"
        )


    # ========================================================
    # AMBIGUOUS DETECTION
    # ========================================================

    # If the best and second candidate are
    # very close, human review is required.

    if (
        len(top) >= 2
        and margin < 0.08
        and best_score >= 0.50
    ):

        review_status = (
            "AMBIGUOUS_REVIEW"
        )


    # ========================================================
    # SAVE
    # ========================================================

    results.append(
        {
            "Current College": current_name,

            "Match Category": "FUZZY",

            "Candidate 1": candidate_values[0],
            "Code 1": candidate_values[1],
            "Token Score 1": candidate_values[2],
            "Character Score 1": candidate_values[3],
            "Combined Score 1": candidate_values[4],

            "Candidate 2": candidate_values[5],
            "Code 2": candidate_values[6],
            "Token Score 2": candidate_values[7],
            "Character Score 2": candidate_values[8],
            "Combined Score 2": candidate_values[9],

            "Candidate 3": candidate_values[10],
            "Code 3": candidate_values[11],
            "Token Score 3": candidate_values[12],
            "Character Score 3": candidate_values[13],
            "Combined Score 3": candidate_values[14],

            "Score Margin": round(
                margin,
                4
            ),

            "Review Status": review_status,

            "Suggested Action": "MANUAL_REVIEW"
        }
    )


print()


# ============================================================
# DATAFRAME
# ============================================================

report = pd.DataFrame(
    results
)


# ============================================================
# DUPLICATE OFFICIAL CODES
# ============================================================

for column in [
    "Code 1",
    "Code 2",
    "Code 3"
]:

    valid_codes = report[
        report[column].notna()
        & (
            report[column]
            != ""
        )
    ][column]

    counts = valid_codes.value_counts()

    duplicate_codes = set(
        counts[
            counts > 1
        ].index
    )


    for index in report.index:

        code = report.at[
            index,
            column
        ]

        if code in duplicate_codes:

            current_status = report.at[
                index,
                "Review Status"
            ]

            if current_status == "SAFE_EXACT":

                report.at[
                    index,
                    "Review Status"
                ] = (
                    "DUPLICATE_TARGET_REVIEW"
                )

                report.at[
                    index,
                    "Suggested Action"
                ] = (
                    "MANUAL_REVIEW"
                )


# ============================================================
# STATUS SORTING
# ============================================================

status_order = {
    "DUPLICATE_TARGET_REVIEW": 0,
    "AMBIGUOUS_REVIEW": 1,
    "STRONG_FUZZY_REVIEW": 2,
    "MEDIUM_FUZZY_REVIEW": 3,
    "WEAK_FUZZY_REVIEW": 4,
    "LIKELY_NO_MATCH": 5,
    "SAFE_EXACT": 6
}


report["_sort"] = (
    report[
        "Review Status"
    ]
    .map(status_order)
    .fillna(99)
)


report = report.sort_values(
    by=[
        "_sort",
        "Combined Score 1"
    ],
    ascending=[
        True,
        False
    ]
)


report = report.drop(
    columns=["_sort"]
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


report.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VERSION 2 REVIEW SUMMARY")
print("=" * 70)

print()

print(
    report[
        "Review Status"
    ]
    .value_counts()
    .to_string()
)


print()

print(
    f"Total colleges: {len(report)}"
)


for status in [
    "SAFE_EXACT",
    "STRONG_FUZZY_REVIEW",
    "MEDIUM_FUZZY_REVIEW",
    "WEAK_FUZZY_REVIEW",
    "AMBIGUOUS_REVIEW",
    "DUPLICATE_TARGET_REVIEW",
    "LIKELY_NO_MATCH"
]:

    count = (
        report[
            "Review Status"
        ]
        == status
    ).sum()

    print(
        f"{status}: {count}"
    )


# ============================================================
# STRONG FUZZY EXAMPLES
# ============================================================

print()
print("=" * 70)
print("VERSION 2 STRONG FUZZY EXAMPLES")
print("=" * 70)


strong = report[
    report[
        "Review Status"
    ]
    == "STRONG_FUZZY_REVIEW"
].head(20)


if strong.empty:

    print(
        "No strong fuzzy matches found."
    )

else:

    for _, row in strong.iterrows():

        print()
        print(
            f"Current: "
            f"{row['Current College']}"
        )

        print(
            f"  1. "
            f"{row['Candidate 1']} "
            f"| {row['Code 1']} "
            f"| Token={row['Token Score 1']} "
            f"| Char={row['Character Score 1']} "
            f"| Final={row['Combined Score 1']}"
        )

        print(
            f"  2. "
            f"{row['Candidate 2']} "
            f"| {row['Code 2']} "
            f"| Token={row['Token Score 2']} "
            f"| Char={row['Character Score 2']} "
            f"| Final={row['Combined Score 2']}"
        )

        print(
            f"  3. "
            f"{row['Candidate 3']} "
            f"| {row['Code 3']} "
            f"| Token={row['Token Score 3']} "
            f"| Char={row['Character Score 3']} "
            f"| Final={row['Combined Score 3']}"
        )


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("VERSION 2 REPORT SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print()
print(
    "NO DATABASE WAS MODIFIED."
)

print(
    "NO CURRENT DATASET WAS MODIFIED."
)

print(
    "NO ML MODEL WAS MODIFIED."
)