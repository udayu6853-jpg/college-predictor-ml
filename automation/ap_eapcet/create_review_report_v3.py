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
    "automation/output/ap_eapcet_review_report_v3.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_CANDIDATES = 3


# ============================================================
# OFFICIAL ENGINEERING BRANCHES
# ============================================================
#
# These are AP EAPCET branch codes that are relevant for
# engineering-college matching.
#
# We are NOT changing the final application branch mapping here.
# This list is only used to improve college-name matching.
# ============================================================

ENGINEERING_BRANCHES = {
    "AI",
    "AID",
    "AIM",
    "ASE",
    "AUT",
    "BDT",
    "BIO",
    "CAD",
    "CAI",
    "CBA",
    "CBC",
    "CCC",
    "CDA",
    "CHE",
    "CIA",
    "CIC",
    "CIT",
    "CIV",
    "CN",
    "CS",
    "CSB",
    "CSBS",
    "CSC",
    "CSD",
    "CSE",
    "CSEB",
    "CSER",
    "CSG",
    "CSM",
    "CSO",
    "CSS",
    "CST",
    "CSW",
    "DS",
    "EBM",
    "ECA",
    "ECE",
    "ECES",
    "ECM",
    "ECT",
    "ECV",
    "EEE",
    "EIE",
    "EII",
    "EVT",
    "FDE",
    "FDT",
    "GDT",
    "GIN",
    "INF",
    "IOT",
    "IST",
    "MAD",
    "MAU",
    "MET",
    "MII",
    "MIN",
    "MMM",
    "MMT",
    "MRB",
    "NAM",
    "PEE",
    "PET",
    "PHD",
    "PHM",
    "RBT",
    "SWE",
}


# ============================================================
# GENERIC WORDS
# ============================================================
#
# These words are common in almost every college name.
# They receive very little importance during matching.
# ============================================================

GENERIC_WORDS = {
    "THE",
    "AND",
    "OF",
    "IN",
    "AT",

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

    "INSTITUTION",
    "INSTITUTIONS",

    "SCHOOL",
    "SCHOOLS",

    "GROUP",

    "OF",

    "FOR",
}


# ============================================================
# COMMON ABBREVIATION NORMALIZATION
# ============================================================

REPLACEMENTS = {
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

    "SCI.": " SCIENCE ",
    "SCI": " SCIENCE ",
}


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(name):

    if pd.isna(name):
        return ""

    name = str(name).upper().strip()

    for old, new in REPLACEMENTS.items():

        name = name.replace(
            old,
            new
        )

    # Remove punctuation.
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

def get_all_tokens(name):

    if not name:
        return set()

    return set(
        name.split()
    )


# ============================================================
# DISTINCTIVE TOKENS
# ============================================================

def get_distinctive_tokens(name):

    tokens = get_all_tokens(
        name
    )

    return {
        token
        for token in tokens
        if (
            token not in GENERIC_WORDS
            and len(token) >= 2
        )
    }


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
# TOKEN SIMILARITY
# ============================================================

def token_similarity(
    current_tokens,
    official_tokens
):

    if not current_tokens:
        return 0.0

    if not official_tokens:
        return 0.0


    intersection = (
        current_tokens
        & official_tokens
    )


    if not intersection:

        return 0.0


    union = (
        current_tokens
        | official_tokens
    )


    jaccard = (
        len(intersection)
        / len(union)
    )


    current_coverage = (
        len(intersection)
        / len(current_tokens)
    )


    official_coverage = (
        len(intersection)
        / len(official_tokens)
    )


    # Distinctive tokens are more important.
    score = (
        0.45 * jaccard
        + 0.35 * current_coverage
        + 0.20 * official_coverage
    )


    return score


# ============================================================
# DISTINCTIVE TOKEN COUNT
# ============================================================

def common_distinctive_tokens(
    current_tokens,
    official_tokens
):

    return sorted(
        current_tokens
        & official_tokens
    )


# ============================================================
# SPECIAL TOKEN BONUS
# ============================================================
#
# If a distinctive token is relatively unusual and matches,
# give a small bonus.
#
# This helps names such as:
#
# ANIL NEERUKONDA
# PACE
# VASAVI
# SUBBA REDDY
# RAMANUJAN
#
# while generic words have already been removed.
# ============================================================

def distinctive_bonus(
    current_tokens,
    official_tokens
):

    common = (
        current_tokens
        & official_tokens
    )

    if not common:
        return 0.0


    long_tokens = {
        token
        for token in common
        if len(token) >= 6
    }


    if not long_tokens:
        return 0.0


    # Maximum bonus: 0.10
    bonus = min(
        0.10,
        0.025 * len(long_tokens)
    )


    return bonus


# ============================================================
# COMBINED SCORE
# ============================================================

def calculate_match_scores(
    current_normalized,
    official_normalized,
    current_tokens,
    official_tokens
):

    token_score = token_similarity(
        current_tokens,
        official_tokens
    )


    char_score = character_similarity(
        current_normalized,
        official_normalized
    )


    bonus = distinctive_bonus(
        current_tokens,
        official_tokens
    )


    # Main score:
    #
    # 65% distinctive tokens
    # 25% character similarity
    # 10% distinctive-token bonus
    #
    combined = (
        0.65 * token_score
        + 0.25 * char_score
        + bonus
    )


    # Never exceed 1.0.
    combined = min(
        combined,
        1.0
    )


    return (
        token_score,
        char_score,
        bonus,
        combined
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print(
    "AP EAPCET COLLEGE REVIEW REPORT - VERSION 3"
)
print("=" * 70)


print()
print(
    "Loading official AP EAPCET data..."
)


official = pd.read_csv(
    OFFICIAL_FILE
)


print(
    f"Official records loaded: "
    f"{len(official)}"
)


print()
print(
    "Loading current EAMCET data..."
)


current = pd.read_csv(
    CURRENT_FILE
)


print(
    f"Current records loaded: "
    f"{len(current)}"
)


# ============================================================
# NORMALIZE OFFICIAL BRANCH CODE
# ============================================================

official["Branch"] = (
    official["Branch"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# FILTER OFFICIAL ENGINEERING RECORDS
# ============================================================

official_engineering = official[
    official["Branch"].isin(
        ENGINEERING_BRANCHES
    )
].copy()


print()
print(
    "Official engineering records: "
    f"{len(official_engineering)}"
)


# ============================================================
# UNIQUE OFFICIAL COLLEGES
# ============================================================

official_colleges = (
    official_engineering[
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


# ============================================================
# UNIQUE CURRENT COLLEGES
# ============================================================

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
    f"Unique official engineering colleges: "
    f"{len(official_colleges)}"
)


print(
    f"Unique current colleges: "
    f"{len(current_colleges)}"
)


# ============================================================
# NORMALIZE NAMES
# ============================================================

official_colleges[
    "Normalized Name"
] = (
    official_colleges[
        "College Name"
    ]
    .apply(
        normalize_name
    )
)


current_colleges[
    "Normalized Name"
] = (
    current_colleges[
        "College Name"
    ]
    .apply(
        normalize_name
    )
)


# ============================================================
# DISTINCTIVE TOKENS
# ============================================================

official_colleges[
    "Distinctive Tokens"
] = (
    official_colleges[
        "Normalized Name"
    ]
    .apply(
        get_distinctive_tokens
    )
)


current_colleges[
    "Distinctive Tokens"
] = (
    current_colleges[
        "Normalized Name"
    ]
    .apply(
        get_distinctive_tokens
    )
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
    ).append(
        row
    )


# ============================================================
# REPORT
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
        "Distinctive Tokens"
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

                "Candidate 1":
                    official_row[
                        "College Name"
                    ],

                "Code 1":
                    official_row[
                        "College Code"
                    ],

                "Common Tokens 1":
                    ", ".join(
                        sorted(
                            current_tokens
                            & official_row[
                                "Distinctive Tokens"
                            ]
                        )
                    ),

                "Token Score 1": 1.0000,

                "Character Score 1": 1.0000,

                "Bonus 1": 0.0000,

                "Final Score 1": 1.0000,

                "Candidate 2": "",
                "Code 2": "",
                "Common Tokens 2": "",
                "Token Score 2": "",
                "Character Score 2": "",
                "Bonus 2": "",
                "Final Score 2": "",

                "Candidate 3": "",
                "Code 3": "",
                "Common Tokens 3": "",
                "Token Score 3": "",
                "Character Score 3": "",
                "Bonus 3": "",
                "Final Score 3": "",

                "Score Margin": "",

                "Review Status":
                    "SAFE_EXACT",

                "Suggested Action":
                    "ACCEPT"
            }
        )

        continue


    # ========================================================
    # FUZZY MATCHING
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
                "Distinctive Tokens"
            ]
        )


        if not official_normalized:
            continue


        (
            token_score,
            char_score,
            bonus,
            final_score
        ) = calculate_match_scores(
            current_normalized,
            official_normalized,
            current_tokens,
            official_tokens
        )


        common_tokens = (
            common_distinctive_tokens(
                current_tokens,
                official_tokens
            )
        )


        candidates.append(
            {
                "name":
                    official_row[
                        "College Name"
                    ],

                "code":
                    official_row[
                        "College Code"
                    ],

                "token_score":
                    token_score,

                "char_score":
                    char_score,

                "bonus":
                    bonus,

                "final_score":
                    final_score,

                "common_tokens":
                    common_tokens
            }
        )


    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda x: (
            x["final_score"],
            x["token_score"],
            x["char_score"]
        ),
        reverse=True
    )


    top = candidates[
        :TOP_CANDIDATES
    ]


    # ========================================================
    # BEST SCORE
    # ========================================================

    if top:

        best_score = top[0][
            "final_score"
        ]

    else:

        best_score = 0.0


    # ========================================================
    # SECOND SCORE
    # ========================================================

    if len(top) >= 2:

        second_score = top[1][
            "final_score"
        ]

    else:

        second_score = 0.0


    score_margin = (
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

                ", ".join(
                    candidate[
                        "common_tokens"
                    ]
                ),

                round(
                    candidate[
                        "token_score"
                    ],
                    4
                ),

                round(
                    candidate[
                        "char_score"
                    ],
                    4
                ),

                round(
                    candidate[
                        "bonus"
                    ],
                    4
                ),

                round(
                    candidate[
                        "final_score"
                    ],
                    4
                )
            ]
        )


    # 7 fields per candidate
    # 3 candidates = 21 fields

    while len(candidate_values) < 21:

        candidate_values.extend(
            [
                "",
                "",
                "",
                "",
                "",
                "",
                ""
            ]
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if best_score >= 0.88:

        confidence = "HIGH"

    elif best_score >= 0.72:

        confidence = "MEDIUM"

    elif best_score >= 0.55:

        confidence = "LOW"

    else:

        confidence = "VERY_LOW"


    # ========================================================
    # REVIEW STATUS
    # ========================================================

    if best_score < 0.55:

        review_status = (
            "LIKELY_NO_MATCH"
        )

    elif (
        len(top) >= 2
        and score_margin < 0.06
    ):

        review_status = (
            "AMBIGUOUS_REVIEW"
        )

    elif best_score >= 0.88:

        review_status = (
            "HIGH_CONFIDENCE_REVIEW"
        )

    elif best_score >= 0.72:

        review_status = (
            "MEDIUM_CONFIDENCE_REVIEW"
        )

    else:

        review_status = (
            "LOW_CONFIDENCE_REVIEW"
        )


    # ========================================================
    # IMPORTANT:
    # FUZZY MATCHES ARE NEVER AUTO-ACCEPTED
    # ========================================================

    suggested_action = (
        "MANUAL_REVIEW"
    )


    # ========================================================
    # ADD RESULT
    # ========================================================

    results.append(
        {
            "Current College":
                current_name,

            "Match Category":
                "FUZZY",

            # Candidate 1
            "Candidate 1":
                candidate_values[0],

            "Code 1":
                candidate_values[1],

            "Common Tokens 1":
                candidate_values[2],

            "Token Score 1":
                candidate_values[3],

            "Character Score 1":
                candidate_values[4],

            "Bonus 1":
                candidate_values[5],

            "Final Score 1":
                candidate_values[6],

            # Candidate 2
            "Candidate 2":
                candidate_values[7],

            "Code 2":
                candidate_values[8],

            "Common Tokens 2":
                candidate_values[9],

            "Token Score 2":
                candidate_values[10],

            "Character Score 2":
                candidate_values[11],

            "Bonus 2":
                candidate_values[12],

            "Final Score 2":
                candidate_values[13],

            # Candidate 3
            "Candidate 3":
                candidate_values[14],

            "Code 3":
                candidate_values[15],

            "Common Tokens 3":
                candidate_values[16],

            "Token Score 3":
                candidate_values[17],

            "Character Score 3":
                candidate_values[18],

            "Bonus 3":
                candidate_values[19],

            "Final Score 3":
                candidate_values[20],

            "Score Margin":
                round(
                    score_margin,
                    4
                ),

            "Confidence":
                confidence,

            "Review Status":
                review_status,

            "Suggested Action":
                suggested_action
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
# DUPLICATE TARGET DETECTION
# ============================================================

code_columns = [
    "Code 1",
    "Code 2",
    "Code 3"
]


for column in code_columns:

    valid_codes = (
        report[
            report[column].notna()
            & (
                report[column]
                != ""
            )
        ][column]
    )


    counts = (
        valid_codes
        .value_counts()
    )


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

            status = report.at[
                index,
                "Review Status"
            ]


            if status == "SAFE_EXACT":

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
# SORTING
# ============================================================

status_order = {

    "DUPLICATE_TARGET_REVIEW": 0,

    "AMBIGUOUS_REVIEW": 1,

    "HIGH_CONFIDENCE_REVIEW": 2,

    "MEDIUM_CONFIDENCE_REVIEW": 3,

    "LOW_CONFIDENCE_REVIEW": 4,

    "LIKELY_NO_MATCH": 5,

    "SAFE_EXACT": 6
}


report["_sort"] = (
    report[
        "Review Status"
    ]
    .map(
        status_order
    )
    .fillna(99)
)


report = report.sort_values(
    by=[
        "_sort",
        "Final Score 1"
    ],
    ascending=[
        True,
        False
    ]
)


report = report.drop(
    columns=[
        "_sort"
    ]
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
print(
    "VERSION 3 REVIEW SUMMARY"
)
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
    f"Total colleges: "
    f"{len(report)}"
)


for status in [
    "SAFE_EXACT",
    "HIGH_CONFIDENCE_REVIEW",
    "MEDIUM_CONFIDENCE_REVIEW",
    "LOW_CONFIDENCE_REVIEW",
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
# HIGH-CONFIDENCE EXAMPLES
# ============================================================

print()
print("=" * 70)
print(
    "VERSION 3 HIGH-CONFIDENCE EXAMPLES"
)
print("=" * 70)


high = report[
    report[
        "Review Status"
    ]
    == "HIGH_CONFIDENCE_REVIEW"
].head(20)


if high.empty:

    print(
        "No high-confidence fuzzy matches."
    )

else:

    for _, row in high.iterrows():

        print()

        print(
            f"Current: "
            f"{row['Current College']}"
        )

        print(
            f"  1. "
            f"{row['Candidate 1']} "
            f"| {row['Code 1']} "
            f"| Tokens={row['Common Tokens 1']} "
            f"| Final={row['Final Score 1']}"
        )

        print(
            f"  2. "
            f"{row['Candidate 2']} "
            f"| {row['Code 2']} "
            f"| Tokens={row['Common Tokens 2']} "
            f"| Final={row['Final Score 2']}"
        )

        print(
            f"  3. "
            f"{row['Candidate 3']} "
            f"| {row['Code 3']} "
            f"| Tokens={row['Common Tokens 3']} "
            f"| Final={row['Final Score 3']}"
        )

        print(
            f"  Margin: "
            f"{row['Score Margin']}"
        )


# ============================================================
# AMBIGUOUS EXAMPLES
# ============================================================

print()
print("=" * 70)
print(
    "VERSION 3 AMBIGUOUS EXAMPLES"
)
print("=" * 70)


ambiguous = report[
    report[
        "Review Status"
    ]
    == "AMBIGUOUS_REVIEW"
].head(15)


if ambiguous.empty:

    print(
        "No ambiguous matches found."
    )

else:

    for _, row in ambiguous.iterrows():

        print()

        print(
            f"Current: "
            f"{row['Current College']}"
        )

        print(
            f"  1. "
            f"{row['Candidate 1']} "
            f"| {row['Code 1']} "
            f"| Final={row['Final Score 1']}"
        )

        print(
            f"  2. "
            f"{row['Candidate 2']} "
            f"| {row['Code 2']} "
            f"| Final={row['Final Score 2']}"
        )

        print(
            f"  3. "
            f"{row['Candidate 3']} "
            f"| {row['Code 3']} "
            f"| Final={row['Final Score 3']}"
        )

        print(
            f"  Margin: "
            f"{row['Score Margin']}"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print(
    "VERSION 3 REPORT SAVED"
)
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