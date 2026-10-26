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
    "automation/output/ap_eapcet_review_report.csv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of official candidates to show
TOP_CANDIDATES = 3


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

    # Remove punctuation
    name = re.sub(
        r"[^A-Z0-9 ]+",
        " ",
        name
    )

    # Remove extra spaces
    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


# ============================================================
# SIMILARITY
# ============================================================

def similarity(name1, name2):

    return SequenceMatcher(
        None,
        name1,
        name2
    ).ratio()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AP EAPCET COLLEGE REVIEW REPORT")
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
    f"Unique official colleges: {len(official_colleges)}"
)

print(
    f"Unique current colleges : {len(current_colleges)}"
)


# ============================================================
# NORMALIZED NAMES
# ============================================================

official_colleges[
    "Normalized Name"
] = official_colleges[
    "College Name"
].apply(
    normalize_name
)

current_colleges[
    "Normalized Name"
] = current_colleges[
    "College Name"
].apply(
    normalize_name
)


# ============================================================
# CREATE OFFICIAL EXACT LOOKUP
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
# CREATE REVIEW REPORT
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


    print(
        f"\rProcessing {number}/{len(current_colleges)}: "
        f"{current_name[:55]}",
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

                "Score 1": 1.0000,

                "Candidate 2": "",

                "Code 2": "",

                "Score 2": "",

                "Candidate 3": "",

                "Code 3": "",

                "Score 3": "",

                "Review Status": "SAFE_EXACT",

                "Suggested Action": "ACCEPT"
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

        if not official_normalized:
            continue

        score = similarity(
            current_normalized,
            official_normalized
        )

        candidates.append(
            {
                "name": official_row[
                    "College Name"
                ],

                "code": official_row[
                    "College Code"
                ],

                "score": score
            }
        )


    # Sort highest similarity first

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # Keep top candidates

    top = candidates[
        :TOP_CANDIDATES
    ]


    # ========================================================
    # FILL CANDIDATES
    # ========================================================

    candidate_values = []


    for candidate in top:

        candidate_values.extend(
            [
                candidate["name"],
                candidate["code"],
                round(
                    candidate["score"],
                    4
                )
            ]
        )


    while len(candidate_values) < 9:

        candidate_values.extend(
            [
                "",
                "",
                ""
            ]
        )


    # ========================================================
    # BEST SCORE
    # ========================================================

    if top:

        best_score = top[0]["score"]

    else:

        best_score = 0


    # ========================================================
    # SECOND SCORE
    # ========================================================

    if len(top) >= 2:

        second_score = top[1]["score"]

    else:

        second_score = 0


    # ========================================================
    # SCORE DIFFERENCE
    # ========================================================

    margin = (
        best_score - second_score
    )


    # ========================================================
    # REVIEW CLASSIFICATION
    # ========================================================

    if best_score >= 0.95:

        review_status = (
            "STRONG_FUZZY_REVIEW"
        )

    elif best_score >= 0.85:

        review_status = (
            "MEDIUM_FUZZY_REVIEW"
        )

    elif best_score >= 0.75:

        review_status = (
            "WEAK_FUZZY_REVIEW"
        )

    else:

        review_status = (
            "LIKELY_NO_MATCH"
        )


    # ========================================================
    # SPECIAL CASE:
    # BEST AND SECOND ARE VERY CLOSE
    # ========================================================

    if (
        len(top) >= 2
        and margin < 0.03
    ):

        review_status = (
            "AMBIGUOUS_REVIEW"
        )


    # ========================================================
    # SUGGESTED ACTION
    # ========================================================

    # Important:
    # No fuzzy match is automatically accepted.

    suggested_action = (
        "MANUAL_REVIEW"
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

            "Score 1": candidate_values[2],

            "Candidate 2": candidate_values[3],

            "Code 2": candidate_values[4],

            "Score 2": candidate_values[5],

            "Candidate 3": candidate_values[6],

            "Code 3": candidate_values[7],

            "Score 3": candidate_values[8],

            "Review Status": review_status,

            "Suggested Action": suggested_action
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

    counts = (
        report[
            report[column].notna()
            & (
                report[column]
                != ""
            )
        ][column]
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

            current_status = report.at[
                index,
                "Review Status"
            ]

            if (
                current_status
                == "SAFE_EXACT"
            ):

                report.at[
                    index,
                    "Review Status"
                ] = "DUPLICATE_TARGET_REVIEW"


# ============================================================
# SORT REPORT
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
        "Score 1"
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
# SAVE CSV
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
print("REVIEW REPORT SUMMARY")
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

print(
    "Safe exact:",
    (
        report[
            "Review Status"
        ]
        == "SAFE_EXACT"
    ).sum()
)

print(
    "Strong fuzzy:",
    (
        report[
            "Review Status"
        ]
        == "STRONG_FUZZY_REVIEW"
    ).sum()
)

print(
    "Ambiguous:",
    (
        report[
            "Review Status"
        ]
        == "AMBIGUOUS_REVIEW"
    ).sum()
)

print(
    "Likely no match:",
    (
        report[
            "Review Status"
        ]
        == "LIKELY_NO_MATCH"
    ).sum()
)


# ============================================================
# PRINT STRONG FUZZY EXAMPLES
# ============================================================

print()
print("=" * 70)
print("STRONG FUZZY REVIEW EXAMPLES")
print("=" * 70)

strong = report[
    report[
        "Review Status"
    ]
    == "STRONG_FUZZY_REVIEW"
].head(20)


if strong.empty:

    print(
        "No strong fuzzy matches."
    )

else:

    print()

    for _, row in strong.iterrows():

        print(
            f"\nCurrent: "
            f"{row['Current College']}"
        )

        print(
            f"  1. "
            f"{row['Candidate 1']} "
            f"| {row['Code 1']} "
            f"| {row['Score 1']}"
        )

        print(
            f"  2. "
            f"{row['Candidate 2']} "
            f"| {row['Code 2']} "
            f"| {row['Score 2']}"
        )

        print(
            f"  3. "
            f"{row['Candidate 3']} "
            f"| {row['Code 3']} "
            f"| {row['Score 3']}"
        )


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 70)
print("REPORT SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)

print()
print(
    "NO DATABASE WAS MODIFIED."
)

print(
    "NO DATASET WAS MODIFIED."
)