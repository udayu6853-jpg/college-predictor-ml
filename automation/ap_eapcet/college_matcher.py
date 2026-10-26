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
    "automation/output/ap_eapcet_college_matching_report.csv"
)


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(name):
    """
    Convert college names into a comparable format.
    """

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
    name = re.sub(r"[^A-Z0-9 ]+", " ", name)

    # Normalize spaces
    name = re.sub(r"\s+", " ", name)

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
print("SAFE AP EAPCET COLLEGE MATCHING")
print("=" * 70)

official = pd.read_csv(
    OFFICIAL_FILE
)

current = pd.read_csv(
    CURRENT_FILE
)

print()
print(
    f"Official records: {len(official)}"
)

print(
    f"Current records : {len(current)}"
)


# ============================================================
# CREATE UNIQUE COLLEGE LISTS
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
    f"Official colleges: {len(official_colleges)}"
)

print(
    f"Current colleges : {len(current_colleges)}"
)


# ============================================================
# NORMALIZED NAMES
# ============================================================

official_colleges["Normalized Name"] = (
    official_colleges["College Name"]
    .apply(normalize_name)
)

current_colleges["Normalized Name"] = (
    current_colleges["College Name"]
    .apply(normalize_name)
)


# ============================================================
# OFFICIAL LOOKUP
# ============================================================

official_exact = {}

for _, row in official_colleges.iterrows():

    normalized = row["Normalized Name"]

    if normalized:

        official_exact.setdefault(
            normalized,
            []
        ).append(row)


# ============================================================
# MATCH CURRENT COLLEGES
# ============================================================

results = []


for _, current_row in current_colleges.iterrows():

    current_name = current_row[
        "College Name"
    ]

    current_normalized = current_row[
        "Normalized Name"
    ]


    # --------------------------------------------------------
    # Candidate list
    # --------------------------------------------------------

    candidates = []


    # ========================================================
    # 1. EXACT NORMALIZED MATCH
    # ========================================================

    exact_candidates = official_exact.get(
        current_normalized,
        []
    )

    if exact_candidates:

        for candidate in exact_candidates:

            candidates.append(
                {
                    "row": candidate,
                    "similarity": 1.0,
                    "match_type": "EXACT"
                }
            )


    # ========================================================
    # 2. FUZZY MATCH
    # ========================================================

    if not candidates:

        for _, official_row in (
            official_colleges.iterrows()
        ):

            official_name = (
                official_row[
                    "Normalized Name"
                ]
            )

            if not official_name:
                continue

            score = similarity(
                current_normalized,
                official_name
            )

            candidates.append(
                {
                    "row": official_row,
                    "similarity": score,
                    "match_type": "FUZZY"
                }
            )


    # ========================================================
    # SORT CANDIDATES
    # ========================================================

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    # ========================================================
    # NO CANDIDATE
    # ========================================================

    if not candidates:

        results.append(
            {
                "Current College": current_name,
                "Official College": "",
                "Official Code": "",
                "Match Type": "NO_MATCH",
                "Similarity": 0,
                "Candidate Rank": "",
                "Duplicate Target": "NO",
                "Confidence": "LOW",
                "Action": "MANUAL_REVIEW"
            }
        )

        continue


    # ========================================================
    # BEST + SECOND BEST
    # ========================================================

    best = candidates[0]

    best_row = best["row"]

    best_score = best["similarity"]

    second_score = (
        candidates[1]["similarity"]
        if len(candidates) > 1
        else 0
    )


    # Difference between first and second candidates
    margin = (
        best_score - second_score
    )


    # ========================================================
    # DETECT DUPLICATE TARGET
    # ========================================================

    duplicate_target = "NO"


    # We will calculate this later globally.
    # For now keep NO.
    # It will be updated after all matches.


    # ========================================================
    # CONFIDENCE / ACTION
    # ========================================================

    if best["match_type"] == "EXACT":

        confidence = "HIGH"
        action = "ACCEPT"

        match_type = "EXACT"


    elif best_score >= 0.92 and margin >= 0.03:

        confidence = "HIGH"
        action = "ACCEPT"

        match_type = "FUZZY_HIGH"


    elif best_score >= 0.85 and margin >= 0.05:

        confidence = "MEDIUM"
        action = "MANUAL_REVIEW"

        match_type = "FUZZY_MEDIUM"


    elif best_score >= 0.75:

        confidence = "LOW"
        action = "MANUAL_REVIEW"

        match_type = "FUZZY_LOW"


    else:

        confidence = "LOW"
        action = "MANUAL_REVIEW"

        match_type = "NO_MATCH"


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append(
        {
            "Current College": current_name,

            "Official College": best_row[
                "College Name"
            ],

            "Official Code": best_row[
                "College Code"
            ],

            "Match Type": match_type,

            "Similarity": round(
                best_score,
                4
            ),

            "Candidate Rank": 1,

            "Duplicate Target": duplicate_target,

            "Confidence": confidence,

            "Action": action
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

report = pd.DataFrame(
    results
)


# ============================================================
# DUPLICATE TARGET DETECTION
# ============================================================

target_counts = (
    report[
        report["Official Code"]
        != ""
    ]["Official Code"]
    .value_counts()
)


for index in report.index:

    code = report.at[
        index,
        "Official Code"
    ]

    if code == "":
        continue

    count = target_counts.get(
        code,
        0
    )

    if count > 1:

        report.at[
            index,
            "Duplicate Target"
        ] = "YES"

        # Any duplicated target must be reviewed.
        report.at[
            index,
            "Action"
        ] = "MANUAL_REVIEW"

        report.at[
            index,
            "Confidence"
        ] = "LOW"


# ============================================================
# SECOND-CANDIDATE CHECK
# ============================================================

# Recalculate candidates for suspicious cases.
# This gives us a useful review flag when two colleges
# have very similar scores.

for index in report.index:

    match_type = report.at[
        index,
        "Match Type"
    ]

    similarity_value = report.at[
        index,
        "Similarity"
    ]

    duplicate = report.at[
        index,
        "Duplicate Target"
    ]

    if duplicate == "YES":

        report.at[
            index,
            "Action"
        ] = "MANUAL_REVIEW"


# ============================================================
# SORT REPORT
# ============================================================

action_order = {
    "MANUAL_REVIEW": 0,
    "ACCEPT": 1
}

report["_sort"] = report[
    "Action"
].map(
    action_order
).fillna(99)

report = report.sort_values(
    by=[
        "_sort",
        "Confidence",
        "Similarity"
    ],
    ascending=[
        True,
        True,
        False
    ]
)

report = report.drop(
    columns=["_sort"]
)


# ============================================================
# SAVE REPORT
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
print("MATCHING SUMMARY")
print("=" * 70)

print()

print(
    report["Match Type"]
    .value_counts()
    .to_string()
)

print()

print(
    f"Total current colleges: {len(report)}"
)

print(
    "Accepted:",
    (
        report["Action"]
        == "ACCEPT"
    ).sum()
)

print(
    "Manual review:",
    (
        report["Action"]
        == "MANUAL_REVIEW"
    ).sum()
)

print(
    "Duplicate targets:",
    (
        report["Duplicate Target"]
        == "YES"
    ).sum()
)


# ============================================================
# REVIEW REQUIRED
# ============================================================

review = report[
    report["Action"]
    == "MANUAL_REVIEW"
].copy()


print()
print("=" * 70)
print("REVIEW REQUIRED")
print("=" * 70)

if len(review) == 0:

    print(
        "No manual review records found."
    )

else:

    print()

    print(
        review[
            [
                "Current College",
                "Official College",
                "Official Code",
                "Match Type",
                "Similarity",
                "Duplicate Target",
                "Confidence",
                "Action"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# FINAL MESSAGE
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