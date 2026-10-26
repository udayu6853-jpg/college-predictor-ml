import os
import re
from difflib import SequenceMatcher
from collections import Counter

import pandas as pd


# ============================================================
# FILES
# ============================================================

OFFICIAL_FILE = (
    "automation/output/ap_eapcet_normalized.csv"
)

CURRENT_FILE = (
    "dataset/eamcet_colleges.csv"
)

OUTPUT_FILE = (
    "automation/output/ap_eapcet_review_report_v4.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_CANDIDATES = 3


# ============================================================
# OFFICIAL ENGINEERING BRANCHES
# ============================================================

ENGINEERING_BRANCHES = {
    "AI", "AID", "AIM", "ASE", "AUT", "BDT", "BIO",
    "CAD", "CAI", "CBA", "CBC", "CCC", "CDA", "CHE",
    "CIA", "CIC", "CIT", "CIV", "CN", "CS", "CSB",
    "CSBS", "CSC", "CSD", "CSE", "CSEB", "CSER",
    "CSG", "CSM", "CSO", "CSS", "CST", "CSW",
    "DS", "EBM", "ECA", "ECE", "ECES", "ECM", "ECT",
    "ECV", "EEE", "EIE", "EII", "EVT", "FDE", "FDT",
    "GDT", "GIN", "INF", "IOT", "IST", "MAD", "MAU",
    "MET", "MII", "MIN", "MMM", "MMT", "MRB", "NAM",
    "PEE", "PET", "PHD", "PHM", "RBT", "SWE"
}


# ============================================================
# GENERIC WORDS
# ============================================================

GENERIC_WORDS = {
    "THE",
    "AND",
    "OF",
    "IN",
    "AT",
    "FOR",

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
    "GROUPS",

    "COLLEGE OF",
}


# ============================================================
# ABBREVIATIONS
# ============================================================

WORD_REPLACEMENTS = {
    "INSTITUTE": {
        "INST",
        "INSTT",
        "INST.",
    },

    "COLLEGE": {
        "COLL",
        "COLL.",
    },

    "ENGINEERING": {
        "ENGG",
        "ENGG.",
        "ENG",
    },

    "TECHNOLOGY": {
        "TECH",
        "TECH.",
    },

    "UNIVERSITY": {
        "UNIV",
        "UNIV.",
    },

    "SCIENCE": {
        "SCI",
        "SCI.",
    },

    "AND": {
        "&"
    }
}


# ============================================================
# BRANCH FAMILY MAPPING
# ============================================================

# Used ONLY as a matching signal.
# This does NOT change the project's final branch mapping.

BRANCH_FAMILIES = {
    "CSE": {
        "CSE",
        "CS",
        "CSC",
        "CSD",
        "CSM",
        "CSO",
        "CSBS",
        "CSB",
        "CSG",
        "CST",
        "CSS",
        "CSW",
        "CSER",
        "CSEB",
        "AI",
        "AID",
        "AIM",
        "CAI",
        "INF",
        "IOT",
        "DS"
    },

    "ECE": {
        "ECE",
        "ECES",
        "ECM",
        "ECT",
        "ECV",
        "ECA"
    },

    "ME": {
        "MEC",
        "MET",
        "MII",
        "MIN",
        "MMM",
        "MMT",
        "MRB"
    }
}


# ============================================================
# TEXT CLEANING
# ============================================================

def basic_clean(value):

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    # Normalize ampersand first.
    text = text.replace("&", " AND ")

    # Convert punctuation to spaces.
    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text
    )

    # Collapse spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# WORD NORMALIZATION
# ============================================================

def normalize_word(word):

    word = word.upper().strip()

    for canonical, variants in WORD_REPLACEMENTS.items():

        if word == canonical:
            return canonical

        if word in variants:
            return canonical

    return word


# ============================================================
# NORMALIZE COLLEGE NAME
# ============================================================

def normalize_name(name):

    cleaned = basic_clean(name)

    if not cleaned:
        return ""

    words = cleaned.split()

    normalized_words = [
        normalize_word(word)
        for word in words
    ]

    return " ".join(
        normalized_words
    )


# ============================================================
# TOKENS
# ============================================================

def all_tokens(name):

    if not name:
        return set()

    return set(
        name.split()
    )


def distinctive_tokens(name):

    tokens = all_tokens(name)

    return {
        token
        for token in tokens
        if (
            token not in GENERIC_WORDS
            and len(token) >= 2
        )
    }


# ============================================================
# ACRONYM EXTRACTION
# ============================================================

def acronym_from_name(name):

    if not name:
        return ""

    words = [
        word
        for word in name.split()
        if word not in GENERIC_WORDS
    ]

    if not words:
        return ""

    return "".join(
        word[0]
        for word in words
        if word
    )


# ============================================================
# LOCATION NORMALIZATION
# ============================================================

LOCATION_REPLACEMENTS = {
    "DISTRICT": "",
    "DIST": "",
    "ROAD": "",
    "RD": "",
    "TOWN": "",
    "CITY": "",
    "VILLAGE": "",
}


def normalize_location(location):

    if pd.isna(location):
        return ""

    text = basic_clean(location)

    for old, new in LOCATION_REPLACEMENTS.items():

        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text
        )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOCATION TOKENS
# ============================================================

def location_tokens(location):

    normalized = normalize_location(
        location
    )

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
    }


# ============================================================
# LOCATION SIMILARITY
# ============================================================

def location_similarity(
    current_location,
    official_location
):

    current = normalize_location(
        current_location
    )

    official = normalize_location(
        official_location
    )

    if not current or not official:
        return 0.0

    current_tokens = location_tokens(
        current
    )

    official_tokens = location_tokens(
        official
    )

    if not current_tokens or not official_tokens:
        return SequenceMatcher(
            None,
            current,
            official
        ).ratio()

    intersection = (
        current_tokens
        & official_tokens
    )

    union = (
        current_tokens
        | official_tokens
    )

    jaccard = (
        len(intersection)
        / len(union)
    )

    character = SequenceMatcher(
        None,
        current,
        official
    ).ratio()

    return (
        0.70 * jaccard
        + 0.30 * character
    )


# ============================================================
# BRANCH EXTRACTION
# ============================================================

def get_official_branch_set(
    official_rows
):

    branches = set()

    for branch in official_rows:

        branch = str(
            branch
        ).strip().upper()

        if branch in ENGINEERING_BRANCHES:

            branches.add(branch)

    return branches


# ============================================================
# BRANCH FAMILY OVERLAP
# ============================================================

def branch_family_overlap(
    current_branches,
    official_branches
):

    if not current_branches:
        return 0.0

    if not official_branches:
        return 0.0

    score_values = []

    for current_branch in current_branches:

        current_branch = (
            str(current_branch)
            .strip()
            .upper()
        )

        matched = False

        for family_name, family_members in (
            BRANCH_FAMILIES.items()
        ):

            if current_branch not in family_members:
                continue

            for official_branch in official_branches:

                if official_branch in family_members:

                    matched = True
                    break

            if matched:
                break

        score_values.append(
            1.0 if matched else 0.0
        )

    if not score_values:
        return 0.0

    return sum(
        score_values
    ) / len(score_values)


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
# TOKEN SCORE
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

    return (
        0.45 * jaccard
        + 0.35 * current_coverage
        + 0.20 * official_coverage
    )


# ============================================================
# ACRONYM SIMILARITY
# ============================================================

def acronym_similarity(
    current_name,
    official_name
):

    current_acronym = acronym_from_name(
        current_name
    )

    official_acronym = acronym_from_name(
        official_name
    )

    if not current_acronym:
        return 0.0

    if not official_acronym:
        return 0.0

    if (
        current_acronym
        == official_acronym
    ):
        return 1.0

    return SequenceMatcher(
        None,
        current_acronym,
        official_acronym
    ).ratio()


# ============================================================
# DISTINCTIVE TOKEN BONUS
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

    return min(
        0.08,
        0.02 * len(long_tokens)
    )


# ============================================================
# COMBINED SCORE
# ============================================================

def calculate_score(
    current_name,
    official_name,
    current_tokens,
    official_tokens,
    current_location,
    official_location,
    current_branches,
    official_branches
):

    token_score = token_similarity(
        current_tokens,
        official_tokens
    )

    char_score = character_similarity(
        current_name,
        official_name
    )

    location_score = location_similarity(
        current_location,
        official_location
    )

    acronym_score = acronym_similarity(
        current_name,
        official_name
    )

    branch_score = branch_family_overlap(
        current_branches,
        official_branches
    )

    bonus = distinctive_bonus(
        current_tokens,
        official_tokens
    )

    # --------------------------------------------------------
    # SCORE WEIGHTS
    # --------------------------------------------------------
    #
    # Name tokens       = 45%
    # Character name    = 15%
    # Location          = 20%
    # Acronym           = 10%
    # Branch overlap    = 10%
    # Distinctive bonus = extra
    #
    # --------------------------------------------------------

    final_score = (
        0.45 * token_score
        + 0.15 * char_score
        + 0.20 * location_score
        + 0.10 * acronym_score
        + 0.10 * branch_score
        + bonus
    )

    final_score = min(
        final_score,
        1.0
    )

    return {
        "token_score": token_score,
        "character_score": char_score,
        "location_score": location_score,
        "acronym_score": acronym_score,
        "branch_score": branch_score,
        "bonus": bonus,
        "final_score": final_score
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print(
    "AP EAPCET COLLEGE REVIEW REPORT - VERSION 4"
)
print("=" * 72)

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
# REQUIRED COLUMNS
# ============================================================

required_official = {
    "College Name",
    "College Code",
    "Branch",
    "Location"
}

required_current = {
    "College Name",
    "Branch",
    "Location"
}


missing_official = (
    required_official
    - set(official.columns)
)

missing_current = (
    required_current
    - set(current.columns)
)


if missing_official:

    raise RuntimeError(
        "Official dataset is missing columns: "
        + ", ".join(
            sorted(
                missing_official
            )
        )
    )


if missing_current:

    raise RuntimeError(
        "Current dataset is missing columns: "
        + ", ".join(
            sorted(
                missing_current
            )
        )
    )


# ============================================================
# NORMALIZE BRANCH
# ============================================================

official["Branch"] = (
    official["Branch"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

current["Branch"] = (
    current["Branch"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# OFFICIAL ENGINEERING RECORDS
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
# CURRENT COLLEGE INFORMATION
# ============================================================

current_college_groups = []

for college_name, group in current.groupby(
    "College Name",
    dropna=True
):

    branches = set(
        group["Branch"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    locations = [
        str(value).strip()
        for value in group["Location"]
        .dropna()
        if str(value).strip()
    ]

    location = (
        locations[0]
        if locations
        else ""
    )

    current_college_groups.append(
        {
            "College Name":
                college_name,

            "Location":
                location,

            "Branches":
                branches
        }
    )


current_colleges = pd.DataFrame(
    current_college_groups
)


# ============================================================
# OFFICIAL COLLEGE INFORMATION
# ============================================================

official_college_groups = []

for college_code, group in (
    official_engineering.groupby(
        "College Code",
        dropna=True
    )
):

    if not str(
        college_code
    ).strip():

        continue

    names = [
        str(value).strip()
        for value in group[
            "College Name"
        ].dropna()
        if str(value).strip()
    ]

    if not names:
        continue

    # Most common name.
    name_counts = Counter(
        names
    )

    college_name = (
        name_counts
        .most_common(1)[0][0]
    )

    locations = [
        str(value).strip()
        for value in group[
            "Location"
        ].dropna()
        if str(value).strip()
    ]

    location = (
        locations[0]
        if locations
        else ""
    )

    branches = set(
        group["Branch"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    official_college_groups.append(
        {
            "College Name":
                college_name,

            "College Code":
                str(
                    college_code
                ).strip(),

            "Location":
                location,

            "Branches":
                branches
        }
    )


official_colleges = pd.DataFrame(
    official_college_groups
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
# PREPARE NAMES
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


official_colleges[
    "Distinctive Tokens"
] = (
    official_colleges[
        "Normalized Name"
    ]
    .apply(
        distinctive_tokens
    )
)

current_colleges[
    "Distinctive Tokens"
] = (
    current_colleges[
        "Normalized Name"
    ]
    .apply(
        distinctive_tokens
    )
)


# ============================================================
# EXACT LOOKUP
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
# MATCHING
# ============================================================

results = []

total = len(
    current_colleges
)


for number, (_, current_row) in enumerate(
    current_colleges.iterrows(),
    start=1
):

    current_name = current_row[
        "College Name"
    ]

    current_location = current_row[
        "Location"
    ]

    current_branches = current_row[
        "Branches"
    ]

    current_normalized = current_row[
        "Normalized Name"
    ]

    current_tokens = current_row[
        "Distinctive Tokens"
    ]


    print(
        f"\rProcessing "
        f"{number}/{total}: "
        f"{str(current_name)[:65]}",
        end=""
    )


    # ========================================================
    # EXACT NORMALIZED MATCH
    # ========================================================

    exact_matches = official_lookup.get(
        current_normalized,
        []
    )


    # If exactly one official code exists,
    # this is a safe exact name match.
    if len(exact_matches) == 1:

        candidate = exact_matches[0]

        score = calculate_score(
            current_normalized,
            candidate[
                "Normalized Name"
            ],
            current_tokens,
            candidate[
                "Distinctive Tokens"
            ],
            current_location,
            candidate[
                "Location"
            ],
            current_branches,
            candidate[
                "Branches"
            ]
        )


        results.append(
            {
                "Current College":
                    current_name,

                "Current Location":
                    current_location,

                "Current Branches":
                    ", ".join(
                        sorted(
                            current_branches
                        )
                    ),

                "Match Category":
                    "EXACT",

                "Candidate 1":
                    candidate[
                        "College Name"
                    ],

                "Code 1":
                    candidate[
                        "College Code"
                    ],

                "Official Location 1":
                    candidate[
                        "Location"
                    ],

                "Official Branches 1":
                    ", ".join(
                        sorted(
                            candidate[
                                "Branches"
                            ]
                        )
                    ),

                "Common Tokens 1":
                    ", ".join(
                        sorted(
                            current_tokens
                            & candidate[
                                "Distinctive Tokens"
                            ]
                        )
                    ),

                "Token Score 1":
                    round(
                        score[
                            "token_score"
                        ],
                        4
                    ),

                "Character Score 1":
                    round(
                        score[
                            "character_score"
                        ],
                        4
                    ),

                "Location Score 1":
                    round(
                        score[
                            "location_score"
                        ],
                        4
                    ),

                "Acronym Score 1":
                    round(
                        score[
                            "acronym_score"
                        ],
                        4
                    ),

                "Branch Score 1":
                    round(
                        score[
                            "branch_score"
                        ],
                        4
                    ),

                "Final Score 1":
                    round(
                        score[
                            "final_score"
                        ],
                        4
                    ),

                "Candidate 2": "",
                "Code 2": "",
                "Official Location 2": "",
                "Official Branches 2": "",
                "Common Tokens 2": "",
                "Token Score 2": "",
                "Character Score 2": "",
                "Location Score 2": "",
                "Acronym Score 2": "",
                "Branch Score 2": "",
                "Final Score 2": "",

                "Candidate 3": "",
                "Code 3": "",
                "Official Location 3": "",
                "Official Branches 3": "",
                "Common Tokens 3": "",
                "Token Score 3": "",
                "Character Score 3": "",
                "Location Score 3": "",
                "Acronym Score 3": "",
                "Branch Score 3": "",
                "Final Score 3": "",

                "Score Margin":
                    "",

                "Review Status":
                    "SAFE_EXACT",

                "Suggested Action":
                    "ACCEPT"
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

        official_name = official_row[
            "Normalized Name"
        ]

        official_tokens = official_row[
            "Distinctive Tokens"
        ]

        official_location = official_row[
            "Location"
        ]

        official_branches = official_row[
            "Branches"
        ]


        if not official_name:
            continue


        score = calculate_score(
            current_normalized,
            official_name,
            current_tokens,
            official_tokens,
            current_location,
            official_location,
            current_branches,
            official_branches
        )


        common_tokens = (
            current_tokens
            & official_tokens
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

                "location":
                    official_location,

                "branches":
                    official_branches,

                "common_tokens":
                    sorted(
                        common_tokens
                    ),

                **score
            }
        )


    # ========================================================
    # SORT CANDIDATES
    # ========================================================

    candidates.sort(
        key=lambda candidate: (
            candidate[
                "final_score"
            ],

            candidate[
                "token_score"
            ],

            candidate[
                "location_score"
            ],

            candidate[
                "character_score"
            ]
        ),
        reverse=True
    )


    top = candidates[
        :TOP_CANDIDATES
    ]


    best_score = (
        top[0]["final_score"]
        if top
        else 0.0
    )


    second_score = (
        top[1]["final_score"]
        if len(top) >= 2
        else 0.0
    )


    score_margin = (
        best_score
        - second_score
    )


    # ========================================================
    # REVIEW STATUS
    # ========================================================

    if best_score < 0.55:

        status = (
            "LIKELY_NO_MATCH"
        )

    elif (
        len(top) >= 2
        and score_margin < 0.05
    ):

        status = (
            "AMBIGUOUS_REVIEW"
        )

    elif best_score >= 0.82:

        status = (
            "HIGH_CONFIDENCE_REVIEW"
        )

    elif best_score >= 0.68:

        status = (
            "MEDIUM_CONFIDENCE_REVIEW"
        )

    else:

        status = (
            "LOW_CONFIDENCE_REVIEW"
        )


    # IMPORTANT:
    # Fuzzy matches are NEVER automatically accepted.

    action = (
        "MANUAL_REVIEW"
    )


    # ========================================================
    # CANDIDATE HELPER
    # ========================================================

    def candidate_value(
        candidate,
        field,
        decimals=True
    ):

        if candidate is None:
            return ""

        value = candidate.get(
            field,
            ""
        )

        if decimals and isinstance(
            value,
            float
        ):

            return round(
                value,
                4
            )

        return value


    candidate_rows = []

    for index in range(
        TOP_CANDIDATES
    ):

        if index < len(top):

            candidate_rows.append(
                top[index]
            )

        else:

            candidate_rows.append(
                None
            )


    c1, c2, c3 = candidate_rows


    # ========================================================
    # RESULT ROW
    # ========================================================

    results.append(
        {
            "Current College":
                current_name,

            "Current Location":
                current_location,

            "Current Branches":
                ", ".join(
                    sorted(
                        current_branches
                    )
                ),

            "Match Category":
                "FUZZY",

            # Candidate 1
            "Candidate 1":
                candidate_value(
                    c1,
                    "name",
                    False
                ),

            "Code 1":
                candidate_value(
                    c1,
                    "code",
                    False
                ),

            "Official Location 1":
                candidate_value(
                    c1,
                    "location",
                    False
                ),

            "Official Branches 1":
                (
                    ", ".join(
                        sorted(
                            c1[
                                "branches"
                            ]
                        )
                    )
                    if c1
                    else ""
                ),

            "Common Tokens 1":
                (
                    ", ".join(
                        c1[
                            "common_tokens"
                        ]
                    )
                    if c1
                    else ""
                ),

            "Token Score 1":
                candidate_value(
                    c1,
                    "token_score"
                ),

            "Character Score 1":
                candidate_value(
                    c1,
                    "character_score"
                ),

            "Location Score 1":
                candidate_value(
                    c1,
                    "location_score"
                ),

            "Acronym Score 1":
                candidate_value(
                    c1,
                    "acronym_score"
                ),

            "Branch Score 1":
                candidate_value(
                    c1,
                    "branch_score"
                ),

            "Final Score 1":
                candidate_value(
                    c1,
                    "final_score"
                ),

            # Candidate 2
            "Candidate 2":
                candidate_value(
                    c2,
                    "name",
                    False
                ),

            "Code 2":
                candidate_value(
                    c2,
                    "code",
                    False
                ),

            "Official Location 2":
                candidate_value(
                    c2,
                    "location",
                    False
                ),

            "Official Branches 2":
                (
                    ", ".join(
                        sorted(
                            c2[
                                "branches"
                            ]
                        )
                    )
                    if c2
                    else ""
                ),

            "Common Tokens 2":
                (
                    ", ".join(
                        c2[
                            "common_tokens"
                        ]
                    )
                    if c2
                    else ""
                ),

            "Token Score 2":
                candidate_value(
                    c2,
                    "token_score"
                ),

            "Character Score 2":
                candidate_value(
                    c2,
                    "character_score"
                ),

            "Location Score 2":
                candidate_value(
                    c2,
                    "location_score"
                ),

            "Acronym Score 2":
                candidate_value(
                    c2,
                    "acronym_score"
                ),

            "Branch Score 2":
                candidate_value(
                    c2,
                    "branch_score"
                ),

            "Final Score 2":
                candidate_value(
                    c2,
                    "final_score"
                ),

            # Candidate 3
            "Candidate 3":
                candidate_value(
                    c3,
                    "name",
                    False
                ),

            "Code 3":
                candidate_value(
                    c3,
                    "code",
                    False
                ),

            "Official Location 3":
                candidate_value(
                    c3,
                    "location",
                    False
                ),

            "Official Branches 3":
                (
                    ", ".join(
                        sorted(
                            c3[
                                "branches"
                            ]
                        )
                    )
                    if c3
                    else ""
                ),

            "Common Tokens 3":
                (
                    ", ".join(
                        c3[
                            "common_tokens"
                        ]
                    )
                    if c3
                    else ""
                ),

            "Token Score 3":
                candidate_value(
                    c3,
                    "token_score"
                ),

            "Character Score 3":
                candidate_value(
                    c3,
                    "character_score"
                ),

            "Location Score 3":
                candidate_value(
                    c3,
                    "location_score"
                ),

            "Acronym Score 3":
                candidate_value(
                    c3,
                    "acronym_score"
                ),

            "Branch Score 3":
                candidate_value(
                    c3,
                    "branch_score"
                ),

            "Final Score 3":
                candidate_value(
                    c3,
                    "final_score"
                ),

            "Score Margin":
                round(
                    score_margin,
                    4
                ),

            "Review Status":
                status,

            "Suggested Action":
                action
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
# DUPLICATE OFFICIAL CODE DETECTION
# ============================================================

code_counts = (
    report[
        report[
            "Code 1"
        ].notna()
        & (
            report[
                "Code 1"
            ].astype(str)
            != ""
        )
    ]["Code 1"]
    .value_counts()
)


duplicate_codes = set(
    code_counts[
        code_counts > 1
    ].index
)


for index in report.index:

    code = report.at[
        index,
        "Code 1"
    ]

    status = report.at[
        index,
        "Review Status"
    ]


    if (
        code in duplicate_codes
        and status == "SAFE_EXACT"
    ):

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
# SORT ORDER
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
print("=" * 72)
print(
    "VERSION 4 REVIEW SUMMARY"
)
print("=" * 72)

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
# HIGH CONFIDENCE
# ============================================================

print()
print("=" * 72)
print(
    "VERSION 4 HIGH-CONFIDENCE EXAMPLES"
)
print("=" * 72)


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
            f"Current location: "
            f"{row['Current Location']}"
        )

        print(
            f"  1. "
            f"{row['Candidate 1']} "
            f"| Code={row['Code 1']}"
        )

        print(
            f"     Location="
            f"{row['Official Location 1']}"
        )

        print(
            f"     Tokens="
            f"{row['Common Tokens 1']}"
        )

        print(
            f"     Token="
            f"{row['Token Score 1']} "
            f"Char="
            f"{row['Character Score 1']} "
            f"Location="
            f"{row['Location Score 1']} "
            f"Branch="
            f"{row['Branch Score 1']} "
            f"Final="
            f"{row['Final Score 1']}"
        )

        print(
            f"  2. "
            f"{row['Candidate 2']} "
            f"| Code={row['Code 2']} "
            f"| Final={row['Final Score 2']}"
        )

        print(
            f"  3. "
            f"{row['Candidate 3']} "
            f"| Code={row['Code 3']} "
            f"| Final={row['Final Score 3']}"
        )

        print(
            f"  Margin="
            f"{row['Score Margin']}"
        )


# ============================================================
# AMBIGUOUS
# ============================================================

print()
print("=" * 72)
print(
    "VERSION 4 AMBIGUOUS EXAMPLES"
)
print("=" * 72)


ambiguous = report[
    report[
        "Review Status"
    ]
    == "AMBIGUOUS_REVIEW"
].head(20)


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
            f"Current location: "
            f"{row['Current Location']}"
        )

        print(
            f"  1. "
            f"{row['Candidate 1']} "
            f"| {row['Code 1']} "
            f"| Location="
            f"{row['Official Location 1']} "
            f"| Final="
            f"{row['Final Score 1']}"
        )

        print(
            f"  2. "
            f"{row['Candidate 2']} "
            f"| {row['Code 2']} "
            f"| Location="
            f"{row['Official Location 2']} "
            f"| Final="
            f"{row['Final Score 2']}"
        )

        print(
            f"  3. "
            f"{row['Candidate 3']} "
            f"| {row['Code 3']} "
            f"| Location="
            f"{row['Official Location 3']} "
            f"| Final="
            f"{row['Final Score 3']}"
        )

        print(
            f"  Margin="
            f"{row['Score Margin']}"
        )


# ============================================================
# NO MATCH EXAMPLES
# ============================================================

print()
print("=" * 72)
print(
    "VERSION 4 LIKELY-NO-MATCH EXAMPLES"
)
print("=" * 72)


no_match = report[
    report[
        "Review Status"
    ]
    == "LIKELY_NO_MATCH"
].head(15)


if no_match.empty:

    print(
        "No likely-no-match colleges."
    )

else:

    for _, row in no_match.iterrows():

        print()

        print(
            f"Current: "
            f"{row['Current College']}"
        )

        print(
            f"Best candidate: "
            f"{row['Candidate 1']}"
        )

        print(
            f"Code: "
            f"{row['Code 1']}"
        )

        print(
            f"Score: "
            f"{row['Final Score 1']}"
        )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 72)
print(
    "VERSION 4 REPORT SAVED"
)
print("=" * 72)

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