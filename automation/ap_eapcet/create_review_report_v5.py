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
    "automation/output/ap_eapcet_college_mapping_review.csv"
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
    "GROUPS"
}


# ============================================================
# WORD NORMALIZATION
# ============================================================

WORD_ALIASES = {

    "INST": "INSTITUTE",
    "INSTT": "INSTITUTE",

    "COLL": "COLLEGE",

    "ENGG": "ENGINEERING",
    "ENG": "ENGINEERING",

    "TECH": "TECHNOLOGY",

    "UNIV": "UNIVERSITY",

    "SCI": "SCIENCE",

    "AND": "AND"
}


# ============================================================
# PHRASE NORMALIZATION
# ============================================================

PHRASE_REPLACEMENTS = [
    (r"\bINST\.?\b", "INSTITUTE"),
    (r"\bINSTT\.?\b", "INSTITUTE"),

    (r"\bCOLL\.?\b", "COLLEGE"),

    (r"\bENGG\.?\b", "ENGINEERING"),
    (r"\bENG\.?\b", "ENGINEERING"),

    (r"\bTECH\.?\b", "TECHNOLOGY"),

    (r"\bUNIV\.?\b", "UNIVERSITY"),

    (r"\bSCI\.?\b", "SCIENCE"),

    (r"&", " AND ")
]


# ============================================================
# BASIC CLEAN
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    text = text.replace(
        "'",
        ""
    )

    text = text.replace(
        "’",
        ""
    )

    for pattern, replacement in PHRASE_REPLACEMENTS:

        text = re.sub(
            pattern,
            f" {replacement} ",
            text
        )

    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZED NAME
# ============================================================

def normalize_name(value):

    text = clean_text(value)

    if not text:
        return ""

    words = []

    for word in text.split():

        word = WORD_ALIASES.get(
            word,
            word
        )

        words.append(
            word
        )

    return " ".join(
        words
    )


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def get_tokens(name):

    if not name:
        return set()

    return set(
        name.split()
    )


def get_distinctive_tokens(name):

    return {
        token
        for token in get_tokens(name)
        if (
            token not in GENERIC_WORDS
            and len(token) >= 2
        )
    }


# ============================================================
# ACRONYM EXTRACTION
# ============================================================

def get_acronym(name):

    if not name:
        return ""

    words = name.split()

    # Ignore generic words.
    useful_words = [
        word
        for word in words
        if word not in GENERIC_WORDS
    ]

    if not useful_words:
        return ""

    return "".join(
        word[0]
        for word in useful_words
        if word
    )


# ============================================================
# COMPACT ACRONYM / INITIAL TOKEN
# ============================================================

def get_initial_tokens(name):

    tokens = get_tokens(name)

    initials = set()

    for token in tokens:

        if len(token) >= 2:

            initials.add(
                token[0]
            )

    return initials


# ============================================================
# LOCATION NORMALIZATION
# ============================================================

LOCATION_ALIASES = {

    "DISTRICT": "",
    "DIST": "",

    "ROAD": "",
    "RD": "",

    "TOWN": "",
    "CITY": "",

    "VILLAGE": "",

    "MANDAL": ""
}


def normalize_location(value):

    if pd.isna(value):
        return ""

    text = clean_text(value)

    if not text:
        return ""

    for old, new in LOCATION_ALIASES.items():

        text = re.sub(
            rf"\b{old}\b",
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

def location_tokens(value):

    normalized = normalize_location(
        value
    )

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 3
    }


# ============================================================
# LOCATION SCORE
# ============================================================

def location_score(
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

    if current == official:
        return 1.0

    current_tokens = location_tokens(
        current
    )

    official_tokens = location_tokens(
        official
    )

    if current_tokens and official_tokens:

        common = (
            current_tokens
            & official_tokens
        )

        if common:

            union = (
                current_tokens
                | official_tokens
            )

            jaccard = (
                len(common)
                / len(union)
            )

            char = SequenceMatcher(
                None,
                current,
                official
            ).ratio()

            return (
                0.75 * jaccard
                + 0.25 * char
            )

    return SequenceMatcher(
        None,
        current,
        official
    ).ratio()


# ============================================================
# TOKEN SCORE
# ============================================================

def token_score(
    current_tokens,
    official_tokens
):

    if not current_tokens:
        return 0.0

    if not official_tokens:
        return 0.0

    common = (
        current_tokens
        & official_tokens
    )

    if not common:
        return 0.0

    union = (
        current_tokens
        | official_tokens
    )

    jaccard = (
        len(common)
        / len(union)
    )

    current_coverage = (
        len(common)
        / len(current_tokens)
    )

    official_coverage = (
        len(common)
        / len(official_tokens)
    )

    return (
        0.45 * jaccard
        + 0.35 * current_coverage
        + 0.20 * official_coverage
    )


# ============================================================
# CHARACTER SCORE
# ============================================================

def character_score(
    current_name,
    official_name
):

    return SequenceMatcher(
        None,
        current_name,
        official_name
    ).ratio()


# ============================================================
# ACRONYM SCORE
# ============================================================

def acronym_score(
    current_name,
    official_name
):

    current_acronym = get_acronym(
        current_name
    )

    official_acronym = get_acronym(
        official_name
    )

    if not current_acronym:
        return 0.0

    if not official_acronym:
        return 0.0

    if current_acronym == official_acronym:
        return 1.0

    # One acronym contained inside another.
    if (
        current_acronym in official_acronym
        or
        official_acronym in current_acronym
    ):

        shorter = min(
            len(current_acronym),
            len(official_acronym)
        )

        longer = max(
            len(current_acronym),
            len(official_acronym)
        )

        if longer:
            return (
                shorter / longer
            )

    return SequenceMatcher(
        None,
        current_acronym,
        official_acronym
    ).ratio()


# ============================================================
# BRANCH FAMILY
# ============================================================

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


def branch_score(
    current_branches,
    official_branches
):

    if not current_branches:
        return 0.0

    if not official_branches:
        return 0.0

    matched = 0

    for current_branch in current_branches:

        current_branch = str(
            current_branch
        ).strip().upper()

        for family in BRANCH_FAMILIES.values():

            if current_branch not in family:
                continue

            if any(
                branch in family
                for branch in official_branches
            ):

                matched += 1
                break

    return (
        matched
        / len(current_branches)
    )


# ============================================================
# SPECIAL NAME SIGNALS
# ============================================================

def special_name_score(
    current_name,
    official_name
):

    current_tokens = (
        get_distinctive_tokens(
            current_name
        )
    )

    official_tokens = (
        get_distinctive_tokens(
            official_name
        )
    )

    common = (
        current_tokens
        & official_tokens
    )

    if not common:
        return 0.0

    score = 0.0

    for token in common:

        if len(token) >= 8:
            score += 0.04

        elif len(token) >= 6:
            score += 0.025

    return min(
        score,
        0.10
    )


# ============================================================
# COMPLETE MATCH SCORE
# ============================================================

def calculate_match(
    current_name,
    official_name,
    current_location,
    official_location,
    current_branches,
    official_branches
):

    current_normalized = normalize_name(
        current_name
    )

    official_normalized = normalize_name(
        official_name
    )

    current_tokens = (
        get_distinctive_tokens(
            current_normalized
        )
    )

    official_tokens = (
        get_distinctive_tokens(
            official_normalized
        )
    )

    token = token_score(
        current_tokens,
        official_tokens
    )

    character = character_score(
        current_normalized,
        official_normalized
    )

    location = location_score(
        current_location,
        official_location
    )

    acronym = acronym_score(
        current_normalized,
        official_normalized
    )

    branches = branch_score(
        current_branches,
        official_branches
    )

    bonus = special_name_score(
        current_normalized,
        official_normalized
    )

    common_tokens = sorted(
        current_tokens
        & official_tokens
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Name remains the strongest signal.
    #
    # Location is used heavily for duplicate names.
    #
    # Acronym helps RVR & JC, Vignan's, etc.
    #
    # Branch is only supporting evidence.
    # --------------------------------------------------------

    final = (
        0.42 * token
        + 0.16 * character
        + 0.22 * location
        + 0.10 * acronym
        + 0.10 * branches
        + bonus
    )

    final = min(
        final,
        1.0
    )

    return {
        "token": token,
        "character": character,
        "location": location,
        "acronym": acronym,
        "branch": branches,
        "bonus": bonus,
        "final": final,
        "common_tokens": common_tokens
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 76)
print(
    "AP EAPCET COLLEGE MAPPING REVIEW - VERSION 5"
)
print("=" * 76)

print()
print(
    "Loading official data..."
)

official = pd.read_csv(
    OFFICIAL_FILE
)

print(
    f"Official records: "
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
    f"Current records: "
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
        "Official dataset missing: "
        + ", ".join(
            sorted(
                missing_official
            )
        )
    )


if missing_current:

    raise RuntimeError(
        "Current dataset missing: "
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
# FILTER OFFICIAL ENGINEERING
# ============================================================

official_engineering = official[
    official["Branch"].isin(
        ENGINEERING_BRANCHES
    )
].copy()


print()
print(
    f"Official engineering records: "
    f"{len(official_engineering)}"
)


# ============================================================
# BUILD CURRENT COLLEGE GROUPS
# ============================================================

current_groups = []

for college_name, group in current.groupby(
    "College Name",
    dropna=True
):

    locations = [
        str(value).strip()
        for value in group["Location"].dropna()
        if str(value).strip()
    ]

    branches = set(
        group["Branch"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    current_groups.append(
        {
            "College Name":
                str(college_name).strip(),

            "Location":
                locations[0]
                if locations
                else "",

            "Branches":
                branches
        }
    )


current_colleges = pd.DataFrame(
    current_groups
)


# ============================================================
# BUILD OFFICIAL COLLEGE GROUPS
# ============================================================

official_groups = []

for college_code, group in (
    official_engineering.groupby(
        "College Code",
        dropna=True
    )
):

    code = str(
        college_code
    ).strip()

    if not code:
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

    name_counter = Counter(
        names
    )

    official_name = (
        name_counter
        .most_common(1)[0][0]
    )

    locations = [
        str(value).strip()
        for value in group[
            "Location"
        ].dropna()
        if str(value).strip()
    ]

    branches = set(
        group["Branch"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    official_groups.append(
        {
            "College Name":
                official_name,

            "College Code":
                code,

            "Location":
                locations[0]
                if locations
                else "",

            "Branches":
                branches
        }
    )


official_colleges = pd.DataFrame(
    official_groups
)


print()
print(
    f"Unique current colleges: "
    f"{len(current_colleges)}"
)

print(
    f"Unique official colleges: "
    f"{len(official_colleges)}"
)


# ============================================================
# NORMALIZED NAMES
# ============================================================

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
    "Normalized Name"
] = (
    official_colleges[
        "College Name"
    ]
    .apply(
        normalize_name
    )
)


# ============================================================
# EXACT NORMALIZED NAME GROUPS
# ============================================================

official_name_groups = {}

for _, row in official_colleges.iterrows():

    key = row[
        "Normalized Name"
    ]

    official_name_groups.setdefault(
        key,
        []
    ).append(
        row
    )


# ============================================================
# MATCHING
# ============================================================

results = []

total = len(
    current_colleges
)


for position, (_, current_row) in enumerate(
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


    print(
        f"\rProcessing "
        f"{position}/{total}: "
        f"{current_name[:65]}",
        end=""
    )


    # ========================================================
    # EXACT NORMALIZED NAME
    # ========================================================

    exact_candidates = (
        official_name_groups.get(
            current_normalized,
            []
        )
    )


    # ========================================================
    # SCORE EXACT CANDIDATES
    # ========================================================

    candidates = []


    if exact_candidates:

        candidate_pool = (
            exact_candidates
        )

    else:

        candidate_pool = (
            official_colleges
        )


    for _, official_row in (
        pd.DataFrame(
            candidate_pool
        ).iterrows()
        if isinstance(
            candidate_pool,
            list
        )
        else candidate_pool.iterrows()
    ):

        match = calculate_match(
            current_name,
            official_row[
                "College Name"
            ],
            current_location,
            official_row[
                "Location"
            ],
            current_branches,
            official_row[
                "Branches"
            ]
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
                    official_row[
                        "Location"
                    ],

                "branches":
                    official_row[
                        "Branches"
                    ],

                **match
            }
        )


    candidates.sort(
        key=lambda item: (
            item["final"],
            item["location"],
            item["token"],
            item["acronym"]
        ),
        reverse=True
    )


    top = candidates[
        :TOP_CANDIDATES
    ]


    if not top:

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

                "Official College":
                    "",

                "Official Code":
                    "",

                "Official Location":
                    "",

                "Official Branches":
                    "",

                "Match Status":
                    "UNMATCHED",

                "Confidence":
                    "NONE",

                "Reason":
                    "No candidate found",

                "Final Score":
                    0,

                "Score Margin":
                    0,

                "Candidate 2":
                    "",

                "Candidate 2 Code":
                    "",

                "Candidate 2 Location":
                    "",

                "Candidate 2 Score":
                    "",

                "Candidate 3":
                    "",

                "Candidate 3 Code":
                    "",

                "Candidate 3 Location":
                    "",

                "Candidate 3 Score":
                    ""
            }
        )

        continue


    best = top[0]


    second_score = (
        top[1]["final"]
        if len(top) >= 2
        else 0
    )


    margin = (
        best["final"]
        - second_score
    )


    # ========================================================
    # DETERMINE MATCH STATUS
    # ========================================================

    same_name_candidates = (
        len(exact_candidates)
    )


    exact_name = (
        normalize_name(
            current_name
        )
        ==
        normalize_name(
            best["name"]
        )
    )


    exact_location = (
        normalize_location(
            current_location
        )
        ==
        normalize_location(
            best["location"]
        )
        and
        normalize_location(
            current_location
        ) != ""
    )


    # --------------------------------------------------------
    # SAFE EXACT
    # --------------------------------------------------------

    if (
        exact_name
        and same_name_candidates == 1
        and exact_location
        and margin >= 0.10
    ):

        status = "SAFE"
        confidence = "HIGH"

        reason = (
            "Exact normalized name + "
            "exact location + unique official code"
        )


    # --------------------------------------------------------
    # EXACT NAME BUT MULTIPLE OFFICIAL COLLEGES
    # --------------------------------------------------------

    elif (
        exact_name
        and same_name_candidates > 1
    ):

        if (
            exact_location
            and margin >= 0.08
        ):

            status = "SAFE"
            confidence = "HIGH"

            reason = (
                "Exact normalized name + "
                "location distinguishes official college"
            )

        else:

            status = "MANUAL_REVIEW"
            confidence = "MEDIUM"

            reason = (
                "Multiple official colleges "
                "share the same normalized name"
            )


    # --------------------------------------------------------
    # VERY STRONG FUZZY
    # --------------------------------------------------------

    elif (
        best["final"] >= 0.78
        and margin >= 0.10
        and (
            exact_location
            or
            best["location"] >= 0.80
            or
            best["acronym"] >= 0.90
        )
    ):

        status = "MANUAL_REVIEW"
        confidence = "HIGH"

        reason = (
            "Strong name/acronym/location evidence; "
            "manual confirmation required"
        )


    # --------------------------------------------------------
    # GOOD FUZZY
    # --------------------------------------------------------

    elif (
        best["final"] >= 0.62
        and margin >= 0.08
    ):

        status = "MANUAL_REVIEW"
        confidence = "MEDIUM"

        reason = (
            "Good candidate but fuzzy match"
        )


    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    elif (
        len(top) >= 2
        and margin < 0.08
    ):

        status = "MANUAL_REVIEW"
        confidence = "LOW"

        reason = (
            "Multiple candidates have similar scores"
        )


    # --------------------------------------------------------
    # UNMATCHED
    # --------------------------------------------------------

    else:

        status = "UNMATCHED"
        confidence = "LOW"

        reason = (
            "No sufficiently reliable automatic match"
        )


    # ========================================================
    # CANDIDATE 2
    # ========================================================

    candidate_2 = (
        top[1]
        if len(top) >= 2
        else None
    )


    # ========================================================
    # CANDIDATE 3
    # ========================================================

    candidate_3 = (
        top[2]
        if len(top) >= 3
        else None
    )


    # ========================================================
    # ADD RESULT
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

            "Official College":
                best["name"],

            "Official Code":
                best["code"],

            "Official Location":
                best["location"],

            "Official Branches":
                ", ".join(
                    sorted(
                        best["branches"]
                    )
                ),

            "Match Status":
                status,

            "Confidence":
                confidence,

            "Reason":
                reason,

            "Common Tokens":
                ", ".join(
                    best["common_tokens"]
                ),

            "Token Score":
                round(
                    best["token"],
                    4
                ),

            "Character Score":
                round(
                    best["character"],
                    4
                ),

            "Location Score":
                round(
                    best["location"],
                    4
                ),

            "Acronym Score":
                round(
                    best["acronym"],
                    4
                ),

            "Branch Score":
                round(
                    best["branch"],
                    4
                ),

            "Final Score":
                round(
                    best["final"],
                    4
                ),

            "Score Margin":
                round(
                    margin,
                    4
                ),

            "Candidate 2":
                candidate_2["name"]
                if candidate_2
                else "",

            "Candidate 2 Code":
                candidate_2["code"]
                if candidate_2
                else "",

            "Candidate 2 Location":
                candidate_2["location"]
                if candidate_2
                else "",

            "Candidate 2 Score":
                round(
                    candidate_2["final"],
                    4
                )
                if candidate_2
                else "",

            "Candidate 3":
                candidate_3["name"]
                if candidate_3
                else "",

            "Candidate 3 Code":
                candidate_3["code"]
                if candidate_3
                else "",

            "Candidate 3 Location":
                candidate_3["location"]
                if candidate_3
                else "",

            "Candidate 3 Score":
                round(
                    candidate_3["final"],
                    4
                )
                if candidate_3
                else ""
        }
    )


print()


# ============================================================
# CREATE REPORT
# ============================================================

report = pd.DataFrame(
    results
)


# ============================================================
# DUPLICATE OFFICIAL CODE CHECK
# ============================================================

accepted_like = report[
    report[
        "Match Status"
    ].isin(
        [
            "SAFE",
            "MANUAL_REVIEW"
        ]
    )
]


code_counts = (
    accepted_like[
        "Official Code"
    ]
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
        "Official Code"
    ]

    status = report.at[
        index,
        "Match Status"
    ]


    if (
        code
        and
        code in duplicate_codes
        and
        status == "SAFE"
    ):

        report.at[
            index,
            "Match Status"
        ] = (
            "MANUAL_REVIEW"
        )

        report.at[
            index,
            "Confidence"
        ] = (
            "MEDIUM"
        )

        report.at[
            index,
            "Reason"
        ] = (
            "Official code is selected by "
            "more than one current college"
        )


# ============================================================
# SORT
# ============================================================

status_order = {

    "MANUAL_REVIEW": 0,

    "UNMATCHED": 1,

    "SAFE": 2
}


report["_sort"] = (
    report[
        "Match Status"
    ]
    .map(
        status_order
    )
    .fillna(99)
)


report = report.sort_values(
    by=[
        "_sort",
        "Final Score"
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
print("=" * 76)
print(
    "VERSION 5 MAPPING REVIEW SUMMARY"
)
print("=" * 76)

print()

print(
    report[
        "Match Status"
    ]
    .value_counts()
    .to_string()
)


print()

print(
    f"Total current colleges: "
    f"{len(report)}"
)


for status in [
    "SAFE",
    "MANUAL_REVIEW",
    "UNMATCHED"
]:

    count = (
        report[
            "Match Status"
        ]
        == status
    ).sum()

    print(
        f"{status}: {count}"
    )


# ============================================================
# SAFE EXAMPLES
# ============================================================

print()
print("=" * 76)
print(
    "SAFE MATCH EXAMPLES"
)
print("=" * 76)


safe = report[
    report[
        "Match Status"
    ]
    == "SAFE"
].head(20)


for _, row in safe.iterrows():

    print()

    print(
        f"{row['Current College']}"
    )

    print(
        f"  -> "
        f"{row['Official College']}"
        f" | {row['Official Code']}"
    )

    print(
        f"  Current location: "
        f"{row['Current Location']}"
    )

    print(
        f"  Official location: "
        f"{row['Official Location']}"
    )

    print(
        f"  Score: "
        f"{row['Final Score']}"
    )

    print(
        f"  Reason: "
        f"{row['Reason']}"
    )


# ============================================================
# MANUAL REVIEW EXAMPLES
# ============================================================

print()
print("=" * 76)
print(
    "MANUAL REVIEW EXAMPLES"
)
print("=" * 76)


manual = report[
    report[
        "Match Status"
    ]
    == "MANUAL_REVIEW"
].head(20)


for _, row in manual.iterrows():

    print()

    print(
        f"Current: "
        f"{row['Current College']}"
    )

    print(
        f"Location: "
        f"{row['Current Location']}"
    )

    print(
        f"Candidate 1: "
        f"{row['Official College']}"
        f" | {row['Official Code']}"
        f" | {row['Official Location']}"
        f" | Score={row['Final Score']}"
    )

    print(
        f"Candidate 2: "
        f"{row['Candidate 2']}"
        f" | {row['Candidate 2 Code']}"
        f" | {row['Candidate 2 Location']}"
        f" | Score={row['Candidate 2 Score']}"
    )

    print(
        f"Candidate 3: "
        f"{row['Candidate 3']}"
        f" | {row['Candidate 3 Code']}"
        f" | {row['Candidate 3 Location']}"
        f" | Score={row['Candidate 3 Score']}"
    )

    print(
        f"Reason: "
        f"{row['Reason']}"
    )


# ============================================================
# UNMATCHED EXAMPLES
# ============================================================

print()
print("=" * 76)
print(
    "UNMATCHED EXAMPLES"
)
print("=" * 76)


unmatched = report[
    report[
        "Match Status"
    ]
    == "UNMATCHED"
].head(20)


for _, row in unmatched.iterrows():

    print()

    print(
        f"Current: "
        f"{row['Current College']}"
    )

    print(
        f"Location: "
        f"{row['Current Location']}"
    )

    print(
        f"Best candidate: "
        f"{row['Official College']}"
    )

    print(
        f"Code: "
        f"{row['Official Code']}"
    )

    print(
        f"Score: "
        f"{row['Final Score']}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 76)
print(
    "VERSION 5 REPORT SAVED"
)
print("=" * 76)

print(
    OUTPUT_FILE
)

print()

print(
    "IMPORTANT:"
)

print(
    "This is a REVIEW file."
)

print(
    "PostgreSQL was NOT modified."
)

print(
    "dataset/eamcet_colleges.csv was NOT modified."
)

print(
    "The ML model was NOT modified."
)

print(
    "No permanent mapping was created yet."
)