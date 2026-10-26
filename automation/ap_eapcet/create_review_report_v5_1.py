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
    "automation/output/ap_eapcet_college_mapping_review_v5_1.csv"
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
# IDENTITY WORDS
#
# These words are NOT treated as harmless differences.
# They can identify a genuinely different institution.
# ============================================================

IDENTITY_WORDS = {
    "UNIVERSITY",
    "WOMEN",
    "WOMENS",
    "WOMAN",
    "PHARMACY",
    "POLYTECHNIC",
    "MEDICAL",
    "DENTAL",
    "LAW",
    "ARCHITECTURE",
    "AGRICULTURAL",
    "AGRICULTURE",
    "ARTS",
    "FINE",
    "MANAGEMENT"
}


# ============================================================
# WORD ALIASES
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

    "MGMT": "MANAGEMENT"
}


# ============================================================
# PHRASE NORMALIZATION
# ============================================================

PHRASE_REPLACEMENTS = [
    (r"\bINSTT?\.?\b", "INSTITUTE"),
    (r"\bCOLL\.?\b", "COLLEGE"),
    (r"\bENGG\.?\b", "ENGINEERING"),
    (r"\bENG\.?\b", "ENGINEERING"),
    (r"\bTECH\.?\b", "TECHNOLOGY"),
    (r"\bUNIV\.?\b", "UNIVERSITY"),
    (r"\bSCI\.?\b", "SCIENCE"),
    (r"\bMGMT\.?\b", "MANAGEMENT"),
    (r"&", " AND ")
]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    # Apostrophes:
    # VIGNAN'S -> VIGNANS
    text = text.replace("'", "")
    text = text.replace("’", "")

    for pattern, replacement in PHRASE_REPLACEMENTS:

        text = re.sub(
            pattern,
            f" {replacement} ",
            text
        )

    # Convert punctuation to spaces.
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
# NORMALIZE NAME
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

        words.append(word)

    return " ".join(words)


# ============================================================
# COMPACT NAME
#
# Removes spaces so:
#
# ANANTHA LAKSHMI
# ANANTHALAKSHMI
#
# can be compared.
# ============================================================

def compact_name(value):

    normalized = normalize_name(value)

    return re.sub(
        r"[^A-Z0-9]",
        "",
        normalized
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
# ACRONYM
# ============================================================

def get_acronym(name):

    if not name:
        return ""

    words = name.split()

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
# INITIAL / ACRONYM COMPACTING
#
# R V R -> RVR
# J C -> JC
# V S M -> VSM
# ============================================================

def get_initial_compact(name):

    normalized = normalize_name(
        name
    )

    words = normalized.split()

    initials = ""

    for word in words:

        if not word:
            continue

        initials += word[0]

    return initials


# ============================================================
# ACRONYM SCORE
# ============================================================

def acronym_score(
    current_name,
    official_name
):

    current_normalized = normalize_name(
        current_name
    )

    official_normalized = normalize_name(
        official_name
    )

    # --------------------------------------------------------
    # Full useful-word acronym
    # --------------------------------------------------------

    current_acronym = get_acronym(
        current_normalized
    )

    official_acronym = get_acronym(
        official_normalized
    )

    # --------------------------------------------------------
    # All-word initials
    # --------------------------------------------------------

    current_initials = get_initial_compact(
        current_normalized
    )

    official_initials = get_initial_compact(
        official_normalized
    )

    scores = []

    if current_acronym and official_acronym:

        if current_acronym == official_acronym:

            scores.append(1.0)

        else:

            scores.append(
                SequenceMatcher(
                    None,
                    current_acronym,
                    official_acronym
                ).ratio()
            )

    if current_initials and official_initials:

        if current_initials == official_initials:

            scores.append(1.0)

        else:

            scores.append(
                SequenceMatcher(
                    None,
                    current_initials,
                    official_initials
                ).ratio()
            )

    if not scores:
        return 0.0

    return max(scores)


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

    # Exact location.
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
# COMPACT SCORE
# ============================================================

def compact_score(
    current_name,
    official_name
):

    current = compact_name(
        current_name
    )

    official = compact_name(
        official_name
    )

    if not current or not official:
        return 0.0

    if current == official:
        return 1.0

    return SequenceMatcher(
        None,
        current,
        official
    ).ratio()


# ============================================================
# BRANCH FAMILIES
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
# IDENTITY WORDS
# ============================================================

def get_identity_words(name):

    tokens = get_tokens(
        normalize_name(name)
    )

    return (
        tokens
        & IDENTITY_WORDS
    )


def identity_penalty(
    current_name,
    official_name
):

    current_identity = (
        get_identity_words(
            current_name
        )
    )

    official_identity = (
        get_identity_words(
            official_name
        )
    )

    # --------------------------------------------------------
    # Important identity mismatch.
    # --------------------------------------------------------

    penalty = 0.0

    # University vs non-university.
    if (
        "UNIVERSITY" in official_identity
        and
        "UNIVERSITY" not in current_identity
    ):
        penalty += 0.16

    if (
        "UNIVERSITY" in current_identity
        and
        "UNIVERSITY" not in official_identity
    ):
        penalty += 0.16

    # Women-specific identity.
    official_women = (
        "WOMEN" in official_identity
        or
        "WOMENS" in official_identity
    )

    current_women = (
        "WOMEN" in current_identity
        or
        "WOMENS" in current_identity
    )

    if official_women != current_women:

        penalty += 0.18

    # Pharmacy.
    if (
        "PHARMACY" in official_identity
        !=
        "PHARMACY" in current_identity
    ):
        penalty += 0.15

    # Polytechnic.
    if (
        "POLYTECHNIC" in official_identity
        !=
        "POLYTECHNIC" in current_identity
    ):
        penalty += 0.15

    # Medical.
    if (
        "MEDICAL" in official_identity
        !=
        "MEDICAL" in current_identity
    ):
        penalty += 0.15

    # Dental.
    if (
        "DENTAL" in official_identity
        !=
        "DENTAL" in current_identity
    ):
        penalty += 0.15

    # Law.
    if (
        "LAW" in official_identity
        !=
        "LAW" in current_identity
    ):
        penalty += 0.15

    return min(
        penalty,
        0.40
    )


# ============================================================
# COMPLETE MATCH
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

    compact = compact_score(
        current_name,
        official_name
    )

    acronym = acronym_score(
        current_name,
        official_name
    )

    location = location_score(
        current_location,
        official_location
    )

    branches = branch_score(
        current_branches,
        official_branches
    )

    penalty = identity_penalty(
        current_name,
        official_name
    )

    # --------------------------------------------------------
    # Exact normalized name.
    # --------------------------------------------------------

    exact_normalized = (
        current_normalized
        ==
        official_normalized
        and
        current_normalized != ""
    )

    # --------------------------------------------------------
    # Exact compact name.
    # --------------------------------------------------------

    exact_compact = (
        compact == 1.0
    )

    # --------------------------------------------------------
    # Final score.
    #
    # Name evidence is dominant.
    # Location is strong.
    # Acronym helps abbreviations.
    # Branch is supporting only.
    # --------------------------------------------------------

    final = (
        0.28 * token
        + 0.13 * character
        + 0.22 * compact
        + 0.18 * acronym
        + 0.15 * location
        + 0.04 * branches
        - penalty
    )

    # Exact normalized name + exact location.
    if (
        exact_normalized
        and
        location == 1.0
    ):
        final = 1.0

    # Exact compact name + exact location.
    elif (
        exact_compact
        and
        location == 1.0
    ):
        final = max(
            final,
            0.98
        )

    final = max(
        0.0,
        min(
            final,
            1.0
        )
    )

    common_tokens = sorted(
        current_tokens
        &
        official_tokens
    )

    return {
        "token": token,
        "character": character,
        "compact": compact,
        "acronym": acronym,
        "location": location,
        "branch": branches,
        "identity_penalty": penalty,
        "final": final,
        "exact_normalized": exact_normalized,
        "exact_compact": exact_compact,
        "common_tokens": common_tokens
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 78)
print(
    "AP EAPCET COLLEGE MAPPING REVIEW - VERSION 5.1"
)
print("=" * 78)

print()
print("Loading official data...")

official = pd.read_csv(
    OFFICIAL_FILE
)

print(
    f"Official records: {len(official)}"
)


print()
print("Loading current EAMCET data...")

current = pd.read_csv(
    CURRENT_FILE
)

print(
    f"Current records: {len(current)}"
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
    -
    set(official.columns)
)

missing_current = (
    required_current
    -
    set(current.columns)
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
# BRANCH NORMALIZATION
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
# ENGINEERING OFFICIAL DATA
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
# CURRENT COLLEGE GROUPS
# ============================================================

current_groups = []

for college_name, group in current.groupby(
    "College Name",
    dropna=True
):

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
# OFFICIAL COLLEGE GROUPS
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
# EXACT NORMALIZED NAME INDEX
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
# MATCH
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
        f"{current_name[:70]}",
        end=""
    )


    # ========================================================
    # EXACT NORMALIZED CANDIDATES
    # ========================================================

    exact_candidates = (
        official_name_groups.get(
            current_normalized,
            []
        )
    )


    # ========================================================
    # CANDIDATE POOL
    # ========================================================

    if exact_candidates:

        candidate_pool = (
            exact_candidates
        )

    else:

        candidate_pool = (
            official_colleges
        )


    candidates = []


    if isinstance(
        candidate_pool,
        list
    ):

        iterator = iter(
            candidate_pool
        )

    else:

        iterator = (
            candidate_pool.iterrows()
        )


    for item in iterator:

        if isinstance(
            candidate_pool,
            list
        ):

            official_row = item

        else:

            _, official_row = item


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
            item["compact"],
            item["acronym"],
            item["token"]
        ),
        reverse=True
    )


    top = candidates[
        :TOP_CANDIDATES
    ]


    if not top:
        continue


    best = top[0]


    second_score = (
        top[1]["final"]
        if len(top) > 1
        else 0.0
    )


    margin = (
        best["final"]
        -
        second_score
    )


    exact_name = (
        best["exact_normalized"]
    )


    exact_location = (
        best["location"] == 1.0
    )


    exact_compact = (
        best["exact_compact"]
    )


    same_name_count = (
        len(exact_candidates)
    )


    # ========================================================
    # STATUS
    # ========================================================

    # --------------------------------------------------------
    # SAFE:
    #
    # Exact normalized name
    # + exact location
    # + unique official college.
    # --------------------------------------------------------

    if (
        exact_name
        and
        exact_location
        and
        same_name_count == 1
    ):

        status = "SAFE"
        confidence = "HIGH"

        reason = (
            "Exact normalized name + "
            "exact location + unique official college"
        )


    # --------------------------------------------------------
    # SAFE:
    #
    # Exact compact name + exact location.
    #
    # Example:
    # R.V.R. -> R V R
    # --------------------------------------------------------

    elif (
        exact_compact
        and
        exact_location
        and
        margin >= 0.10
        and
        best["identity_penalty"] == 0
    ):

        status = "SAFE"
        confidence = "HIGH"

        reason = (
            "Exact compact name + "
            "exact location + no identity conflict"
        )


    # --------------------------------------------------------
    # SAFE:
    #
    # Extremely strong acronym + name + location.
    # --------------------------------------------------------

    elif (
        best["acronym"] >= 0.98
        and
        best["compact"] >= 0.90
        and
        best["location"] >= 0.90
        and
        margin >= 0.10
        and
        best["identity_penalty"] == 0
    ):

        status = "SAFE"
        confidence = "HIGH"

        reason = (
            "Very strong compact/acronym/name/location match"
        )


    # --------------------------------------------------------
    # DUPLICATE NORMALIZED NAME
    # --------------------------------------------------------

    elif (
        exact_name
        and
        same_name_count > 1
    ):

        if (
            exact_location
            and
            margin >= 0.08
        ):

            status = "SAFE"
            confidence = "HIGH"

            reason = (
                "Exact normalized name + "
                "location distinguishes duplicate official names"
            )

        else:

            status = "MANUAL_REVIEW"
            confidence = "MEDIUM"

            reason = (
                "Multiple official colleges share "
                "the same normalized name"
            )


    # --------------------------------------------------------
    # STRONG FUZZY
    # --------------------------------------------------------

    elif (
        best["final"] >= 0.78
        and
        margin >= 0.10
        and
        (
            best["location"] >= 0.85
            or
            best["acronym"] >= 0.95
        )
    ):

        status = "MANUAL_REVIEW"
        confidence = "HIGH"

        reason = (
            "Strong fuzzy candidate; "
            "manual confirmation required"
        )


    # --------------------------------------------------------
    # MEDIUM FUZZY
    # --------------------------------------------------------

    elif (
        best["final"] >= 0.60
        and
        margin >= 0.08
    ):

        status = "MANUAL_REVIEW"
        confidence = "MEDIUM"

        reason = (
            "Good candidate but not safe for "
            "automatic mapping"
        )


    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    elif (
        len(top) >= 2
        and
        margin < 0.08
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


    candidate_2 = (
        top[1]
        if len(top) >= 2
        else None
    )

    candidate_3 = (
        top[2]
        if len(top) >= 3
        else None
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

            "Compact Score":
                round(
                    best["compact"],
                    4
                ),

            "Acronym Score":
                round(
                    best["acronym"],
                    4
                ),

            "Location Score":
                round(
                    best["location"],
                    4
                ),

            "Branch Score":
                round(
                    best["branch"],
                    4
                ),

            "Identity Penalty":
                round(
                    best["identity_penalty"],
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
# DATAFRAME
# ============================================================

report = pd.DataFrame(
    results
)


# ============================================================
# SORT
# ============================================================

status_order = {
    "SAFE": 0,
    "MANUAL_REVIEW": 1,
    "UNMATCHED": 2
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
print("=" * 78)
print(
    "VERSION 5.1 MAPPING REVIEW SUMMARY"
)
print("=" * 78)

print()

print(
    report[
        "Match Status"
    ]
    .value_counts()
    .to_string()
)


print()

for status in [
    "SAFE",
    "MANUAL_REVIEW",
    "UNMATCHED"
]:

    count = (
        report[
            "Match Status"
        ]
        ==
        status
    ).sum()

    print(
        f"{status}: {count}"
    )


print(
    f"Total current colleges: "
    f"{len(report)}"
)


# ============================================================
# SAFE EXAMPLES
# ============================================================

print()
print("=" * 78)
print(
    "SAFE MATCH EXAMPLES"
)
print("=" * 78)


safe = report[
    report[
        "Match Status"
    ]
    == "SAFE"
].head(25)


if safe.empty:

    print(
        "No SAFE matches."
    )

else:

    for _, row in safe.iterrows():

        print()

        print(
            f"Current: "
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
            f"  Compact score: "
            f"{row['Compact Score']}"
        )

        print(
            f"  Acronym score: "
            f"{row['Acronym Score']}"
        )

        print(
            f"  Final score: "
            f"{row['Final Score']}"
        )

        print(
            f"  Reason: "
            f"{row['Reason']}"
        )


# ============================================================
# MANUAL EXAMPLES
# ============================================================

print()
print("=" * 78)
print(
    "MANUAL REVIEW EXAMPLES"
)
print("=" * 78)


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
print("=" * 78)
print(
    "UNMATCHED EXAMPLES"
)
print("=" * 78)


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
print("=" * 78)
print(
    "VERSION 5.1 REPORT SAVED"
)
print("=" * 78)

print(
    OUTPUT_FILE
)

print()

print(
    "IMPORTANT:"
)

print(
    "This is still a REVIEW file."
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
    "No permanent mapping was created."
)