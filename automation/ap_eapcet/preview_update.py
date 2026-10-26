import os
import re
import pandas as pd


OFFICIAL_FILE = os.path.join(
    "automation",
    "output",
    "ap_eapcet_normalized.csv"
)

CURRENT_FILE = os.path.join(
    "dataset",
    "eamcet_colleges.csv"
)

OUTPUT_DIR = os.path.join(
    "automation",
    "output"
)


# ============================================================
# BRANCH MAPPING
# ============================================================

BRANCH_MAPPING = {
    "CSE": "CSE",

    "ECE": "ECE",

    "MEC": "ME",

    # AI / ML related official branches
    "CSM": "AIML",
    "CSD": "AIML",
    "INF": "AIML",
    "AIM": "AIML",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# NUMBER NORMALIZATION
# ============================================================

def normalize_number(value):

    if pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("Loading official AP EAPCET data...")

    official = pd.read_csv(
        OFFICIAL_FILE
    )

    print(
        f"Official rows: {len(official):,}"
    )

    print()
    print("Loading current project EAMCET data...")

    current = pd.read_csv(
        CURRENT_FILE
    )

    print(
        f"Current rows: {len(current):,}"
    )

    return official, current


# ============================================================
# PREPARE OFFICIAL DATA
# ============================================================

def prepare_official_data(df):

    df = df.copy()

    # Keep only branches supported
    df["Official Branch"] = (
        df["Branch"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Project Branch"] = (
        df["Official Branch"]
        .map(BRANCH_MAPPING)
    )

    df = df[
        df["Project Branch"].notna()
    ].copy()

    # Normalize college name
    df["College Key"] = (
        df["College Name"]
        .map(normalize_text)
    )

    # Normalize branch
    df["Branch Key"] = (
        df["Project Branch"]
        .map(normalize_text)
    )

    # Create matching key
    df["Match Key"] = (
        df["College Key"]
        + "||"
        + df["Branch Key"]
    )

    return df


# ============================================================
# PREPARE CURRENT DATA
# ============================================================

def prepare_current_data(df):

    df = df.copy()

    df["College Key"] = (
        df["College Name"]
        .map(normalize_text)
    )

    df["Branch Key"] = (
        df["Branch"]
        .map(normalize_text)
    )

    df["Match Key"] = (
        df["College Key"]
        + "||"
        + df["Branch Key"]
    )

    return df


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    print("=" * 70)
    print("AP EAPCET SAFE UPDATE PREVIEW")
    print("=" * 70)

    if not os.path.exists(
        OFFICIAL_FILE
    ):
        print()
        print(
            "ERROR: Official normalized file not found."
        )
        return

    if not os.path.exists(
        CURRENT_FILE
    ):
        print()
        print(
            "ERROR: Current EAMCET dataset not found."
        )
        return

    official, current = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    official = prepare_official_data(
        official
    )

    current = prepare_current_data(
        current
    )

    print()
    print(
        "Official supported records:"
    )
    print(
        len(official)
    )

    print()
    print(
        "Official supported branches:"
    )

    print(
        sorted(
            official["Project Branch"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    official = official.drop_duplicates(
        subset=["Match Key"],
        keep="first"
    )

    current = current.drop_duplicates(
        subset=["Match Key"],
        keep="first"
    )

    # --------------------------------------------------------
    # CURRENT KEYS
    # --------------------------------------------------------

    current_keys = set(
        current["Match Key"]
    )

    official_keys = set(
        official["Match Key"]
    )

    # --------------------------------------------------------
    # NEW
    # --------------------------------------------------------

    new_keys = (
        official_keys
        - current_keys
    )

    new_records = official[
        official["Match Key"].isin(
            new_keys
        )
    ].copy()

    # --------------------------------------------------------
    # COMMON
    # --------------------------------------------------------

    common_keys = (
        official_keys
        & current_keys
    )

    official_common = (
        official[
            official["Match Key"].isin(
                common_keys
            )
        ]
        .set_index("Match Key")
    )

    current_common = (
        current[
            current["Match Key"].isin(
                common_keys
            )
        ]
        .set_index("Match Key")
    )

    # --------------------------------------------------------
    # CHANGE DETECTION
    # --------------------------------------------------------

    changed = []
    unchanged = []

    for key in common_keys:

        official_row = (
            official_common.loc[key]
        )

        current_row = (
            current_common.loc[key]
        )

        official_cutoff = normalize_number(
            official_row.get(
                "Cutoff Rank"
            )
        )

        current_cutoff = normalize_number(
            current_row.get(
                "Cutoff Rank"
            )
        )

        official_fee = normalize_number(
            official_row.get(
                "Eamcet Fees"
            )
        )

        current_fee = normalize_number(
            current_row.get(
                "Eamcet Fees"
            )
        )

        cutoff_changed = (
            official_cutoff
            !=
            current_cutoff
        )

        fee_changed = (
            official_fee
            !=
            current_fee
        )

        if (
            cutoff_changed
            or fee_changed
        ):

            changed.append(
                {
                    "College Name":
                        official_row[
                            "College Name"
                        ],

                    "Branch":
                        official_row[
                            "Project Branch"
                        ],

                    "Official Branch":
                        official_row[
                            "Official Branch"
                        ],

                    "Old Cutoff":
                        current_cutoff,

                    "New Cutoff":
                        official_cutoff,

                    "Old Fee":
                        current_fee,

                    "New Fee":
                        official_fee,

                    "Cutoff Changed":
                        cutoff_changed,

                    "Fee Changed":
                        fee_changed
                }
            )

        else:

            unchanged.append(
                {
                    "College Name":
                        official_row[
                            "College Name"
                        ],

                    "Branch":
                        official_row[
                            "Project Branch"
                        ]
                }
            )

    changed_records = pd.DataFrame(
        changed
    )

    unchanged_records = pd.DataFrame(
        unchanged
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    new_file = os.path.join(
        OUTPUT_DIR,
        "ap_eapcet_preview_new.csv"
    )

    changed_file = os.path.join(
        OUTPUT_DIR,
        "ap_eapcet_preview_changed.csv"
    )

    unchanged_file = os.path.join(
        OUTPUT_DIR,
        "ap_eapcet_preview_unchanged.csv"
    )

    new_records.to_csv(
        new_file,
        index=False
    )

    changed_records.to_csv(
        changed_file,
        index=False
    )

    unchanged_records.to_csv(
        unchanged_file,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAFE UPDATE PREVIEW")
    print("=" * 70)

    print()
    print(
        f"Current project records : {len(current):,}"
    )

    print(
        f"Official supported      : {len(official):,}"
    )

    print(
        f"Common records          : {len(common_keys):,}"
    )

    print(
        f"New records             : {len(new_records):,}"
    )

    print(
        f"Changed records         : {len(changed_records):,}"
    )

    print(
        f"Unchanged records       : {len(unchanged_records):,}"
    )

    # --------------------------------------------------------
    # BRANCH SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NEW RECORDS BY BRANCH")
    print("=" * 70)

    if len(new_records) > 0:

        print(
            new_records[
                "Project Branch"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "No new records."
        )

    print()
    print("=" * 70)
    print("CHANGED RECORDS BY BRANCH")
    print("=" * 70)

    if len(changed_records) > 0:

        print(
            changed_records[
                "Branch"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "No changed records."
        )

    # --------------------------------------------------------
    # SAMPLE NEW
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NEW RECORD SAMPLE")
    print("=" * 70)

    if len(new_records) > 0:

        print(
            new_records[
                [
                    "College Name",
                    "Project Branch",
                    "Official Branch",
                    "Cutoff Rank",
                    "Eamcet Fees"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No new records."
        )

    # --------------------------------------------------------
    # SAMPLE CHANGES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHANGED RECORD SAMPLE")
    print("=" * 70)

    if len(changed_records) > 0:

        print(
            changed_records
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No changed records."
        )

    # --------------------------------------------------------
    # REPORT FILES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREVIEW FILES")
    print("=" * 70)

    print()
    print(new_file)

    print(changed_file)

    print(unchanged_file)

    print()
    print("=" * 70)
    print("NO DATABASE WAS MODIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()