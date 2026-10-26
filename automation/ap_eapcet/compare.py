import os
import pandas as pd


NORMALIZED_FILE = os.path.join(
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


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def clean_number(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def main():

    print("=" * 70)
    print("AP EAPCET DATA COMPARISON")
    print("=" * 70)

    # ---------------------------------------------------------
    # CHECK FILES
    # ---------------------------------------------------------

    if not os.path.exists(NORMALIZED_FILE):
        print()
        print("ERROR: Normalized AP EAPCET file not found.")
        print(NORMALIZED_FILE)
        return

    if not os.path.exists(CURRENT_FILE):
        print()
        print("ERROR: Current EAMCET dataset not found.")
        print(CURRENT_FILE)
        return

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print()
    print("Loading official AP EAPCET data...")

    official = pd.read_csv(
        NORMALIZED_FILE
    )

    print(
        f"Official records: {len(official):,}"
    )

    print()
    print("Loading current project EAMCET data...")

    current = pd.read_csv(
        CURRENT_FILE
    )

    print(
        f"Current records: {len(current):,}"
    )

    # ---------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # ---------------------------------------------------------

    official.columns = [
        str(column).strip()
        for column in official.columns
    ]

    current.columns = [
        str(column).strip()
        for column in current.columns
    ]

    # ---------------------------------------------------------
    # NORMALIZE VALUES
    # ---------------------------------------------------------

    official["College Key"] = (
        official["College Name"]
        .map(clean_text)
    )

    official["Branch Key"] = (
        official["Branch"]
        .map(clean_text)
    )

    current["College Key"] = (
        current["College Name"]
        .map(clean_text)
    )

    current["Branch Key"] = (
        current["Branch"]
        .map(clean_text)
    )

    # ---------------------------------------------------------
    # CREATE UNIQUE KEY
    # ---------------------------------------------------------

    official["Record Key"] = (
        official["College Key"]
        + "||"
        + official["Branch Key"]
    )

    current["Record Key"] = (
        current["College Key"]
        + "||"
        + current["Branch Key"]
    )

    # ---------------------------------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------------------------------

    official = official.drop_duplicates(
        subset=["Record Key"]
    )

    current = current.drop_duplicates(
        subset=["Record Key"]
    )

    # ---------------------------------------------------------
    # KEY SETS
    # ---------------------------------------------------------

    official_keys = set(
        official["Record Key"]
    )

    current_keys = set(
        current["Record Key"]
    )

    new_keys = official_keys - current_keys

    removed_keys = current_keys - official_keys

    common_keys = (
        official_keys & current_keys
    )

    # ---------------------------------------------------------
    # NEW RECORDS
    # ---------------------------------------------------------

    new_records = official[
        official["Record Key"].isin(new_keys)
    ].copy()

    # ---------------------------------------------------------
    # REMOVED RECORDS
    # ---------------------------------------------------------

    removed_records = current[
        current["Record Key"].isin(removed_keys)
    ].copy()

    # ---------------------------------------------------------
    # COMPARE COMMON RECORDS
    # ---------------------------------------------------------

    official_common = (
        official[
            official["Record Key"].isin(common_keys)
        ]
        .set_index("Record Key")
    )

    current_common = (
        current[
            current["Record Key"].isin(common_keys)
        ]
        .set_index("Record Key")
    )

    changed_rows = []

    for key in common_keys:

        official_row = official_common.loc[key]

        current_row = current_common.loc[key]

        official_cutoff = clean_number(
            official_row.get("Cutoff Rank")
        )

        current_cutoff = clean_number(
            current_row.get("Cutoff Rank")
        )

        official_fee = clean_number(
            official_row.get("Eamcet Fees")
        )

        current_fee = clean_number(
            current_row.get("Eamcet Fees")
        )

        cutoff_changed = (
            official_cutoff != current_cutoff
        )

        fee_changed = (
            official_fee != current_fee
        )

        if cutoff_changed or fee_changed:

            changed_rows.append(
                {
                    "College Name":
                        official_row.get(
                            "College Name",
                            ""
                        ),

                    "Branch":
                        official_row.get(
                            "Branch",
                            ""
                        ),

                    "Old Cutoff Rank":
                        current_cutoff,

                    "New Cutoff Rank":
                        official_cutoff,

                    "Old Eamcet Fees":
                        current_fee,

                    "New Eamcet Fees":
                        official_fee,

                    "Cutoff Changed":
                        cutoff_changed,

                    "Fee Changed":
                        fee_changed
                }
            )

    changed_records = pd.DataFrame(
        changed_rows
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"Official records       : {len(official):,}"
    )

    print(
        f"Current records        : {len(current):,}"
    )

    print(
        f"Common records         : {len(common_keys):,}"
    )

    print(
        f"New records            : {len(new_records):,}"
    )

    print(
        f"Changed records        : {len(changed_records):,}"
    )

    print(
        f"Removed records        : {len(removed_records):,}"
    )

    # ---------------------------------------------------------
    # BRANCH ANALYSIS
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("OFFICIAL BRANCHES")
    print("=" * 70)

    official_branches = sorted(
        official["Branch"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    for branch in official_branches:
        count = len(
            official[
                official["Branch"]
                .astype(str)
                .str.strip()
                == branch
            ]
        )

        print(
            f"{branch:<10} {count:>5} records"
        )

    # ---------------------------------------------------------
    # SAVE REPORTS
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    new_file = os.path.join(
        OUTPUT_DIR,
        "ap_eapcet_new_records.csv"
    )

    changed_file = os.path.join(
        OUTPUT_DIR,
        "ap_eapcet_changed_records.csv"
    )

    removed_file = os.path.join(
        OUTPUT_DIR,
        "ap_eapcet_removed_records.csv"
    )

    new_records.to_csv(
        new_file,
        index=False
    )

    changed_records.to_csv(
        changed_file,
        index=False
    )

    removed_records.to_csv(
        removed_file,
        index=False
    )

    # ---------------------------------------------------------
    # DISPLAY SAMPLES
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("NEW RECORD SAMPLE")
    print("=" * 70)

    if len(new_records) > 0:

        print(
            new_records[
                [
                    "College Name",
                    "Branch",
                    "Cutoff Rank",
                    "Eamcet Fees"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    else:

        print("No new records.")

    print()
    print("=" * 70)
    print("CHANGED RECORD SAMPLE")
    print("=" * 70)

    if len(changed_records) > 0:

        print(
            changed_records
            .head(10)
            .to_string(index=False)
        )

    else:

        print("No changed records.")

    print()
    print("=" * 70)
    print("REMOVED RECORD SAMPLE")
    print("=" * 70)

    if len(removed_records) > 0:

        print(
            removed_records[
                [
                    "College Name",
                    "Branch"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    else:

        print("No removed records.")

    # ---------------------------------------------------------
    # OUTPUT FILES
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("REPORT FILES")
    print("=" * 70)

    print()
    print(
        "New:"
    )
    print(
        new_file
    )

    print()
    print(
        "Changed:"
    )
    print(
        changed_file
    )

    print()
    print(
        "Removed:"
    )
    print(
        removed_file
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()