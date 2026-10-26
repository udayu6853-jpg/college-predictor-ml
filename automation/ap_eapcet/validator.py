import pandas as pd


# ============================================================
# AP EAPCET VALIDATOR
# ============================================================

REQUIRED_COLUMNS = [
    "College Name",
    "Branch",
    "Cutoff Rank",
    "Eamcet Fees",
    "Location",
    "State"
]


def validate_ap_eapcet(df):

    print()
    print("Validating normalized AP EAPCET data...")

    if df.empty:

        raise ValueError(
            "AP EAPCET dataset is empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # CUTOFF VALIDATION
    # --------------------------------------------------------

    invalid_cutoff = (
        df["Cutoff Rank"]
        .isna()
        .sum()
    )

    if invalid_cutoff:

        print(
            f"Warning: {invalid_cutoff} "
            "records have invalid cutoff."
        )

    df = df[
        df["Cutoff Rank"].notna()
    ]

    df = df[
        df["Cutoff Rank"] > 0
    ]

    # --------------------------------------------------------
    # FEE VALIDATION
    # --------------------------------------------------------

    df["Eamcet Fees"] = pd.to_numeric(
        df["Eamcet Fees"],
        errors="coerce"
    )

    df = df[
        df["Eamcet Fees"].notna()
    ]

    df = df[
        df["Eamcet Fees"] >= 0
    ]

    # --------------------------------------------------------
    # COLLEGE NAME
    # --------------------------------------------------------

    df = df[
        df["College Name"]
        .astype(str)
        .str.strip()
        .str.len()
        > 0
    ]

    # --------------------------------------------------------
    # BRANCH
    # --------------------------------------------------------

    df = df[
        df["Branch"]
        .astype(str)
        .str.strip()
        .str.len()
        > 0
    ]

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_count = (
        df.duplicated(
            subset=[
                "College Code",
                "Branch"
            ]
        )
        .sum()
    )

    if duplicate_count:

        print(
            f"Removing {duplicate_count} "
            "duplicate records."
        )

        df = df.drop_duplicates(
            subset=[
                "College Code",
                "Branch"
            ],
            keep="first"
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    print()
    print(
        f"Valid records: {len(df):,}"
    )

    print(
        f"Unique colleges: "
        f"{df['College Code'].nunique():,}"
    )

    print(
        f"Unique branches: "
        f"{df['Branch'].nunique():,}"
    )

    return df