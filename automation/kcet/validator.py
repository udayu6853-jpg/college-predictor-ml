import pandas as pd


# ============================================================
# KCET DATA VALIDATION
# ============================================================

REQUIRED_COLUMNS = [
    "College Name",
    "Branch",
    "Cutoff Rank"
]


def validate_kcet_dataframe(df):

    print()
    print("Validating KCET data...")

    if df.empty:

        raise ValueError(
            "KCET dataset is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    ).copy()

    # Clean text fields
    df["College Name"] = (
        df["College Name"]
        .astype(str)
        .str.strip()
    )

    df["Branch"] = (
        df["Branch"]
        .astype(str)
        .str.strip()
    )

    # Convert cutoff to numeric
    df["Cutoff Rank"] = pd.to_numeric(
        df["Cutoff Rank"],
        errors="coerce"
    )

    # Remove invalid cutoff rows
    df = df[
        df["Cutoff Rank"].notna()
    ]

    # Rank must be positive
    df = df[
        df["Cutoff Rank"] > 0
    ]

    print(
        f"Valid KCET rows: {len(df):,}"
    )

    return df