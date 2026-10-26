import os
import pandas as pd


FILE = os.path.join(
    "automation",
    "output",
    "ap_eapcet_normalized.csv"
)


def main():

    print("=" * 70)
    print("AP EAPCET OFFICIAL DATA AUDIT")
    print("=" * 70)

    df = pd.read_csv(FILE)

    print()
    print("TOTAL RECORDS:", len(df))

    print()
    print("=" * 70)
    print("UNIQUE COLLEGES")
    print("=" * 70)

    print(
        df["College Name"]
        .astype(str)
        .str.strip()
        .str.upper()
        .nunique()
    )

    print()
    print("=" * 70)
    print("BRANCHES")
    print("=" * 70)

    branch_counts = (
        df.groupby("Branch")["College Name"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print(
        branch_counts.to_string()
    )

    print()
    print("=" * 70)
    print("COLLEGE FEES")
    print("=" * 70)

    fee_summary = (
        df.groupby("Branch")["Eamcet Fees"]
        .agg(
            [
                "count",
                "min",
                "max",
                "mean"
            ]
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        fee_summary.to_string()
    )

    print()
    print("=" * 70)
    print("SUPPORTED PROJECT BRANCHES")
    print("=" * 70)

    supported = [
        "CSE",
        "ECE",
        "MEC",
        "AIM",
        "CSD",
        "CSM",
        "INF"
    ]

    supported_df = df[
        df["Branch"].isin(
            supported
        )
    ]

    print(
        "Records:",
        len(supported_df)
    )

    print(
        "Colleges:",
        supported_df[
            "College Name"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .nunique()
    )

    print()
    print("=" * 70)
    print("SAMPLE COLLEGES")
    print("=" * 70)

    colleges = (
        df[
            [
                "College Code",
                "College Name",
                "Location",
                "Branch",
                "Cutoff Rank",
                "Eamcet Fees"
            ]
        ]
        .drop_duplicates()
        .head(30)
    )

    print(
        colleges.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()