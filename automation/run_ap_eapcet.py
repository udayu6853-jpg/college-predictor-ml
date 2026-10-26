from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from automation.ap_eapcet.fetch import (
    download_last_rank_file
)

from automation.ap_eapcet.parser import (
    read_official_excel,
    build_dataframe,
    normalize_ap_eapcet
)

from automation.ap_eapcet.validator import (
    validate_ap_eapcet
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "automation"
    / "output"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ap_eapcet_normalized.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 70)
    print("COLLEGE PREDICTOR")
    print("AP EAPCET AUTOMATED DATA PIPELINE")
    print("#" * 70)

    # --------------------------------------------------------
    # 1. DOWNLOAD
    # --------------------------------------------------------

    excel_bytes = (
        download_last_rank_file()
    )

    # --------------------------------------------------------
    # 2. READ EXCEL
    # --------------------------------------------------------

    raw = read_official_excel(
        excel_bytes
    )

    # --------------------------------------------------------
    # 3. FIND HEADER + BUILD DATAFRAME
    # --------------------------------------------------------

    df = build_dataframe(
        raw
    )

    print()
    print(
        f"Raw AP records: {len(df):,}"
    )

    # --------------------------------------------------------
    # 4. NORMALIZE
    # --------------------------------------------------------

    normalized = normalize_ap_eapcet(
        df
    )

    print()
    print(
        f"Normalized records: "
        f"{len(normalized):,}"
    )

    # --------------------------------------------------------
    # 5. VALIDATE
    # --------------------------------------------------------

    validated = validate_ap_eapcet(
        normalized
    )

    # --------------------------------------------------------
    # 6. CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 7. SAVE
    # --------------------------------------------------------

    validated.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("AUTOMATION SUCCESSFUL")
    print("=" * 70)

    print()
    print(
        f"Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        f"Records saved: "
        f"{len(validated):,}"
    )

    print()
    print("Sample records:")

    print(
        validated[
            [
                "College Name",
                "Branch",
                "Cutoff Rank",
                "Eamcet Fees",
                "Location"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":

    main()