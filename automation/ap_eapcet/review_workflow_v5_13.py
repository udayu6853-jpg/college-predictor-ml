
"""
AP EAPCET Manual Review Workflow v5.13

Input:
    ap_eapcet_manual_review_priority_v5_12.csv

Outputs:
    ap_eapcet_human_review_v5_13.csv
    ap_eapcet_confirmed_mapping_v5_13.csv
    ap_eapcet_review_validation_v5_13.txt

Safety:
- Does not modify V5.10 or V5.12 files.
- Does not automatically confirm mappings.
- Preserves existing reviewer fields when identity keys match.
- Exports only complete, explicitly confirmed mappings.
"""

from pathlib import Path
import pandas as pd


VERSION = "5.13"

REVIEW_FIELDS = [
    "Reviewer Decision",
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewer Notes",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
]

ALLOWED_DECISIONS = {
    "PENDING",
    "CONFIRMED",
    "REJECTED",
    "NEEDS_MORE_EVIDENCE",
    "NO_MATCH",
}

IDENTITY_FIELDS = [
    "Current Canonical Name",
    "Current Canonical Location",
    "Current College Name",
    "Current Location",
]


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def record_key(row):
    return " | ".join(
        clean(row.get(field, "")).casefold()
        for field in IDENTITY_FIELDS
    )


def preserve_reviews(previous_path):
    """Read prior review fields without modifying the source."""
    preserved = {}

    if not previous_path.exists():
        return preserved

    previous = pd.read_csv(
        previous_path,
        dtype=str,
        keep_default_na=False,
    )

    if not all(field in previous.columns for field in IDENTITY_FIELDS):
        print(
            "WARNING: Previous review file lacks identity columns. "
            "Its reviewer entries will not be imported."
        )
        return preserved

    for _, row in previous.iterrows():
        key = record_key(row)

        if not key.replace("|", "").strip():
            continue

        values = {
            field: clean(row.get(field, ""))
            for field in REVIEW_FIELDS
            if field in previous.columns
        }

        # Do not silently overwrite different reviewer entries
        # if duplicate identity keys exist.
        if key in preserved and preserved[key] != values:
            print(
                "WARNING: Conflicting previous review entries found "
                "for one identity key. Review this college manually."
            )
            continue

        preserved[key] = values

    return preserved


def validate_reviews(df):
    errors = []
    warnings = []

    for index, row in df.iterrows():
        decision = clean(
            row.get("Reviewer Decision", "PENDING")
        ).upper()

        college = clean(row.get("Current College Name", ""))

        if decision not in ALLOWED_DECISIONS:
            errors.append(
                f"Row {index + 2} ({college}): "
                f"invalid decision '{decision}'."
            )
            continue

        if decision == "CONFIRMED":
            required = [
                "Verified Official Name",
                "Verified Location",
                "Verified Official Code",
                "Evidence URL",
            ]

            missing = [
                field for field in required
                if not clean(row.get(field, ""))
            ]

            if missing:
                errors.append(
                    f"Row {index + 2} ({college}): "
                    "CONFIRMED but missing "
                    + ", ".join(missing)
                )

            evidence = clean(row.get("Evidence URL", ""))

            if evidence and not evidence.lower().startswith(
                ("https://", "http://")
            ):
                warnings.append(
                    f"Row {index + 2} ({college}): "
                    "evidence URL is not HTTP/HTTPS."
                )

    return errors, warnings


def build_confirmed_file(df):
    """Export only complete, explicitly human-confirmed rows."""
    output_columns = [
        "Current College Name",
        "Current Location",
        "Verified Official Name",
        "Verified Location",
        "Verified Official Code",
        "Evidence URL",
        "Reviewer Decision",
        "Reviewed By",
        "Review Date (YYYY-MM-DD)",
        "Reviewer Notes",
        "Review Priority",
        "Review Priority Score",
    ]

    rows = []

    for _, row in df.iterrows():
        decision = clean(
            row.get("Reviewer Decision", "")
        ).upper()

        if decision != "CONFIRMED":
            continue

        required = [
            "Verified Official Name",
            "Verified Location",
            "Verified Official Code",
            "Evidence URL",
        ]

        if not all(clean(row.get(field, "")) for field in required):
            continue

        rows.append({
            column: clean(row.get(column, ""))
            for column in output_columns
        })

    return pd.DataFrame(rows, columns=output_columns)


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "automation" / "output"

    input_path = (
        output_dir
        / "ap_eapcet_manual_review_priority_v5_12.csv"
    )

    previous_path = (
        output_dir / "ap_eapcet_human_review_v5_10.csv"
    )

    review_path = (
        output_dir / "ap_eapcet_human_review_v5_13.csv"
    )

    confirmed_path = (
        output_dir / "ap_eapcet_confirmed_mapping_v5_13.csv"
    )

    summary_path = (
        output_dir / "ap_eapcet_review_validation_v5_13.txt"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"V5.12 input not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = [
        "Current College Name",
        "Current Location",
        "Review Priority",
        "Review Priority Score",
    ]

    missing = [
        field for field in required_columns
        if field not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required V5.12 columns are missing: "
            + ", ".join(missing)
        )

    previous_reviews = preserve_reviews(previous_path)

    for field in REVIEW_FIELDS:
        if field not in df.columns:
            df[field] = ""

    imported = 0

    for index, row in df.iterrows():
        old_values = previous_reviews.get(record_key(row), {})

        for field, value in old_values.items():
            # Only import non-empty prior reviewer entries.
            # A blank prior cell must not erase current data.
            if value:
                df.at[index, field] = value
                imported += 1

        if not clean(df.at[index, "Reviewer Decision"]):
            df.at[index, "Reviewer Decision"] = "PENDING"

    errors, warnings = validate_reviews(df)

    df.to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig",
    )

    confirmed_df = build_confirmed_file(df)

    confirmed_df.to_csv(
        confirmed_path,
        index=False,
        encoding="utf-8-sig",
    )

    pending_count = int(
        df["Reviewer Decision"].str.upper().eq("PENDING").sum()
    )

    summary = [
        f"AP EAPCET REVIEW WORKFLOW V{VERSION}",
        "=" * 55,
        f"Input records: {len(df)}",
        f"Pending decisions: {pending_count}",
        f"Imported reviewer field values: {imported}",
        f"Complete confirmed mappings exported: {len(confirmed_df)}",
        f"Validation errors: {len(errors)}",
        f"Validation warnings: {len(warnings)}",
        "",
        "VALIDATION ERRORS:",
        *(errors if errors else ["None"]),
        "",
        "WARNINGS:",
        *(warnings if warnings else ["None"]),
        "",
        "Safety: No mappings were automatically confirmed.",
        "Official evidence still requires human verification.",
    ]

    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print(f"\nV{VERSION} workflow completed.")
    print(f"Input records: {len(df)}")
    print(f"Pending decisions: {pending_count}")
    print(f"Confirmed mappings exported: {len(confirmed_df)}")
    print(f"Validation errors: {len(errors)}")
    print(f"Validation warnings: {len(warnings)}")
    print(f"\nReview file: {review_path}")
    print(f"Confirmed file: {confirmed_path}")
    print(f"Summary file: {summary_path}")

    if errors:
        print("\nReview the validation errors before relying on confirmations.")

    print("\nNo mappings were automatically confirmed.")


if __name__ == "__main__":
    main()
