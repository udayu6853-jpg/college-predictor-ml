
from pathlib import Path
import hashlib
import pandas as pd


# ---------------------------------------------------------
# AP EAPCET V5.16 — Stable IDs and Duplicate Review
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

V515_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_15.csv"
V516_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_16.csv"

DUPLICATE_REPORT = (
    OUTPUT_DIR / "ap_eapcet_duplicate_identity_report_v5_16.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "ap_eapcet_review_validation_summary_v5_16.txt"
)

ID_COLUMN = "Stable Review ID"

IDENTITY_COLUMNS = [
    "Current College Name",
    "Current Location",
    "Current Canonical Name",
    "Current Canonical Location",
]

REVIEW_FIELDS = [
    "Reviewer Decision",
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewer Notes",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
    "Decision Reason",
]


def normalize(value):
    """Normalize text for duplicate comparison."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().casefold().split())


def make_identity_key(row):
    """Create a comparison key from the available identity fields."""
    return "||".join(
        normalize(row.get(column, ""))
        for column in IDENTITY_COLUMNS
    )


def make_stable_id(identity_key, occurrence):
    """Generate a deterministic ID for an identity and its occurrence."""
    digest = hashlib.sha256(
        identity_key.encode("utf-8")
    ).hexdigest()[:12].upper()

    return f"AP16-{digest}-{occurrence:03d}"


def load_input():
    """
    First run: read V5.15.
    Later runs: read V5.16 itself so its manual edits are preserved.
    """
    if V516_FILE.exists():
        return V516_FILE

    if V515_FILE.exists():
        return V515_FILE

    raise FileNotFoundError(
        "Neither the V5.15 nor V5.16 review CSV was found."
    )


def assign_stable_ids(df):
    """Preserve existing IDs and assign IDs to records without one."""
    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = ""

    df[ID_COLUMN] = df[ID_COLUMN].fillna("").astype(str).str.strip()

    occurrence_counts = {}
    used_ids = set(df.loc[df[ID_COLUMN] != "", ID_COLUMN])

    for index in df.index:
        if df.at[index, ID_COLUMN]:
            continue

        key = make_identity_key(df.loc[index])

        # Include row content if all identity fields are empty.
        if not key.replace("|", "").strip():
            key = "EMPTY-IDENTITY||" + "||".join(
                normalize(df.at[index, column])
                for column in df.columns
                if column != ID_COLUMN
            )

        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
        occurrence = occurrence_counts[key]

        candidate_id = make_stable_id(key, occurrence)

        # Avoid accidental collision with IDs already in the file.
        while candidate_id in used_ids:
            occurrence += 1
            candidate_id = make_stable_id(key, occurrence)

        df.at[index, ID_COLUMN] = candidate_id
        used_ids.add(candidate_id)

    return df


def build_duplicate_report(df):
    """Report repeated names and repeated composite identities."""
    report = df.copy()

    name_keys = report["Current College Name"].map(normalize)

    identity_keys = report.apply(make_identity_key, axis=1)

    report["_normalized_name"] = name_keys
    report["_identity_key"] = identity_keys

    report["Duplicate Name Count"] = (
        report["_normalized_name"].map(
            report["_normalized_name"].value_counts()
        )
    )

    report["Duplicate Identity Count"] = (
        report["_identity_key"].map(
            report["_identity_key"].value_counts()
        )
    )

    repeated_name = report["Duplicate Name Count"] > 1
    repeated_identity = report["Duplicate Identity Count"] > 1

    report["Duplicate Review Flag"] = ""

    report.loc[
        repeated_name,
        "Duplicate Review Flag"
    ] = "REPEATED COLLEGE NAME - CHECK LOCATION"

    report.loc[
        repeated_identity,
        "Duplicate Review Flag"
    ] = report.loc[
        repeated_identity,
        "Duplicate Review Flag"
    ].apply(
        lambda existing:
            (existing + "; " if existing else "")
            + "REPEATED COMPOSITE IDENTITY - INVESTIGATE"
    )

    report = report.loc[
        repeated_name | repeated_identity
    ].copy()

    report.drop(
        columns=["_normalized_name", "_identity_key"],
        inplace=True,
        errors="ignore",
    )

    # Put useful review fields first.
    preferred = [
        ID_COLUMN,
        "Current College Name",
        "Current Location",
        "Current Canonical Name",
        "Current Canonical Location",
        "Reviewer Decision",
        "Duplicate Name Count",
        "Duplicate Identity Count",
        "Duplicate Review Flag",
        "Decision Reason",
    ]

    ordered = [column for column in preferred if column in report.columns]
    remaining = [column for column in report.columns if column not in ordered]

    return report[ordered + remaining]


def write_summary(df, duplicate_report, source_file):
    decisions = (
        df["Reviewer Decision"]
        if "Reviewer Decision" in df.columns
        else pd.Series(dtype=str)
    )

    normalized_names = df["Current College Name"].map(normalize)

    identity_keys = df.apply(make_identity_key, axis=1)

    repeated_name_rows = int(
        (normalized_names.map(normalized_names.value_counts()) > 1).sum()
    )

    repeated_identity_rows = int(
        (identity_keys.map(identity_keys.value_counts()) > 1).sum()
    )

    name_group_count = int(
        normalized_names[normalized_names.duplicated(keep=False)].nunique()
    )

    identity_group_count = int(
        identity_keys[identity_keys.duplicated(keep=False)].nunique()
    )

    duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
    missing_ids = int(df[ID_COLUMN].eq("").sum())

    lines = [
        "AP EAPCET V5.16 Review Workflow Summary",
        "=" * 48,
        f"Input file: {source_file}",
        f"Review records: {len(df)}",
        f"Unique college names: {normalized_names.nunique()}",
        f"Stable ID column: {ID_COLUMN}",
        f"Missing stable IDs: {missing_ids}",
        f"Duplicate stable IDs: {duplicate_ids}",
        "",
        "Duplicate inspection:",
        f"Repeated name groups: {name_group_count}",
        f"Rows in repeated name groups: {repeated_name_rows}",
        f"Repeated composite identity groups: {identity_group_count}",
        f"Rows in repeated composite identity groups: {repeated_identity_rows}",
        f"Rows included in duplicate report: {len(duplicate_report)}",
        "",
        "Decision counts:",
    ]

    if decisions.empty:
        lines.append("Reviewer Decision column not found.")
    else:
        counts = decisions.fillna("").astype(str).value_counts(dropna=False)
        for decision, count in counts.items():
            lines.append(f"  {decision or '(blank)'}: {count}")

    lines.extend([
        "",
        "Safety rules:",
        "- V5.15 input is not modified.",
        "- Existing V5.16 review edits are retained on reruns.",
        "- Existing stable IDs are preserved.",
        "- Duplicate names are flagged, not deleted.",
        "- No mappings are automatically confirmed.",
        "- Duplicate names do not necessarily mean duplicate colleges.",
        "",
        f"Review queue: {V516_FILE}",
        f"Duplicate report: {DUPLICATE_REPORT}",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_file = load_input()

    df = pd.read_csv(source_file, dtype=str, keep_default_na=False)
    df.columns = df.columns.astype(str).str.strip()

    # Required fields for reliable identity review.
    missing_identity_columns = [
        column for column in IDENTITY_COLUMNS
        if column not in df.columns
    ]

    if missing_identity_columns:
        raise ValueError(
            "Missing required identity columns: "
            + ", ".join(missing_identity_columns)
        )

    if "Current College Name" not in df.columns:
        raise ValueError("Current College Name column is required.")

    # Preserve existing records and review fields.
    original_count = len(df)
    df = assign_stable_ids(df)

    # Do not auto-confirm or alter reviewer decisions.
    duplicate_report = build_duplicate_report(df)

    # Write the new review file; V5.15 remains untouched.
    df.to_csv(V516_FILE, index=False, encoding="utf-8-sig")
    duplicate_report.to_csv(
        DUPLICATE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    write_summary(df, duplicate_report, source_file)

    print("\nV5.16 workflow completed.")
    print(f"Input file: {source_file}")
    print(f"Review records: {original_count}")
    print(f"Output records: {len(df)}")
    print(f"Stable IDs missing: {df[ID_COLUMN].eq('').sum()}")
    print(f"Duplicate stable IDs: {df[ID_COLUMN].duplicated().sum()}")
    print(f"Rows in duplicate report: {len(duplicate_report)}")

    print("\nCreated files:")
    print(f"Review queue: {V516_FILE}")
    print(f"Duplicate report: {DUPLICATE_REPORT}")
    print(f"Summary: {SUMMARY_FILE}")

    print("\nNo mappings were automatically confirmed.")


if __name__ == "__main__":
    main()
