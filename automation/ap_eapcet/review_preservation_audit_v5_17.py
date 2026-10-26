
from pathlib import Path
import pandas as pd


# =========================================================
# AP EAPCET V5.17 — Review Preservation Audit
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

CURRENT_FILE = (
    OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_16.csv"
)

BASELINE_FILE = (
    OUTPUT_DIR / "ap_eapcet_review_baseline_v5_17.csv"
)

CHANGE_REPORT = (
    OUTPUT_DIR / "ap_eapcet_review_changes_v5_17.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "ap_eapcet_review_audit_summary_v5_17.txt"
)

ID_COLUMN = "Stable Review ID"

TRACKED_FIELDS = [
    "Reviewer Decision",
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewer Notes",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
    "Decision Reason",
    "Current College Name",
    "Current Location",
    "Current Canonical Name",
    "Current Canonical Location",
]


def load_csv(path):
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    ).fillna("")


def write_summary(lines):
    SUMMARY_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            f"Current review queue not found: {CURRENT_FILE}"
        )

    current = load_csv(CURRENT_FILE)

    if ID_COLUMN not in current.columns:
        raise ValueError(
            f"Required column '{ID_COLUMN}' is missing."
        )

    if current[ID_COLUMN].str.strip().eq("").any():
        raise ValueError(
            "Some records have missing Stable Review ID values. "
            "Resolve these before auditing."
        )

    if current[ID_COLUMN].duplicated().any():
        duplicates = current.loc[
            current[ID_COLUMN].duplicated(keep=False),
            ID_COLUMN,
        ].tolist()

        raise ValueError(
            "Stable Review ID values must be unique. "
            f"Duplicate IDs: {duplicates[:10]}"
        )

    # First run: create the baseline once.
    # Later runs: never overwrite it automatically.
    if not BASELINE_FILE.exists():
        current.to_csv(
            BASELINE_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        empty_changes = pd.DataFrame(
            columns=[
                "Change Type",
                ID_COLUMN,
                "Field",
                "Previous Value",
                "Current Value",
            ]
        )

        empty_changes.to_csv(
            CHANGE_REPORT,
            index=False,
            encoding="utf-8-sig",
        )

        write_summary([
            "AP EAPCET V5.17 Review Audit Summary",
            "=" * 44,
            "Status: BASELINE CREATED",
            f"Baseline records: {len(current)}",
            f"Current records: {len(current)}",
            "Added records: 0",
            "Removed records: 0",
            "Changed field values: 0",
            "",
            "The baseline was created from the current V5.16 file.",
            "No changes can be detected until the current file changes.",
            "The baseline will not be overwritten automatically.",
            "",
            f"Baseline: {BASELINE_FILE}",
            f"Current queue: {CURRENT_FILE}",
            f"Change report: {CHANGE_REPORT}",
        ])

        print("\nV5.17 baseline created successfully.")
        print(f"Baseline records: {len(current)}")
        print(f"Current records: {len(current)}")
        print("Changes detected: 0")
        print(f"Baseline: {BASELINE_FILE}")
        print(f"Summary: {SUMMARY_FILE}")
        print("\nRun this script again after making review edits.")
        return

    baseline = load_csv(BASELINE_FILE)

    for label, frame in [
        ("Baseline", baseline),
        ("Current file", current),
    ]:
        if ID_COLUMN not in frame.columns:
            raise ValueError(
                f"{label} is missing '{ID_COLUMN}'."
            )

        if frame[ID_COLUMN].str.strip().eq("").any():
            raise ValueError(
                f"{label} contains missing stable IDs."
            )

        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(
                f"{label} contains duplicate stable IDs."
            )

    baseline_ids = set(baseline[ID_COLUMN])
    current_ids = set(current[ID_COLUMN])

    added_ids = sorted(current_ids - baseline_ids)
    removed_ids = sorted(baseline_ids - current_ids)
    common_ids = sorted(baseline_ids & current_ids)

    baseline_by_id = baseline.set_index(ID_COLUMN, drop=False)
    current_by_id = current.set_index(ID_COLUMN, drop=False)

    changes = []

    # Report records added after the baseline.
    for record_id in added_ids:
        changes.append({
            "Change Type": "RECORD_ADDED",
            ID_COLUMN: record_id,
            "Field": "*",
            "Previous Value": "",
            "Current Value": "Record exists in current file",
        })

    # Report records removed after the baseline.
    for record_id in removed_ids:
        changes.append({
            "Change Type": "RECORD_REMOVED",
            ID_COLUMN: record_id,
            "Field": "*",
            "Previous Value": "Record exists in baseline",
            "Current Value": "",
        })

    # Compare review and identity fields for common IDs.
    fields = [
        field for field in TRACKED_FIELDS
        if field in baseline.columns and field in current.columns
    ]

    # Also compare other shared columns for unexpected changes.
    extra_fields = [
        field for field in baseline.columns
        if field in current.columns
        and field not in fields
        and field != ID_COLUMN
    ]

    fields.extend(extra_fields)

    for record_id in common_ids:
        old_row = baseline_by_id.loc[record_id]
        new_row = current_by_id.loc[record_id]

        for field in fields:
            old_value = str(old_row[field])
            new_value = str(new_row[field])

            if old_value != new_value:
                change_type = (
                    "REVIEW_FIELD_CHANGED"
                    if field in TRACKED_FIELDS
                    else "OTHER_FIELD_CHANGED"
                )

                changes.append({
                    "Change Type": change_type,
                    ID_COLUMN: record_id,
                    "Field": field,
                    "Previous Value": old_value,
                    "Current Value": new_value,
                })

    # Report columns added or removed from the CSV structure.
    baseline_columns = set(baseline.columns)
    current_columns = set(current.columns)

    for field in sorted(current_columns - baseline_columns):
        changes.append({
            "Change Type": "COLUMN_ADDED",
            ID_COLUMN: "",
            "Field": field,
            "Previous Value": "",
            "Current Value": "Column exists in current file",
        })

    for field in sorted(baseline_columns - current_columns):
        changes.append({
            "Change Type": "COLUMN_REMOVED",
            ID_COLUMN: "",
            "Field": field,
            "Previous Value": "Column exists in baseline",
            "Current Value": "",
        })

    report_columns = [
        "Change Type",
        ID_COLUMN,
        "Field",
        "Previous Value",
        "Current Value",
    ]

    report = pd.DataFrame(changes, columns=report_columns)

    report.to_csv(
        CHANGE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    changed_review_fields = sum(
        change["Change Type"] == "REVIEW_FIELD_CHANGED"
        for change in changes
    )

    other_changes = sum(
        change["Change Type"] == "OTHER_FIELD_CHANGED"
        for change in changes
    )

    lines = [
        "AP EAPCET V5.17 Review Audit Summary",
        "=" * 44,
        "Status: AUDIT COMPLETED",
        f"Baseline records: {len(baseline)}",
        f"Current records: {len(current)}",
        f"Common stable IDs: {len(common_ids)}",
        f"Added records: {len(added_ids)}",
        f"Removed records: {len(removed_ids)}",
        f"Review/identity field changes: {changed_review_fields}",
        f"Other field changes: {other_changes}",
        f"Total change report rows: {len(report)}",
        "",
        "Interpretation:",
        "- Changes are reported, not automatically reverted.",
        "- Review edits may be intentional; inspect the report.",
        "- A clean report means no tracked differences were detected.",
        "- The baseline is not overwritten automatically.",
        "- No college mappings are automatically confirmed.",
        "",
        f"Baseline: {BASELINE_FILE}",
        f"Current queue: {CURRENT_FILE}",
        f"Change report: {CHANGE_REPORT}",
    ]

    write_summary(lines)

    print("\nV5.17 audit completed.")
    print(f"Baseline records: {len(baseline)}")
    print(f"Current records: {len(current)}")
    print(f"Added records: {len(added_ids)}")
    print(f"Removed records: {len(removed_ids)}")
    print(f"Review/identity field changes: {changed_review_fields}")
    print(f"Other field changes: {other_changes}")
    print(f"Total change report rows: {len(report)}")

    print("\nCreated/updated files:")
    print(f"Baseline: {BASELINE_FILE}")
    print(f"Change report: {CHANGE_REPORT}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
