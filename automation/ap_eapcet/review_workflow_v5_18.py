from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =========================================================
# AP EAPCET V5.18 - Controlled Manual Review Workflow
# =========================================================
# Safety design:
# - Does not modify the V5.16 review queue.
# - Does not overwrite the V5.17 baseline.
# - Creates a separate "last acknowledged" snapshot.
# - Generates a review-change report.
# - Appends changes to a log only when --acknowledge-changes is used.
# - Never automatically confirms a college mapping.

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "automation" / "output"

QUEUE_FILE = OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_16.csv"
BASELINE_FILE = OUTPUT_DIR / "ap_eapcet_review_baseline_v5_17.csv"

SNAPSHOT_FILE = OUTPUT_DIR / "ap_eapcet_review_snapshot_v5_18.csv"
CHANGE_REPORT = OUTPUT_DIR / "ap_eapcet_review_changes_v5_18.csv"
VALIDATION_REPORT = OUTPUT_DIR / "ap_eapcet_review_validation_issues_v5_18.csv"
CHANGE_LOG = OUTPUT_DIR / "ap_eapcet_review_change_log_v5_18.csv"
SUMMARY_FILE = OUTPUT_DIR / "ap_eapcet_review_workflow_summary_v5_18.txt"

ID_COLUMN = "Stable Review ID"
DECISION_COLUMN = "Reviewer Decision"

REVIEW_FIELDS = {
    "Reviewer Decision",
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewer Notes",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
    "Decision Reason",
}

CONFIRMATION_FIELDS = [
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
    "Reviewed By",
    "Review Date (YYYY-MM-DD)",
]

ALLOWED_DECISIONS = {"PENDING", "CONFIRMED", "REJECTED", "NEEDS_REVIEW"}

CHANGE_COLUMNS = [
    "Detected At (UTC)",
    "Change Type",
    ID_COLUMN,
    "Field",
    "Previous Value",
    "Current Value",
]

VALIDATION_COLUMNS = [
    ID_COLUMN,
    "Current College Name",
    "Reviewer Decision",
    "Issue",
    "Field",
    "Value",
]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig").fillna("")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def validate_ids(df: pd.DataFrame, label: str) -> None:
    if ID_COLUMN not in df.columns:
        raise ValueError(f"{label} is missing required column: {ID_COLUMN}")

    ids = df[ID_COLUMN].astype(str).str.strip()
    if ids.eq("").any():
        raise ValueError(f"{label} has {int(ids.eq('').sum())} missing Stable Review ID value(s).")

    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(f"{label} has duplicate Stable Review ID value(s): {duplicates[:10]}")


def make_validation_report(df: pd.DataFrame) -> pd.DataFrame:
    issues = []

    if DECISION_COLUMN not in df.columns:
        return pd.DataFrame([{
            ID_COLUMN: "",
            "Current College Name": "",
            "Reviewer Decision": "",
            "Issue": f"Required column missing: {DECISION_COLUMN}",
            "Field": DECISION_COLUMN,
            "Value": "",
        }], columns=VALIDATION_COLUMNS)

    for _, row in df.iterrows():
        record_id = str(row.get(ID_COLUMN, "")).strip()
        college_name = str(row.get("Current College Name", "")).strip()
        decision = str(row.get(DECISION_COLUMN, "")).strip().upper()

        def add_issue(issue: str, field: str, value: str = "") -> None:
            issues.append({
                ID_COLUMN: record_id,
                "Current College Name": college_name,
                "Reviewer Decision": decision,
                "Issue": issue,
                "Field": field,
                "Value": value,
            })

        if not decision:
            add_issue("Reviewer Decision is blank.", DECISION_COLUMN)
            continue

        if decision not in ALLOWED_DECISIONS:
            add_issue(
                "Decision value is not in the V5.18 allowed set: "
                + ", ".join(sorted(ALLOWED_DECISIONS)),
                DECISION_COLUMN,
                decision,
            )
            continue

        if decision == "CONFIRMED":
            for field in CONFIRMATION_FIELDS:
                value = str(row.get(field, "")).strip()
                if not value:
                    add_issue("Confirmed mapping requires this field.", field, value)

            evidence = str(row.get("Evidence URL", "")).strip()
            if evidence and not evidence.lower().startswith(("https://", "http://")):
                add_issue("Evidence URL must start with http:// or https://.", "Evidence URL", evidence)

            review_date = str(row.get("Review Date (YYYY-MM-DD)", "")).strip()
            if review_date:
                try:
                    parsed = datetime.strptime(review_date, "%Y-%m-%d")
                    if parsed.strftime("%Y-%m-%d") != review_date:
                        raise ValueError
                except ValueError:
                    add_issue("Review date must use YYYY-MM-DD format.", "Review Date (YYYY-MM-DD)", review_date)

        elif decision in {"REJECTED", "NEEDS_REVIEW"}:
            if not str(row.get("Reviewed By", "")).strip():
                add_issue(f"{decision} decision requires reviewer identity.", "Reviewed By")
            if not str(row.get("Review Date (YYYY-MM-DD)", "")).strip():
                add_issue(f"{decision} decision requires a review date.", "Review Date (YYYY-MM-DD)")
            if not str(row.get("Decision Reason", "")).strip():
                add_issue(f"{decision} decision requires a reason.", "Decision Reason")

    return pd.DataFrame(issues, columns=VALIDATION_COLUMNS)


def compare_frames(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    validate_ids(previous, "Acknowledged snapshot")
    validate_ids(current, "Current queue")

    previous_ids = set(previous[ID_COLUMN].astype(str))
    current_ids = set(current[ID_COLUMN].astype(str))
    previous_by_id = previous.set_index(ID_COLUMN, drop=False)
    current_by_id = current.set_index(ID_COLUMN, drop=False)

    changes = []
    detected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for record_id in sorted(current_ids - previous_ids):
        changes.append({
            "Detected At (UTC)": detected_at,
            "Change Type": "RECORD_ADDED",
            ID_COLUMN: record_id,
            "Field": "*",
            "Previous Value": "",
            "Current Value": "Record exists in current queue",
        })

    for record_id in sorted(previous_ids - current_ids):
        changes.append({
            "Detected At (UTC)": detected_at,
            "Change Type": "RECORD_REMOVED",
            ID_COLUMN: record_id,
            "Field": "*",
            "Previous Value": "Record exists in acknowledged snapshot",
            "Current Value": "",
        })

    shared_columns = [
        col for col in previous.columns
        if col in current.columns and col != ID_COLUMN
    ]
    for record_id in sorted(previous_ids & current_ids):
        old_row = previous_by_id.loc[record_id]
        new_row = current_by_id.loc[record_id]
        for field in shared_columns:
            old_value = str(old_row[field])
            new_value = str(new_row[field])
            if old_value != new_value:
                changes.append({
                    "Detected At (UTC)": detected_at,
                    "Change Type": "REVIEW_FIELD_CHANGED" if field in REVIEW_FIELDS else "OTHER_FIELD_CHANGED",
                    ID_COLUMN: record_id,
                    "Field": field,
                    "Previous Value": old_value,
                    "Current Value": new_value,
                })

    for field in sorted(set(current.columns) - set(previous.columns)):
        changes.append({
            "Detected At (UTC)": detected_at,
            "Change Type": "COLUMN_ADDED",
            ID_COLUMN: "",
            "Field": field,
            "Previous Value": "",
            "Current Value": "Column exists in current queue",
        })
    for field in sorted(set(previous.columns) - set(current.columns)):
        changes.append({
            "Detected At (UTC)": detected_at,
            "Change Type": "COLUMN_REMOVED",
            ID_COLUMN: "",
            "Field": field,
            "Previous Value": "Column exists in acknowledged snapshot",
            "Current Value": "",
        })

    return pd.DataFrame(changes, columns=CHANGE_COLUMNS)


def append_to_log(changes: pd.DataFrame) -> None:
    if changes.empty:
        return
    if CHANGE_LOG.exists():
        existing = load_csv(CHANGE_LOG)
        combined = pd.concat([existing, changes], ignore_index=True)
    else:
        combined = changes.copy()
    save_csv(combined, CHANGE_LOG)


def write_summary(lines: list[str]) -> None:
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and validate AP EAPCET manual review edits without modifying the review queue."
    )
    parser.add_argument(
        "--acknowledge-changes",
        action="store_true",
        help="After inspecting the report, append changes to the permanent log and advance the V5.18 snapshot.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not QUEUE_FILE.exists():
        raise FileNotFoundError(f"Review queue not found: {QUEUE_FILE}")

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"V5.17 baseline not found: {BASELINE_FILE}. "
            "V5.18 will not create or replace the V5.17 baseline."
        )

    current = load_csv(QUEUE_FILE)
    validate_ids(current, "Current review queue")

    # The V5.17 baseline is read-only and is used as a safety reference.
    baseline = load_csv(BASELINE_FILE)
    validate_ids(baseline, "V5.17 baseline")

    if not SNAPSHOT_FILE.exists():
        save_csv(current, SNAPSHOT_FILE)
        validation = make_validation_report(current)
        save_csv(validation, VALIDATION_REPORT)
        save_csv(pd.DataFrame(columns=CHANGE_COLUMNS), CHANGE_REPORT)
        write_summary([
            "AP EAPCET V5.18 Controlled Review Summary",
            "=" * 48,
            "Status: INITIAL SNAPSHOT CREATED",
            f"Current queue records: {len(current)}",
            f"V5.17 baseline records: {len(baseline)}",
            "Changes detected: 0 (first V5.18 run establishes its own snapshot)",
            f"Validation issues: {len(validation)}",
            "No review queue data was changed.",
            "The V5.17 baseline was not changed.",
            "",
            f"Review queue: {QUEUE_FILE}",
            f"V5.17 baseline: {BASELINE_FILE}",
            f"V5.18 snapshot: {SNAPSHOT_FILE}",
            f"Validation report: {VALIDATION_REPORT}",
        ])
        print("\nV5.18 initial snapshot created.")
        print(f"Current records: {len(current)}")
        print(f"Validation issues: {len(validation)}")
        print(f"Snapshot: {SNAPSHOT_FILE}")
        print("No review queue data or V5.17 baseline was changed.")
        return

    previous = load_csv(SNAPSHOT_FILE)
    validate_ids(previous, "V5.18 acknowledged snapshot")

    changes = compare_frames(previous, current)
    validation = make_validation_report(current)
    save_csv(changes, CHANGE_REPORT)
    save_csv(validation, VALIDATION_REPORT)

    if args.acknowledge_changes:
        if not validation.empty:
            print("\nChanges NOT acknowledged because validation issues exist.")
            print(f"Review {VALIDATION_REPORT} and correct the queue before acknowledging.")
            print(f"Detected changes remain in: {CHANGE_REPORT}")
            return

        # Backup snapshot before advancing it; never touch the original V5.17 baseline.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = SNAPSHOT_FILE.with_name(f"{SNAPSHOT_FILE.stem}_backup_{stamp}.csv")
        shutil.copy2(SNAPSHOT_FILE, backup)
        append_to_log(changes)
        save_csv(current, SNAPSHOT_FILE)
        status = "CHANGES ACKNOWLEDGED; SNAPSHOT ADVANCED"
        print("\nV5.18 changes acknowledged.")
        print(f"Snapshot backup: {backup}")
        print(f"Permanent change log: {CHANGE_LOG}")
    else:
        status = "AUDIT ONLY - NO SNAPSHOT ADVANCE"
        print("\nV5.18 audit completed (read-only mode).")
        print("No permanent log append or snapshot update was performed.")
        print("After inspecting the report, use --acknowledge-changes to record and accept valid changes.")

    write_summary([
        "AP EAPCET V5.18 Controlled Review Summary",
        "=" * 48,
        f"Status: {status}",
        f"Current queue records: {len(current)}",
        f"Acknowledged snapshot records: {len(previous)}",
        f"Detected change report rows: {len(changes)}",
        f"Validation issues: {len(validation)}",
        "The review queue is never edited by this script.",
        "The V5.17 baseline is never edited by this script.",
        "Detected changes are not automatically reverted.",
        "No college mapping is automatically confirmed.",
        "",
        f"Review queue: {QUEUE_FILE}",
        f"V5.17 baseline: {BASELINE_FILE}",
        f"V5.18 snapshot: {SNAPSHOT_FILE}",
        f"Change report: {CHANGE_REPORT}",
        f"Validation report: {VALIDATION_REPORT}",
        f"Permanent change log: {CHANGE_LOG}",
    ])

    print(f"Current records: {len(current)}")
    print(f"Detected change rows: {len(changes)}")
    print(f"Validation issues: {len(validation)}")
    print(f"Change report: {CHANGE_REPORT}")
    print(f"Validation report: {VALIDATION_REPORT}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
