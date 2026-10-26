"""
College Predictor — Confirmed AP EAPCET Mapping Audit v5.11

Reads:
  automation/output/ap_eapcet_human_review_v5_10.csv

Creates:
  automation/output/ap_eapcet_mapping_validation_issues_v5_11.csv
  automation/output/ap_eapcet_mapping_validation_summary_v5_11.txt

This is an audit-only script. It does not change the review CSV, source datasets,
database, trained model, or live app. It checks consistency; it does NOT prove
that a college mapping is factually correct.
"""

from pathlib import Path
from collections import Counter
from datetime import date
import pandas as pd


VERSION = "5.11"
REVIEW_FILE = "ap_eapcet_human_review_v5_10.csv"
ISSUES_FILE = "ap_eapcet_mapping_validation_issues_v5_11.csv"
SUMMARY_FILE = "ap_eapcet_mapping_validation_summary_v5_11.txt"

ALLOWED_DECISIONS = {
    "", "PENDING", "CONFIRMED", "REJECTED",
    "NEEDS_MORE_EVIDENCE", "NO_MATCH"
}
REQUIRED_CONFIRMATION_FIELDS = [
    "Verified Official Name",
    "Verified Location",
    "Verified Official Code",
    "Evidence URL",
]


def txt(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def add_issue(issues, row_number, row, severity, issue, details=""):
    issues.append({
        "CSV Row": row_number,
        "Severity": severity,
        "Current College Name": txt(row.get("Current College Name", "")),
        "Current Location": txt(row.get("Current Location", "")),
        "Reviewer Decision": txt(row.get("Reviewer Decision", "")).upper(),
        "Verified Official Name": txt(row.get("Verified Official Name", "")),
        "Verified Location": txt(row.get("Verified Location", "")),
        "Verified Official Code": txt(row.get("Verified Official Code", "")),
        "Evidence URL": txt(row.get("Evidence URL", "")),
        "Issue": issue,
        "Details": details,
    })


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "automation" / "output"
    review_path = output_dir / REVIEW_FILE
    issues_path = output_dir / ISSUES_FILE
    summary_path = output_dir / SUMMARY_FILE

    if not review_path.exists():
        print("ERROR: Review CSV not found:")
        print(f"  {review_path}")
        print("Run create_verified_mapping_v5_10.py first.")
        return

    try:
        df = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    except Exception as exc:
        print(f"ERROR: Could not read review CSV: {exc}")
        return

    required_columns = [
        "Current College Name",
        "Current Location",
        "Reviewer Decision",
        *REQUIRED_CONFIRMATION_FIELDS,
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        print("ERROR: Review CSV is missing required columns:")
        for col in missing_columns:
            print(f"  - {col}")
        print("Make sure this is ap_eapcet_human_review_v5_10.csv.")
        return

    issues = []
    decisions = []
    confirmed_rows = []

    for idx, row in df.iterrows():
        row_number = idx + 2  # CSV header is line 1
        decision = txt(row.get("Reviewer Decision", "")).upper()
        decisions.append(decision)

        if decision not in ALLOWED_DECISIONS:
            add_issue(
                issues, row_number, row, "ERROR", "INVALID_DECISION",
                f"Allowed values: {', '.join(sorted(ALLOWED_DECISIONS - {''}))}"
            )
            continue

        if decision == "CONFIRMED":
            confirmed_rows.append(row)
            missing = [
                field for field in REQUIRED_CONFIRMATION_FIELDS
                if not txt(row.get(field, ""))
            ]
            if missing:
                add_issue(
                    issues, row_number, row, "ERROR",
                    "CONFIRMED_FIELDS_MISSING",
                    "Missing: " + ", ".join(missing)
                )

            url = txt(row.get("Evidence URL", ""))
            if url and not url.lower().startswith(("https://", "http://")):
                add_issue(
                    issues, row_number, row, "ERROR", "EVIDENCE_URL_INVALID",
                    "Evidence URL should begin with https:// or http://."
                )

            review_date = txt(row.get("Review Date (YYYY-MM-DD)", ""))
            if review_date:
                try:
                    date.fromisoformat(review_date)
                except ValueError:
                    add_issue(
                        issues, row_number, row, "WARNING", "REVIEW_DATE_INVALID",
                        "Use YYYY-MM-DD format."
                    )

            if not txt(row.get("Reviewed By", "")):
                add_issue(
                    issues, row_number, row, "WARNING", "REVIEWER_NOT_RECORDED",
                    "Reviewed By is blank."
                )

        elif decision == "PENDING" or decision == "":
            add_issue(
                issues, row_number, row, "INFO", "NOT_YET_REVIEWED",
                "This row has not been human-confirmed."
            )

    # Duplicate verified institution codes across CONFIRMED rows.
    code_to_rows = {}
    for idx, row in df.iterrows():
        if txt(row.get("Reviewer Decision", "")).upper() != "CONFIRMED":
            continue
        code = txt(row.get("Verified Official Code", "")).casefold()
        if code:
            code_to_rows.setdefault(code, []).append((idx + 2, row))

    for code, matches in code_to_rows.items():
        if len(matches) > 1:
            descriptions = [
                f"row {n}: {txt(r.get('Current College Name', ''))}"
                for n, r in matches
            ]
            for row_number, row in matches:
                add_issue(
                    issues, row_number, row, "ERROR",
                    "DUPLICATE_CONFIRMED_OFFICIAL_CODE",
                    f"Code '{txt(row.get('Verified Official Code', ''))}' "
                    f"appears in multiple confirmed rows: " + "; ".join(descriptions)
                )

    # Same current college/location mapped to more than one verified identity.
    identity_to_rows = {}
    for idx, row in df.iterrows():
        if txt(row.get("Reviewer Decision", "")).upper() != "CONFIRMED":
            continue
        current_key = (
            txt(row.get("Current College Name", "")).casefold(),
            txt(row.get("Current Location", "")).casefold(),
        )
        identity = (
            txt(row.get("Verified Official Name", "")).casefold(),
            txt(row.get("Verified Location", "")).casefold(),
            txt(row.get("Verified Official Code", "")).casefold(),
        )
        identity_to_rows.setdefault(current_key, set()).add(identity)

    for idx, row in df.iterrows():
        if txt(row.get("Reviewer Decision", "")).upper() != "CONFIRMED":
            continue
        current_key = (
            txt(row.get("Current College Name", "")).casefold(),
            txt(row.get("Current Location", "")).casefold(),
        )
        identities = identity_to_rows.get(current_key, set())
        if len(identities) > 1:
            add_issue(
                issues, idx + 2, row, "ERROR",
                "CONFLICTING_MAPPING_FOR_CURRENT_COLLEGE",
                "The same current college name/location has multiple verified identities."
            )

    issues_df = pd.DataFrame(
        issues,
        columns=[
            "CSV Row", "Severity", "Current College Name", "Current Location",
            "Reviewer Decision", "Verified Official Name", "Verified Location",
            "Verified Official Code", "Evidence URL", "Issue", "Details"
        ]
    )
    issues_df.to_csv(issues_path, index=False, encoding="utf-8-sig")

    decision_counts = Counter(decisions)
    severity_counts = Counter(issues_df["Severity"]) if not issues_df.empty else Counter()
    issue_counts = Counter(issues_df["Issue"]) if not issues_df.empty else Counter()

    fully_completed = 0
    for row in confirmed_rows:
        if all(txt(row.get(field, "")) for field in REQUIRED_CONFIRMATION_FIELDS):
            fully_completed += 1

    summary_lines = [
        f"College Predictor — AP EAPCET Mapping Audit v{VERSION}",
        "=" * 58,
        f"Review CSV: {review_path}",
        f"Total review rows: {len(df)}",
        f"CONFIRMED decisions: {decision_counts.get('CONFIRMED', 0)}",
        f"CONFIRMED rows with all required fields: {fully_completed}",
        f"PENDING decisions: {decision_counts.get('PENDING', 0) + decision_counts.get('', 0)}",
        f"REJECTED decisions: {decision_counts.get('REJECTED', 0)}",
        f"NEEDS_MORE_EVIDENCE decisions: {decision_counts.get('NEEDS_MORE_EVIDENCE', 0)}",
        f"NO_MATCH decisions: {decision_counts.get('NO_MATCH', 0)}",
        "",
        f"Total audit findings: {len(issues_df)}",
        f"Errors: {severity_counts.get('ERROR', 0)}",
        f"Warnings: {severity_counts.get('WARNING', 0)}",
        f"Info findings: {severity_counts.get('INFO', 0)}",
        "",
        "Findings by type:",
    ]
    if issue_counts:
        for issue, count in sorted(issue_counts.items()):
            summary_lines.append(f"  {issue}: {count}")
    else:
        summary_lines.append("  No consistency issues found.")

    summary_lines += [
        "",
        f"Issues CSV: {issues_path}",
        "",
        "Interpretation:",
        "- This script checks completeness and consistency only.",
        "- No match is declared factually correct just because it passes these checks.",
        "- Review ERROR findings before using mappings.",
        "- Verify official names, locations, and institution codes against official sources.",
        "- This script does not modify the review CSV, source datasets, model, database, or app.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\nCollege Predictor — AP EAPCET Mapping Audit v{VERSION}")
    print("-" * 58)
    print(f"Review rows:             {len(df)}")
    print(f"Confirmed decisions:     {decision_counts.get('CONFIRMED', 0)}")
    print(f"Complete confirmations:  {fully_completed}")
    print(f"Audit findings:          {len(issues_df)}")
    print(f"Errors:                  {severity_counts.get('ERROR', 0)}")
    print(f"Warnings:                {severity_counts.get('WARNING', 0)}")
    print(f"Info findings:           {severity_counts.get('INFO', 0)}")
    print(f"\nIssues CSV: {issues_path}")
    print(f"Summary:    {summary_path}")
    print("\nAudit completed. No project data or application files were modified.")


if __name__ == "__main__":
    main()
