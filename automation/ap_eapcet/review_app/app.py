
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, abort

app = Flask(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "automation" / "output"

WORKSHEET_FILE = (
    OUTPUT_DIR / "ap_eapcet_manual_review_worksheet_v5_19.csv"
)

QUEUE_FILE = (
    OUTPUT_DIR / "ap_eapcet_manual_review_queue_v5_16.csv"
)

# Read-only review interface for the first version.
# This application does not change the source CSV files.


def load_review_data():
    """Load the prioritized worksheet without modifying it."""

    if not WORKSHEET_FILE.exists():
        raise FileNotFoundError(
            f"Review worksheet not found: {WORKSHEET_FILE}"
        )

    df = pd.read_csv(
        WORKSHEET_FILE,
        dtype=str,
        keep_default_na=False
    )

    # Use the stable ID to identify each review record.
    if "Stable Review ID" not in df.columns:
        raise ValueError(
            "The worksheet is missing the Stable Review ID column."
        )

    return df


def get_value(row, *column_names, default=""):
    """Return the first available value from a row."""

    for column in column_names:
        if column in row.index:
            value = str(row[column]).strip()

            if value:
                return value

    return default


def get_candidate_columns(df):
    """Find the available candidate-related columns."""

    candidate_groups = []

    for number in range(1, 6):
        group = {
            "number": number,
            "name": f"Candidate {number} Name",
            "location": f"Candidate {number} Location",
            "code": f"Candidate {number} Code",
            "similarity": f"Candidate {number} Similarity",
            "score": f"Candidate {number} Score",
        }

        available = [
            column
            for column in group.values()
            if isinstance(column, str) and column in df.columns
        ]

        if available:
            candidate_groups.append(group)

    return candidate_groups


@app.route("/")
def index():
    try:
        df = load_review_data()
    except (FileNotFoundError, ValueError) as error:
        return (
            f"<h2>Review data error</h2><pre>{error}</pre>",
            500
        )

    # Search and filter values
    search_text = request.args.get("q", "").strip().lower()
    selected_category = request.args.get("category", "").strip()
    selected_decision = request.args.get("decision", "").strip()

    # Make missing review fields safe to display
    if "Reviewer Decision" not in df.columns:
        df["Reviewer Decision"] = "PENDING"

    df["Reviewer Decision"] = (
        df["Reviewer Decision"].astype(str).str.strip()
        .replace("", "PENDING")
    )

    # Search by stable ID, college name, or location
    if search_text:
        searchable_columns = [
            column for column in [
                "Stable Review ID",
                "Current College Name",
                "Current Location",
                "Current Canonical Name",
                "Current Canonical Location",
            ]
            if column in df.columns
        ]

        mask = pd.Series(False, index=df.index)

        for column in searchable_columns:
            mask |= df[column].astype(str).str.lower().str.contains(
                search_text,
                regex=False,
                na=False
            )

        df = df[mask]

    # Filter by manual review category
    if selected_category and "Manual Review Category" in df.columns:
        df = df[
            df["Manual Review Category"] == selected_category
        ]

    # Filter by current decision
    if selected_decision:
        df = df[
            df["Reviewer Decision"].str.upper()
            == selected_decision.upper()
        ]

    # Summary counts are based on the complete worksheet.
    all_df = load_review_data()

    if "Reviewer Decision" not in all_df.columns:
        all_df["Reviewer Decision"] = "PENDING"

    all_df["Reviewer Decision"] = (
        all_df["Reviewer Decision"].astype(str).str.strip()
        .replace("", "PENDING")
    )

    total_records = len(all_df)
    pending_count = (
        all_df["Reviewer Decision"].str.upper() == "PENDING"
    ).sum()
    confirmed_count = (
        all_df["Reviewer Decision"].str.upper() == "CONFIRMED"
    ).sum()
    rejected_count = (
        all_df["Reviewer Decision"].str.upper() == "REJECTED"
    ).sum()
    needs_review_count = (
        all_df["Reviewer Decision"].str.upper() == "NEEDS_REVIEW"
    ).sum()

    categories = []

    if "Manual Review Category" in all_df.columns:
        categories = sorted(
            value
            for value in all_df["Manual Review Category"].unique()
            if value
        )

    records = df.to_dict(orient="records")

    return render_template(
        "index.html",
        records=records,
        columns=df.columns.tolist(),
        candidate_groups=get_candidate_columns(all_df),
        categories=categories,
        search_text=request.args.get("q", ""),
        selected_category=selected_category,
        selected_decision=selected_decision,
        total_records=total_records,
        visible_records=len(records),
        pending_count=int(pending_count),
        confirmed_count=int(confirmed_count),
        rejected_count=int(rejected_count),
        needs_review_count=int(needs_review_count),
    )


@app.route("/health")
def health():
    """Check whether the review application can load its data."""

    try:
        df = load_review_data()

        return {
            "status": "ok",
            "worksheet": WORKSHEET_FILE.name,
            "records": len(df),
            "read_only": True,
        }
    except (FileNotFoundError, ValueError) as error:
        return {
            "status": "error",
            "message": str(error),
        }, 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5051,
        debug=False
    )
