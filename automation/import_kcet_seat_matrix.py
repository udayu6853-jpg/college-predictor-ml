
import sys
from pathlib import Path

import pandas as pd
from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_db_connection


CSV_PATH = (
    PROJECT_ROOT
    / "automation"
    / "source_samples"
    / "kcet_engineering_seat_matrix_2026.csv"
)

TABLE = "kcet_seat_matrix_2026"

COLUMN_MAP = {
    "Source Year": "source_year",
    "PDF Page": "pdf_page",
    "College Code": "college_code",
    "College Name": "college_name",
    "Course Code": "course_code",
    "Course Name": "course_name",
    "Seat Type": "seat_type",
    "General - C-1": "general_c1",
    "General - 2A": "general_2a",
    "General - 2B": "general_2b",
    "General - 3A": "general_3a",
    "General - 3B": "general_3b",
    "General - ST": "general_st",
    "General - SCA": "general_sca",
    "General - SCB": "general_scb",
    "General - SCC": "general_scc",
    "General - SCC (20%)": "general_scc_20",
    "General - GM": "general_gm",
    "Kannada - C-1": "kannada_c1",
    "Kannada - 2A": "kannada_2a",
    "Kannada - 2B": "kannada_2b",
    "Kannada - 3A": "kannada_3a",
    "Kannada - 3B": "kannada_3b",
    "Kannada - ST": "kannada_st",
    "Kannada - SCA": "kannada_sca",
    "Kannada - SCB": "kannada_scb",
    "Kannada - SCC": "kannada_scc",
    "Kannada - SCC (20%)": "kannada_scc_20",
    "Kannada - GM": "kannada_gm",
    "Rural - C-1": "rural_c1",
    "Rural - 2A": "rural_2a",
    "Rural - 2B": "rural_2b",
    "Rural - 3A": "rural_3a",
    "Rural - 3B": "rural_3b",
    "Rural - ST": "rural_st",
    "Rural - SCA": "rural_sca",
    "Rural - SCB": "rural_scb",
    "Rural - SCC": "rural_scc",
    "Rural - SCC (20%)": "rural_scc_20",
    "Rural - GM": "rural_gm",
    "Total HK-RK": "total_hk_rk",
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    missing = set(COLUMN_MAP) - set(df.columns)
    unexpected = set(df.columns) - set(COLUMN_MAP)

    if missing or unexpected:
        raise ValueError(
            f"CSV column mismatch. Missing={sorted(missing)}, "
            f"Unexpected={sorted(unexpected)}"
        )

    if df.duplicated(
        ["Source Year", "College Code", "Course Code", "Seat Type"]
    ).any():
        raise ValueError("CSV contains duplicate unique keys.")

    if not df["Seat Type"].isin(["HK", "RK"]).all():
        raise ValueError("Unexpected seat type found.")

    if df[list(COLUMN_MAP)].isna().any().any():
        raise ValueError("CSV contains missing values.")

    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    columns = list(COLUMN_MAP.values())

    insert_query = sql.SQL("""
        INSERT INTO {} ({})
        VALUES ({})
        ON CONFLICT (source_year, college_code, course_code, seat_type)
        DO NOTHING
    """).format(
        sql.Identifier(TABLE),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )


    connection = get_db_connection()

    try:
        with connection.transaction():
            before = connection.execute(
                sql.SQL("SELECT COUNT(*) AS n FROM {}").format(
                    sql.Identifier(TABLE)
                )
            ).fetchone()["n"]

            with connection.cursor() as cursor:
                cursor.executemany(
                    insert_query,
                    df.itertuples(index=False, name=None),
                )

        # The transaction has committed successfully at this point.
        after = connection.execute(
            sql.SQL("SELECT COUNT(*) AS n FROM {}").format(
                sql.Identifier(TABLE)
            )
        ).fetchone()["n"]

        print("CSV records:", len(df))
        print("Records already in table:", before)
        print("Records now in table:", after)
        print("New records inserted:", after - before)
        print("Duplicate keys were skipped.")
        print("Live college CSVs and ML model were not modified.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
