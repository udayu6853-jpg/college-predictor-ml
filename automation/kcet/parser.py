import pandas as pd
from io import StringIO


# ============================================================
# KCET TABLE PARSER
# ============================================================

def parse_kcet_html(html):

    print()
    print("Parsing KCET data...")

    try:

        tables = pd.read_html(
            StringIO(html)
        )

    except ValueError:

        print(
            "No HTML tables were found."
        )

        return []

    print(
        f"Tables found: {len(tables)}"
    )

    results = []

    for table_number, table in enumerate(
        tables,
        start=1
    ):

        print(
            f"Table {table_number}: "
            f"{len(table)} rows"
        )

        table.columns = [
            str(column).strip()
            for column in table.columns
        ]

        results.append(table)

    return results