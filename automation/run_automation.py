import os
import sys
from pathlib import Path

import pandas as pd


# Make project root available
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from automation.kcet.fetch import (
    fetch_kcet_source
)

from automation.kcet.parser import (
    parse_kcet_html
)


from automation.kcet.validator import (
    validate_kcet_dataframe
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "automation"
    / "output"
)


def run_kcet():

    print()
    print("#" * 60)
    print("COLLEGE PREDICTOR AUTOMATION")
    print("#" * 60)

    print()
    print("Starting KCET collector...")

    response = fetch_kcet_source()

    tables = parse_kcet_html(
        response.text
    )

    if not tables:

        print(
            "No KCET tables were extracted."
        )

        return

    # For the first prototype,
    # process the first discovered table.
    df = tables[0].copy()

    try:

        df = validate_kcet_dataframe(
            df
        )

    except ValueError as error:

        print()
        print(
            "VALIDATION ERROR:"
        )

        print(error)

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "kcet_latest.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"KCET data saved to:"
    )

    print(
        output_file
    )

    print()
    print(
        "KCET automation completed."
    )


if __name__ == "__main__":

    run_kcet()