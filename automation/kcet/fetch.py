import os
import requests


# ============================================================
# KCET OFFICIAL DATA COLLECTOR
# ============================================================

KCET_SOURCE_URL = os.getenv(
    "KCET_SOURCE_URL",
    ""
)


def fetch_kcet_source():

    if not KCET_SOURCE_URL:

        raise RuntimeError(
            "KCET_SOURCE_URL is not configured."
        )

    print("=" * 60)
    print("KCET AUTOMATION")
    print("=" * 60)

    print(
        f"Source: {KCET_SOURCE_URL}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/154.0 Safari/537.36"
        )
    }

    response = requests.get(
        KCET_SOURCE_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    print(
        f"Downloaded: "
        f"{len(response.content):,} bytes"
    )

    return response