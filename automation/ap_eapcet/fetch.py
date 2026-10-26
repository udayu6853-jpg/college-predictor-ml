import requests


# ============================================================
# AP EAPCET OFFICIAL SOURCE
# ============================================================

OFFICIAL_EAPCET_HOME = (
    "https://cets.apsche.ap.gov.in/"
    "EAPCET25/Eapcet/EAPCET_HomePage.aspx"
)

# This is the official APSCHE Last Rank Excel currently
# linked from the AP EAPCET official page.
OFFICIAL_LAST_RANK_XLS = (
    "https://apsche.ap.gov.in/Pdf/"
    "APEAMCET2024LASTRANKDETAILSNONSW.XLS"
)


def download_last_rank_file():

    print("=" * 70)
    print("AP EAPCET AUTOMATION")
    print("=" * 70)

    print()
    print("Official source:")
    print(OFFICIAL_EAPCET_HOME)

    print()
    print("Downloading official Last Rank Excel...")

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
        OFFICIAL_LAST_RANK_XLS,
        headers=headers,
        timeout=60
    )

    response.raise_for_status()

    print(
        "Download successful."
    )

    print(
        f"Downloaded bytes: "
        f"{len(response.content):,}"
    )

    return response.content