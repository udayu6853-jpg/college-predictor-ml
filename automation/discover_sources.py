
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests


SOURCES = {
    "AP EAPCET 2026": (
        "https://cets.apsche.ap.gov.in/EAPCET/"
        "Eapcet/EAPCET_HomePage.aspx"
    ),
    "Karnataka UGCET 2026": (
        "https://cetonline.karnataka.gov.in/kea/ugcet2026"
    ),
}

# Keywords used to identify likely cutoff-rank documents.
CUTOFF_KEYWORDS = (
    "cutoff rank",
    "cutoff",
    "last rank",
    "last ranks",
    "closing rank",
    "opening rank",
    "rank-wise",
    "rank wise",
    "last_rank",
    "cutoff_rank",
)

# Exclude document types that are unlikely to contain cutoff ranks.
EXCLUDE_KEYWORDS = (
    "candidate",
    "candidates",
    "ph_candidates",
    "vacant_seat",
    "vacant seat",
    "timetable",
    "archive",
    "nursing",
)

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current_href = None
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attrs = dict(attrs)
            self.current_href = attrs.get("href", "")
            self.current_text = []

    def handle_data(self, data):
        if self.current_href is not None:
            self.current_text.append(data.strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.current_href is not None:
            text = " ".join(
                part for part in self.current_text if part
            )
            self.links.append((text, self.current_href))
            self.current_href = None
            self.current_text = []


def inspect_page(name, url):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)
    print("Page:", url)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()

        print("HTTP status:", response.status_code)
        print("Final URL:", response.url)

        parser = LinkParser()
        parser.feed(response.text)

        matches = []

        for text, href in parser.links:
            if not href:
                continue

            full_url = urljoin(response.url, href)
            parsed_url = urlparse(full_url)

            # Ignore non-HTTP links.
            if parsed_url.scheme not in ("http", "https"):
                continue

            searchable = f"{text} {href}".lower()

            is_cutoff_candidate = any(
                keyword in searchable
                for keyword in CUTOFF_KEYWORDS
            )

            is_excluded = any(
                keyword in searchable
                for keyword in EXCLUDE_KEYWORDS
            )
            
            # Temporary discovery mode: collect all HTTP/HTTPS links.
            matches.append((text, full_url))


        # Remove duplicate links while preserving their order.
        matches = list(dict.fromkeys(matches))

        if not matches:
            print("No likely cutoff-rank links found in the page HTML.")
            print(
                "The page may use JavaScript or different document names."
            )
            return

        print(f"\nLikely cutoff-rank links: {len(matches)}\n")

        for text, link in matches:
            print("Text:", text or "(no visible text)")
            print("URL: ", link)
            print()

    except requests.RequestException as error:
        print("Request failed:", error)


def main():
    for name, url in SOURCES.items():
        inspect_page(name, url)

    print("Discovery finished. No datasets were changed.")


if __name__ == "__main__":
    main()
