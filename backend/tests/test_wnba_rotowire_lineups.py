from pathlib import Path

from src.scrapers.rotowire_starters_scraper import WNBADailyLineups

FIXTURES = Path(__file__).parent / "fixtures"


def test_expected_starters_by_abbr_preserves_order_and_positions(monkeypatch):
    html = (FIXTURES / "rotowire_wnba_lineups_sea_atl.html").read_text()

    class FakeLineups(WNBADailyLineups):
        def _get_soup(self):
            from bs4 import BeautifulSoup

            return BeautifulSoup(html, "html.parser")

    scraped = FakeLineups()
    by_abbr = scraped.expected_starters_by_abbr()
    atl = by_abbr["ATL"]
    assert [p["name"] for p in atl] == [
        "Allisha Gray",
        "Jordin Canada",
        "Rhyne Howard",
        "Naz Hillmon",
        "Angel Reese",
    ]
    assert atl[-1]["position"] == "F"
    assert "Madina Okot" not in [p["name"] for p in atl]
    assert len(by_abbr["SEA"]) == 5
