import asyncio
from pathlib import Path

import app.services.wnba_rotowire_lineups as rw
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
    sea = by_abbr["SEA"]
    assert len(sea) == 5
    assert [p["name"] for p in sea] == [
        "Natisha Hiedeman",
        "Jade Melbourne",
        "Flau'jae Johnson",
        "Awa Fam",
        "Dominique Malonga",
    ]
    assert sea[-1]["position"] == "C"
    assert "Ebony Hoffman" not in [p["name"] for p in sea]


def test_get_rotowire_starters_for_matchup_sea_atl(monkeypatch):
    rw.clear_rotowire_lineups_cache()
    html = (FIXTURES / "rotowire_wnba_lineups_sea_atl.html").read_text()

    def fake_fetch():
        from bs4 import BeautifulSoup
        from src.scrapers.rotowire_starters_scraper import WNBADailyLineups

        class Fake(WNBADailyLineups):
            def _get_soup(self):
                return BeautifulSoup(html, "html.parser")

        return Fake().expected_starters_by_abbr()

    monkeypatch.setattr(rw, "_scrape_starters_by_abbr", fake_fetch)
    result = asyncio.run(
        rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL")
    )
    assert result is not None
    assert result["home"][-1]["name"] == "Angel Reese"
    assert len(result["away"]) == 5
    assert result["away"][-1]["name"] == "Dominique Malonga"


def test_get_rotowire_starters_returns_none_if_team_missing(monkeypatch):
    rw.clear_rotowire_lineups_cache()
    monkeypatch.setattr(rw, "_scrape_starters_by_abbr", lambda: {"ATL": [{"name": "X", "position": "F"}] * 5})
    result = asyncio.run(
        rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL")
    )
    assert result is None


def test_rotowire_cache_reuses_scrape(monkeypatch):
    rw.clear_rotowire_lineups_cache()
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        five = [{"name": f"P{i}", "position": "G"} for i in range(5)]
        return {"SEA": five, "ATL": five}

    monkeypatch.setattr(rw, "_scrape_starters_by_abbr", fake_fetch)
    asyncio.run(rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL"))
    asyncio.run(rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL"))
    assert calls["n"] == 1


def test_get_rotowire_starters_aliases_espn_wsh_to_was(monkeypatch):
    """ESPN game detail uses WSH; RotoWire lineup__abbr is WAS."""
    rw.clear_rotowire_lineups_cache()
    five_dal = [{"name": f"Dal{i}", "position": "G"} for i in range(5)]
    five_was = [{"name": f"Was{i}", "position": "G"} for i in range(5)]
    five_was[-1] = {"name": "Shakira Austin", "position": "C"}

    monkeypatch.setattr(
        rw,
        "_scrape_starters_by_abbr",
        lambda: {"DAL": five_dal, "WAS": five_was},
    )
    result = asyncio.run(
        rw.get_rotowire_starters_for_matchup(away_abbr="DAL", home_abbr="WSH")
    )
    assert result is not None
    assert result["home"][-1]["name"] == "Shakira Austin"
    assert len(result["away"]) == 5
