"""Unit tests for Underdog sport → league mapping."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRAPER_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "scrapers" / "underdog_scraper.py"
)


def _load_scraper():
    spec = importlib.util.spec_from_file_location("underdog_scraper", _SCRAPER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["underdog_scraper"] = mod
    spec.loader.exec_module(mod)
    return mod


ud = _load_scraper()


class TestSportToLeague:
    def test_nba_case_insensitive(self) -> None:
        assert ud.sport_to_league("NBA") == "nba"
        assert ud.sport_to_league("nba") == "nba"
        assert ud.sport_to_league("  nba  ") == "nba"

    def test_wnba_case_insensitive(self) -> None:
        assert ud.sport_to_league("WNBA") == "wnba"
        assert ud.sport_to_league("wnba") == "wnba"
        assert ud.sport_to_league("  wnba  ") == "wnba"

    def test_unknown_sports_return_none(self) -> None:
        assert ud.sport_to_league("unknown") is None
        assert ud.sport_to_league("MLB") is None
        assert ud.sport_to_league("") is None
