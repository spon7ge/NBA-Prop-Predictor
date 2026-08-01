"""Unit tests for PrizePicks scraper helpers (no live network)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest

_SCRAPER_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "scrapers" / "prizepicks_scraper.py"
)


def _load_scraper():
    spec = importlib.util.spec_from_file_location("prizepicks_scraper", _SCRAPER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prizepicks_scraper"] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load_scraper()


class TestIsDatadomeChallenge:
    def test_detects_captcha_delivery_url_in_body(self) -> None:
        body = (
            '{"url":"https://geo.captcha-delivery.com/captcha/'
            '?initialCid=AHrlqAAAAAMACmknY9EDriAA_Jy_tQ=="}'
        )
        assert pp.is_datadome_challenge(body) is True

    def test_detects_x_datadome_header(self) -> None:
        assert (
            pp.is_datadome_challenge("{}", headers={"X-DataDome": "protected"}) is True
        )

    def test_ignores_normal_json(self) -> None:
        assert pp.is_datadome_challenge('{"data":[]}') is False
        assert pp.is_datadome_challenge("", headers={}) is False


class TestIsBotChallenge:
    def test_datadome_counts(self) -> None:
        assert pp.is_bot_challenge('{"url":"https://geo.captcha-delivery.com/x"}') is True

    def test_perimeterx_still_counts(self) -> None:
        assert pp.is_bot_challenge("px-cloud challenge") is True

    def test_clean_body_false(self) -> None:
        assert pp.is_bot_challenge('{"data":[{"type":"projection"}]}') is False


class TestBuildFetchFailureMessage:
    def test_mentions_datadome_and_playwright(self) -> None:
        msg = pp.build_fetch_failure_message()
        assert "DataDome" in msg
        assert "playwright" in msg.lower()
        assert "PRIZEPICKS_COOKIE" in msg
        # PerimeterX should not be the primary framing
        assert not msg.strip().startswith("PerimeterX")
        assert "PerimeterX" not in msg.split("Troubleshooting")[0]


class TestHeadlessFromEnv:
    def test_default_headed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRIZEPICKS_HEADLESS", raising=False)
        assert pp.headless_from_env() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_headless(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("PRIZEPICKS_HEADLESS", value)
        assert pp.headless_from_env() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_falsy_headless(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("PRIZEPICKS_HEADLESS", value)
        assert pp.headless_from_env() is False


class TestLeagues:
    def test_default_leagues_are_nba_and_wnba(self) -> None:
        assert pp.DEFAULT_LEAGUES == (("NBA", 7), ("WNBA", 3))

    def test_projections_api_url(self) -> None:
        url = pp.projections_api_url(3)
        assert "league_id=3" in url
        assert "per_page=250" in url
        assert url.startswith("https://api.prizepicks.com/projections?")

    def test_extract_sets_league(self) -> None:
        payload = {
            "data": [
                {
                    "type": "projection",
                    "attributes": {
                        "line_score": 22.5,
                        "stat_type": "Points",
                        "odds_type": "standard",
                        "updated_at": "2026-07-31T00:00:00Z",
                        "description": "Fallback",
                    },
                    "relationships": {
                        "new_player": {"data": {"id": "p1"}},
                    },
                }
            ],
            "included": [
                {
                    "type": "new_player",
                    "id": "p1",
                    "attributes": {"name": "A'ja Wilson"},
                }
            ],
        }
        rows = pp.extract_projections(payload, league="WNBA")
        assert len(rows) == 1
        assert rows[0].league == "WNBA"
        assert rows[0].player == "A'ja Wilson"


class TestResolveOutputPath:
    def test_includes_league_slug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRIZEPICKS_OUTPUT", raising=False)
        when = datetime(2026, 7, 31, 18, 2, 38, tzinfo=pp._OUTPUT_TZ)
        path = pp.resolve_output_path("WNBA", when=when)
        assert path.endswith("prizepicks_wnba_2026-07-31_180238.json")
        assert path.startswith(pp._DEFAULT_OUTPUT_DIR)

    def test_nba_slug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRIZEPICKS_OUTPUT", raising=False)
        when = datetime(2026, 7, 31, 18, 2, 38, tzinfo=pp._OUTPUT_TZ)
        path = pp.resolve_output_path("NBA", when=when)
        assert "prizepicks_nba_2026-07-31_180238.json" in path

    def test_league_slug_helper(self) -> None:
        assert pp.league_slug("WNBA") == "wnba"
        assert pp.league_slug("NBA") == "nba"
