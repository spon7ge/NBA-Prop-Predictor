from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import sharp_odds as svc

FIXTURE = Path(__file__).parent / "fixtures" / "sharp_wnba_odds.json"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_normalize_picks_favorite_spread_and_total():
    rows = json.loads(FIXTURE.read_text())["data"]
    games = svc.normalize_sharp_odds(rows)
    atl = next(g for g in games if g.home_abbrev == "ATL")
    assert atl.away_abbrev == "SEA"
    assert atl.spread_team_abbrev == "ATL"
    assert atl.spread_line == -12.5
    assert atl.total == 179.5
    assert atl.game_date == "2026-07-31"


def test_normalize_handles_missing_home_object_and_favorite_away():
    rows = json.loads(FIXTURE.read_text())["data"]
    games = svc.normalize_sharp_odds(rows)
    was = next(g for g in games if g.home_abbrev == "WAS")
    assert was.away_abbrev == "DAL"
    assert was.spread_team_abbrev == "DAL"
    assert was.spread_line == -3.5
    assert was.total == 167.5


def test_normalize_ignores_halves_and_alternates():
    rows = json.loads(FIXTURE.read_text())["data"]
    games = svc.normalize_sharp_odds(rows)
    assert {g.home_abbrev for g in games} == {"ATL", "WAS"}
    assert len(games) == 2


def test_normalize_parses_game_date_from_event_id():
    rows = json.loads(FIXTURE.read_text())["data"]
    games = svc.normalize_sharp_odds(rows, sportsbook="draftkings")
    atl = next(g for g in games if g.home_abbrev == "ATL")
    assert atl.game_date == "2026-07-31"
    assert atl.sportsbook == "draftkings"


def test_normalize_omits_game_date_when_event_id_has_none():
    rows = [
        {
            "event_id": "wnba_no_date_here",
            "is_main_line": True,
            "market_type": "total_points",
            "line": 170.5,
            "home": {"abbreviation": "ATL"},
            "away": {"abbreviation": "SEA"},
            "home_team": "ATL Dream",
            "away_team": "SEA Storm",
        }
    ]
    games = svc.normalize_sharp_odds(rows, sportsbook="fanduel")
    assert len(games) == 1
    assert games[0].game_date is None
    assert games[0].sportsbook == "fanduel"


def test_merge_odds_prefer_primary_keeps_dk_over_fd():
    from app.schemas.wnba_odds import WnbaOddsGame

    dk = [
        WnbaOddsGame(
            home_abbrev="ATL",
            away_abbrev="SEA",
            spread_team_abbrev="ATL",
            spread_line=-12.5,
            total=179.5,
            game_date="2026-07-31",
            sportsbook="draftkings",
        )
    ]
    fd = [
        WnbaOddsGame(
            home_abbrev="ATL",
            away_abbrev="SEA",
            spread_team_abbrev="ATL",
            spread_line=-11.5,
            total=180.5,
            game_date="2026-07-31",
            sportsbook="fanduel",
        )
    ]
    merged = svc.merge_odds_prefer_primary(dk, fd)
    assert len(merged) == 1
    assert merged[0].sportsbook == "draftkings"
    assert merged[0].spread_line == -12.5


def test_merge_odds_prefer_primary_fills_missing_game_from_fd():
    from app.schemas.wnba_odds import WnbaOddsGame

    dk = [
        WnbaOddsGame(
            home_abbrev="ATL",
            away_abbrev="SEA",
            spread_line=-12.5,
            total=179.5,
            game_date="2026-07-31",
            sportsbook="draftkings",
            spread_team_abbrev="ATL",
        )
    ]
    fd = [
        WnbaOddsGame(
            home_abbrev="WAS",
            away_abbrev="DAL",
            spread_line=-3.5,
            total=167.5,
            game_date="2026-07-31",
            sportsbook="fanduel",
            spread_team_abbrev="DAL",
        )
    ]
    merged = svc.merge_odds_prefer_primary(dk, fd)
    assert {g.home_abbrev for g in merged} == {"ATL", "WAS"}
    was = next(g for g in merged if g.home_abbrev == "WAS")
    assert was.sportsbook == "fanduel"


def test_get_today_odds_uses_fd_when_dk_fetch_fails():
    fd_rows = json.loads(FIXTURE.read_text())["data"]

    async def fake_fetch(sportsbook: str = "draftkings"):
        if sportsbook == "draftkings":
            raise RuntimeError("dk down")
        return fd_rows

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_odds_rows", side_effect=fake_fetch),
    ):
        body = __import__("asyncio").run(svc.get_today_odds())

    assert len(body.games) >= 1
    assert all(g.sportsbook == "fanduel" for g in body.games)


def test_odds_route_returns_games_when_fetch_ok():
    payload = json.loads(FIXTURE.read_text())

    async def fake_fetch(sportsbook: str = "draftkings"):
        return payload["data"]

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_odds_rows", side_effect=fake_fetch),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/odds/today")

    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["sportsbook"] == "draftkings"
    assert body["as_of"]
    assert len(body["games"]) == 2
    atl = next(g for g in body["games"] if g["home_abbrev"] == "ATL")
    assert atl["spread_team_abbrev"] == "ATL"
    assert atl["spread_line"] == -12.5
    assert atl["total"] == 179.5


def test_odds_route_empty_when_no_key():
    with patch.object(svc, "SHARP_API_KEY", None):
        client = TestClient(app)
        res = client.get("/api/wnba/odds/today")

    assert res.status_code == 200
    body = res.json()
    assert body["games"] == []
    assert body["error"]


def test_odds_route_stale_cache_on_error():
    payload = json.loads(FIXTURE.read_text())

    async def ok(sportsbook: str = "draftkings"):
        return payload["data"]

    async def boom(sportsbook: str = "draftkings"):
        raise RuntimeError("sharp down")

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_odds_rows", side_effect=ok),
    ):
        client = TestClient(app)
        assert client.get("/api/wnba/odds/today").status_code == 200

    svc._cache["expires_at"] = 0

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_odds_rows", side_effect=boom),
    ):
        res = client.get("/api/wnba/odds/today")

    assert res.status_code == 200
    assert len(res.json()["games"]) == 2
