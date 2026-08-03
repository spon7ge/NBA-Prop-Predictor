from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.mlb_odds import MlbOddsGame, MlbOddsResponse
from app.services import mlb_odds as mlb_svc
from app.services import sharp_odds as sharp_svc


@pytest.fixture(autouse=True)
def clear_caches():
    mlb_svc._cache.clear()
    sharp_svc._cache.clear()
    yield
    mlb_svc._cache.clear()
    sharp_svc._cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_normalize_mlb_run_line_and_total_runs():
    rows = [
        {
            "event_id": "mlb_nyy_bos_2026-08-02_a1",
            "is_main_line": True,
            "market_type": "run_line",
            "line": -1.5,
            "team_side": "away",
            "home": {"abbreviation": "BOS", "name": "Boston Red Sox"},
            "away": {"abbreviation": "NYY", "name": "New York Yankees"},
            "home_team": "BOS Red Sox",
            "away_team": "NYY Yankees",
        },
        {
            "event_id": "mlb_nyy_bos_2026-08-02_a1",
            "is_main_line": True,
            "market_type": "total_runs",
            "line": 8.5,
            "home": {"abbreviation": "BOS"},
            "away": {"abbreviation": "NYY"},
            "home_team": "BOS Red Sox",
            "away_team": "NYY Yankees",
        },
    ]
    games = sharp_svc.normalize_sharp_odds(
        rows, sportsbook="draftkings", wnba_aliases=False
    )
    assert len(games) == 1
    g = games[0]
    assert g.away_abbrev == "NYY"
    assert g.home_abbrev == "BOS"
    assert g.spread_team_abbrev == "NYY"
    assert g.spread_line == -1.5
    assert g.total == 8.5
    assert g.game_date == "2026-08-02"
    assert g.sportsbook == "draftkings"


def test_normalize_mlb_keeps_wsh_not_was():
    rows = [
        {
            "event_id": "mlb_wsh_atl_2026-08-02_b2",
            "is_main_line": True,
            "market_type": "total_runs",
            "line": 9.0,
            "home": {"abbreviation": "ATL"},
            "away": {"abbreviation": "WSH"},
            "home_team": "ATL Braves",
            "away_team": "WSH Nationals",
        }
    ]
    games = sharp_svc.normalize_sharp_odds(rows, wnba_aliases=False)
    assert games[0].away_abbrev == "WSH"


def test_mlb_odds_today_ok(client):
    sample = MlbOddsResponse(
        as_of="2026-08-02T12:00:00Z",
        games=[
            MlbOddsGame(
                home_abbrev="BOS",
                away_abbrev="NYY",
                spread_team_abbrev="NYY",
                spread_line=-1.5,
                total=8.5,
                game_date="2026-08-02",
                sportsbook="draftkings",
            )
        ],
    )
    with patch(
        "app.api.routes.mlb_odds.get_today_odds",
        new=AsyncMock(return_value=sample),
    ):
        res = client.get("/api/mlb/odds/today")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["games"][0]["away_abbrev"] == "NYY"
    assert body["games"][0]["total"] == 8.5
