from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_player as svc

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(autouse=True)
def clear_player_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_format_pct_handles_fraction_and_percent():
    assert svc.format_pct(0.482) == "48.2"
    assert svc.format_pct(48.2) == "48.2"
    assert svc.format_pct(None) is None


def test_made_attempt():
    assert svc.made_attempt(11, 20) == "11-20"


def test_normalize_player_happy_path():
    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.player_id == "1628932"
    assert result.name == "A'ja Wilson"
    assert result.position  # from info
    assert result.team_abbrev == "LVA"
    assert result.averages.pts  # one-decimal string
    assert result.averages.fg_pct
    assert result.averages.fg3_pct
    assert len(result.games) >= 6
    g0 = result.games[0]
    assert g0.fg  # "m-a"
    assert g0.three_pt
    assert g0.ft
    assert result.source_label == "stats.wnba.com"
    assert result.headshot_url  # non-empty CDN URL containing player_id


def test_normalize_unknown_player_returns_none():
    result = svc.normalize_wnba_player(
        player_id="99999999",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is None


def test_player_route_200_no_store():
    async def fake_get(player_id: str):
        return svc.normalize_wnba_player(
            player_id="1628932",
            season=2026,
            dash=_load("stats_wnba_player_dash.json"),
            info=_load("stats_wnba_player_info.json"),
            gamelog=_load("stats_wnba_player_gamelog.json"),
        )

    with patch("app.api.routes.wnba_player.get_wnba_player", side_effect=fake_get):
        client = TestClient(app)
        res = client.get("/api/wnba/player/1628932")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.json()["name"] == "A'ja Wilson"


def test_player_route_404():
    async def missing(player_id: str):
        raise HTTPException(status_code=404, detail="Player not found")

    with patch("app.api.routes.wnba_player.get_wnba_player", side_effect=missing):
        client = TestClient(app)
        res = client.get("/api/wnba/player/999")
    assert res.status_code == 404
    assert res.headers.get("cache-control") == "no-store"


def test_player_route_502_cold():
    async def boom(*_a, **_k):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom), \
         patch.object(svc, "fetch_commonplayerinfo", side_effect=boom), \
         patch.object(svc, "fetch_playergamelog", side_effect=boom):
        client = TestClient(app)
        res = client.get("/api/wnba/player/1628932")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"


def test_get_wnba_player_uses_cache():
    calls = {"dash": 0, "info": 0, "gamelog": 0}

    async def fake_dash(season: int):
        calls["dash"] += 1
        return _load("stats_wnba_player_dash.json")

    async def fake_info(player_id: str):
        calls["info"] += 1
        return _load("stats_wnba_player_info.json")

    async def fake_gamelog(player_id: str, season: int):
        calls["gamelog"] += 1
        return _load("stats_wnba_player_gamelog.json")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=fake_dash), \
         patch.object(svc, "fetch_commonplayerinfo", side_effect=fake_info), \
         patch.object(svc, "fetch_playergamelog", side_effect=fake_gamelog), \
         patch.object(svc, "current_wnba_season_year", return_value=2026):
        first = asyncio.run(svc.get_wnba_player("1628932"))
        second = asyncio.run(svc.get_wnba_player("1628932"))
    assert first.name == "A'ja Wilson"
    assert second.name == "A'ja Wilson"
    assert calls == {"dash": 1, "info": 1, "gamelog": 1}


def test_get_wnba_player_stale_while_error():
    async def ok_dash(season: int):
        return _load("stats_wnba_player_dash.json")

    async def ok_info(player_id: str):
        return _load("stats_wnba_player_info.json")

    async def ok_gamelog(player_id: str, season: int):
        return _load("stats_wnba_player_gamelog.json")

    async def boom(*_a, **_k):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=ok_dash), \
         patch.object(svc, "fetch_commonplayerinfo", side_effect=ok_info), \
         patch.object(svc, "fetch_playergamelog", side_effect=ok_gamelog), \
         patch.object(svc, "current_wnba_season_year", return_value=2026):
        first = asyncio.run(svc.get_wnba_player("1628932"))
    assert first.name == "A'ja Wilson"
    svc._cache["1628932"]["expires_at"] = 0

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom), \
         patch.object(svc, "fetch_commonplayerinfo", side_effect=boom), \
         patch.object(svc, "fetch_playergamelog", side_effect=boom), \
         patch.object(svc, "current_wnba_season_year", return_value=2026):
        stale = asyncio.run(svc.get_wnba_player("1628932"))
    assert stale.name == "A'ja Wilson"


def test_get_wnba_player_404_when_missing():
    async def fake_dash(season: int):
        return _load("stats_wnba_player_dash.json")

    async def fake_info(player_id: str):
        return _load("stats_wnba_player_info.json")

    async def fake_gamelog(player_id: str, season: int):
        return _load("stats_wnba_player_gamelog.json")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=fake_dash), \
         patch.object(svc, "fetch_commonplayerinfo", side_effect=fake_info), \
         patch.object(svc, "fetch_playergamelog", side_effect=fake_gamelog), \
         patch.object(svc, "current_wnba_season_year", return_value=2026):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(svc.get_wnba_player("999"))
    assert exc_info.value.status_code == 404
