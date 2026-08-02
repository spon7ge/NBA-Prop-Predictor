from __future__ import annotations

import asyncio
import json
from datetime import date
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
    svc._refresh_locks.clear()
    svc._refresh_locks_loop = None
    yield
    svc._cache.clear()
    svc._refresh_locks.clear()
    svc._refresh_locks_loop = None


def test_format_pct_handles_fraction_and_percent():
    assert svc.format_pct(0.482) == "48.2"
    assert svc.format_pct(48.2) == "48.2"
    assert svc.format_pct(None) is None


def test_made_attempt():
    assert svc.made_attempt(11, 20) == "11-20"


def test_format_height():
    assert svc.format_height("5-10") == "5' 10\""
    assert svc.format_height("6-4") == "6' 4\""
    assert svc.format_height(None) is None


def test_format_birthdate_with_age():
    assert svc.format_birthdate(
        "2003-01-29T00:00:00", today=date(2026, 8, 1)
    ) == "1/29/2003 (23)"


def test_format_position_expands_abbrev():
    assert svc.format_position("G") == "Guard"
    assert svc.format_position("F") == "Forward"
    assert svc.format_position("C") == "Center"
    assert svc.format_position("Guard") == "Guard"


def test_format_draft():
    assert (
        svc.format_draft("2026", "1", "2", "MIN")
        == "2026: Rd 1, Pk 2 (MIN)"
    )
    assert svc.format_draft(None, None, None, "MIN") is None


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


def test_normalize_includes_bio_fields():
    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.jersey == "22"
    assert result.position == "Center"
    assert result.height == "6' 4\""
    assert result.college == "South Carolina"
    assert result.draft_info == "2018: Rd 1, Pk 1 (LVA)"
    assert result.birthdate  # contains year and age parens
    assert "1996" in result.birthdate
    assert "(" in result.birthdate and ")" in result.birthdate


def test_normalize_sparse_info_bio_fields_none():
    """Missing HEIGHT/SCHOOL must yield None bio fields; player still returns."""
    info = _load("stats_wnba_player_info.json")
    headers = [str(h) for h in info["resultSets"][0]["headers"]]
    height_i = headers.index("HEIGHT")
    school_i = headers.index("SCHOOL")
    for row in info["resultSets"][0]["rowSet"]:
        row[height_i] = None
        row[school_i] = None

    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=info,
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.name == "A'ja Wilson"
    assert result.height is None
    assert result.college is None
    assert result.jersey == "22"
    assert result.position == "Center"


def test_normalize_unknown_player_returns_none():
    result = svc.normalize_wnba_player(
        player_id="99999999",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is None


def test_normalize_sparse_averages_returns_player():
    """Missing/unparseable avg fields must not 404 — use display fallbacks."""
    dash = _load("stats_wnba_player_dash.json")
    headers = [str(h) for h in dash["resultSets"][0]["headers"]]
    fg_pct_i = headers.index("FG_PCT")
    fg3_pct_i = headers.index("FG3_PCT")
    ast_i = headers.index("AST")
    player_i = headers.index("PLAYER_ID")
    for row in dash["resultSets"][0]["rowSet"]:
        if str(row[player_i]) == "1628932":
            row[fg_pct_i] = None
            row[fg3_pct_i] = "n/a"
            row[ast_i] = None
            break

    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=dash,
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.name == "A'ja Wilson"
    assert result.averages.fg_pct == "—"
    assert result.averages.fg3_pct == "—"
    assert result.averages.ast == "0.0"
    assert result.averages.pts  # still parsed from fixture


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


def test_refresh_locks_are_per_player_id():
    """Distinct player_ids must not share a single process-wide lock."""

    async def run():
        lock_a = svc._get_refresh_lock("111")
        lock_b = svc._get_refresh_lock("222")
        assert lock_a is not lock_b
        assert svc._get_refresh_lock("111") is lock_a

        # Concurrent holders of different locks must both proceed (no global serialize).
        order: list[str] = []
        started = asyncio.Event()
        release_a = asyncio.Event()

        async def hold_a():
            async with lock_a:
                order.append("a_enter")
                started.set()
                await release_a.wait()
                order.append("a_exit")

        async def hold_b():
            await started.wait()
            async with lock_b:
                order.append("b_enter")
                order.append("b_exit")

        task_a = asyncio.create_task(hold_a())
        task_b = asyncio.create_task(hold_b())
        await started.wait()
        # Give hold_b a turn; with a global lock it would block until a exits.
        await asyncio.sleep(0.01)
        assert "b_enter" in order
        release_a.set()
        await asyncio.gather(task_a, task_b)
        assert order == ["a_enter", "b_enter", "b_exit", "a_exit"]

    asyncio.run(run())
