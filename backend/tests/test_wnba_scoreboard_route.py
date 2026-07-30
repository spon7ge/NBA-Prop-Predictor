from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.wnba_scoreboard import WnbaScoreboardResponse
from app.services import wnba_scoreboard as svc

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_scoreboard_today_returns_no_store_and_games():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    stats = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())

    async def fake_fetch_espn(date_et: str):
        return espn

    async def fake_fetch_stats(date_et: str):
        return stats

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["date"]
    assert len(body["games"]) == 1
    assert body["games"][0]["league"] == "wnba"


def test_scoreboard_stale_while_error_when_both_fail_after_success():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())

    async def ok_espn(date_et: str):
        return espn

    async def ok_stats(date_et: str):
        return {"scoreboard": {"games": []}}

    async def boom(date_et: str):
        raise RuntimeError("upstream down")

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=ok_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=ok_stats),
    ):
        client = TestClient(app)
        assert client.get("/api/wnba/scoreboard/today").status_code == 200

    svc._cache["expires_at"] = 0  # force TTL expiry

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=boom),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=boom),
    ):
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert len(res.json()["games"]) >= 1


def test_scoreboard_502_no_store_when_both_fail_with_empty_cache():
    async def boom(date_et: str):
        raise RuntimeError("upstream down")

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=boom),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=boom),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"


def test_scoreboard_cache_misses_on_different_et_date():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    stats = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())
    today = "2026-07-30"
    yesterday = "2026-07-29"

    svc._cache["response"] = WnbaScoreboardResponse(
        date=yesterday,
        games=[],
        fetched_at="2026-07-29T12:00:00-04:00",
    )
    svc._cache["date"] = yesterday
    svc._cache["expires_at"] = time.time() + 3600

    async def fake_fetch_espn(date_et: str):
        assert date_et == today
        return espn

    async def fake_fetch_stats(date_et: str):
        assert date_et == today
        return stats

    with (
        patch.object(svc, "today_et_date", return_value=today),
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert res.json()["date"] == today
    assert len(res.json()["games"]) == 1


def test_scoreboard_stale_while_error_refuses_yesterday_slate():
    yesterday = "2026-07-29"
    today = "2026-07-30"

    svc._cache["response"] = WnbaScoreboardResponse(
        date=yesterday,
        games=[],
        fetched_at="2026-07-29T12:00:00-04:00",
    )
    svc._cache["date"] = yesterday
    svc._cache["expires_at"] = 0

    async def boom(date_et: str):
        raise RuntimeError("upstream down")

    with (
        patch.object(svc, "today_et_date", return_value=today),
        patch.object(svc, "fetch_espn_scoreboard", side_effect=boom),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=boom),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
