from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.routes import wnba_scoreboard as route
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
        assert date_et in (today, yesterday)
        return espn

    async def fake_fetch_stats(date_et: str):
        assert date_et in (today, yesterday)
        return stats

    with (
        patch.object(svc, "slate_et_date", return_value=today),
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert res.json()["date"] == today
    assert len(res.json()["games"]) == 1


BAD_ESPN_SCORE = {
    "events": [
        {
            "id": "401749001",
            "date": "2026-07-29T23:00Z",
            "competitions": [
                {
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "not-a-number",
                            "team": {"abbreviation": "DAL", "displayName": "Wings"},
                        },
                        {
                            "homeAway": "away",
                            "score": "36",
                            "team": {"abbreviation": "ATL", "displayName": "Dream"},
                        },
                    ]
                }
            ],
            "status": {
                "type": {"state": "in", "name": "STATUS_IN_PROGRESS"},
                "period": 3,
                "displayClock": "7:13",
            },
        }
    ]
}


def test_scoreboard_still_200_when_one_source_payload_is_malformed():
    stats = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())

    async def bad_espn(date_et: str):
        return BAD_ESPN_SCORE  # int("not-a-number") raises during normalize

    async def ok_stats(date_et: str):
        return stats

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=bad_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=ok_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert len(body["games"]) == 1
    assert body["games"][0]["id"] == "1022600123"  # stats-only survivor


def test_scoreboard_still_200_when_stats_payload_is_not_an_object():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())

    async def ok_espn(date_et: str):
        return espn

    async def junk_stats(date_et: str):
        return ["unexpected", "list"]

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=ok_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=junk_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert len(res.json()["games"]) == 1
    assert res.json()["games"][0]["id"] == "espn-401749001"


def test_scoreboard_502_no_store_when_both_payloads_malformed_and_no_cache():
    async def bad_espn(date_et: str):
        return BAD_ESPN_SCORE

    async def junk_stats(date_et: str):
        return 12345  # not a mapping at all

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=bad_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=junk_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"


def test_scoreboard_serves_stale_cache_when_both_payloads_malformed():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())

    async def ok_espn(date_et: str):
        return espn

    async def empty_stats(date_et: str):
        return {"scoreboard": {"games": []}}

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=ok_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=empty_stats),
    ):
        client = TestClient(app)
        assert client.get("/api/wnba/scoreboard/today").status_code == 200

    svc._cache["expires_at"] = 0  # force TTL expiry

    async def bad_espn(date_et: str):
        return BAD_ESPN_SCORE

    async def junk_stats(date_et: str):
        return 12345

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=bad_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=junk_stats),
    ):
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert len(res.json()["games"]) == 1


def test_concurrent_cache_misses_share_one_upstream_refresh():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    calls = {"espn": 0, "stats": 0}

    async def slow_espn(date_et: str):
        calls["espn"] += 1
        await asyncio.sleep(0.05)
        return espn

    async def slow_stats(date_et: str):
        calls["stats"] += 1
        await asyncio.sleep(0.05)
        return {"scoreboard": {"games": []}}

    async def run():
        with (
            patch.object(svc, "fetch_espn_scoreboard", side_effect=slow_espn),
            patch.object(svc, "fetch_stats_scoreboard", side_effect=slow_stats),
        ):
            return await asyncio.gather(
                *(svc.get_today_scoreboard() for _ in range(5))
            )

    responses = asyncio.run(run())
    # One shared refresh fetches today + yesterday (overnight carryover).
    assert calls == {"espn": 2, "stats": 2}
    assert all(len(r.games) == 1 for r in responses)


def test_scoreboard_502_no_store_on_unexpected_service_error():
    async def explode():
        raise ValueError("unexpected internal failure")

    with patch.object(route, "get_today_scoreboard", side_effect=explode):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"


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
        patch.object(svc, "slate_et_date", return_value=today),
        patch.object(svc, "fetch_espn_scoreboard", side_effect=boom),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=boom),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"


def test_scoreboard_includes_overnight_live_from_previous_et_day():
    today = "2026-07-30"
    yesterday = "2026-07-29"

    tonight = {
        "events": [
            {
                "id": "401857099",
                "date": "2026-07-31T00:00Z",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": None,
                                "team": {
                                    "abbreviation": "TOR",
                                    "displayName": "Toronto Tempo",
                                },
                            },
                            {
                                "homeAway": "away",
                                "score": None,
                                "team": {
                                    "abbreviation": "MIN",
                                    "displayName": "Minnesota Lynx",
                                },
                            },
                        ]
                    }
                ],
                "status": {
                    "type": {
                        "state": "pre",
                        "completed": False,
                        "name": "STATUS_SCHEDULED",
                        "shortDetail": "8:00 PM ET",
                    },
                    "period": 0,
                    "displayClock": "0.0",
                },
            }
        ]
    }
    still_live = {
        "events": [
            {
                "id": "401857098",
                "date": "2026-07-30T02:00Z",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": "80",
                                "team": {
                                    "abbreviation": "PHX",
                                    "displayName": "Phoenix Mercury",
                                },
                            },
                            {
                                "homeAway": "away",
                                "score": "77",
                                "team": {
                                    "abbreviation": "GS",
                                    "displayName": "Golden State Valkyries",
                                },
                            },
                        ]
                    }
                ],
                "status": {
                    "type": {
                        "state": "in",
                        "completed": False,
                        "name": "STATUS_IN_PROGRESS",
                        "shortDetail": "2:09 - 4th",
                    },
                    "period": 4,
                    "displayClock": "2:09",
                },
            }
        ]
    }

    async def fake_fetch_espn(date_et: str):
        return tonight if date_et == today else still_live

    async def fake_fetch_stats(date_et: str):
        return {"scoreboard": {"games": []}}

    with (
        patch.object(svc, "slate_et_date", return_value=today),
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")

    assert res.status_code == 200
    body = res.json()
    assert body["date"] == today
    statuses = {g["away"]["abbrev"]: g["status"] for g in body["games"]}
    assert statuses["GS"] == "live"
    assert statuses["MIN"] == "scheduled"
