import json
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_game_detail as svc

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_game_detail_200_no_store():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())

    async def fake_fetch(espn_event_id: str):
        return payload

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401857098")
    assert res.status_code == 200
    assert res.headers["Cache-Control"] == "no-store"
    body = res.json()
    assert body["espn_event_id"] == "401857098"
    assert body["away"]["abbrev"] == "GS"


def test_game_detail_route_includes_win_probability():
    payload = load_fixture("espn_wnba_summary.json")
    payload.update(load_fixture("espn_wnba_summary_with_winprobability.json"))

    async def fake_fetch(espn_event_id: str):
        return payload

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        response = client.get("/api/wnba/games/401857098")

    assert response.status_code == 200
    body = response.json()
    assert body["win_probability"]["summary"] is None
    assert body["win_probability"]["timeline"][-1]["home_win_pct"] == 54
    assert body["win_probability"]["timeline"][-1]["away_win_pct"] == 46
    assert body["win_probability"]["timeline"][-1]["id"] == "40185709811"
    assert body["win_probability"]["team_stats"][0]["label"] == "Field goal %"


def test_game_detail_route_allows_null_win_probability():
    payload = load_fixture("espn_wnba_summary.json")

    async def fake_fetch(espn_event_id: str):
        return payload

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        response = client.get("/api/wnba/games/401857098")

    assert response.status_code == 200
    assert response.json()["win_probability"] is None


def test_game_detail_404_when_espn_says_not_found():
    async def fake_fetch(espn_event_id: str):
        return {"code": 404, "message": "Not found"}

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401999999")
    assert res.status_code == 404
    assert res.headers.get("Cache-Control") == "no-store"


def test_game_detail_404_when_espn_payload_is_empty():
    async def fake_fetch(espn_event_id: str):
        return {}

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401999999")
    assert res.status_code == 404
    assert res.headers.get("Cache-Control") == "no-store"


def test_game_detail_404_when_espn_http_status_is_not_found():
    request = httpx.Request("GET", svc.ESPN_SUMMARY_URL)
    response = httpx.Response(404, request=request)

    async def fake_fetch(espn_event_id: str):
        raise httpx.HTTPStatusError(
            "Not Found",
            request=request,
            response=response,
        )

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401999999")
    assert res.status_code == 404
    assert res.headers.get("Cache-Control") == "no-store"


def test_game_detail_stale_while_error():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())

    async def ok(espn_event_id: str):
        return payload

    with patch.object(svc, "fetch_espn_summary", side_effect=ok):
        svc.clear_game_detail_cache()
        assert client.get("/api/wnba/games/401857098").status_code == 200

    async def boom(espn_event_id: str):
        raise RuntimeError("down")

    svc._cache["401857098"]["expires_at"] = 0
    with patch.object(svc, "fetch_espn_summary", side_effect=boom) as fetch:
        res = client.get("/api/wnba/games/401857098")
    fetch.assert_called_once_with("401857098")
    assert res.status_code == 200
    assert res.json()["espn_event_id"] == "401857098"


def test_game_detail_502_when_never_cached():
    async def boom(espn_event_id: str):
        raise RuntimeError("down")

    with patch.object(svc, "fetch_espn_summary", side_effect=boom):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401857098")
    assert res.status_code == 502


def test_game_detail_404_for_malformed_id_without_calling_espn():
    async def fake_fetch(espn_event_id: str):
        raise AssertionError("ESPN should not be called for a malformed id")

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch) as fetch:
        svc.clear_game_detail_cache()
        for bad_id in ("999", "abc", "not-an-id"):
            res = client.get(f"/api/wnba/games/{bad_id}")
            assert res.status_code == 404
            assert res.headers.get("Cache-Control") == "no-store"
    fetch.assert_not_called()


def test_game_detail_negative_cache_avoids_repeat_espn_calls():
    async def fake_fetch(espn_event_id: str):
        return {"code": 404, "message": "Not found"}

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch) as fetch:
        svc.clear_game_detail_cache()
        first = client.get("/api/wnba/games/401999999")
        second = client.get("/api/wnba/games/401999999")

    assert first.status_code == 404
    assert second.status_code == 404
    fetch.assert_called_once_with("401999999")
