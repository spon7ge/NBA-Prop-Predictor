import json
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_game_detail as svc

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


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


def test_game_detail_404_when_espn_says_not_found():
    async def fake_fetch(espn_event_id: str):
        return {"code": 404, "message": "Not found"}

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/999")
    assert res.status_code == 404
    assert res.headers.get("Cache-Control") == "no-store"


def test_game_detail_404_when_espn_payload_is_empty():
    async def fake_fetch(espn_event_id: str):
        return {}

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/999")
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
        res = client.get("/api/wnba/games/999")
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
