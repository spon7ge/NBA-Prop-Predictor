from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_standings as svc

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_standings_returns_no_store_and_conferences():
    payload = json.loads((FIXTURES / "espn_wnba_standings.json").read_text())

    async def fake_fetch():
        return payload

    with patch.object(svc, "fetch_espn_standings", side_effect=fake_fetch):
        client = TestClient(app)
        res = client.get("/api/wnba/standings")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["season"] == 2026
    assert body["conferences"][0]["key"] == "east"
    assert body["conferences"][0]["teams"][0]["abbrev"] == "IND"
    assert body["conferences"][1]["key"] == "west"


def test_standings_uses_cache_within_ttl():
    payload = json.loads((FIXTURES / "espn_wnba_standings.json").read_text())
    calls = {"n": 0}

    async def fake_fetch():
        calls["n"] += 1
        return payload

    with patch.object(svc, "fetch_espn_standings", side_effect=fake_fetch):
        client = TestClient(app)
        assert client.get("/api/wnba/standings").status_code == 200
        assert client.get("/api/wnba/standings").status_code == 200
    assert calls["n"] == 1


def test_standings_stale_while_error():
    payload = json.loads((FIXTURES / "espn_wnba_standings.json").read_text())

    async def ok():
        return payload

    async def boom():
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_espn_standings", side_effect=ok):
        client = TestClient(app)
        assert client.get("/api/wnba/standings").status_code == 200

    svc._cache["expires_at"] = 0

    with patch.object(svc, "fetch_espn_standings", side_effect=boom):
        res = client.get("/api/wnba/standings")
    assert res.status_code == 200
    assert res.json()["conferences"][0]["teams"][0]["abbrev"] == "IND"


def test_standings_502_no_store_when_cold():
    async def boom():
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_espn_standings", side_effect=boom):
        client = TestClient(app)
        res = client.get("/api/wnba/standings")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
