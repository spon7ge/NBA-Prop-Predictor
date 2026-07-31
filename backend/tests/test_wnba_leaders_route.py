from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_leaders as svc

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_leaders_returns_no_store_and_categories():
    payload = json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )

    async def fake_fetch(season: int):
        return payload

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=fake_fetch):
        client = TestClient(app)
        res = client.get("/api/wnba/leaders")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["pace"] == "per_game"
    assert len(body["categories"]) == 6
    assert body["categories"][0]["key"] == "points"
    assert body["categories"][0]["leaders"][0]["name"] == "A'ja Wilson"


def test_leaders_uses_cache_within_ttl():
    payload = json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )
    calls = {"n": 0}

    async def fake_fetch(season: int):
        calls["n"] += 1
        return payload

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=fake_fetch):
        client = TestClient(app)
        assert client.get("/api/wnba/leaders").status_code == 200
        assert client.get("/api/wnba/leaders").status_code == 200
    assert calls["n"] == 1


def test_leaders_stale_while_error():
    payload = json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )

    async def ok(season: int):
        return payload

    async def boom(season: int):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=ok):
        client = TestClient(app)
        assert client.get("/api/wnba/leaders").status_code == 200

    svc._cache["expires_at"] = 0

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom):
        res = client.get("/api/wnba/leaders")
    assert res.status_code == 200
    assert res.json()["categories"][0]["leaders"][0]["name"] == "A'ja Wilson"


def test_leaders_502_no_store_when_cold():
    async def boom(season: int):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom):
        client = TestClient(app)
        res = client.get("/api/wnba/leaders")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
