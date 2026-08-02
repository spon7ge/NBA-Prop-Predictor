from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.wnba_futures import WnbaFuturesResponse
from app.services import wnba_futures as svc

FIXTURES = Path(__file__).parent / "fixtures"
FUTURES_FIXTURE = FIXTURES / "espn_wnba_futures.json"

TEAMS_BY_ID = {
    "8": {
        "id": "8",
        "abbreviation": "NYL",
        "displayName": "New York Liberty",
        "logos": [{"href": "https://example.com/nyl.png"}],
    },
    "17": {
        "id": "17",
        "abbreviation": "LAS",
        "displayName": "Los Angeles Sparks",
        "logos": [{"href": "https://example.com/las.png"}],
    },
    "5": {
        "id": "5",
        "abbreviation": "IND",
        "displayName": "Indiana Fever",
        "logos": [{"href": "https://example.com/ind.png"}],
    },
    "9": {
        "id": "9",
        "abbreviation": "PHX",
        "displayName": "Phoenix Mercury",
        "logos": [{"href": "https://example.com/phx.png"}],
    },
}


def _team_id_from_ref(ref: str) -> str | None:
    match = re.search(r"/teams/(\d+)", ref)
    return match.group(1) if match else None


async def _fake_resolve(ref_or_id: str, client: httpx.AsyncClient) -> dict | None:
    team_id = _team_id_from_ref(ref_or_id) or ref_or_id
    raw = TEAMS_BY_ID.get(str(team_id))
    if raw is None:
        return None
    logos = raw.get("logos") or []
    logo_url = logos[0].get("href") if logos else None
    return {
        "id": raw["id"],
        "abbreviation": raw["abbreviation"],
        "displayName": raw["displayName"],
        "logo_url": logo_url,
    }


def test_display_name_maps_winner():
    assert svc.display_name_for_market("WNBA - Winner") == "Finals Winner"
    assert svc.display_name_for_market("Other Market") == "Other Market"


def test_parse_american_odds():
    assert svc.parse_american_odds("+250") == 250
    assert svc.parse_american_odds("-150") == -150
    assert svc.parse_american_odds("even") is None


def test_normalize_sorts_favorites_first_and_maps_teams(monkeypatch):
    payload = json.loads(FUTURES_FIXTURE.read_text())

    async def fake_resolve(ref_or_id: str, client: httpx.AsyncClient) -> dict | None:
        return await _fake_resolve(ref_or_id, client)

    monkeypatch.setattr(svc, "resolve_team", fake_resolve)

    async def run():
        async with httpx.AsyncClient() as client:
            return await svc.normalize_futures_payload(payload, 2026, client)

    result = asyncio.run(run())

    assert result.season == 2026
    assert result.error is None
    assert len(result.markets) == 1

    market = result.markets[0]
    assert market.id == "8146"
    assert market.name == "WNBA - Winner"
    assert market.display_name == "Finals Winner"
    assert market.provider == "ESPN BET"
    assert len(market.entries) == 4

    odds = [entry.odds_american for entry in market.entries]
    assert odds == ["+250", "+290", "+290", "+600"]

    favorite = market.entries[0]
    assert favorite.team_id == "8"
    assert favorite.abbrev == "NYL"
    assert favorite.name == "New York Liberty"
    assert favorite.logo_url == "https://example.com/nyl.png"
    assert favorite.odds_american == "+250"


@pytest.fixture(autouse=True)
def clear_futures_cache():
    svc._cache.clear()
    svc._team_cache.clear()
    yield
    svc._cache.clear()
    svc._team_cache.clear()


def test_normalize_embedded_team_skips_http():
    """Embedded team with id/abbrev/name must normalize without HTTP."""
    payload = {
        "items": [
            {
                "id": "8146",
                "name": "WNBA - Winner",
                "futures": [
                    {
                        "provider": {"name": "ESPN BET", "active": 1},
                        "books": [
                            {
                                "value": "+250",
                                "team": {
                                    "id": "8",
                                    "abbreviation": "NYL",
                                    "displayName": "New York Liberty",
                                    "logos": [
                                        {"href": "https://a.espncdn.com/nyl.png"}
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        ]
    }

    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(
            AssertionError(f"unexpected HTTP: {request.url}")
        )
    )

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await svc.normalize_futures_payload(payload, 2026, client)

    result = asyncio.run(run())
    assert len(result.markets) == 1
    entry = result.markets[0].entries[0]
    assert entry.team_id == "8"
    assert entry.abbrev == "NYL"
    assert entry.name == "New York Liberty"
    assert entry.logo_url == "https://a.espncdn.com/nyl.png"
    assert entry.odds_american == "+250"


def test_resolve_team_rejects_non_espn_ref_host():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "8",
                "abbreviation": "EVL",
                "displayName": "Evil Team",
                "logos": [{"href": "https://evil.example.com/logo.png"}],
            },
        )

    transport = httpx.MockTransport(handler)

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await svc.resolve_team(
                "https://evil.example.com/v2/sports/basketball/leagues/wnba/teams/8",
                client,
            )

    assert asyncio.run(run()) is None
    assert calls == []


def test_futures_route_ok(monkeypatch):
    payload = json.loads(FUTURES_FIXTURE.read_text())

    async def fake_resolve(ref_or_id: str, client: httpx.AsyncClient) -> dict | None:
        return await _fake_resolve(ref_or_id, client)

    monkeypatch.setattr(svc, "resolve_team", fake_resolve)

    async def fake_fetch(season: int):
        return payload

    with patch.object(svc, "fetch_espn_futures", side_effect=fake_fetch):
        client = TestClient(app)
        res = client.get("/api/wnba/futures")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    assert res.json()["markets"][0]["display_name"] == "Finals Winner"


def test_get_wnba_futures_uses_cache(monkeypatch):
    payload = json.loads(FUTURES_FIXTURE.read_text())
    calls = {"n": 0}

    async def fake_resolve(ref_or_id: str, client: httpx.AsyncClient) -> dict | None:
        return await _fake_resolve(ref_or_id, client)

    monkeypatch.setattr(svc, "resolve_team", fake_resolve)

    async def fake_fetch(season: int):
        calls["n"] += 1
        return payload

    with patch.object(svc, "fetch_espn_futures", side_effect=fake_fetch):
        asyncio.run(svc.get_wnba_futures())
        asyncio.run(svc.get_wnba_futures())
    assert calls["n"] == 1


def test_get_wnba_futures_stale_while_error(monkeypatch):
    payload = json.loads(FUTURES_FIXTURE.read_text())

    async def fake_resolve(ref_or_id: str, client: httpx.AsyncClient) -> dict | None:
        return await _fake_resolve(ref_or_id, client)

    monkeypatch.setattr(svc, "resolve_team", fake_resolve)

    async def ok(season: int):
        return payload

    async def boom(season: int):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_espn_futures", side_effect=ok):
        asyncio.run(svc.get_wnba_futures())

    svc._cache["expires_at"] = 0

    with patch.object(svc, "fetch_espn_futures", side_effect=boom):
        result = asyncio.run(svc.get_wnba_futures())
    assert isinstance(result, WnbaFuturesResponse)
    assert result.markets[0].display_name == "Finals Winner"


def test_futures_route_502_no_store_when_cold():
    async def boom(season: int):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_espn_futures", side_effect=boom):
        client = TestClient(app)
        res = client.get("/api/wnba/futures")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
