from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx

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
