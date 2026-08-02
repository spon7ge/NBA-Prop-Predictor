from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.schemas.wnba_futures import (
    WnbaFuturesEntry,
    WnbaFuturesMarket,
    WnbaFuturesResponse,
)

logger = logging.getLogger(__name__)

FUTURES_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
    "wnba/seasons/{season}/futures"
)
CACHE_TTL_SECONDS = 300.0

_team_cache: dict[str, dict[str, Any]] = {}


def display_name_for_market(name: str) -> str:
    if name.strip() == "WNBA - Winner":
        return "Finals Winner"
    return name.strip() or "Futures"


def parse_american_odds(value: str) -> int | None:
    text = str(value or "").strip().replace("−", "-")
    if not text:
        return None
    if text[0] not in "+-" and not text.isdigit():
        return None
    try:
        return int(text)
    except ValueError:
        return None


def pick_provider(futures: list[dict]) -> dict | None:
    if not futures:
        return None

    def is_active(entry: dict) -> bool:
        provider = entry.get("provider") or {}
        active = provider.get("active")
        return active in (1, True, "1")

    for entry in futures:
        if not isinstance(entry, dict):
            continue
        provider = entry.get("provider") or {}
        name = str(provider.get("name") or "").lower()
        if is_active(entry) and "espn bet" in name:
            return entry

    for entry in futures:
        if isinstance(entry, dict) and is_active(entry):
            return entry

    first = futures[0]
    return first if isinstance(first, dict) else None


def _logo_url(team: dict[str, Any]) -> str | None:
    logos = team.get("logos") or []
    if not isinstance(logos, list):
        return None
    for logo in logos:
        if not isinstance(logo, dict):
            continue
        href = str(logo.get("href") or "").strip()
        if href:
            return href
    return None


def _normalize_team_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    team_id = str(payload.get("id") or "").strip()
    abbrev = str(payload.get("abbreviation") or "").strip().upper()
    name = str(payload.get("displayName") or "").strip()
    if not team_id or not abbrev or not name:
        return None
    return {
        "id": team_id,
        "abbreviation": abbrev,
        "displayName": name,
        "logo_url": _logo_url(payload),
    }


def _team_ref(book: dict[str, Any]) -> str | None:
    team = book.get("team") or {}
    if not isinstance(team, dict):
        return None
    ref = str(team.get("$ref") or "").strip()
    if ref:
        return ref
    team_id = str(team.get("id") or "").strip()
    if team_id:
        return team_id
    return None


async def resolve_team(ref_or_id: str, client: httpx.AsyncClient) -> dict | None:
    ref = str(ref_or_id or "").strip()
    if not ref:
        return None

    cache_key = ref.replace("http://", "https://")
    if cache_key in _team_cache:
        return _team_cache[cache_key]

    if ref.startswith("http://") or ref.startswith("https://"):
        url = ref.replace("http://", "https://", 1)
        try:
            res = await client.get(url)
            res.raise_for_status()
            payload = res.json()
        except Exception:
            logger.warning("Failed to resolve WNBA team ref: %s", ref)
            return None
    else:
        payload = {"id": ref}

    normalized = _normalize_team_payload(payload)
    if normalized is not None:
        _team_cache[cache_key] = normalized
    return normalized


def _sort_key(entry: WnbaFuturesEntry) -> tuple[int, int]:
    parsed = parse_american_odds(entry.odds_american)
    if parsed is None:
        return (1, 0)
    return (0, parsed)


async def normalize_futures_payload(
    payload: dict[str, Any],
    season: int,
    client: httpx.AsyncClient,
) -> WnbaFuturesResponse:
    markets: list[WnbaFuturesMarket] = []

    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue

        market_id = str(item.get("id") or "").strip()
        market_name = str(item.get("name") or "").strip()
        if not market_id or not market_name:
            continue

        provider_blob = pick_provider(item.get("futures") or [])
        if provider_blob is None:
            continue

        provider = provider_blob.get("provider") or {}
        provider_name = str(provider.get("name") or "").strip() or "Unknown"

        entries: list[WnbaFuturesEntry] = []
        for book in provider_blob.get("books") or []:
            if not isinstance(book, dict):
                continue

            odds = str(book.get("value") or "").strip()
            if not odds:
                continue

            ref = _team_ref(book)
            if ref is None:
                continue

            team = await resolve_team(ref, client)
            if team is None:
                continue

            entries.append(
                WnbaFuturesEntry(
                    team_id=str(team["id"]),
                    abbrev=str(team["abbreviation"]),
                    name=str(team["displayName"]),
                    logo_url=team.get("logo_url"),
                    odds_american=odds,
                )
            )

        entries.sort(key=_sort_key)
        markets.append(
            WnbaFuturesMarket(
                id=market_id,
                name=market_name,
                display_name=display_name_for_market(market_name),
                provider=provider_name,
                entries=entries,
            )
        )

    as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    return WnbaFuturesResponse(season=season, as_of=as_of, markets=markets)
