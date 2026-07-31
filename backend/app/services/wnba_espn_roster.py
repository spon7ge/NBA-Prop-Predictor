from __future__ import annotations

import time
import unicodedata

import httpx

from app.schemas.wnba_game_detail import GameDetailStarter

ESPN_ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{team_id}/roster"
)
ESPN_TIMEOUT_SECONDS = 8.0
ROSTER_CACHE_TTL_SECONDS = 600

_roster_cache: dict[str, dict] = {}


def clear_roster_cache() -> None:
    _roster_cache.clear()


def norm_player_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold().strip()


def roster_player_index(payload: dict) -> dict[str, dict[str, str | None]]:
    index: dict[str, dict[str, str | None]] = {}
    for athlete in payload.get("athletes") or []:
        if not isinstance(athlete, dict):
            continue
        display_name = str(athlete.get("displayName") or "").strip()
        if not display_name:
            continue
        jersey_raw = athlete.get("jersey")
        jersey = str(jersey_raw).strip() if jersey_raw is not None else None
        position_block = athlete.get("position") or {}
        position = None
        if isinstance(position_block, dict):
            position = str(position_block.get("abbreviation") or "").strip() or None
        index[norm_player_name(display_name)] = {
            "jersey": jersey,
            "position": position,
        }
    return index


def enrich_starters(
    starters: list[dict],
    index: dict[str, dict[str, str | None]],
) -> list[GameDetailStarter]:
    enriched: list[GameDetailStarter] = []
    for starter in starters:
        name = str(starter.get("name") or "").strip()
        rw_position = str(starter.get("position") or "").strip()
        roster_entry = index.get(norm_player_name(name), {})
        jersey = roster_entry.get("jersey")
        position = rw_position or roster_entry.get("position")
        enriched.append(
            GameDetailStarter(jersey=jersey, name=name, position=position or None)
        )
    return enriched


async def fetch_espn_roster(team_id: str) -> dict:
    url = ESPN_ROSTER_URL.format(team_id=team_id)
    async with httpx.AsyncClient(timeout=ESPN_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def get_roster_index(team_id: str) -> dict[str, dict[str, str | None]]:
    now = time.time()
    cached = _roster_cache.get(team_id)
    if cached and float(cached["expires_at"]) > now:
        return cached["index"]  # type: ignore[return-value]
    payload = await fetch_espn_roster(team_id)
    index = roster_player_index(payload)
    _roster_cache[team_id] = {"expires_at": now + ROSTER_CACHE_TTL_SECONDS, "index": index}
    return index
