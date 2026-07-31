from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.schemas.wnba_leaders import (
    LeaderCategoryKey,
    WnbaLeaderCategory,
    WnbaLeaderRow,
    WnbaLeadersResponse,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
STATS_URL = "https://stats.wnba.com/stats/leaguedashplayerstats"
STATS_TIMEOUT_SECONDS = 10.0
CACHE_TTL_SECONDS = 10 * 60

_cache: dict = {}  # response, expires_at, season
_refresh_lock: asyncio.Lock | None = None
_refresh_lock_loop: asyncio.AbstractEventLoop | None = None

_CATEGORY_SPECS: list[tuple[LeaderCategoryKey, str, str, str]] = [
    # key, label, display_stat, upstream_header
    ("points", "Points", "PTS", "PTS"),
    ("rebounds", "Rebounds", "REB", "REB"),
    ("assists", "Assists", "AST", "AST"),
    ("steals", "Steals", "STL", "STL"),
    ("blocks", "Blocks", "BLK", "BLK"),
    ("three_pointers", "3-Pointers", "3PM", "FG3M"),
]

TOP_N = 10


def _rows_as_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return []
    block = sets[0] or {}
    headers = [str(h) for h in (block.get("headers") or [])]
    if not headers:
        return []
    out: list[dict[str, Any]] = []
    for raw in block.get("rowSet") or []:
        if not isinstance(raw, (list, tuple)):
            continue
        out.append({headers[i]: raw[i] for i in range(min(len(headers), len(raw)))})
    return out


def _format_value(raw: Any) -> str | None:
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    return f"{num:.1f}"


def _leader_row(rank: int, player: dict[str, Any], value: str) -> WnbaLeaderRow | None:
    player_id = player.get("PLAYER_ID")
    name = str(player.get("PLAYER_NAME") or "").strip()
    abbrev = str(player.get("TEAM_ABBREVIATION") or "").strip().upper()
    gp_raw = player.get("GP")
    try:
        gp = int(gp_raw)
    except (TypeError, ValueError):
        return None
    if player_id is None or not name or not abbrev:
        return None
    return WnbaLeaderRow(
        rank=rank,
        player_id=str(player_id),
        name=name,
        team_abbrev=abbrev,
        gp=gp,
        value=value,
    )


def normalize_leaguedashplayerstats(
    payload: dict[str, Any], *, season: int
) -> WnbaLeadersResponse:
    players = _rows_as_dicts(payload)
    categories: list[WnbaLeaderCategory] = []
    for key, label, stat, header in _CATEGORY_SPECS:
        scored: list[tuple[float, dict[str, Any], str]] = []
        for player in players:
            formatted = _format_value(player.get(header))
            if formatted is None:
                continue
            scored.append((float(formatted), player, formatted))
        scored.sort(key=lambda item: item[0], reverse=True)
        leaders: list[WnbaLeaderRow] = []
        for _num, player, formatted in scored:
            row = _leader_row(len(leaders) + 1, player, formatted)
            if row is None:
                continue
            leaders.append(row)
            if len(leaders) >= TOP_N:
                break
        categories.append(
            WnbaLeaderCategory(
                key=key,
                label=label,
                stat=stat,
                leaders=leaders,
            )
        )
    return WnbaLeadersResponse(season=season, pace="per_game", categories=categories)


def current_wnba_season_year() -> int:
    return datetime.now(ET).year


def _get_refresh_lock() -> asyncio.Lock:
    global _refresh_lock, _refresh_lock_loop
    loop = asyncio.get_running_loop()
    if _refresh_lock is None or _refresh_lock_loop is not loop:
        _refresh_lock = asyncio.Lock()
        _refresh_lock_loop = loop
    return _refresh_lock


async def fetch_leaguedashplayerstats(season: int) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.wnba.com/",
        "Accept": "application/json",
    }
    params = {
        "LastNGames": "0",
        "LeagueID": "10",
        "MeasureType": "Base",
        "Month": "0",
        "OpponentTeamID": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": str(season),
        "SeasonType": "Regular Season",
        "TeamID": "0",
    }
    async with httpx.AsyncClient(
        timeout=STATS_TIMEOUT_SECONDS, headers=headers
    ) as client:
        res = await client.get(STATS_URL, params=params)
        res.raise_for_status()
        return res.json()


def _fresh_cached() -> WnbaLeadersResponse | None:
    cached = _cache.get("response")
    if cached is None:
        return None
    if _cache.get("season") != current_wnba_season_year():
        return None
    if time.time() >= float(_cache.get("expires_at") or 0):
        return None
    return cached


async def get_wnba_leaders() -> WnbaLeadersResponse:
    fresh = _fresh_cached()
    if fresh is not None:
        return fresh

    lock = _get_refresh_lock()
    async with lock:
        fresh = _fresh_cached()
        if fresh is not None:
            return fresh
        season = current_wnba_season_year()
        try:
            payload = await fetch_leaguedashplayerstats(season)
            response = normalize_leaguedashplayerstats(payload, season=season)
        except Exception:
            stale = _cache.get("response")
            if stale is not None and _cache.get("season") == season:
                logger.warning("WNBA leaders refresh failed; serving stale cache")
                return stale
            raise
        _cache["response"] = response
        _cache["expires_at"] = time.time() + CACHE_TTL_SECONDS
        _cache["season"] = season
        return response
