from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.schemas.wnba_standings import (
    ConferenceKey,
    WnbaStandingsConference,
    WnbaStandingsResponse,
    WnbaStandingsRow,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
ESPN_URL = "https://site.api.espn.com/apis/v2/sports/basketball/wnba/standings"
ESPN_TIMEOUT_SECONDS = 10.0
CACHE_TTL_SECONDS = 10 * 60

_cache: dict = {}  # response, expires_at, season
_refresh_lock: asyncio.Lock | None = None
_refresh_lock_loop: asyncio.AbstractEventLoop | None = None

_CONF_BY_ABBREV: dict[str, tuple[ConferenceKey, str]] = {
    "E": ("east", "Eastern Conference"),
    "W": ("west", "Western Conference"),
}


def _stat_map(stats: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in stats or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if name:
            out[name] = raw
    return out


def _display(stats: dict[str, dict[str, Any]], *names: str) -> str | None:
    for name in names:
        block = stats.get(name)
        if not block:
            continue
        value = block.get("displayValue")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _int_stat(stats: dict[str, dict[str, Any]], name: str) -> int | None:
    block = stats.get(name)
    if not block:
        return None
    raw = block.get("value", block.get("displayValue"))
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


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


def _row_from_entry(entry: dict[str, Any]) -> WnbaStandingsRow | None:
    team = entry.get("team") or {}
    if not isinstance(team, dict):
        return None
    team_id = str(team.get("id") or "").strip()
    abbrev = str(team.get("abbreviation") or "").strip().upper()
    name = str(team.get("displayName") or "").strip()
    if not team_id or not abbrev or not name:
        return None
    stats = _stat_map(entry.get("stats") or [])
    rank = _int_stat(stats, "playoffSeed")
    wins = _int_stat(stats, "wins")
    losses = _int_stat(stats, "losses")
    wl = _display(stats, "overall")
    pct = _display(stats, "winPercent")
    gb = _display(stats, "gamesBehind")
    home = _display(stats, "Home")
    away = _display(stats, "Road")
    l10 = _display(stats, "Last Ten Games")
    diff = _display(stats, "pointDifferential", "differential")
    streak = _display(stats, "streak")
    if (
        rank is None
        or wins is None
        or losses is None
        or not wl
        or not pct
        or not gb
        or not home
        or not away
        or not l10
        or not diff
        or not streak
    ):
        return None
    return WnbaStandingsRow(
        rank=rank,
        team_id=team_id,
        abbrev=abbrev,
        name=name,
        logo_url=_logo_url(team),
        wins=wins,
        losses=losses,
        wl=wl,
        pct=pct,
        gb=gb,
        home=home,
        away=away,
        l10=l10,
        diff=diff,
        streak=streak,
    )


def _season_year(payload: dict[str, Any]) -> int | None:
    season = payload.get("season")
    if isinstance(season, dict):
        year = season.get("year")
        try:
            return int(year)
        except (TypeError, ValueError):
            return None
    if isinstance(season, int):
        return season
    return None


def normalize_espn_standings(payload: dict[str, Any]) -> WnbaStandingsResponse:
    season = _season_year(payload)
    if season is None:
        raise ValueError("ESPN standings payload missing season.year")

    by_key: dict[ConferenceKey, WnbaStandingsConference] = {}
    for child in payload.get("children") or []:
        if not isinstance(child, dict):
            continue
        abbrev = str(child.get("abbreviation") or "").strip().upper()
        mapped = _CONF_BY_ABBREV.get(abbrev)
        if mapped is None:
            name = str(child.get("name") or "").lower()
            if "eastern" in name:
                mapped = ("east", "Eastern Conference")
            elif "western" in name:
                mapped = ("west", "Western Conference")
            else:
                continue
        key, default_label = mapped
        label = str(child.get("name") or "").strip() or default_label
        standings = child.get("standings") or {}
        entries = standings.get("entries") if isinstance(standings, dict) else []
        teams: list[WnbaStandingsRow] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            row = _row_from_entry(entry)
            if row is not None:
                teams.append(row)
        by_key[key] = WnbaStandingsConference(key=key, label=label, teams=teams)

    conferences: list[WnbaStandingsConference] = []
    for key in ("east", "west"):
        if key in by_key:
            conferences.append(by_key[key])
    return WnbaStandingsResponse(season=season, conferences=conferences)


def current_wnba_season_year() -> int:
    return datetime.now(ET).year


def _get_refresh_lock() -> asyncio.Lock:
    global _refresh_lock, _refresh_lock_loop
    loop = asyncio.get_running_loop()
    if _refresh_lock is None or _refresh_lock_loop is not loop:
        _refresh_lock = asyncio.Lock()
        _refresh_lock_loop = loop
    return _refresh_lock


async def fetch_espn_standings() -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(
        timeout=ESPN_TIMEOUT_SECONDS, headers=headers
    ) as client:
        res = await client.get(ESPN_URL)
        res.raise_for_status()
        return res.json()


def _fresh_cached() -> WnbaStandingsResponse | None:
    cached = _cache.get("response")
    if cached is None:
        return None
    if _cache.get("season") != current_wnba_season_year():
        return None
    if time.time() >= float(_cache.get("expires_at") or 0):
        return None
    return cached


async def get_wnba_standings() -> WnbaStandingsResponse:
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
            payload = await fetch_espn_standings()
            response = normalize_espn_standings(payload)
        except Exception:
            stale = _cache.get("response")
            if stale is not None and _cache.get("season") == season:
                logger.warning("WNBA standings refresh failed; serving stale cache")
                return stale
            raise
        _cache["response"] = response
        _cache["expires_at"] = time.time() + CACHE_TTL_SECONDS
        _cache["season"] = response.season
        return response
