from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from app.schemas.mlb_scoreboard import (
    GameStatus,
    MlbGame,
    MlbScoreboardResponse,
    MlbTeam,
)

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

TEAM_LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
TIMEOUT_SECONDS = 12.0

# Slate flips at 3:00 AM ET so late West Coast finishes stay on yesterday's board.
SLATE_ROLLOVER_HOUR_ET = 3

_NON_RESULT_KEYWORDS = ("postponed", "cancelled", "canceled", "suspended")

DATED_CACHE_TTL_SECONDS = 300

_cache: dict = {}  # keys: response, expires_at, date
_date_cache: dict[str, dict] = {}
_refresh_lock: asyncio.Lock | None = None
_refresh_lock_loop: asyncio.AbstractEventLoop | None = None


def _parse_start(start: str) -> datetime | None:
    raw = str(start or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed


def format_tip_label(start: str) -> str | None:
    """Render a scheduled first pitch as an ET wall-clock label, e.g. ``7:00 PM ET``."""
    parsed = _parse_start(start)
    if parsed is None:
        return None
    local = parsed.astimezone(ET)
    return f"{local.strftime('%I:%M %p').lstrip('0')} ET"


def _team_logo_url(team_id: int | None) -> str | None:
    if team_id is None:
        return None
    return TEAM_LOGO.format(id=team_id)


def _mlb_team_record(side: dict) -> str | None:
    record = side.get("leagueRecord") or {}
    if not isinstance(record, dict):
        return None
    wins = record.get("wins")
    losses = record.get("losses")
    if wins is None or losses is None:
        return None
    return f"{wins}-{losses}"


def _mlb_status(
    status: dict,
    linescore: dict | None,
    game_date: str,
) -> tuple[GameStatus, str]:
    detailed = str(status.get("detailedState") or "").strip()
    detailed_lower = detailed.lower()

    for keyword in _NON_RESULT_KEYWORDS:
        if keyword in detailed_lower:
            return "scheduled", detailed

    abstract = str(status.get("abstractGameState") or "").strip()
    if abstract == "Final":
        return "final", "Final"
    if abstract == "Live":
        if isinstance(linescore, dict):
            inning_state = str(linescore.get("inningState") or "").strip()
            inning_ordinal = str(linescore.get("currentInningOrdinal") or "").strip()
            if inning_state and inning_ordinal:
                return "live", f"{inning_state} {inning_ordinal}"
        return "live", "Live"

    tip = format_tip_label(game_date)
    return "scheduled", tip or detailed or "Scheduled"


def _mlb_venue(game: dict) -> tuple[str | None, str | None]:
    venue = game.get("venue") or {}
    if not isinstance(venue, dict):
        return None, None
    name = venue.get("name")
    address = venue.get("address") or {}
    city = address.get("city") if isinstance(address, dict) else None
    return (str(name) if name else None, str(city) if city else None)


def normalize_mlb_schedule(payload: dict, *, date_et: str) -> list[MlbGame]:
    games: list[MlbGame] = []
    for day in payload.get("dates") or []:
        if not isinstance(day, dict):
            continue
        for game in day.get("games") or []:
            if not isinstance(game, dict):
                continue
            game_pk = game.get("gamePk")
            if game_pk is None:
                continue
            pk_str = str(game_pk)
            game_date = str(game.get("gameDate") or "")
            status, label = _mlb_status(
                game.get("status") or {},
                game.get("linescore"),
                game_date,
            )
            venue, venue_city = _mlb_venue(game)
            teams = game.get("teams") or {}
            away_side = teams.get("away") or {}
            home_side = teams.get("home") or {}

            def team(side: dict) -> MlbTeam:
                team_info = side.get("team") or {}
                team_id = team_info.get("id")
                raw_score = side.get("score")
                score = (
                    int(raw_score)
                    if raw_score is not None and status != "scheduled"
                    else None
                )
                return MlbTeam(
                    abbrev=str(team_info.get("abbreviation") or ""),
                    name=str(team_info.get("name") or ""),
                    score=score,
                    record=_mlb_team_record(side),
                    logo_url=_team_logo_url(team_id),
                )

            games.append(
                MlbGame(
                    id=f"mlb-{pk_str}",
                    mlb_game_pk=pk_str,
                    status=status,
                    status_label=label,
                    away=team(away_side),
                    home=team(home_side),
                    start_time_et=game_date,
                    venue=venue,
                    venue_city=venue_city,
                )
            )
    return sorted(games, key=lambda g: (g.start_time_et or "", g.id))


def slate_et_date(*, now: datetime | None = None) -> str:
    """ET calendar date used for ``/scoreboard/today`` (lags until 3:00 AM ET)."""
    current = now.astimezone(ET) if now is not None else datetime.now(ET)
    if current.hour < SLATE_ROLLOVER_HOUR_ET:
        return (current.date() - timedelta(days=1)).isoformat()
    return current.date().isoformat()


def cache_ttl_seconds(games: list[MlbGame]) -> int:
    if any(g.status in ("live", "halftime") for g in games):
        return 30
    return 60


async def fetch_mlb_schedule(date_et: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        r = await client.get(
            SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": date_et,
                "hydrate": "team,linescore",
            },
        )
        r.raise_for_status()
        return r.json()


def _cache_valid_for_today() -> MlbScoreboardResponse | None:
    cached = _cache.get("response")
    if cached is None:
        return None
    if _cache.get("date") != slate_et_date():
        return None
    return cached


def _fresh_cached_response() -> MlbScoreboardResponse | None:
    cached = _cache_valid_for_today()
    if cached is None:
        return None
    if time.time() >= float(_cache.get("expires_at") or 0):
        return None
    return cached


def _refresh_lock_for_loop() -> asyncio.Lock:
    """Serializes upstream refreshes so concurrent cache misses share one fetch."""
    global _refresh_lock, _refresh_lock_loop
    loop = asyncio.get_running_loop()
    if _refresh_lock is None or _refresh_lock_loop is not loop:
        _refresh_lock = asyncio.Lock()
        _refresh_lock_loop = loop
    return _refresh_lock


async def get_today_scoreboard() -> MlbScoreboardResponse:
    fresh = _fresh_cached_response()
    if fresh is not None:
        return fresh
    async with _refresh_lock_for_loop():
        fresh = _fresh_cached_response()
        if fresh is not None:
            return fresh
        return await _refresh_today_scoreboard()


async def _refresh_today_scoreboard() -> MlbScoreboardResponse:
    now = time.time()
    date_et = slate_et_date()
    cached = _cache_valid_for_today()

    try:
        payload = await fetch_mlb_schedule(date_et)
        games = normalize_mlb_schedule(payload, date_et=date_et)
    except Exception as exc:
        if cached is not None:
            logger.warning(
                "MLB scoreboard refresh failed; serving stale cache (%s): %s",
                date_et,
                exc,
            )
            return cached
        raise RuntimeError(f"No usable MLB scoreboard source for {date_et}") from exc

    response = MlbScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    _cache["response"] = response
    _cache["expires_at"] = now + cache_ttl_seconds(games)
    _cache["date"] = date_et
    return response


async def get_scoreboard_for_date(date_et: str) -> MlbScoreboardResponse:
    """Scoreboard for one ET calendar day (no overnight live carryover)."""
    cached = _date_cache.get(date_et)
    if cached is not None and time.time() < float(cached.get("expires_at") or 0):
        return cached["response"]

    try:
        payload = await fetch_mlb_schedule(date_et)
        games = normalize_mlb_schedule(payload, date_et=date_et)
    except Exception as exc:
        if cached is not None:
            logger.warning(
                "MLB dated scoreboard refresh failed; serving stale (%s): %s",
                date_et,
                exc,
            )
            return cached["response"]
        raise RuntimeError(f"No usable MLB scoreboard source for {date_et}") from exc

    response = MlbScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    _date_cache[date_et] = {
        "response": response,
        "expires_at": time.time() + DATED_CACHE_TTL_SECONDS,
    }
    return response
