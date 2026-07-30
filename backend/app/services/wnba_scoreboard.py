from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import overload
from zoneinfo import ZoneInfo

import httpx

from app.schemas.wnba_scoreboard import (
    GameStatus,
    WnbaGame,
    WnbaScoreboardResponse,
    WnbaTeam,
)

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
STATS_URL = "https://stats.wnba.com/stats/scoreboardv3"

_cache: dict = {}  # keys: response, expires_at, date

_STATUS_MAP = {1: "scheduled", 2: "live", 3: "final"}


def _espn_status(status_block: dict) -> tuple[GameStatus, str]:
    typ = status_block.get("type") or {}
    name = str(typ.get("name") or "")
    state = str(typ.get("state") or "")
    short = str(typ.get("shortDetail") or typ.get("detail") or "")
    period = status_block.get("period")
    clock = str(status_block.get("displayClock") or "").strip()

    if typ.get("completed") or name == "STATUS_FINAL" or state == "post":
        return "final", "Final"
    if "HALFTIME" in name.upper() or short.lower() == "halftime":
        return "halftime", "Halftime"
    if state == "in" or name == "STATUS_IN_PROGRESS":
        # Prefer compact Qn clock when period + clock present
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", short or "Live"
    # scheduled
    label = short or "Scheduled"
    return "scheduled", label


def normalize_espn_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    games: list[WnbaGame] = []
    for event in payload.get("events") or []:
        comps = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in (comps.get("competitors") or [])}
        away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
        status, label = _espn_status(event.get("status") or {})
        start = str(event.get("date") or "")

        def team(c: dict) -> WnbaTeam:
            t = c.get("team") or {}
            raw = c.get("score")
            score = int(raw) if raw not in (None, "") else None
            return WnbaTeam(
                abbrev=str(t.get("abbreviation") or ""),
                name=str(t.get("displayName") or ""),
                score=score if status != "scheduled" else None,
            )

        games.append(
            WnbaGame(
                id=f"espn-{event.get('id')}",
                status=status,
                status_label=label,
                away=team(away_c),
                home=team(home_c),
                start_time_et=start,
            )
        )
    return games


def _parse_iso_clock(game_clock: str | None) -> str | None:
    if not game_clock:
        return None
    # PT7M10.00S → 7:10
    m = re.match(r"PT(?:(\d+)M)?(?:(\d+)(?:\.\d+)?S)?", game_clock)
    if not m:
        return None
    mins = int(m.group(1) or 0)
    secs = int(float(m.group(2) or 0))
    return f"{mins}:{secs:02d}"


def _stats_status(game: dict) -> tuple[GameStatus, str]:
    code = int(game.get("gameStatus") or 1)
    text = str(game.get("gameStatusText") or "").strip()
    if code == 3 or text.lower() == "final":
        return "final", "Final"
    if "half" in text.lower():
        return "halftime", "Halftime"
    if code == 2:
        period = game.get("period")
        clock = _parse_iso_clock(game.get("gameClock"))
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", text or "Live"
    return "scheduled", text or "Scheduled"


def normalize_stats_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    board = payload.get("scoreboard") or payload
    games: list[WnbaGame] = []
    for g in board.get("games") or []:
        status, label = _stats_status(g)
        away, home = g.get("awayTeam") or {}, g.get("homeTeam") or {}

        def team(t: dict) -> WnbaTeam:
            city = str(t.get("teamCity") or "").strip()
            name = str(t.get("teamName") or "").strip()
            full = f"{city} {name}".strip()
            raw = t.get("score")
            score = int(raw) if raw is not None and status != "scheduled" else None
            return WnbaTeam(
                abbrev=str(t.get("teamTricode") or ""),
                name=full,
                score=score,
            )

        games.append(
            WnbaGame(
                id=str(g.get("gameId")),
                status=status,
                status_label=label,
                away=team(away),
                home=team(home),
                start_time_et=str(g.get("gameTimeUTC") or ""),
            )
        )
    return games


def _match_key(game: WnbaGame) -> tuple[str, str]:
    return (game.away.abbrev.upper(), game.home.abbrev.upper())


_STATUS_RANK: dict[GameStatus, int] = {
    "scheduled": 0,
    "live": 1,
    "halftime": 1,
    "final": 2,
}


@overload
def prefer_complete(a: str, b: str) -> str: ...


@overload
def prefer_complete(a: int | None, b: int | None) -> int | None: ...


def prefer_complete(a: str | int | None, b: str | int | None) -> str | int | None:
    """Return the more complete field value; ties prefer ``b`` (stats source)."""
    if isinstance(a, str) or isinstance(b, str):
        left = str(a or "")
        right = str(b or "")
        if not left:
            return right
        if not right:
            return left
        return right if len(right) >= len(left) else left
    if a is None:
        return b
    if b is None:
        return a
    return b


def _prefer_status_and_label(a: WnbaGame, b: WnbaGame) -> tuple[GameStatus, str]:
    rank_a = _STATUS_RANK[a.status]
    rank_b = _STATUS_RANK[b.status]
    if rank_b > rank_a:
        return b.status, b.status_label
    if rank_a > rank_b:
        return a.status, a.status_label
    if a.status != b.status:
        return b.status, b.status_label
    label = prefer_complete(a.status_label, b.status_label)
    return a.status, str(label)


def merge_games(espn: list[WnbaGame], stats: list[WnbaGame]) -> list[WnbaGame]:
    by_key: dict[tuple[str, str], WnbaGame] = {}
    for g in espn:
        by_key[_match_key(g)] = g
    for g in stats:
        key = _match_key(g)
        if key not in by_key:
            by_key[key] = g
            continue
        a = by_key[key]
        game_id = g.id if not g.id.startswith("espn-") else a.id
        if a.id.startswith("espn-") and not g.id.startswith("espn-"):
            game_id = g.id
        status, status_label = _prefer_status_and_label(a, g)
        by_key[key] = WnbaGame(
            id=game_id,
            status=status,
            status_label=status_label,
            away=WnbaTeam(
                abbrev=str(prefer_complete(a.away.abbrev, g.away.abbrev)),
                name=str(prefer_complete(a.away.name, g.away.name)),
                score=prefer_complete(a.away.score, g.away.score),
            ),
            home=WnbaTeam(
                abbrev=str(prefer_complete(a.home.abbrev, g.home.abbrev)),
                name=str(prefer_complete(a.home.name, g.home.name)),
                score=prefer_complete(a.home.score, g.home.score),
            ),
            start_time_et=str(prefer_complete(a.start_time_et, g.start_time_et)),
        )
    return sorted(by_key.values(), key=lambda g: g.start_time_et or g.id)


def cache_ttl_seconds(games: list[WnbaGame]) -> int:
    if any(g.status in ("live", "halftime") for g in games):
        return 30
    return 60


def today_et_date() -> str:
    return datetime.now(ET).date().isoformat()


async def fetch_espn_scoreboard(date_et: str) -> dict:
    dates = date_et.replace("-", "")
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(ESPN_URL, params={"dates": dates})
        r.raise_for_status()
        return r.json()


async def fetch_stats_scoreboard(date_et: str) -> dict:
    # MM/DD/YYYY for stats.wnba.com
    y, m, d = date_et.split("-")
    game_date = f"{m}/{d}/{y}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.wnba.com/",
        "Origin": "https://www.wnba.com",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
        r = await client.get(
            STATS_URL,
            params={"GameDate": game_date, "LeagueID": "10"},
        )
        r.raise_for_status()
        return r.json()


def _cache_valid_for_today() -> WnbaScoreboardResponse | None:
    cached = _cache.get("response")
    if cached is None:
        return None
    if _cache.get("date") != today_et_date():
        return None
    return cached


async def get_today_scoreboard() -> WnbaScoreboardResponse:
    now = time.time()
    date_et = today_et_date()
    cached = _cache_valid_for_today()
    if cached is not None and now < float(_cache.get("expires_at") or 0):
        return cached
    espn_payload, stats_payload = None, None
    results = await asyncio.gather(
        fetch_espn_scoreboard(date_et),
        fetch_stats_scoreboard(date_et),
        return_exceptions=True,
    )
    if isinstance(results[0], Exception):
        logger.warning("ESPN scoreboard fetch failed: %s", results[0])
    else:
        espn_payload = results[0]
    if isinstance(results[1], Exception):
        logger.warning("stats.wnba scoreboard fetch failed: %s", results[1])
    else:
        stats_payload = results[1]

    if espn_payload is None and stats_payload is None:
        if cached is not None:
            return cached  # stale-while-error (same ET date only)
        raise RuntimeError("Both WNBA scoreboard upstreams failed")

    espn_games = (
        normalize_espn_scoreboard(espn_payload, date_et=date_et)
        if espn_payload is not None
        else []
    )
    stats_games = (
        normalize_stats_scoreboard(stats_payload, date_et=date_et)
        if stats_payload is not None
        else []
    )
    games = merge_games(espn_games, stats_games)
    response = WnbaScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(tz=ET).isoformat(),
    )
    _cache["response"] = response
    _cache["expires_at"] = now + cache_ttl_seconds(games)
    _cache["date"] = date_et
    return response
