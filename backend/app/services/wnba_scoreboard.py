from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import date, datetime, timedelta
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

# ESPN is the primary source, so it gets the longer budget. stats.wnba.com is
# supplementary and occasionally hangs, so it is cut off early rather than
# holding the whole response back.
ESPN_TIMEOUT_SECONDS = 8.0
STATS_TIMEOUT_SECONDS = 4.0

_cache: dict = {}  # keys: response, expires_at, date

DATED_CACHE_TTL_SECONDS = 300
_date_cache: dict[str, dict] = {}

_refresh_lock: asyncio.Lock | None = None
_refresh_lock_loop: asyncio.AbstractEventLoop | None = None

# ESPN uses shorter tricodes than stats.wnba.com for several teams. Canonical
# form is the stats.wnba.com spelling so both sources key identically in merges.
_ABBREV_ALIASES = {
    "GS": "GSV",
    "LA": "LAS",
    "LV": "LVA",
    "NY": "NYL",
    "PHX": "PHO",
    "POR": "PDX",  # Sharp/ESPN POR vs stats.wnba.com PDX (Portland Fire)
    "WSH": "WAS",
}

# States that look "post" to ESPN's generic state field but are not results.
_ESPN_NON_RESULT_LABELS = {
    "STATUS_POSTPONED": "Postponed",
    "STATUS_CANCELED": "Canceled",
    "STATUS_CANCELLED": "Canceled",
    "STATUS_SUSPENDED": "Suspended",
    "STATUS_DELAYED": "Delayed",
}

# Tip times this close on the same ET day are treated as the same game when
# tricodes alone fail to match.
TIP_MATCH_WINDOW_SECONDS = 15 * 60


def canonical_abbrev(abbrev: str) -> str:
    """Map a team tricode to its canonical (stats.wnba.com) spelling."""
    upper = str(abbrev or "").strip().upper()
    return _ABBREV_ALIASES.get(upper, upper)


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
    """Render a scheduled tip time as an ET wall-clock label, e.g. ``7:00 PM ET``."""
    parsed = _parse_start(start)
    if parsed is None:
        return None
    local = parsed.astimezone(ET)
    return f"{local.strftime('%I:%M %p').lstrip('0')} ET"


def _espn_status(status_block: dict, start: str) -> tuple[GameStatus, str]:
    typ = status_block.get("type") or {}
    name = str(typ.get("name") or "").upper()
    state = str(typ.get("state") or "")
    short = str(typ.get("shortDetail") or typ.get("detail") or "")
    period = status_block.get("period")
    clock = str(status_block.get("displayClock") or "").strip()

    # Checked before the generic post/completed branch: a postponed game has no
    # result and must never be labelled "Final".
    if name in _ESPN_NON_RESULT_LABELS:
        return "scheduled", _ESPN_NON_RESULT_LABELS[name]
    if typ.get("completed") or name == "STATUS_FINAL" or state == "post":
        return "final", "Final"
    if "HALFTIME" in name or short.lower() == "halftime":
        return "halftime", "Halftime"
    if state == "in" or name == "STATUS_IN_PROGRESS":
        # Prefer compact Qn clock when period + clock present
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", short or "Live"
    return "scheduled", format_tip_label(start) or short or "Scheduled"


def _espn_team_record(competitor: dict) -> str | None:
    for rec in competitor.get("records") or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("type") == "total" or str(rec.get("name") or "").lower() in {
            "total",
            "overall",
            "ytd",
        }:
            summary = rec.get("summary")
            if summary:
                return str(summary)
    for rec in competitor.get("records") or []:
        if isinstance(rec, dict) and rec.get("summary"):
            return str(rec["summary"])
    return None


def _espn_venue(comps: dict) -> tuple[str | None, str | None]:
    venue = comps.get("venue") or {}
    if not isinstance(venue, dict):
        return None, None
    name = venue.get("fullName") or venue.get("name")
    city = (venue.get("address") or {}).get("city") if isinstance(
        venue.get("address"), dict
    ) else None
    return (str(name) if name else None, str(city) if city else None)


def _team_logo_from_espn(team: dict) -> str | None:
    logo = str(team.get("logo") or "").strip()
    return logo or None


def normalize_espn_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    games: list[WnbaGame] = []
    for event in payload.get("events") or []:
        comps = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in (comps.get("competitors") or [])}
        away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
        start = str(event.get("date") or "")
        status, label = _espn_status(event.get("status") or {}, start)
        venue, venue_city = _espn_venue(comps)

        def team(c: dict) -> WnbaTeam:
            t = c.get("team") or {}
            raw = c.get("score")
            score = int(raw) if raw not in (None, "") else None
            return WnbaTeam(
                abbrev=canonical_abbrev(str(t.get("abbreviation") or "")),
                name=str(t.get("displayName") or ""),
                score=score if status != "scheduled" else None,
                record=_espn_team_record(c),
                logo_url=_team_logo_from_espn(t),
            )

        raw_id = str(event.get("id") or "").strip()
        games.append(
            WnbaGame(
                id=f"espn-{raw_id}" if raw_id else "espn-unknown",
                espn_event_id=raw_id or None,
                status=status,
                status_label=label,
                away=team(away_c),
                home=team(home_c),
                start_time_et=start,
                venue=venue,
                venue_city=venue_city,
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
    tip = format_tip_label(str(game.get("gameTimeUTC") or ""))
    return "scheduled", tip or text or "Scheduled"


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
                abbrev=canonical_abbrev(str(t.get("teamTricode") or "")),
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


MatchKey = tuple[str, ...]


def _match_key(game: WnbaGame) -> MatchKey:
    return (canonical_abbrev(game.away.abbrev), canonical_abbrev(game.home.abbrev))


def _canonical_teams(game: WnbaGame) -> set[str]:
    return {canonical_abbrev(game.away.abbrev), canonical_abbrev(game.home.abbrev)}


def _tip_time_match(
    game: WnbaGame, candidates: dict[MatchKey, WnbaGame]
) -> MatchKey | None:
    """Find an unclaimed ESPN game tipping within the window on the same ET day.

    Fallback for when tricodes disagree in a way the alias map does not cover.
    A candidate sharing a team tricode always wins; a window match with no
    shared tricode is only trusted when it is unambiguous.
    """
    target = _parse_start(game.start_time_et)
    if target is None:
        return None
    target_day = target.astimezone(ET).date()
    teams = _canonical_teams(game)

    ranked: list[tuple[int, float, MatchKey]] = []
    for key, candidate in candidates.items():
        other = _parse_start(candidate.start_time_et)
        if other is None or other.astimezone(ET).date() != target_day:
            continue
        delta = abs((other - target).total_seconds())
        if delta > TIP_MATCH_WINDOW_SECONDS:
            continue
        shared = len(teams & _canonical_teams(candidate))
        ranked.append((-shared, delta, key))

    if not ranked:
        return None
    ranked.sort()
    best_shared, _, best_key = ranked[0]
    if best_shared == 0 and len(ranked) > 1:
        return None
    return best_key


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
    by_key: dict[MatchKey, WnbaGame] = {}
    for g in espn:
        by_key[_match_key(g)] = g
    # ESPN games not yet claimed by a stats game, so each pairs at most once.
    unclaimed = dict(by_key)

    for g in stats:
        key = _match_key(g)
        match = key if key in unclaimed else _tip_time_match(g, unclaimed)
        if match is None:
            if espn:
                logger.warning(
                    "No ESPN counterpart for stats game %s (%s@%s, tip %s)",
                    g.id,
                    g.away.abbrev,
                    g.home.abbrev,
                    g.start_time_et,
                )
            # Widen the key on collision so a second stats game with the same
            # tricode pair cannot silently evict an already-merged game.
            by_key[key if key not in by_key else (*key, g.id)] = g
            continue
        a = unclaimed.pop(match)
        game_id = g.id if not g.id.startswith("espn-") else a.id
        if a.id.startswith("espn-") and not g.id.startswith("espn-"):
            game_id = g.id
        status, status_label = _prefer_status_and_label(a, g)
        by_key[match] = WnbaGame(
            id=game_id,
            espn_event_id=a.espn_event_id or g.espn_event_id,
            status=status,
            status_label=status_label,
            away=WnbaTeam(
                abbrev=str(prefer_complete(a.away.abbrev, g.away.abbrev)),
                name=str(prefer_complete(a.away.name, g.away.name)),
                score=prefer_complete(a.away.score, g.away.score),
                record=prefer_complete(a.away.record, g.away.record) or None,
                logo_url=prefer_complete(a.away.logo_url, g.away.logo_url) or None,
            ),
            home=WnbaTeam(
                abbrev=str(prefer_complete(a.home.abbrev, g.home.abbrev)),
                name=str(prefer_complete(a.home.name, g.home.name)),
                score=prefer_complete(a.home.score, g.home.score),
                record=prefer_complete(a.home.record, g.home.record) or None,
                logo_url=prefer_complete(a.home.logo_url, g.home.logo_url) or None,
            ),
            start_time_et=str(prefer_complete(a.start_time_et, g.start_time_et)),
            venue=prefer_complete(a.venue, g.venue) or None,
            venue_city=prefer_complete(a.venue_city, g.venue_city) or None,
        )
    return sorted(by_key.values(), key=lambda g: g.start_time_et or g.id)


def cache_ttl_seconds(games: list[WnbaGame]) -> int:
    if any(g.status in ("live", "halftime") for g in games):
        return 30
    return 60


def today_et_date() -> str:
    return datetime.now(ET).date().isoformat()


# The slate should flip at midnight Pacific, not midnight Eastern. That keeps
# late-evening West Coast users on the same slate until their local day ends.
SLATE_ROLLOVER_HOUR_ET = 3


def slate_et_date(*, now: datetime | None = None) -> str:
    """ET calendar date used for ``/scoreboard/today`` (lags until 3:00 AM ET)."""
    current = now.astimezone(ET) if now is not None else datetime.now(ET)
    if current.hour < SLATE_ROLLOVER_HOUR_ET:
        return (current.date() - timedelta(days=1)).isoformat()
    return current.date().isoformat()


def previous_et_date(date_et: str) -> str:
    return (date.fromisoformat(date_et) - timedelta(days=1)).isoformat()


def combine_with_overnight_carryover(
    today_games: list[WnbaGame],
    yesterday_games: list[WnbaGame],
) -> list[WnbaGame]:
    """Keep yesterday's still-in-progress games after the ET date rolls over.

    Late West Coast tips often finish after midnight ET. Without carryover they
    vanish from ``/scoreboard/today`` the moment the calendar flips.
    """
    today_keys = {_match_key(g) for g in today_games}
    today_espn_ids = {g.espn_event_id for g in today_games if g.espn_event_id}
    carryover: list[WnbaGame] = []
    for game in yesterday_games:
        if game.status not in ("live", "halftime"):
            continue
        if _match_key(game) in today_keys:
            continue
        if game.espn_event_id and game.espn_event_id in today_espn_ids:
            continue
        carryover.append(game)
    return sorted(
        carryover + today_games, key=lambda g: g.start_time_et or g.id
    )


async def fetch_espn_scoreboard(date_et: str) -> dict:
    dates = date_et.replace("-", "")
    async with httpx.AsyncClient(timeout=ESPN_TIMEOUT_SECONDS) as client:
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
    async with httpx.AsyncClient(
        timeout=STATS_TIMEOUT_SECONDS, headers=headers
    ) as client:
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
    if _cache.get("date") != slate_et_date():
        return None
    return cached


def _fresh_cached_response() -> WnbaScoreboardResponse | None:
    cached = _cache_valid_for_today()
    if cached is None:
        return None
    if time.time() >= float(_cache.get("expires_at") or 0):
        return None
    return cached


def _refresh_lock_for_loop() -> asyncio.Lock:
    """Serializes upstream refreshes so concurrent cache misses share one fetch.

    Created lazily because an ``asyncio.Lock`` binds to the loop that first
    waits on it; re-created if called from a different loop (test clients spin
    up a fresh loop per request).
    """
    global _refresh_lock, _refresh_lock_loop
    loop = asyncio.get_running_loop()
    if _refresh_lock is None or _refresh_lock_loop is not loop:
        _refresh_lock = asyncio.Lock()
        _refresh_lock_loop = loop
    return _refresh_lock


async def get_today_scoreboard() -> WnbaScoreboardResponse:
    fresh = _fresh_cached_response()
    if fresh is not None:
        return fresh
    # Single-flight: concurrent misses queue here and reuse the winner's refresh
    # instead of each firing their own pair of upstream requests.
    async with _refresh_lock_for_loop():
        fresh = _fresh_cached_response()
        if fresh is not None:
            return fresh
        return await _refresh_today_scoreboard()


async def _merged_games_for_date(date_et: str) -> tuple[list[WnbaGame], bool]:
    """Fetch + normalize + merge one ET date. Returns (games, any_source_usable)."""
    espn_payload, stats_payload = None, None
    results = await asyncio.gather(
        fetch_espn_scoreboard(date_et),
        fetch_stats_scoreboard(date_et),
        return_exceptions=True,
    )
    if isinstance(results[0], Exception):
        logger.warning("ESPN scoreboard fetch failed (%s): %s", date_et, results[0])
    else:
        espn_payload = results[0]
    if isinstance(results[1], Exception):
        logger.warning(
            "stats.wnba scoreboard fetch failed (%s): %s", date_et, results[1]
        )
    else:
        stats_payload = results[1]

    espn_games: list[WnbaGame] = []
    stats_games: list[WnbaGame] = []
    espn_usable = False
    stats_usable = False

    if espn_payload is not None:
        try:
            espn_games = normalize_espn_scoreboard(espn_payload, date_et=date_et)
            espn_usable = True
        except Exception as exc:
            logger.warning(
                "ESPN scoreboard payload unusable (%s): %s", date_et, exc
            )
    if stats_payload is not None:
        try:
            stats_games = normalize_stats_scoreboard(stats_payload, date_et=date_et)
            stats_usable = True
        except Exception as exc:
            logger.warning(
                "stats.wnba scoreboard payload unusable (%s): %s", date_et, exc
            )

    if not espn_usable and not stats_usable:
        return [], False
    return merge_games(espn_games, stats_games), True


async def get_scoreboard_for_date(date_et: str) -> WnbaScoreboardResponse:
    """Scoreboard for one ET calendar day (no overnight live carryover)."""
    cached = _date_cache.get(date_et)
    if cached is not None and time.time() < float(cached.get("expires_at") or 0):
        return cached["response"]

    games, usable = await _merged_games_for_date(date_et)
    if not usable:
        if cached is not None:
            return cached["response"]
        raise RuntimeError(f"No usable WNBA scoreboard source for {date_et}")

    response = WnbaScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(tz=ET).isoformat(),
    )
    _date_cache[date_et] = {
        "response": response,
        "expires_at": time.time() + DATED_CACHE_TTL_SECONDS,
    }
    return response


async def _refresh_today_scoreboard() -> WnbaScoreboardResponse:
    now = time.time()
    date_et = slate_et_date()
    yesterday = previous_et_date(date_et)
    cached = _cache_valid_for_today()

    today_result, yesterday_result = await asyncio.gather(
        _merged_games_for_date(date_et),
        _merged_games_for_date(yesterday),
    )
    today_games, today_usable = today_result
    yesterday_games, yesterday_usable = yesterday_result

    if not today_usable:
        if cached is not None:
            return cached  # stale-while-error (same ET date only)
        raise RuntimeError("No usable WNBA scoreboard source")

    games = combine_with_overnight_carryover(
        today_games,
        yesterday_games if yesterday_usable else [],
    )
    response = WnbaScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(tz=ET).isoformat(),
    )
    _cache["response"] = response
    _cache["expires_at"] = now + cache_ttl_seconds(games)
    _cache["date"] = date_et
    return response
