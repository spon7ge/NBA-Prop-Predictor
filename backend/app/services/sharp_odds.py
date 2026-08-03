from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import SHARP_API_KEY
from app.schemas.wnba_odds import WnbaOddsGame, WnbaOddsResponse
from app.services.wnba_scoreboard import canonical_abbrev

logger = logging.getLogger(__name__)

SHARP_ODDS_URL = "https://api.sharpapi.io/api/v1/odds"
CACHE_TTL_SECONDS = 45.0
FETCH_TIMEOUT_SECONDS = 8.0
MAX_PAGES = 3
PAGE_LIMIT = 200

_MAIN_MARKETS = frozenset(
    {"point_spread", "total_points", "run_line", "total_runs"}
)
_SPREAD_MARKETS = frozenset({"point_spread", "run_line"})
_TOTAL_MARKETS = frozenset({"total_points", "total_runs"})

# Fallback when Sharp omits nested team objects (seen for some Mystics rows).
_NAME_TO_ABBREV = {
    "atlanta dream": "ATL",
    "atl dream": "ATL",
    "chicago sky": "CHI",
    "chi sky": "CHI",
    "connecticut sun": "CON",
    "con sun": "CON",
    "dallas wings": "DAL",
    "dal wings": "DAL",
    "golden state valkyries": "GSV",
    "gs valkyries": "GSV",
    "indiana fever": "IND",
    "ind fever": "IND",
    "las vegas aces": "LVA",
    "lv aces": "LVA",
    "los angeles sparks": "LAS",
    "la sparks": "LAS",
    "minnesota lynx": "MIN",
    "min lynx": "MIN",
    "new york liberty": "NYL",
    "ny liberty": "NYL",
    "phoenix mercury": "PHO",
    "phx mercury": "PHO",
    "portland fire": "PDX",
    "por fire": "PDX",
    "pdx fire": "PDX",
    "seattle storm": "SEA",
    "sea storm": "SEA",
    "washington mystics": "WAS",
    "was mystics": "WAS",
    "wsh mystics": "WAS",
    # MLB (Sharp / books occasionally omit nested team objects)
    "arizona diamondbacks": "AZ",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "oakland athletics": "ATH",
    "athletics": "ATH",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st. louis cardinals": "STL",
    "st louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}

_TRICODE_RE = re.compile(r"^[A-Z]{2,3}$")
_EVENT_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")

_cache: dict[str, Any] = {}  # response, expires_at


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canon_abbrev(abbrev: str, *, wnba_aliases: bool) -> str:
    upper = str(abbrev or "").strip().upper()
    if not wnba_aliases:
        return upper
    return canonical_abbrev(upper)


def _abbrev_from_team_blob(
    team: dict[str, Any] | None,
    team_label: str | None,
    *,
    wnba_aliases: bool = True,
) -> str | None:
    if isinstance(team, dict):
        raw = str(team.get("abbreviation") or "").strip().upper()
        if raw:
            return _canon_abbrev(raw, wnba_aliases=wnba_aliases)
        name = str(team.get("name") or "").strip().lower()
        if name and name in _NAME_TO_ABBREV:
            return _canon_abbrev(_NAME_TO_ABBREV[name], wnba_aliases=wnba_aliases)

    label = str(team_label or "").strip()
    if not label:
        return None

    # "WAS Mystics" / "ATL Dream" → leading tricode
    first = label.split()[0].upper()
    if _TRICODE_RE.match(first):
        return _canon_abbrev(first, wnba_aliases=wnba_aliases)

    mapped = _NAME_TO_ABBREV.get(label.lower())
    if mapped:
        return _canon_abbrev(mapped, wnba_aliases=wnba_aliases)
    return None


def _game_date_from_event_id(event_id: str | None) -> str | None:
    if not event_id:
        return None
    match = _EVENT_DATE_RE.search(str(event_id))
    return match.group(1) if match else None


def _spread_side_abbrev(row: dict[str, Any], home: str, away: str) -> str | None:
    side = str(row.get("team_side") or row.get("selection_type") or "").lower()
    if side == "home":
        return home
    if side == "away":
        return away
    selection = str(row.get("selection") or "").strip().lower()
    if selection:
        mapped = _NAME_TO_ABBREV.get(selection)
        if mapped:
            return canonical_abbrev(mapped)
        # Selection may be "ATL Dream"
        first = selection.split()[0].upper()
        if _TRICODE_RE.match(first):
            return canonical_abbrev(first)
    return None


def normalize_sharp_odds(
    rows: list[dict[str, Any]],
    sportsbook: str | None = None,
    *,
    wnba_aliases: bool = True,
) -> list[WnbaOddsGame]:
    """Collapse Sharp odds rows into one game record with favorite spread + total."""
    by_event: dict[str, dict[str, Any]] = {}

    for row in rows:
        if not row.get("is_main_line", False):
            continue
        market = str(row.get("market_type") or "")
        if market not in _MAIN_MARKETS:
            continue

        home = _abbrev_from_team_blob(
            row.get("home"), row.get("home_team"), wnba_aliases=wnba_aliases
        )
        away = _abbrev_from_team_blob(
            row.get("away"), row.get("away_team"), wnba_aliases=wnba_aliases
        )
        if not home or not away:
            logger.debug(
                "Skipping Sharp row without abbrevs: event=%s market=%s",
                row.get("event_id"),
                market,
            )
            continue

        event_id = str(row.get("event_id") or f"{away}@{home}")
        bucket = by_event.setdefault(
            event_id,
            {
                "home_abbrev": home,
                "away_abbrev": away,
                "game_date": _game_date_from_event_id(row.get("event_id")),
                "spreads": [],
                "totals": [],
            },
        )

        line = row.get("line")
        if line is None:
            continue
        try:
            line_f = float(line)
        except (TypeError, ValueError):
            continue

        if market in _SPREAD_MARKETS:
            team = _spread_side_abbrev(row, home, away)
            if team:
                bucket["spreads"].append((team, line_f))
        elif market in _TOTAL_MARKETS:
            bucket["totals"].append(line_f)

    games: list[WnbaOddsGame] = []
    for bucket in by_event.values():
        spread_team: str | None = None
        spread_line: float | None = None
        spreads: list[tuple[str, float]] = bucket["spreads"]
        if spreads:
            favorites = [(t, ln) for t, ln in spreads if ln < 0]
            pick = favorites[0] if favorites else spreads[0]
            spread_team, spread_line = pick[0], pick[1]

        total: float | None = None
        if bucket["totals"]:
            total = bucket["totals"][0]

        if spread_line is None and total is None:
            continue

        games.append(
            WnbaOddsGame(
                home_abbrev=bucket["home_abbrev"],
                away_abbrev=bucket["away_abbrev"],
                spread_team_abbrev=spread_team,
                spread_line=spread_line,
                total=total,
                game_date=bucket.get("game_date"),
                sportsbook=sportsbook,
            )
        )

    games.sort(key=lambda g: (g.home_abbrev, g.away_abbrev))
    return games


def _odds_merge_key(game: WnbaOddsGame) -> tuple[str, str, str]:
    return (game.away_abbrev, game.home_abbrev, game.game_date or "")


def merge_odds_prefer_primary(
    primary: list[WnbaOddsGame],
    fallback: list[WnbaOddsGame],
) -> list[WnbaOddsGame]:
    by_key: dict[tuple[str, str, str], WnbaOddsGame] = {}
    for game in primary:
        by_key[_odds_merge_key(game)] = game
    for game in fallback:
        key = _odds_merge_key(game)
        if key not in by_key:
            by_key[key] = game
    games = list(by_key.values())
    games.sort(key=lambda g: (g.game_date or "", g.home_abbrev, g.away_abbrev))
    return games


async def fetch_sharp_odds_rows(
    sportsbook: str = "draftkings",
    *,
    league: str = "wnba",
    market: str = "point_spread,total_points",
) -> list[dict[str, Any]]:
    if not SHARP_API_KEY:
        raise RuntimeError("SHARP_API_KEY is not configured")

    headers = {"X-API-Key": SHARP_API_KEY, "Accept": "application/json"}
    params_base = {
        "league": league,
        "sportsbook": sportsbook,
        "market": market,
        "is_main_line": "true",
        "limit": str(PAGE_LIMIT),
    }

    rows: list[dict[str, Any]] = []
    offset = 0
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
        for _ in range(MAX_PAGES):
            params = {**params_base, "offset": str(offset)}
            res = await client.get(SHARP_ODDS_URL, headers=headers, params=params)
            res.raise_for_status()
            payload = res.json()
            chunk = payload.get("data") or []
            if not isinstance(chunk, list):
                break
            rows.extend(chunk)

            pagination = (
                payload.get("pagination")
                or (payload.get("meta") or {}).get("pagination")
                or {}
            )
            if not pagination.get("has_more"):
                break
            # Sharp requires a cursor past ~offset 500; next_offset becomes null.
            next_offset = pagination.get("next_offset")
            if next_offset is None:
                break
            offset = int(next_offset)

    return rows


async def get_today_odds() -> WnbaOddsResponse:
    now = time.monotonic()
    cached = _cache.get("response")
    expires_at = float(_cache.get("expires_at") or 0)
    if cached is not None and now < expires_at:
        return cached

    if not SHARP_API_KEY:
        return WnbaOddsResponse(
            as_of=_utcnow_iso(),
            games=[],
            error="SHARP_API_KEY is not configured",
        )

    try:
        dk_result, fd_result = await asyncio.gather(
            fetch_sharp_odds_rows("draftkings"),
            fetch_sharp_odds_rows("fanduel"),
            return_exceptions=True,
        )
        errors: list[str] = []
        dk_games: list[WnbaOddsGame] = []
        fd_games: list[WnbaOddsGame] = []
        if isinstance(dk_result, BaseException):
            errors.append(f"draftkings: {dk_result}")
        else:
            dk_games = normalize_sharp_odds(dk_result, sportsbook="draftkings")
        if isinstance(fd_result, BaseException):
            errors.append(f"fanduel: {fd_result}")
        else:
            fd_games = normalize_sharp_odds(fd_result, sportsbook="fanduel")

        if not dk_games and not fd_games:
            if errors:
                raise RuntimeError("; ".join(errors))
            games = []
        else:
            games = merge_odds_prefer_primary(dk_games, fd_games)

        response = WnbaOddsResponse(
            as_of=_utcnow_iso(),
            games=games,
            error="; ".join(errors) if errors else None,
        )
        _cache["response"] = response
        _cache["expires_at"] = now + CACHE_TTL_SECONDS
        return response
    except Exception as exc:
        logger.warning("Sharp WNBA odds unavailable: %s", exc)
        if cached is not None:
            # Preserve last good payload; attach error for debug.
            return WnbaOddsResponse(
                as_of=cached.as_of,
                sportsbook=cached.sportsbook,
                games=cached.games,
                error=str(exc),
            )
        return WnbaOddsResponse(
            as_of=_utcnow_iso(),
            games=[],
            error=str(exc),
        )
