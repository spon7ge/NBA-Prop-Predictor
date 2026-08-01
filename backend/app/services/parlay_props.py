from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.config import PARLAY_API_KEY
from app.schemas.wnba_props import (
    PROP_SPORTSBOOKS,
    WnbaPropBookQuote,
    WnbaPropLine,
    WnbaPropsResponse,
)
from app.services.parlay_client import parlay_get
from app.services.wnba_espn_roster import get_roster_index, norm_player_name
from app.services.wnba_scoreboard import canonical_abbrev
from app.services.wnba_team_names import abbrev_from_team_name
from src.odds.parlay_main_lines import select_parlay_main_lines

logger = logging.getLogger(__name__)

SPORT_KEY = "basketball_wnba"
# Markets that Parlay currently lists for WNBA (steals/blocks/turnovers often absent).
PROP_MARKETS = ",".join(
    [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_pra",
        "player_pts_rebs",
        "player_pts_asts",
        "player_rebs_asts",
        "player_pts_rebs_asts",
        "player_double_double",
        "player_triple_double",
        "player_points_rebounds",
        "player_points_assists",
        "player_assists_rebounds",
        "player_points_rebounds_assists",
        "player_three_pointers",
        "player_three_pointers_made",
    ]
)

CACHE_TTL_SECONDS = 45.0
FETCH_TIMEOUT_SECONDS = 12.0
ESPN_TIMEOUT_SECONDS = 8.0
ESPN_TEAMS_CACHE_TTL_SECONDS = 600.0
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams"
)
PROPS_LIMIT = 10000

_cache: dict[str, Any] = {}
_espn_teams_cache: dict[str, Any] = {}

PlayerTeamIndex = dict[str, tuple[str, str | None]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _stat_label(row: dict[str, Any]) -> str:
    market_label = str(row.get("market") or "").strip()
    if market_label:
        return market_label
    market = str(row.get("market_key") or "")
    if market.startswith("player_"):
        return market[len("player_") :].replace("_", " ").title()
    return market.replace("_", " ").title() or "Unknown"


def normalize_parlay_props(
    rows: list[dict[str, Any]],
    player_teams: PlayerTeamIndex | None = None,
) -> list[WnbaPropLine]:
    """Collapse Parlay prop rows into one line per player + market + side."""
    teams = player_teams or {}
    main_rows = select_parlay_main_lines(rows)
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in main_rows:
        book = str(row.get("bookmaker") or "").lower().strip()
        player = str(row.get("player") or "").strip()
        market = str(row.get("market_key") or "").strip()
        try:
            line_f = float(row["line"])
        except (KeyError, TypeError, ValueError):
            continue
        over_raw = row.get("over_price")
        under_raw = row.get("under_price")
        sides: list[tuple[str, int]] = []
        for side, raw in (("over", over_raw), ("under", under_raw)):
            if raw is None:
                continue
            try:
                sides.append((side, int(raw)))
            except (TypeError, ValueError):
                continue
        if not sides:
            continue

        home = abbrev_from_team_name(str(row.get("home_team") or ""))
        away = abbrev_from_team_name(str(row.get("away_team") or ""))
        event_abbrevs = {a for a in (home, away) if a}
        game_date = str(row.get("game_date") or "").strip() or None
        commence_time = str(row.get("commence_time") or "").strip() or None

        for side, price in sides:
            key = (player, market, side)
            bucket = buckets.setdefault(
                key,
                {
                    "player_name": player,
                    "stat": _stat_label(row),
                    "market_type": market,
                    "side": side,
                    "game_date": game_date,
                    "commence_time": commence_time,
                    **{book_id: None for book_id in PROP_SPORTSBOOKS},
                    "event_abbrevs": set(),
                },
            )
            bucket[book] = WnbaPropBookQuote(line=line_f, odds_american=price)
            bucket["event_abbrevs"].update(event_abbrevs)
            if not bucket.get("commence_time") and commence_time:
                bucket["commence_time"] = commence_time
            if not bucket.get("game_date") and game_date:
                bucket["game_date"] = game_date

    props: list[WnbaPropLine] = []
    for bucket in buckets.values():
        if all(bucket[book_id] is None for book_id in PROP_SPORTSBOOKS):
            continue

        team_abbrev: str | None = None
        logo_url: str | None = None
        hit = teams.get(norm_player_name(bucket["player_name"]))
        if hit:
            team_abbrev, logo_url = hit

        props.append(
            WnbaPropLine(
                player_name=bucket["player_name"],
                team_abbrev=team_abbrev,
                logo_url=logo_url,
                stat=bucket["stat"],
                market_type=bucket["market_type"],
                side=bucket["side"],
                game_date=bucket.get("game_date"),
                commence_time=bucket.get("commence_time"),
                **{book_id: bucket[book_id] for book_id in PROP_SPORTSBOOKS},
            )
        )

    props.sort(
        key=lambda p: (
            p.player_name.lower(),
            p.market_type,
            0 if p.side == "over" else 1,
        )
    )
    return props


async def fetch_parlay_prop_rows() -> list[dict[str, Any]]:
    # Parlay returns 400 UNKNOWN_BOOKMAKER if `bet365` is passed in
    # `bookmakers=` even though bet365 rows appear when unfiltered. Omit the
    # bookmakers filter and keep only our display books client-side.
    payload = await parlay_get(
        f"/sports/{SPORT_KEY}/props",
        params={
            "markets": PROP_MARKETS,
            "limit": PROPS_LIMIT,
        },
        timeout=FETCH_TIMEOUT_SECONDS,
    )
    if not isinstance(payload, list):
        raise RuntimeError("Parlay props response was not a list")
    allowed = frozenset(PROP_SPORTSBOOKS)
    return [
        row
        for row in payload
        if isinstance(row, dict)
        and str(row.get("bookmaker") or "").lower().strip() in allowed
    ]


async def _espn_teams_by_abbrev() -> dict[str, dict[str, str | None]]:
    import httpx

    now = time.monotonic()
    expires_at = float(_espn_teams_cache.get("expires_at") or 0)
    cached = _espn_teams_cache.get("by_abbrev")
    if cached is not None and now < expires_at:
        return cached

    async with httpx.AsyncClient(timeout=ESPN_TIMEOUT_SECONDS) as client:
        res = await client.get(ESPN_TEAMS_URL)
        res.raise_for_status()
        payload = res.json()

    by_abbrev: dict[str, dict[str, str | None]] = {}
    sports = payload.get("sports") or []
    leagues = (sports[0].get("leagues") or []) if sports else []
    teams = (leagues[0].get("teams") or []) if leagues else []
    for entry in teams:
        team = entry.get("team") if isinstance(entry, dict) else None
        if not isinstance(team, dict):
            continue
        abbrev = canonical_abbrev(str(team.get("abbreviation") or ""))
        team_id = str(team.get("id") or "").strip()
        if not abbrev or not team_id:
            continue
        logo_url: str | None = None
        for logo in team.get("logos") or []:
            if not isinstance(logo, dict):
                continue
            href = str(logo.get("href") or "").strip()
            if href:
                logo_url = href
                break
        by_abbrev[abbrev] = {"id": team_id, "logo_url": logo_url}

    _espn_teams_cache["by_abbrev"] = by_abbrev
    _espn_teams_cache["expires_at"] = now + ESPN_TEAMS_CACHE_TTL_SECONDS
    return by_abbrev


async def build_player_team_index(rows: list[dict[str, Any]]) -> PlayerTeamIndex:
    abbrevs: set[str] = set()
    for row in rows:
        for key in ("home_team", "away_team"):
            abbrev = abbrev_from_team_name(str(row.get(key) or ""))
            if abbrev:
                abbrevs.add(abbrev)

    if not abbrevs:
        return {}

    try:
        espn_teams = await _espn_teams_by_abbrev()
    except Exception as exc:
        logger.warning("ESPN WNBA teams unavailable for prop logos: %s", exc)
        return {}

    team_ids: list[tuple[str, str, str | None]] = []
    for abbrev in abbrevs:
        meta = espn_teams.get(abbrev)
        if not meta or not meta.get("id"):
            continue
        team_ids.append((abbrev, str(meta["id"]), meta.get("logo_url")))

    if not team_ids:
        return {}

    async def one(
        abbrev: str, team_id: str, logo: str | None
    ) -> tuple[str, str, str | None, dict]:
        try:
            index = await get_roster_index(team_id)
        except Exception as exc:
            logger.debug("Roster fetch failed for %s (%s): %s", abbrev, team_id, exc)
            index = {}
        return abbrev, team_id, logo, index

    results = await asyncio.gather(
        *(one(abbrev, team_id, logo) for abbrev, team_id, logo in team_ids)
    )

    player_teams: PlayerTeamIndex = {}
    for abbrev, _team_id, logo, index in results:
        for name in index:
            player_teams.setdefault(name, (abbrev, logo))
    return player_teams


async def get_today_props() -> WnbaPropsResponse:
    now = time.monotonic()
    cached = _cache.get("response")
    expires_at = float(_cache.get("expires_at") or 0)
    if cached is not None and now < expires_at:
        return cached

    if not PARLAY_API_KEY:
        return WnbaPropsResponse(
            as_of=_utcnow_iso(),
            props=[],
            error="PARLAY_API_KEY is not configured",
        )

    try:
        rows = await fetch_parlay_prop_rows()
        try:
            from src.odds.load_snapshots import maybe_persist_parlay_props

            maybe_persist_parlay_props(rows, league="wnba")
        except Exception as exc:
            logger.warning("Parlay props snapshot persist skipped: %s", exc)
        try:
            player_teams = await build_player_team_index(rows)
        except Exception as exc:
            logger.warning("Prop team enrichment failed: %s", exc)
            player_teams = {}
        props = normalize_parlay_props(rows, player_teams=player_teams)
        response = WnbaPropsResponse(as_of=_utcnow_iso(), props=props)
        _cache["response"] = response
        _cache["expires_at"] = now + CACHE_TTL_SECONDS
        return response
    except Exception as exc:
        logger.warning("Parlay WNBA props unavailable: %s", exc)
        if cached is not None:
            return WnbaPropsResponse(
                as_of=cached.as_of,
                sportsbooks=cached.sportsbooks,
                props=cached.props,
                error=str(exc),
            )
        return WnbaPropsResponse(
            as_of=_utcnow_iso(),
            props=[],
            error=str(exc),
        )
