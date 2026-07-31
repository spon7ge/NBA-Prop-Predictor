from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import SHARP_API_KEY
from app.schemas.wnba_props import WnbaPropBookQuote, WnbaPropLine, WnbaPropsResponse
from app.services.wnba_espn_roster import get_roster_index, norm_player_name
from app.services.wnba_scoreboard import canonical_abbrev

logger = logging.getLogger(__name__)

SHARP_ODDS_URL = "https://api.sharpapi.io/api/v1/odds"
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams"
)
CACHE_TTL_SECONDS = 45.0
FETCH_TIMEOUT_SECONDS = 8.0
ESPN_TIMEOUT_SECONDS = 8.0
ESPN_TEAMS_CACHE_TTL_SECONDS = 600.0
MAX_PAGES = 10
PAGE_LIMIT = 200

_VALID_SIDES = frozenset({"over", "under"})
_VALID_BOOKS = frozenset({"fanduel", "draftkings"})

_cache: dict[str, Any] = {}  # response, expires_at
_espn_teams_cache: dict[str, Any] = {}  # by_abbrev, expires_at

# Player team lookup: normalized name -> (abbrev, logo_url)
PlayerTeamIndex = dict[str, tuple[str, str | None]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _stat_label(row: dict[str, Any]) -> str:
    category = str(row.get("stat_category") or "").strip()
    if category:
        return category.replace("_", " ").title()

    market = str(row.get("market_type") or "")
    if market.startswith("player_"):
        return market[len("player_") :].replace("_", " ").title()
    return market.replace("_", " ").title() or "Unknown"


def _player_name(row: dict[str, Any]) -> str | None:
    name = str(row.get("player_name") or "").strip()
    if name:
        return name
    selection = str(row.get("selection") or "").strip()
    return selection or None


def _team_blob_abbrev_logo(blob: Any) -> tuple[str | None, str | None]:
    if not isinstance(blob, dict):
        return None, None
    raw = str(blob.get("abbreviation") or "").strip()
    abbrev = canonical_abbrev(raw) if raw else None
    logo = str(blob.get("logo") or "").strip() or None
    return abbrev, logo


def _event_team_candidates(row: dict[str, Any]) -> dict[str, str | None]:
    """Map canonical abbrev -> preferred logo URL from Sharp home/away blobs."""
    out: dict[str, str | None] = {}
    for key in ("home", "away"):
        abbrev, logo = _team_blob_abbrev_logo(row.get(key))
        if abbrev:
            out[abbrev] = logo
    return out


def normalize_sharp_props(
    rows: list[dict[str, Any]],
    player_teams: PlayerTeamIndex | None = None,
) -> list[WnbaPropLine]:
    """Collapse Sharp prop rows into one line per player + market + side."""
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    teams = player_teams or {}

    for row in rows:
        if not row.get("is_main_line", False):
            continue
        market = str(row.get("market_type") or "")
        if not market.startswith("player_"):
            continue

        side = str(row.get("selection_type") or "").lower()
        if side not in _VALID_SIDES:
            continue

        book = str(row.get("sportsbook") or "").lower()
        if book not in _VALID_BOOKS:
            continue

        player = _player_name(row)
        if not player:
            continue

        line_raw = row.get("line")
        odds_raw = row.get("odds_american")
        if line_raw is None or odds_raw is None:
            continue
        try:
            line_f = float(line_raw)
            odds_i = int(odds_raw)
        except (TypeError, ValueError):
            continue

        key = (player, market, side)
        bucket = buckets.setdefault(
            key,
            {
                "player_name": player,
                "stat": _stat_label(row),
                "market_type": market,
                "side": side,
                "fanduel": None,
                "draftkings": None,
                "event_logos": _event_team_candidates(row),
            },
        )
        quote = WnbaPropBookQuote(line=line_f, odds_american=odds_i)
        bucket[book] = quote
        # Prefer any Sharp logos seen for this player's event(s).
        for abbrev, logo in _event_team_candidates(row).items():
            if abbrev not in bucket["event_logos"] or not bucket["event_logos"][abbrev]:
                bucket["event_logos"][abbrev] = logo

    props: list[WnbaPropLine] = []
    for bucket in buckets.values():
        if bucket["fanduel"] is None and bucket["draftkings"] is None:
            continue

        team_abbrev: str | None = None
        logo_url: str | None = None
        hit = teams.get(norm_player_name(bucket["player_name"]))
        if hit:
            team_abbrev, logo_url = hit
            # Prefer Sharp CDN logo when available for this abbrev.
            sharp_logo = bucket["event_logos"].get(team_abbrev)
            if sharp_logo:
                logo_url = sharp_logo

        props.append(
            WnbaPropLine(
                player_name=bucket["player_name"],
                team_abbrev=team_abbrev,
                logo_url=logo_url,
                stat=bucket["stat"],
                market_type=bucket["market_type"],
                side=bucket["side"],
                fanduel=bucket["fanduel"],
                draftkings=bucket["draftkings"],
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


async def fetch_sharp_prop_rows() -> list[dict[str, Any]]:
    if not SHARP_API_KEY:
        raise RuntimeError("SHARP_API_KEY is not configured")

    headers = {"X-API-Key": SHARP_API_KEY, "Accept": "application/json"}

    async def fetch_book(
        client: httpx.AsyncClient, sportsbook: str
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        for _ in range(MAX_PAGES):
            params = {
                "league": "wnba",
                "sportsbook": sportsbook,
                "market": "props",
                "is_main_line": "true",
                "limit": str(PAGE_LIMIT),
                "offset": str(offset),
            }
            try:
                res = await client.get(
                    SHARP_ODDS_URL, headers=headers, params=params
                )
                res.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Sharp rejects deep offset (>~500) with 400; keep what we have.
                if rows and exc.response is not None and exc.response.status_code in {
                    400,
                    404,
                }:
                    logger.warning(
                        "Stopping Sharp %s props pagination after %s rows: %s",
                        sportsbook,
                        len(rows),
                        exc,
                    )
                    break
                raise
            payload = res.json()
            chunk = payload.get("data") or []
            if not isinstance(chunk, list) or not chunk:
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

    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
        # Fetch books separately so FanDuel volume cannot crowd out DraftKings,
        # and so we stay under Sharp's offset pagination limit.
        book_rows = await asyncio.gather(
            fetch_book(client, "fanduel"),
            fetch_book(client, "draftkings"),
        )
    return [row for chunk in book_rows for row in chunk]


async def _espn_teams_by_abbrev() -> dict[str, dict[str, str | None]]:
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
    """Map player names to team abbrev + logo using ESPN rosters for event teams."""
    abbrevs: set[str] = set()
    sharp_logos: dict[str, str | None] = {}
    for row in rows:
        for abbrev, logo in _event_team_candidates(row).items():
            abbrevs.add(abbrev)
            if logo and not sharp_logos.get(abbrev):
                sharp_logos[abbrev] = logo

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
        logo = sharp_logos.get(abbrev) or meta.get("logo_url")
        team_ids.append((abbrev, str(meta["id"]), logo))

    if not team_ids:
        return {}

    async def one(abbrev: str, team_id: str, logo: str | None) -> tuple[str, str, str | None, dict]:
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

    if not SHARP_API_KEY:
        return WnbaPropsResponse(
            as_of=_utcnow_iso(),
            props=[],
            error="SHARP_API_KEY is not configured",
        )

    try:
        rows = await fetch_sharp_prop_rows()
        try:
            player_teams = await build_player_team_index(rows)
        except Exception as exc:
            logger.warning("Prop team enrichment failed: %s", exc)
            player_teams = {}
        props = normalize_sharp_props(rows, player_teams=player_teams)
        response = WnbaPropsResponse(as_of=_utcnow_iso(), props=props)
        _cache["response"] = response
        _cache["expires_at"] = now + CACHE_TTL_SECONDS
        return response
    except Exception as exc:
        logger.warning("Sharp WNBA props unavailable: %s", exc)
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
