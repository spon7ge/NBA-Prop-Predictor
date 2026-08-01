from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import SHARP_API_KEY
from app.schemas.wnba_props import (
    PROP_SPORTSBOOKS,
    WnbaPropBookQuote,
    WnbaPropLine,
    WnbaPropsResponse,
)
from app.services.odds_snapshots import fetch_latest_prizepicks, fetch_latest_underdog
from app.services.wnba_espn_roster import get_roster_index, norm_player_name
from app.services.wnba_scoreboard import canonical_abbrev

logger = logging.getLogger(__name__)

SHARP_ODDS_URL = "https://api.sharpapi.io/api/v1/odds"
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams"
)
CACHE_TTL_SECONDS = 45.0
FETCH_TIMEOUT_SECONDS = 12.0
ESPN_TIMEOUT_SECONDS = 8.0
ESPN_TEAMS_CACHE_TTL_SECONDS = 600.0
MAX_PAGES = 10
PAGE_LIMIT = 200

_VALID_SIDES = frozenset({"over", "under"})
# Sharp API books only — PrizePicks / Underdog come from Supabase snapshots.
SHARP_PROP_SPORTSBOOKS = ("fanduel", "draftkings")
_SHARP_BOOKS = frozenset(SHARP_PROP_SPORTSBOOKS)

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
        if book not in _SHARP_BOOKS:
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
                **{book_id: None for book_id in PROP_SPORTSBOOKS},
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
        if all(bucket[book_id] is None for book_id in PROP_SPORTSBOOKS):
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
                prizepicks=bucket["prizepicks"],
                underdog=bucket["underdog"],
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


def _stat_key_from_sharp_market(market_type: str) -> str:
    market = market_type.strip().lower()
    if market.startswith("player_"):
        return market[len("player_") :]
    return market


def _stat_key_from_pp_stat_type(stat_type: str) -> str:
    return stat_type.strip().lower().replace(" ", "_").replace("+", "_")


def _stat_key_from_ud_stat_name(stat_name: str) -> str:
    return stat_name.strip().lower().replace(" ", "_")


def _ud_stat_label(stat_name: str) -> str:
    return stat_name.replace("_", " ").title()


def _prop_merge_key(player_name: str, stat_key: str, side: str) -> tuple[str, str, str]:
    return (norm_player_name(player_name), stat_key, side)


def _bucket_from_prop(prop: WnbaPropLine) -> dict[str, Any]:
    return {
        "player_name": prop.player_name,
        "stat": prop.stat,
        "market_type": prop.market_type,
        "side": prop.side,
        "team_abbrev": prop.team_abbrev,
        "logo_url": prop.logo_url,
        "fanduel": prop.fanduel,
        "draftkings": prop.draftkings,
        "prizepicks": prop.prizepicks,
        "underdog": prop.underdog,
    }


def _bucket_has_any_book(bucket: dict[str, Any]) -> bool:
    return any(bucket.get(book_id) is not None for book_id in PROP_SPORTSBOOKS)


def _apply_roster(bucket: dict[str, Any], teams: PlayerTeamIndex) -> None:
    if bucket.get("team_abbrev"):
        return
    hit = teams.get(norm_player_name(bucket["player_name"]))
    if hit:
        bucket["team_abbrev"], bucket["logo_url"] = hit


def merge_snapshot_props(
    sharp_props: list[WnbaPropLine],
    pp_rows: list[dict[str, Any]],
    ud_rows: list[dict[str, Any]],
    player_teams: PlayerTeamIndex | None = None,
) -> list[WnbaPropLine]:
    """Merge Supabase PrizePicks / Underdog snapshots into Sharp-normalized props."""
    teams = player_teams or {}
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

    for prop in sharp_props:
        stat_key = _stat_key_from_sharp_market(prop.market_type)
        key = _prop_merge_key(prop.player_name, stat_key, prop.side)
        buckets[key] = _bucket_from_prop(prop)

    for row in pp_rows:
        player = str(row.get("player_name") or "").strip()
        stat_type = str(row.get("stat_type") or "").strip()
        line_raw = row.get("line_score")
        if not player or not stat_type or line_raw is None:
            continue
        try:
            line_f = float(line_raw)
        except (TypeError, ValueError):
            continue

        stat_key = _stat_key_from_pp_stat_type(stat_type)
        quote = WnbaPropBookQuote(line=line_f, odds_american=None)
        for side in _VALID_SIDES:
            key = _prop_merge_key(player, stat_key, side)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {
                    "player_name": player,
                    "stat": stat_type,
                    "market_type": f"prizepicks:{stat_type}",
                    "side": side,
                    "team_abbrev": None,
                    "logo_url": None,
                    **{book_id: None for book_id in PROP_SPORTSBOOKS},
                }
                buckets[key] = bucket
            bucket["prizepicks"] = quote
            _apply_roster(bucket, teams)

    for row in ud_rows:
        player = str(row.get("player_name") or "").strip()
        stat_name = str(row.get("stat_name") or "").strip()
        side = str(row.get("side") or "").lower()
        line_raw = row.get("line_score")
        if not player or not stat_name or side not in _VALID_SIDES or line_raw is None:
            continue
        try:
            line_f = float(line_raw)
        except (TypeError, ValueError):
            continue

        odds_raw = row.get("american_price")
        odds_i: int | None
        if odds_raw is None:
            odds_i = None
        else:
            try:
                odds_i = int(odds_raw)
            except (TypeError, ValueError):
                odds_i = None

        stat_key = _stat_key_from_ud_stat_name(stat_name)
        key = _prop_merge_key(player, stat_key, side)
        quote = WnbaPropBookQuote(line=line_f, odds_american=odds_i)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "player_name": player,
                "stat": _ud_stat_label(stat_name),
                "market_type": f"underdog:{stat_name}",
                "side": side,
                "team_abbrev": None,
                "logo_url": None,
                **{book_id: None for book_id in PROP_SPORTSBOOKS},
            }
            buckets[key] = bucket
        bucket["underdog"] = quote
        _apply_roster(bucket, teams)

    props: list[WnbaPropLine] = []
    for bucket in buckets.values():
        if not _bucket_has_any_book(bucket):
            continue
        props.append(
            WnbaPropLine(
                player_name=bucket["player_name"],
                team_abbrev=bucket.get("team_abbrev"),
                logo_url=bucket.get("logo_url"),
                stat=bucket["stat"],
                market_type=bucket["market_type"],
                side=bucket["side"],
                fanduel=bucket["fanduel"],
                draftkings=bucket["draftkings"],
                prizepicks=bucket["prizepicks"],
                underdog=bucket["underdog"],
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

    # Fetch books one-at-a-time so free-tier rate limits (FanDuel volume)
    # do not starve DraftKings on a parallel burst.
    rows: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
        for book in SHARP_PROP_SPORTSBOOKS:
            try:
                rows.extend(await fetch_book(client, book))
            except Exception as exc:
                logger.warning("Sharp %s props fetch failed: %s", book, exc)
                errors.append(exc)

    if not rows and errors:
        raise errors[0]
    return rows


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
        sharp_props = normalize_sharp_props(rows, player_teams=player_teams)
        pp_rows = fetch_latest_prizepicks("wnba")
        ud_rows = fetch_latest_underdog("wnba")
        props = merge_snapshot_props(
            sharp_props, pp_rows, ud_rows, player_teams=player_teams
        )
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
