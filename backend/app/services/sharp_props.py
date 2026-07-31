from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import SHARP_API_KEY
from app.schemas.wnba_props import WnbaPropBookQuote, WnbaPropLine, WnbaPropsResponse

logger = logging.getLogger(__name__)

SHARP_ODDS_URL = "https://api.sharpapi.io/api/v1/odds"
CACHE_TTL_SECONDS = 45.0
FETCH_TIMEOUT_SECONDS = 8.0
MAX_PAGES = 10
PAGE_LIMIT = 200

_VALID_SIDES = frozenset({"over", "under"})
_VALID_BOOKS = frozenset({"fanduel", "draftkings"})

_cache: dict[str, Any] = {}  # response, expires_at


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


def normalize_sharp_props(rows: list[dict[str, Any]]) -> list[WnbaPropLine]:
    """Collapse Sharp prop rows into one line per player + market + side."""
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

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
            },
        )
        quote = WnbaPropBookQuote(line=line_f, odds_american=odds_i)
        bucket[book] = quote

    props: list[WnbaPropLine] = []
    for bucket in buckets.values():
        if bucket["fanduel"] is None and bucket["draftkings"] is None:
            continue
        props.append(
            WnbaPropLine(
                player_name=bucket["player_name"],
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
    params_base = {
        "league": "wnba",
        "sportsbook": "draftkings,fanduel",
        "market": "props",
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

            pagination = (payload.get("meta") or {}).get("pagination") or {}
            if not pagination.get("has_more"):
                break
            next_offset = pagination.get("next_offset")
            if next_offset is None:
                break
            offset = int(next_offset)

    return rows


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
        props = normalize_sharp_props(rows)
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
