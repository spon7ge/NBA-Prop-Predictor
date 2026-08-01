from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.config import PARLAY_API_KEY
from app.schemas.wnba_odds import WnbaOddsGame, WnbaOddsResponse
from app.services.parlay_client import parlay_get
from app.services.wnba_team_names import abbrev_from_team_name

logger = logging.getLogger(__name__)

SPORT_KEY = "basketball_wnba"
CACHE_TTL_SECONDS = 45.0
FETCH_TIMEOUT_SECONDS = 8.0

# Prefer sharper books first; early slates often lack Pinnacle/DK while softer
# books (and Novig) already post spreads/totals.
BOOK_PREFERENCE: tuple[str, ...] = (
    "pinnacle",
    "draftkings",
    "novig",
    "fanduel",
    "bet365",
    "caesars",
    "betmgm",
)

_cache: dict[str, Any] = {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _markets_by_key(bookmaker: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for market in bookmaker.get("markets") or []:
        if not isinstance(market, dict):
            continue
        key = str(market.get("key") or "").strip()
        if key:
            out[key] = market
    return out


def _pick_spread(
    market: dict[str, Any] | None,
) -> tuple[str | None, float | None]:
    if not market:
        return None, None
    outcomes = market.get("outcomes") or []
    spreads: list[tuple[str, float]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = str(outcome.get("name") or "").strip()
        point = outcome.get("point")
        if not name or point is None:
            continue
        abbrev = abbrev_from_team_name(name)
        if not abbrev:
            continue
        try:
            spreads.append((abbrev, float(point)))
        except (TypeError, ValueError):
            continue
    if not spreads:
        return None, None
    favorites = [(t, ln) for t, ln in spreads if ln < 0]
    pick = favorites[0] if favorites else spreads[0]
    return pick[0], pick[1]


def _pick_total(market: dict[str, Any] | None) -> float | None:
    if not market:
        return None
    for outcome in market.get("outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        point = outcome.get("point")
        if point is None:
            continue
        try:
            return float(point)
        except (TypeError, ValueError):
            continue
    return None


def _bookmakers_by_key(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for book in event.get("bookmakers") or []:
        if not isinstance(book, dict):
            continue
        key = str(book.get("key") or "").lower().strip()
        if key:
            out[key] = book
    return out


def _ordered_books(
    books: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Preferred books first, then any remaining bookmakers."""
    ordered: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for key in BOOK_PREFERENCE:
        book = books.get(key)
        if book is not None:
            ordered.append((key, book))
            seen.add(key)
    for key, book in books.items():
        if key not in seen:
            ordered.append((key, book))
    return ordered


def normalize_parlay_odds(events: list[dict[str, Any]]) -> tuple[list[WnbaOddsGame], str]:
    """
    Collapse Parlay/TOA events into favorite spread + total.

    Prefer Pinnacle, then DraftKings, then other available books for each market.
    Returns (games, sportsbook_label) for the sharpest book that supplied a market.
    """
    games: list[WnbaOddsGame] = []
    used_books: set[str] = set()

    for event in events:
        if not isinstance(event, dict):
            continue
        home = abbrev_from_team_name(str(event.get("home_team") or ""))
        away = abbrev_from_team_name(str(event.get("away_team") or ""))
        if not home or not away:
            continue

        books = _bookmakers_by_key(event)
        spread_team: str | None = None
        spread_line: float | None = None
        total: float | None = None

        for book_key, book in _ordered_books(books):
            markets = _markets_by_key(book)
            if spread_line is None and "spreads" in markets:
                spread_team, spread_line = _pick_spread(markets.get("spreads"))
                if spread_line is not None:
                    used_books.add(book_key)
            if total is None and "totals" in markets:
                total = _pick_total(markets.get("totals"))
                if total is not None:
                    used_books.add(book_key)
            if spread_line is not None and total is not None:
                break

        if spread_line is None and total is None:
            continue

        games.append(
            WnbaOddsGame(
                home_abbrev=home,
                away_abbrev=away,
                spread_team_abbrev=spread_team,
                spread_line=spread_line,
                total=total,
            )
        )

    games.sort(key=lambda g: (g.home_abbrev, g.away_abbrev))
    sportsbook = "draftkings"
    for preferred in BOOK_PREFERENCE:
        if preferred in used_books:
            sportsbook = preferred
            break
    else:
        if used_books:
            sportsbook = sorted(used_books)[0]
    return games, sportsbook


async def fetch_parlay_odds_events() -> list[dict[str, Any]]:
    # Parlay often returns [] when bookmakers= is set (even for books that
    # appear in the unfiltered payload). Fetch all books and prefer client-side.
    payload = await parlay_get(
        f"/sports/{SPORT_KEY}/odds",
        params={
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
        },
        timeout=FETCH_TIMEOUT_SECONDS,
    )
    if not isinstance(payload, list):
        raise RuntimeError("Parlay odds response was not a list")
    return [row for row in payload if isinstance(row, dict)]


async def get_today_odds() -> WnbaOddsResponse:
    now = time.monotonic()
    cached = _cache.get("response")
    expires_at = float(_cache.get("expires_at") or 0)
    if cached is not None and now < expires_at:
        return cached

    if not PARLAY_API_KEY:
        return WnbaOddsResponse(
            as_of=_utcnow_iso(),
            games=[],
            error="PARLAY_API_KEY is not configured",
        )

    try:
        events = await fetch_parlay_odds_events()
        games, sportsbook = normalize_parlay_odds(events)
        response = WnbaOddsResponse(
            as_of=_utcnow_iso(), sportsbook=sportsbook, games=games
        )
        _cache["response"] = response
        _cache["expires_at"] = now + CACHE_TTL_SECONDS
        return response
    except Exception as exc:
        logger.warning("Parlay WNBA odds unavailable: %s", exc)
        if cached is not None:
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
