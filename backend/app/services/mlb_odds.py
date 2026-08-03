from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.config import SHARP_API_KEY
from app.schemas.mlb_odds import MlbOddsGame, MlbOddsResponse
from app.schemas.wnba_odds import WnbaOddsGame
from app.services.sharp_odds import (
    fetch_sharp_odds_rows,
    merge_odds_prefer_primary,
    normalize_sharp_odds,
)

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 45.0
MLB_MARKETS = "run_line,total_runs"

_cache: dict[str, Any] = {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _to_mlb_games(games: list[WnbaOddsGame]) -> list[MlbOddsGame]:
    return [MlbOddsGame.model_validate(g.model_dump()) for g in games]


async def get_today_odds() -> MlbOddsResponse:
    now = time.monotonic()
    cached = _cache.get("response")
    expires_at = float(_cache.get("expires_at") or 0)
    if cached is not None and now < expires_at:
        return cached

    if not SHARP_API_KEY:
        return MlbOddsResponse(
            as_of=_utcnow_iso(),
            games=[],
            error="SHARP_API_KEY is not configured",
        )

    try:
        dk_result, fd_result = await asyncio.gather(
            fetch_sharp_odds_rows(
                "draftkings", league="mlb", market=MLB_MARKETS
            ),
            fetch_sharp_odds_rows(
                "fanduel", league="mlb", market=MLB_MARKETS
            ),
            return_exceptions=True,
        )
        errors: list[str] = []
        dk_games: list[WnbaOddsGame] = []
        fd_games: list[WnbaOddsGame] = []
        if isinstance(dk_result, BaseException):
            errors.append(f"draftkings: {dk_result}")
        else:
            dk_games = normalize_sharp_odds(
                dk_result, sportsbook="draftkings", wnba_aliases=False
            )
        if isinstance(fd_result, BaseException):
            errors.append(f"fanduel: {fd_result}")
        else:
            fd_games = normalize_sharp_odds(
                fd_result, sportsbook="fanduel", wnba_aliases=False
            )

        if not dk_games and not fd_games:
            if errors:
                raise RuntimeError("; ".join(errors))
            games: list[MlbOddsGame] = []
        else:
            games = _to_mlb_games(merge_odds_prefer_primary(dk_games, fd_games))

        response = MlbOddsResponse(
            as_of=_utcnow_iso(),
            games=games,
            error="; ".join(errors) if errors else None,
        )
        _cache["response"] = response
        _cache["expires_at"] = now + CACHE_TTL_SECONDS
        return response
    except Exception as exc:
        logger.warning("Sharp MLB odds unavailable: %s", exc)
        if cached is not None:
            return MlbOddsResponse(
                as_of=cached.as_of,
                sportsbook=cached.sportsbook,
                games=cached.games,
                error=str(exc),
            )
        return MlbOddsResponse(
            as_of=_utcnow_iso(),
            games=[],
            error=str(exc),
        )
