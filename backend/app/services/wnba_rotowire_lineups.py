from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROTOWIRE_TTL_SECONDS = 180

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_cache: dict[str, object] = {"date_et": None, "expires_at": 0.0, "by_abbr": None}


def clear_rotowire_lineups_cache() -> None:
    _cache.update({"date_et": None, "expires_at": 0.0, "by_abbr": None})


def _scrape_starters_by_abbr() -> dict[str, list[dict[str, str | None]]]:
    from src.scrapers.rotowire_starters_scraper import WNBADailyLineups

    return WNBADailyLineups().expected_starters_by_abbr()


def _cached_by_abbr() -> dict[str, list[dict[str, str | None]]] | None:
    now = time.time()
    date_et = datetime.now(ET).strftime("%Y-%m-%d")
    if (
        _cache["by_abbr"] is not None
        and _cache["date_et"] == date_et
        and float(_cache["expires_at"]) > now
    ):
        return _cache["by_abbr"]  # type: ignore[return-value]
    try:
        by_abbr = _scrape_starters_by_abbr()
    except Exception:
        return None
    _cache["date_et"] = date_et
    _cache["expires_at"] = now + ROTOWIRE_TTL_SECONDS
    _cache["by_abbr"] = by_abbr
    return by_abbr


async def get_rotowire_starters_for_matchup(
    *, away_abbr: str, home_abbr: str
) -> dict[str, list[dict[str, str | None]]] | None:
    by_abbr = await asyncio.to_thread(_cached_by_abbr)
    if not by_abbr:
        return None
    away = by_abbr.get(away_abbr.upper())
    home = by_abbr.get(home_abbr.upper())
    if not away or not home or len(away) != 5 or len(home) != 5:
        return None
    return {"away": away, "home": home}
