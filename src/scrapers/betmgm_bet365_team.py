"""
NBA full-game spread + totals from Odds-API.io (v3) for BetMGM and Bet365.

Default save dir: ``data/props/365+mgm_team_lines/`` (``365+mgm_YYYYMMDD_HHMMSS.json``).

Env: ``API_KEY_IO_2`` (or ``ODDS_API_IO_KEY``). Optional ``ODDS_API_IO_TIMEOUT``,
``ODDS_API_IO_RETRIES``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.scrapers.team_odds import (
    TeamOddsIoScraper,
    filter_odds_payloads_team_lines_only,
    flatten_team_lines_records,
)

DEFAULT_BOOKMAKERS = ("BetMGM", "Bet365")


def default_team_lines_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "props"
        / "365+mgm_team_lines"
    )


def _resolve_api_key(explicit: str | None) -> str:
    key = (
        explicit
        or os.environ.get("API_KEY_IO_2", "")
        or os.environ.get("ODDS_API_IO_KEY", "")
    )
    if not key:
        raise ValueError(
            "Set API_KEY_IO_2 (or ODDS_API_IO_KEY) or pass api_key= to fetchTeamLines."
        )
    return key


def fetchTeamLines(
    api_key: str | None = None,
    date: str | None = None,
    save_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[dict[str, Any]] | None, Path | None]:
    """
    Fetch Spread + Totals for NBA (BetMGM, Bet365); write JSON under ``save_dir``
    (default ``data/props/365+mgm_team_lines``).

    Returns ``(records, path)`` where ``records`` are flattened rows (same as the
    ``records`` array in the saved file), or ``(None, None)`` on error.
    """
    try:
        key = _resolve_api_key(api_key)
    except ValueError as e:
        print(str(e))
        return None, None

    root_dir = Path(save_dir) if save_dir is not None else default_team_lines_dir()
    root_dir.mkdir(parents=True, exist_ok=True)

    pulled = datetime.now(timezone.utc)
    date_s = date if date is not None else pulled.strftime("%Y%m%d")
    time_s = pulled.strftime("%H%M%S")

    try:
        scraper = TeamOddsIoScraper(
            api_key=key,
            bookmakers=DEFAULT_BOOKMAKERS,
        )
        payload = scraper.run()
    except Exception as e:
        print(f"Error fetching team lines from Odds-API.io: {e}")
        return None, None

    team_payloads = filter_odds_payloads_team_lines_only(payload)
    records = flatten_team_lines_records(team_payloads, odds_format="american")

    body: dict[str, Any] = {
        "pulled_at_utc": pulled.isoformat(),
        "bookmakers": list(DEFAULT_BOOKMAKERS),
        "events_fetched": len(scraper.events),
        "odds_format": "american",
        "markets_included": ["Spread", "Totals"],
        "odds_payload_count": len(team_payloads),
        "record_count": len(records),
        "records": records,
    }

    file_path = root_dir / f"365+mgm_{date_s}_{time_s}.json"
    file_path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved NBA team lines to {file_path}")
    print(
        f"Total games with team lines: {len(team_payloads)} "
        f"(events_fetched={len(scraper.events)} records={len(records)})"
    )

    return records, file_path


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv:
        load_dotenv()

    fetchTeamLines()
