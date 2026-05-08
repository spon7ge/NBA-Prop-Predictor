"""
Fetch NBA events and odds from Odds-API.io (v3) for selected bookmakers (BetMGM, Bet365).

Default save dir: ``data/props/365+mgm_props/`` (files like ``365+mgm_YYYYMMDD_HHMMSS.json``).

Docs: https://api.odds-api.io/v3 — use GET /bookmakers for exact name spelling.

Env (optional): ``API_KEY_IO_2`` for this scraper (separate from FD/DK pulls on ``API_KEY_IO_1``).
Fallback ``ODDS_API_IO_KEY`` if set. Also: ``ODDS_API_IO_TIMEOUT`` (default 60), ``ODDS_API_IO_RETRIES`` (default 4).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.odds-api.io/v3"
DEFAULT_SPORT = "basketball"
DEFAULT_BOOKMAKERS = ("BetMGM", "Bet365")
MULTI_BATCH = 10


def _request_timeout() -> float:
    return float(os.environ.get("ODDS_API_IO_TIMEOUT", "60"))


def _request_retries() -> int:
    return max(1, int(os.environ.get("ODDS_API_IO_RETRIES", "4")))


def _get_json(
    url: str,
    params: dict[str, Any],
    timeout: float | None = None,
    *,
    retries: int | None = None,
) -> Any:
    """GET JSON with retries on connection / read timeouts (transient network issues)."""
    if timeout is None:
        timeout = _request_timeout()
    if retries is None:
        retries = _request_retries()
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            raise
        except (
            requests.ConnectTimeout,
            requests.ConnectionError,
            requests.ReadTimeout,
        ) as e:
            last_exc = e
            if attempt + 1 >= retries:
                break
            time.sleep(min(2.0**attempt, 30.0))
    raise last_exc  # type: ignore[misc]


def _event_is_nba(event: dict) -> bool:
    lg = event.get("league") or {}
    name = (lg.get("name") or "").lower()
    slug = (lg.get("slug") or "").lower()
    if "wnba" in name or "wnba" in slug:
        return False
    if "g league" in name or "summer league" in name:
        return False
    return "nba" in name or "nba" in slug


def _pick_nba_league_slug(api_key: str, sport: str) -> str | None:
    leagues = _get_json(
        f"{BASE_URL}/leagues",
        {"apiKey": api_key, "sport": sport},
    )
    if not isinstance(leagues, list):
        return None
    rows = []
    for row in leagues:
        name = (row.get("name") or "").lower()
        slug = (row.get("slug") or "").lower()
        if "wnba" in name or "wnba" in slug:
            continue
        if "g league" in name or "summer league" in name or "gleague" in slug:
            continue
        if "nba" not in name and "nba" not in slug:
            continue
        rows.append(row)
    rows.sort(key=lambda L: L.get("eventsCount") or 0, reverse=True)
    return rows[0]["slug"] if rows else None


def fetch_nba_events(
    api_key: str,
    *,
    sport: str = DEFAULT_SPORT,
    league_slug: str | None = None,
    status: str = "pending,live",
) -> list[dict]:
    """
    Return all events from /events for NBA (league auto-resolved if league_slug omitted).
    """
    params_base: dict[str, Any] = {
        "apiKey": api_key,
        "sport": sport,
        "status": status,
    }
    slugs: list[str] = []
    if league_slug:
        slugs.append(league_slug)
    auto = _pick_nba_league_slug(api_key, sport)
    if auto and auto not in slugs:
        slugs.append(auto)

    for slug in slugs:
        try:
            data = _get_json(
                f"{BASE_URL}/events",
                {**params_base, "league": slug},
            )
            if isinstance(data, list):
                return data
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise

    data = _get_json(f"{BASE_URL}/events", params_base)
    if not isinstance(data, list):
        return []
    return [e for e in data if _event_is_nba(e)]


def fetch_odds_multi(
    api_key: str,
    event_ids: list[int],
    bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
) -> list[dict]:
    """
    One /odds/multi call per up to MULTI_BATCH event IDs. Returns merged list of odds payloads.
    """
    if not event_ids:
        return []
    bookmakers_param = ",".join(bookmakers)
    out: list[dict] = []
    for i in range(0, len(event_ids), MULTI_BATCH):
        chunk = event_ids[i : i + MULTI_BATCH]
        batch = _get_json(
            f"{BASE_URL}/odds/multi",
            {
                "apiKey": api_key,
                "eventIds": ",".join(str(x) for x in chunk),
                "bookmakers": bookmakers_param,
            },
        )
        if isinstance(batch, list):
            out.extend(batch)
    return out


def _is_player_prop_market(market: dict) -> bool:
    """Odds-API.io groups NBA player markets under name \"Player Props\"."""
    name = (market.get("name") or "").strip().lower()
    return name == "player props" or name == "player prop"


def filter_bookmakers_player_props_only(bookmakers: dict[str, Any]) -> dict[str, Any]:
    """Keep only Player Props market blocks per bookmaker."""
    out: dict[str, Any] = {}
    for bk, markets in bookmakers.items():
        if not isinstance(markets, list):
            continue
        kept = [
            m for m in markets if isinstance(m, dict) and _is_player_prop_market(m)
        ]
        if kept:
            out[bk] = kept
    return out


def filter_odds_payloads_player_props_only(payloads: list[dict]) -> list[dict]:
    """Return /odds-style payloads with non–player-prop markets removed; drop games with none."""
    filtered: list[dict] = []
    for p in payloads:
        bmap = p.get("bookmakers")
        if not isinstance(bmap, dict):
            continue
        fb = filter_bookmakers_player_props_only(bmap)
        if not fb:
            continue
        row = {k: v for k, v in p.items() if k != "bookmakers"}
        row["bookmakers"] = fb
        filtered.append(row)
    return filtered


def decimal_to_american(value: Any) -> Any:
    """Convert European decimal odds string/number to American (e.g. 1.91 -> -111, 2.50 -> +150).

    Non-numeric values like \"N/A\" are returned unchanged.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        up = raw.upper()
        if up in ("N/A", "NA", "", "-"):
            return raw if raw else None
        try:
            d = float(raw.replace(",", "."))
        except ValueError:
            return value
    else:
        try:
            d = float(value)
        except (TypeError, ValueError):
            return value
    if d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1) * 100))
    return int(round(-100 / (d - 1)))


def _split_prop_label(label: str) -> tuple[str, str]:
    """Parse API labels like \"Max Strus (Points)\" -> name, market."""
    label = (label or "").strip()
    m = re.match(r"^(.+) \((.+)\)\s*$", label)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return label, ""


def flatten_player_props_records(
    payloads: list[dict],
    *,
    odds_format: str = "american",
) -> list[dict[str, Any]]:
    """
    One row per bookmaker / prop line with NAME, MARKET, LINE, OVER, UNDER
    plus BOOKMAKER, EVENT_ID, HOME, AWAY, START for context.

    odds_format: \"american\" (default) or \"decimal\" for OVER/UNDER columns.
    """
    use_american = odds_format.lower() == "american"
    records: list[dict[str, Any]] = []
    for ev in payloads:
        eid = ev.get("id")
        home = ev.get("home")
        away = ev.get("away")
        start = ev.get("date")
        bmap = ev.get("bookmakers") or {}
        if not isinstance(bmap, dict):
            continue
        for bookmaker, markets in bmap.items():
            if not isinstance(markets, list):
                continue
            for market in markets:
                if not isinstance(market, dict):
                    continue
                for line in market.get("odds") or []:
                    if not isinstance(line, dict):
                        continue
                    raw_label = line.get("label")
                    name, mkt = _split_prop_label(
                        raw_label if isinstance(raw_label, str) else ""
                    )
                    over = line.get("over")
                    under = line.get("under")
                    if use_american:
                        over = decimal_to_american(over)
                        under = decimal_to_american(under)
                    records.append(
                        {
                            "NAME": name,
                            "MARKET": mkt,
                            "LINE": line.get("hdp"),
                            "OVER": over,
                            "UNDER": under,
                            "BOOKMAKER": bookmaker,
                            "EVENT_ID": eid,
                            "HOME": home,
                            "AWAY": away,
                            "START": start,
                        }
                    )
    return records


class OddsIoScraper:
    """
    Pull every NBA event the API returns for the configured status, then attach
    BetMGM + Bet365 odds via /odds/multi.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        bookmakers: tuple[str, ...] = DEFAULT_BOOKMAKERS,
        sport: str = DEFAULT_SPORT,
        league_slug: str | None = None,
        status: str = "pending,live",
    ):
        self.api_key = (
            api_key
            or os.environ.get("API_KEY_IO_2", "")
            or os.environ.get("ODDS_API_IO_KEY", "")
        )
        if not self.api_key:
            raise ValueError(
                "Set API_KEY_IO_2 (or ODDS_API_IO_KEY) or pass api_key= to OddsIoScraper."
            )
        self.bookmakers = bookmakers
        self.sport = sport
        self.league_slug = league_slug
        self.status = status
        self.events: list[dict] = []
        self.odds_by_event: list[dict] = []

    def run(self) -> list[dict]:
        self.events = fetch_nba_events(
            self.api_key,
            sport=self.sport,
            league_slug=self.league_slug,
            status=self.status,
        )
        ids = [e["id"] for e in self.events if e.get("id") is not None]
        self.odds_by_event = fetch_odds_multi(
            self.api_key,
            ids,
            bookmakers=self.bookmakers,
        )
        return self.odds_by_event

    @property
    def events_with_odds(self) -> list[dict]:
        """Last `run()` odds payloads (same shape as /odds)."""
        return self.odds_by_event


def default_player_lines_dir() -> Path:
    """``data/props/365+mgm_props`` under project root."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "props" / "365+mgm_props"


def save_odds_io_pull_json(
    *,
    events: list[dict],
    odds: list[dict],
    bookmakers: tuple[str, ...],
    out_dir: Path | None = None,
    pulled_at: datetime | None = None,
    player_props_only: bool = True,
    odds_format: str = "american",
) -> Path:
    """Write odds payloads to timestamped JSON (default: ``data/props/365+mgm_props``).

    Filenames: ``365+mgm_YYYYMMDD_HHMMSS.json`` (player props) or ``365+mgm_full_YYYYMMDD_HHMMSS.json`` (full odds).

    By default saves only markets named \"Player Props\" (NBA); omits the full events list.
    Set player_props_only=False to persist unfiltered /odds payloads and include events.

    odds_format: \"american\" (default) or \"decimal\" for record OVER/UNDER when player_props_only.
    """
    pulled = pulled_at or datetime.now(timezone.utc)
    out_dir = out_dir or default_player_lines_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    date_s = pulled.strftime("%Y%m%d")
    time_s = pulled.strftime("%H%M%S")
    name = (
        f"365+mgm_{date_s}_{time_s}.json"
        if player_props_only
        else f"365+mgm_full_{date_s}_{time_s}.json"
    )
    path = out_dir / name
    odds_out = (
        filter_odds_payloads_player_props_only(odds) if player_props_only else odds
    )
    body: dict[str, Any] = {
        "pulled_at_utc": pulled.isoformat(),
        "bookmakers": list(bookmakers),
        "events_fetched": len(events),
        "player_props_only": player_props_only,
    }
    if player_props_only:
        body["odds_format"] = odds_format.lower()
        records = flatten_player_props_records(odds_out, odds_format=odds_format)
        body["odds_payload_count"] = len(odds_out)
        body["record_count"] = len(records)
        body["records"] = records
    else:
        body["odds_payload_count"] = len(odds_out)
        body["odds"] = odds_out
        body["events"] = events
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    scraper = OddsIoScraper()
    payload = scraper.run()
    props = filter_odds_payloads_player_props_only(payload)
    recs = flatten_player_props_records(props, odds_format="american")
    out_path = save_odds_io_pull_json(
        events=scraper.events,
        odds=payload,
        bookmakers=scraper.bookmakers,
    )
    print(
        f"events_fetched={len(scraper.events)} "
        f"raw_odds_payloads={len(payload)} "
        f"props_payloads={len(props)} "
        f"records={len(recs)}"
    )
    print(f"saved {out_path}")