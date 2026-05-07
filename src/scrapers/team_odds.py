"""
Fetch NBA team markets (Spread, game Totals) from Odds-API.io (v3).

Docs: https://api.odds-api.io/v3

Env: API_KEY_IO_1 or ODDS_API_IO_KEY; optional ODDS_API_IO_TIMEOUT, ODDS_API_IO_RETRIES
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.odds-api.io/v3"
DEFAULT_SPORT = "basketball"
DEFAULT_BOOKMAKERS = ("FanDuel", "DraftKings")
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


def _is_team_core_market(market: dict) -> bool:
    """Full-game spread and main total only (not Spread 1Q / Totals HT, etc.)."""
    name = (market.get("name") or "").strip()
    return name in ("Spread", "Totals")


def filter_bookmakers_team_lines_only(bookmakers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bk, markets in bookmakers.items():
        if not isinstance(markets, list):
            continue
        kept = [m for m in markets if isinstance(m, dict) and _is_team_core_market(m)]
        if kept:
            out[bk] = kept
    return out


def filter_odds_payloads_team_lines_only(payloads: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for p in payloads:
        bmap = p.get("bookmakers")
        if not isinstance(bmap, dict):
            continue
        fb = filter_bookmakers_team_lines_only(bmap)
        if not fb:
            continue
        row = {k: v for k, v in p.items() if k != "bookmakers"}
        row["bookmakers"] = fb
        filtered.append(row)
    return filtered


def decimal_to_american(value: Any) -> Any:
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


def flatten_team_lines_records(
    payloads: list[dict],
    *,
    odds_format: str = "american",
) -> list[dict[str, Any]]:
    """Rows: Spread → LINE, HOME_ODDS, AWAY_ODDS; Totals → LINE, OVER, UNDER (+ game meta)."""
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
                api_market = (market.get("name") or "").strip()
                if api_market not in ("Spread", "Totals"):
                    continue
                for line in market.get("odds") or []:
                    if not isinstance(line, dict):
                        continue
                    base = {
                        "MARKET": api_market,
                        "LINE": line.get("hdp"),
                        "BOOKMAKER": bookmaker,
                        "EVENT_ID": eid,
                        "HOME": home,
                        "AWAY": away,
                        "START": start,
                    }
                    if api_market == "Spread":
                        ho = line.get("home")
                        ao = line.get("away")
                        if use_american:
                            ho = decimal_to_american(ho)
                            ao = decimal_to_american(ao)
                        records.append(
                            {
                                **base,
                                "HOME_ODDS": ho,
                                "AWAY_ODDS": ao,
                                "OVER": None,
                                "UNDER": None,
                            }
                        )
                    else:
                        o = line.get("over")
                        u = line.get("under")
                        if use_american:
                            o = decimal_to_american(o)
                            u = decimal_to_american(u)
                        records.append(
                            {
                                **base,
                                "HOME_ODDS": None,
                                "AWAY_ODDS": None,
                                "OVER": o,
                                "UNDER": u,
                            }
                        )
    return records


class TeamOddsIoScraper:
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
            or os.environ.get("API_KEY_IO_1", "")
            or os.environ.get("ODDS_API_IO_KEY", "")
        )
        if not self.api_key:
            raise ValueError(
                "Set API_KEY_IO_1 (or ODDS_API_IO_KEY) or pass api_key=."
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


def default_team_lines_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "team_lines"
    )


def save_team_lines_json(
    *,
    events: list[dict],
    odds: list[dict],
    bookmakers: tuple[str, ...],
    out_dir: Path | None = None,
    pulled_at: datetime | None = None,
    odds_format: str = "american",
) -> Path:
    pulled = pulled_at or datetime.now(timezone.utc)
    out_dir = out_dir or default_team_lines_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pulled.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"NBA_team_lines_io_{stamp}.json"

    team_payloads = filter_odds_payloads_team_lines_only(odds)
    records = flatten_team_lines_records(team_payloads, odds_format=odds_format)

    body: dict[str, Any] = {
        "pulled_at_utc": pulled.isoformat(),
        "bookmakers": list(bookmakers),
        "events_fetched": len(events),
        "odds_format": odds_format.lower(),
        "markets_included": ["Spread", "Totals"],
        "odds_payload_count": len(team_payloads),
        "record_count": len(records),
        "records": records,
    }
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    scraper = TeamOddsIoScraper()
    payload = scraper.run()
    team_only = filter_odds_payloads_team_lines_only(payload)
    recs = flatten_team_lines_records(team_only, odds_format="american")
    out_path = save_team_lines_json(
        events=scraper.events,
        odds=payload,
        bookmakers=scraper.bookmakers,
    )
    print(
        f"events_fetched={len(scraper.events)} "
        f"team_line_payloads={len(team_only)} "
        f"records={len(recs)}"
    )
    print(f"saved {out_path}")
 