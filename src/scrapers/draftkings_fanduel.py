"""
Fetch NBA events and odds from Odds-API.io (v3) for FanDuel + DraftKings player props.

Default save dir: ``data/props/dk+fd_props/`` (files like ``dk+fd_YYYYMMDD_HHMMSS.json``).

**Schema parity:** The JSON envelope and ``records[]`` rows match ``betmgm_bet365.py``.
Prop filtering and flattening helpers should be updated together in both files (or deduped
into shared code later).

Docs: https://api.odds-api.io/v3 — use GET /bookmakers for exact name spelling.

Env (optional): ``API_KEY_IO_1`` or ``ODDS_API_IO_KEY``; ``ODDS_API_IO_TIMEOUT`` (default 60),
``ODDS_API_IO_RETRIES`` (default 4).
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


_TEAM_SIDE_TAIL = frozenset({"home", "away", "draw", "none", "tie", "pk"})
_UMBRELLA_MARKET_NAMES = frozenset({"player props", "player prop"})


def _market_title_suggests_player_stat_props(name: str) -> bool:
    """True when Odds names a bucket clearly about individual stat lines (often Bet365/BetMGM)."""
    lc = (name or "").strip().lower()
    if not lc:
        return False
    if any(x in lc for x in ("team total", "match total", "game total")):
        return False
    if any(x in lc for x in ("winner", "handicap ", " handicap", "spread", "money line")):
        return False

    stat_bits = (
        "point",
        "assist",
        "rebound",
        "three",
        "steal",
        "block",
        "turnover",
        "triple",
        "double",
        "fantasy",
        "performance",
        "minute",
        "first basket",
    )
    if "player" in lc:
        return any(s in lc for s in stat_bits)
    # Title-only alternate lines
    return bool(re.search(r"\b(alternate|alternative)\s+.*\b(points|assists|rebounds)", lc))


def _is_player_prop_market(market: dict) -> bool:
    """Treat a market bucket as NBA player props for filtering.

    Odds-API.io usually nests props under markets named ``Player Props``, but some
    books use other titles or omit the umbrella. In those cases we still keep the
    block when most outcome ``label`` strings look like ``Player Name (Market)``.

    Labels like spreads ``Away (+7.5)`` are ignored (pure numeric parentheses).
    """
    if not isinstance(market, dict):
        return False
    raw_name = (market.get("name") or "").strip()
    name = raw_name.lower()
    if name in ("player props", "player prop"):
        return True
    if "player props" in name or name.endswith("player prop"):
        return True
    if _market_title_suggests_player_stat_props(raw_name):
        return True

    odds = market.get("odds")
    if not isinstance(odds, list) or not odds:
        return False
    lines = [o for o in odds if isinstance(o, dict)]
    if not lines:
        return False
    n = len(lines)
    with_stat = sum(
        1
        for ln in lines
        if isinstance(ln.get("label"), str)
        and bool(_split_prop_label(ln["label"])[1])
    )
    return with_stat > 0 and with_stat >= max(1, (4 * n + 9) // 10)


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
    """Parse API labels like \"Max Strus (Points)\" -> name, market.

    Strings like spread labels \"Team Name (+7.5)\" are not split (returns empty market)
    so we do not classify them as player-prop labels.
    """
    label = (label or "").strip()
    m = re.match(r"^(.+) \((.+)\)\s*$", label)
    if not m:
        return label, ""
    mkt_raw = (m.group(2) or "").strip()
    mkt_nop_fracs = (
        mkt_raw.replace("\u00bd", "")
        .replace("\u00bc", "")
        .replace("\u00be", "")
        .replace("½", "")
        .replace("¼", "")
        .replace("¾", "")
        .strip()
    )
    if re.fullmatch(r"[+\- 0-9.]+", mkt_nop_fracs):
        return label, ""
    if mkt_raw.strip().lower() in _TEAM_SIDE_TAIL:
        return label, ""
    return m.group(1).strip(), mkt_raw


def _infer_market_from_freeform_label(label: str) -> str:
    """Best-effort stat tag when label is not ``Name (Stat)`` (common for Bet365)."""
    low = (label or "").lower()
    if not low:
        return ""
    checks = (
        (r"\bdouble[\s\-]?double\b", "Double+Double"),
        (r"\btriple[\s\-]?double\b", "Triple+Double"),
        (r"\b3[- ]?point|three[- ]?pointer|\bfg\b.*3", "3 Point FG"),
        (r"\bpoints?\b|\bpts\b", "Points"),
        (r"\brebounds?\b|\brebs?\b", "Rebounds"),
        (r"\bassists?\b|\basts?\b", "Assists"),
        (r"\bblocks?\b", "Blocks"),
        (r"\bsteals?\b", "Steals"),
        (r"\bfirst basket\b", "First Basket"),
    )
    for pat, disp in checks:
        if re.search(pat, low):
            return disp
    return ""


def _resolve_flat_market(
    market: dict,
    line: dict,
    mkt_from_label: str,
) -> str:
    """Fill MARKET when Odds-API uses plain labels under a generic umbrella (often Bet365)."""
    if mkt_from_label:
        return mkt_from_label
    for k in (
        "betTypeName",
        "categoryName",
        "groupName",
        "marketName",
        "selectionName",
        "subMarketName",
    ):
        v = line.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    bucket = (market.get("name") or "").strip()
    bl = bucket.lower()
    if bucket and bl not in _UMBRELLA_MARKET_NAMES:
        return bucket
    raw = line.get("label")
    if isinstance(raw, str):
        inferred = _infer_market_from_freeform_label(raw)
        if inferred:
            return inferred
    return "Player Props"


def flatten_player_props_records(
    payloads: list[dict],
    *,
    odds_format: str = "american",
) -> list[dict[str, Any]]:
    """
    One row per bookmaker / prop line with NAME, MARKET, LINE, OVER, UNDER
    plus BOOKMAKER, EVENT_ID, HOME, AWAY, START for context.

    ``MARKET`` is usually parsed from ``label`` as ``Name (Stat)``. If missing,
    we use the market bucket title, optional line metadata fields, or fall back
    to the string ``Player Props`` for generic umbrellas (common on some books).

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
                    mkt = _resolve_flat_market(market, line, mkt)
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
    FanDuel + DraftKings odds via /odds/multi.
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
            or os.environ.get("API_KEY_IO_1", "")
            or os.environ.get("ODDS_API_IO_KEY", "")
        )
        if not self.api_key:
            raise ValueError(
                "Set API_KEY_IO_1 (or ODDS_API_IO_KEY) or pass api_key= to OddsIoScraper."
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
    """``data/props/dk+fd_props`` under project root."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "props" / "dk+fd_props"


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
    """Write odds payloads to timestamped JSON (default: ``data/props/dk+fd_props``).

    Filenames: ``dk+fd_YYYYMMDD_HHMMSS.json`` (player props) or ``dk+fd_full_YYYYMMDD_HHMMSS.json`` (full odds).

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
        f"dk+fd_{date_s}_{time_s}.json"
        if player_props_only
        else f"dk+fd_full_{date_s}_{time_s}.json"
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