"""
DraftKings NBA Player Props Scraper — concurrent fetch
-------------------------------------------------------
Fetches all prop types in parallel and saves to JSON.

Usage:
    python draftkings.py                    # data/props/draftkings/draftkings_YYYY-MM-DD_HHMMSS.json
    python draftkings.py -o my.json
    python draftkings.py --list-types       # show all market names returned
"""

import argparse
import json
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUTPUT_DIR = os.path.join(_ROOT, "data", "props", "draftkings")
_OUTPUT_TZ = ZoneInfo("America/Los_Angeles")

LEAGUE_ID = "42648"

SUBCATEGORY_IDS = [
    "16477",  # Player Points
    "16478",  # Player Rebounds
    "16479",  # Player Assists
    "16480",  # Player Threes
    "16481",  # Player Steals
    "16482",  # Player Blocks
    "16483",  # Player Turnovers
    "16484",  # Player PRA (Pts+Reb+Ast)
    "16485",  # Player PR
    "16486",  # Player PA
    "16487",  # Player RA
]

URL = (
    "https://sportsbook-nash.draftkings.com/sites/US-SB/api/sportscontent"
    "/controldata/league/leagueSubcategory/v1/markets"
)

HEADERS = {
    "accept":                  "*/*",
    "accept-language":         "en-US,en;q=0.5",
    "content-type":            "application/json; charset=utf-8",
    "origin":                  "https://sportsbook.draftkings.com",
    "priority":                "u=1, i",
    "referer":                 "https://sportsbook.draftkings.com/",
    "sec-ch-ua":               '"Chromium";v="146", "Not-A.Brand";v="24", "Brave";v="146"',
    "sec-ch-ua-mobile":        "?0",
    "sec-ch-ua-platform":      '"macOS"',
    "sec-fetch-dest":          "empty",
    "sec-fetch-mode":          "cors",
    "sec-fetch-site":          "same-site",
    "sec-gpc":                 "1",
    "user-agent":              (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "x-client-feature":        "leagueSubcategory",
    "x-client-name":           "web",
    "x-client-page":           "league",
    "x-client-version":        "2618.2.1.19",
    "x-client-widget-name":    "cms",
    "x-client-widget-version": "2.10.12",
}

GAME_LINE_TYPES = {"moneyline", "spread", "total", "game lines"}

# Known stat suffixes DraftKings appends to player names in market names
STAT_SUFFIXES = [
    "Points", "Rebounds", "Assists", "Threes", "3-Pointers",
    "Blocks", "Steals", "Turnovers", "Double Double", "Triple Double",
    "Pts+Reb+Ast", "Pts+Reb", "Pts+Ast", "Reb+Ast",
    "PRA", "PR", "PA", "RA", "Minutes",
]
# Longest first so we don't partially match a shorter suffix
STAT_SUFFIXES.sort(key=len, reverse=True)

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_odds(value) -> int | None:
    """
    Convert DraftKings odds to a plain integer.
    Handles unicode minus sign (U+2212 −) and string/float inputs.
    e.g. "−335" -> -335, "-115" -> -115, 110 -> 110
    """
    if value is None:
        return None
    s = str(value).replace("\u2212", "-").replace("\u002d", "-").strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def normalize_market_name(market_name: str) -> str:
    """Strip trailing alternate-line markers: "Points +" -> "Points"."""
    return re.sub(r"\s*\+$", "", market_name).strip()


def split_market_name(market_name: str) -> tuple[str, str]:
    """
    Split "Donovan Mitchell Assists" -> ("Donovan Mitchell", "Assists").
    Normalizes trailing "+" before matching so "Points +" -> stat "Points".
    Falls back to (normalized_name, "") if no known suffix is found.
    """
    name = normalize_market_name(market_name)
    for suffix in STAT_SUFFIXES:
        if name.endswith(suffix):
            player = name[: -len(suffix)].strip()
            return player, suffix
    return name, ""


def parse_alternate_line(label: str) -> float | None:
    """
    Extract numeric line from alternate-line labels like "3+", "24.5+", "10+".
    Returns None if label doesn't match the pattern.
    """
    m = re.match(r"^(\d+(?:\.\d+)?)\+$", label.strip())
    return float(m.group(1)) if m else None


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_subcategory(subcategory_id):
    params = {
        "isBatchable":  "false",
        "templateVars": LEAGUE_ID,
        "eventsQuery": (
            f"$filter=leagueId eq '{LEAGUE_ID}' AND "
            f"clientMetadata/Subcategories/any(s: s/Id eq '{subcategory_id}')"
        ),
        "marketsQuery": (
            f"$filter=clientMetadata/subCategoryId eq '{subcategory_id}' AND "
            f"tags/all(t: t ne 'SportcastBetBuilder')"
        ),
        "include": "Events",
        "entity":  "events",
    }
    try:
        resp = SESSION.get(URL, params=params, timeout=15)
        resp.raise_for_status()
        return subcategory_id, resp.json()
    except Exception as e:
        print(f"  [SKIP] subcategory {subcategory_id}: {e}")
        return subcategory_id, None


def fetch_all(subcategory_ids, workers=8):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_subcategory, sid): sid for sid in subcategory_ids}
        for future in as_completed(futures):
            sid, data = future.result()
            if data:
                results[sid] = data
    return results


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse(data):
    events  = {e["id"]: e["name"] for e in data.get("events", [])}
    markets = {
        m["id"]: {"name": m["name"], "event_id": m["eventId"]}
        for m in data.get("markets", [])
    }

    props = []
    for sel in data.get("selections", []):
        market = markets.get(sel.get("marketId"))
        if not market:
            continue

        market_name = market["name"]
        if market_name.lower() in GAME_LINE_TYPES:
            continue

        label = (sel.get("label") or "").strip()

        # ── Determine prop type (Over/Under vs alternate line) ──────────────
        alt_line = parse_alternate_line(label)

        if "Over" in label:
            # Standard over/under: "Donovan Mitchell Over"
            side   = "over"
            player = label.replace(" Over", "").strip()
            line   = sel.get("points")

        elif "Under" in label:
            # Standard over/under: "Donovan Mitchell Under"
            side   = "under"
            player = label.replace(" Under", "").strip()
            line   = sel.get("points")

        elif alt_line is not None:
            # Alternate line: label is "3+", player lives in the market name
            side   = "over"                          # "3+" is always an over
            player, _ = split_market_name(market_name)
            line   = alt_line

        else:
            side   = None
            player = label
            line   = sel.get("points")

        # ── Extract clean stat type from market name ────────────────────────
        _, stat = split_market_name(market_name)
        if not stat:
            # Fallback: strip player name prefix if known
            stat = market_name[len(player):].strip() if player and market_name.startswith(player) else market_name

        props.append({
            "player":       player,
            "side":         side,
            "prop_type":    stat,
            "line":         line,
            "odds":         normalize_odds(
                                sel.get("price", {}).get("american")
                                or sel.get("price", {}).get("americanOdds")
                                or sel.get("price", {}).get("odds")
                                or (sel.get("displayOdds") or {}).get("american")
                                or sel.get("trueOdds")
                                or sel.get("odds")
                            ),
            "event":        events.get(market["event_id"], "Unknown"),
            "market_id":    sel.get("marketId"),
            "selection_id": sel.get("id"),
        })

    return props


# ── Output ────────────────────────────────────────────────────────────────────

def _draftkings_output_filename():
    return datetime.now(_OUTPUT_TZ).strftime("draftkings_%Y-%m-%d_%H%M%S.json")


def _default_output_path():
    os.makedirs(_DEFAULT_OUTPUT_DIR, exist_ok=True)
    return os.path.join(_DEFAULT_OUTPUT_DIR, _draftkings_output_filename())


def save(props, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    output = {
        "source":     "DraftKings",
        "league":     "NBA",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count":      len(props),
        "props":      props,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(props)} props -> {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DraftKings NBA player props -> JSON")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Concurrent threads (default: 8)")
    parser.add_argument("--list-types", action="store_true", help="Print all market types returned")
    args = parser.parse_args()

    print(f"Fetching {len(SUBCATEGORY_IDS)} subcategories with {args.workers} threads...")
    t0 = datetime.now()

    all_data = fetch_all(SUBCATEGORY_IDS, workers=args.workers)

    if args.list_types:
        all_types = set()
        for data in all_data.values():
            all_types.update(m["name"] for m in data.get("markets", []))
        print(f"\nMarket types across all subcategories ({len(all_types)} total):")
        for t in sorted(all_types):
            print(f"  * {t}")
        return

    all_props = []
    seen_ids  = set()
    for data in all_data.values():
        for prop in parse(data):
            uid = prop["selection_id"]
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_props.append(prop)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Done in {elapsed:.1f}s — {len(all_props)} props from {len(all_data)} subcategories")

    if not all_props:
        print("No props found. Run with --list-types to inspect.")
        return

    out_path = os.path.expanduser(args.output) if args.output else _default_output_path()
    save(all_props, out_path)


if __name__ == "__main__":
    main()