"""
BettingPros NBA Prop Bets Scraper
Fetches each market separately to get all props.
"""

import requests
import json
import csv
from datetime import date, timedelta

BASE_URL = "https://api.bettingpros.com/v3/props"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bettingpros.com",
    "Referer": "https://www.bettingpros.com/",
}

MARKET_NAMES = {
    156: "Points",
    151: "Assists",
    157: "Rebounds",
    335: "Pts+Ast",
    336: "Pts+Reb",
    337: "Reb+Ast",
    338: "Pts+Reb+Ast",
    152: "Steals",
    160: "Blocks",
    162: "3-Pointers Made",
}


def fetch_market(target_date: str, market_id: int, limit: int = 100, offset: int = 0) -> dict:
    params = {
        "limit": limit,
        "offset": offset,
        "sport": "NBA",
        "market_id": market_id,
        "date": target_date,
        "include_selections": "false",
        "include_filter_graphs": "false",
        "data_points": 8,
        "min_odds": -1000,
        "max_odds": 1000,
        "ev_threshold_min": -0.4,
        "ev_threshold_max": 0.4,
    }
    resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_props(data: dict) -> list[dict]:
    rows = []
    for prop in data.get("props", []):
        proj = prop.get("projection") or {}
        rows.append({
            "player": prop.get("participant", {}).get("name", "Unknown"),
            "prop":   MARKET_NAMES.get(prop.get("market_id"), prop.get("market_id")),
            "line":   prop.get("over", {}).get("line"),
            "proj":   proj.get("value"),
            "side":   proj.get("recommended_side"),
            "diff":   proj.get("diff"),
        })
    return rows


def scrape_all(target_date: str) -> list[dict]:
    all_rows = []

    print(f"Fetching props for {target_date}...")
    for market_id, market_name in MARKET_NAMES.items():
        seen   = set()
        offset = 0

        while True:
            data = fetch_market(target_date, market_id, limit=100, offset=offset)
            rows = parse_props(data)

            if not rows:
                break

            new_rows = []
            for r in rows:
                key = (r["player"], r["prop"], r["line"])
                if key not in seen:
                    seen.add(key)
                    new_rows.append(r)

            if not new_rows:
                break

            all_rows.extend(new_rows)
            offset += 100

        print(f"  {market_name:<20} → {len([r for r in all_rows if r['prop'] == market_name])} props")

    print(f"\n  Total: {len(all_rows)} props")
    return all_rows


def save_csv(rows: list[dict], filename: str):
    if not rows:
        print("No data to save.")
        return
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {filename}")


def save_json(rows: list[dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {len(rows)} rows → {filename}")



# if __name__ == "__main__":
#     START_DATE    = date(2025, 4, 1)
#     END_DATE      = date(2025, 4, 15)
#     OUTPUT_FORMAT = "csv"
#     OUTPUT_NAME   = "nba_props"

#     current = START_DATE
#     while current <= END_DATE:
#         target = str(current)
#         print(f"\n{'='*50}\nProcessing {target}\n{'='*50}")
        
#         rows = scrape_all(target)
        
#         if rows:
#             if OUTPUT_FORMAT in ("csv", "both"):
#                 save_csv(rows, f"NBA-Prop-Predictor/data/backtest/historical_odds/{target}.csv")
#         else:
#             print(f"  No props found for {target}, skipping.")
        
#         current += timedelta(days=1)
#     print("\nDone!")