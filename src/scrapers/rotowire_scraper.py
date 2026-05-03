"""
Rotowire NBA Betting Archive Scraper
=====================================
Scrapes Game, Tipoff, Season, Score, Over/Under, Home Line
from https://www.rotowire.com/betting/nba/tables/games-archive.php
filtered to Season 2025 only.

REQUIREMENTS (install once):
    pip install playwright pandas
    playwright install chromium

RUN:
    python rotowire_nba_scraper.py

OUTPUT:
    rotowire_nba_2025.csv

To scrape a different season, change TARGET_SEASON below.
"""

import asyncio
import csv
import json

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    raise SystemExit("Run:  pip install playwright && playwright install chromium")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

PAGE_URL      = "https://www.rotowire.com/betting/nba/archive.php"
API_URL       = "https://www.rotowire.com/betting/nba/tables/games-archive.php"
TARGET_SEASON = "2025"
OUTPUT_FILE   = f"rotowire_nba_{TARGET_SEASON}.csv"
HEADLESS      = True  # Set False to watch the browser


async def fetch_data(page: Page) -> list:
    """Load archive.php (for session/cookies), then fetch the API directly."""

    # Capture the request as-is so we get any auth headers/cookies automatically
    captured = {}

    async def on_response(response):
        if "games-archive.php" in response.url and "body" not in captured:
            captured["url"] = response.url
            captured["body"] = await response.body()

    page.on("response", on_response)

    print(f"Loading {PAGE_URL} …")
    await page.goto(PAGE_URL, wait_until="networkidle", timeout=60_000)
    await asyncio.sleep(3)

    if "body" not in captured:
        # Fallback: fetch directly using the page's session cookies
        print(f"Fetching {API_URL} directly …")
        result = await page.evaluate(f"""
            async () => {{
                const r = await fetch({json.dumps(API_URL)}, {{
                    credentials: 'include',
                    headers: {{'X-Requested-With': 'XMLHttpRequest'}}
                }});
                return await r.text();
            }}
        """)
        return json.loads(result)

    body = captured["body"]
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")

    return json.loads(body)


def build_rows(records: list, season: str) -> list[dict]:
    """Map raw API fields to the 6 output columns, filtered by season."""
    out = []
    for r in records:
        if r.get("season") != season:
            continue

        # Game: "BOS @ CLE"  (visitor @ home, clean — no HTML)
        away = r.get("visit_team_abbrev", "")
        home = r.get("home_team_abbrev", "")
        game = f"{away} @ {home}"

        out.append({
            "Game":       game,
            "Tipoff":     r.get("tipoff", ""),
            "Season":     r.get("season", ""),
            "Score":      r.get("score", ""),        # e.g. "99-102"
            "Over_Under": r.get("game_over_under", ""),
            "Home_Line":  r.get("line", ""),
        })
    return out


def save(rows: list[dict], path: str):
    fieldnames = ["Game", "Tipoff", "Season", "Score", "Over_Under", "Home_Line"]
    if not rows:
        print("No rows found for season", TARGET_SEASON)
        return

    if HAS_PANDAS:
        df = pd.DataFrame(rows, columns=fieldnames)
        df.to_csv(path, index=False)
        print(f"\n✓ Saved {len(df):,} rows → {path}")
        print(df.head(10).to_string(index=False))
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Saved {len(rows):,} rows → {path}")
        for r in rows[:5]:
            print(r)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
        )
        page = await ctx.new_page()
        records = await fetch_data(page)
        await browser.close()

    print(f"Total records from API: {len(records):,}")
    rows = build_rows(records, TARGET_SEASON)
    print(f"Season {TARGET_SEASON} rows: {len(rows):,}")
    save(rows, OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())