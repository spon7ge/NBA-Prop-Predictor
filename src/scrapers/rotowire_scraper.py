"""
Rotowire NBA Betting Archive Scraper
=====================================
Scrapes Game, Tipoff, Season, Score, Over/Under, Home Line
from https://www.rotowire.com/betting/nba/archive.php
filtered by API season year (e.g. "2025").

REQUIREMENTS (install once):
    pip install playwright pandas
    playwright install chromium

RUN (from repo root):
    python -m src.scrapers.rotowire_scraper --season 2025 -o data/raw/rotowire/rotowire_nba_2025.csv

OUTPUT:
    CSV with columns: Game, Tipoff, Season, Score, Over_Under, Home_Line
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    raise SystemExit("Run:  pip install playwright && playwright install chromium")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

PAGE_URL = "https://www.rotowire.com/betting/nba/archive.php"
API_URL = "https://www.rotowire.com/betting/nba/tables/games-archive.php"
TARGET_SEASON = "2025"
OUTPUT_FILE = f"rotowire_nba_{TARGET_SEASON}.csv"
HEADLESS = True  # Set False to watch the browser


async def fetch_data(page: Page) -> list:
    """Load archive.php (for session/cookies), then fetch the API directly."""

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

        away = r.get("visit_team_abbrev", "")
        home = r.get("home_team_abbrev", "")
        game = f"{away} @ {home}"

        out.append({
            "Game":       game,
            "Tipoff":     r.get("tipoff", ""),
            "Season":     r.get("season", ""),
            "Score":      r.get("score", ""),
            "Over_Under": r.get("game_over_under", ""),
            "Home_Line":  r.get("line", ""),
        })
    return out


def save(rows: list[dict], path: str, *, season_label: str = "") -> None:
    fieldnames = ["Game", "Tipoff", "Season", "Score", "Over_Under", "Home_Line"]
    sl = season_label or TARGET_SEASON
    if not rows:
        print("No rows found for season", sl)
        return

    if HAS_PANDAS:
        df = pd.DataFrame(rows, columns=fieldnames)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\n✓ Saved {len(df):,} rows → {path}")
        print(df.head(10).to_string(index=False))
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Saved {len(rows):,} rows → {path}")
        for r in rows[:5]:
            print(r)


async def run_scrape(
    *,
    season: str | None = None,
    output_file: str | Path | None = None,
    headless: bool | None = None,
) -> Path:
    """
    Fetch Rotowire games-archive for ``season`` (API season string, e.g. \"2025\")
    and write CSV to ``output_file`` or ``rotowire_nba_{season}.csv`` in cwd.
    """
    s = season or TARGET_SEASON
    out = Path(output_file) if output_file else Path(f"rotowire_nba_{s}.csv")
    hl = HEADLESS if headless is None else headless

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=hl)
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
    rows = build_rows(records, s)
    print(f"Season {s} rows: {len(rows):,}")
    save(rows, str(out), season_label=s)
    return out


def _cli() -> None:
    p = argparse.ArgumentParser(description="Scrape Rotowire NBA games archive.")
    p.add_argument(
        "--season",
        default=TARGET_SEASON,
        help=f'API season filter (default: {TARGET_SEASON!r})',
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: rotowire_nba_{season}.csv)",
    )
    p.add_argument(
        "--headed",
        action="store_true",
        help="Run browser with UI (default: headless)",
    )
    args = p.parse_args()
    asyncio.run(
        run_scrape(
            season=args.season,
            output_file=args.output,
            headless=not args.headed,
        )
    )


if __name__ == "__main__":
    _cli()
