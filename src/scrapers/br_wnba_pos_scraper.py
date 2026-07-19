"""Scrape WNBA player name / position from Basketball-Reference per-game tables."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup, Comment
from curl_cffi import requests

BASE_URL = "https://www.basketball-reference.com/wnba/years/{year}_per_game.html"
OUT_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "player_positions"
)

# WNBA pages use ``per_game``; NBA-style pages use ``per_game_stats``.
_TABLE_IDS = ("per_game", "per_game_stats")


def find_table(soup: BeautifulSoup, table_ids: tuple[str, ...] = _TABLE_IDS):
    for table_id in table_ids:
        table = soup.find("table", {"id": table_id})
        if table:
            return table

    # BR sometimes wraps secondary tables in HTML comments.
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        for table_id in table_ids:
            if table_id not in comment:
                continue
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.find("table", {"id": table_id})
            if table:
                return table
    return None


def _cell_text(row, data_stat: str) -> str | None:
    cell = row.find(["td", "th"], attrs={"data-stat": data_stat})
    if not cell:
        return None
    link = cell.find("a")
    if link is not None:
        return link.get_text(strip=True) or None
    return cell.get_text(strip=True) or None


def scrape_players(year: str | int) -> list[dict[str, str]]:
    url = BASE_URL.format(year=year)
    print(f"Fetching {url} ...")
    time.sleep(2)

    response = requests.get(url, impersonate="chrome", timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = find_table(soup)
    if not table:
        raise RuntimeError(
            "Could not find the WNBA per-game stats table "
            f"(tried ids: {', '.join(_TABLE_IDS)})."
        )

    players: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in table.find("tbody").find_all("tr"):
        classes = row.get("class") or []
        if "thead" in classes:
            continue

        # Prefer the player link text — the th dumps the whole row when flattened.
        name = _cell_text(row, "player") or _cell_text(row, "name_display")
        if not name:
            continue

        pos = _cell_text(row, "pos") or "N/A"
        age = _cell_text(row, "age") or "N/A"

        if name in seen:
            continue
        seen.add(name)
        players.append({"name": name, "pos": pos, "age": age})

    return players


def save_csv(players: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "pos", "age"])
        writer.writeheader()
        writer.writerows(players)
    print(f"Saved {len(players)} players to {path}")


if __name__ == "__main__":
    year = sys.argv[1] if len(sys.argv) > 1 else "2024"

    players = scrape_players(year)

    print(f"\n{'NAME':<30} {'POS':<6} {'AGE'}")
    print("-" * 45)
    for p in players[:20]:
        print(f"{p['name']:<30} {p['pos']:<6} {p['age']}")
    print(f"\n... {len(players)} total players found.")

    save_csv(players, OUT_DIR / f"wnba_{year}_players.csv")
