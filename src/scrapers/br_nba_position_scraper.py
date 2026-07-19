from curl_cffi import requests  # mimics real browser TLS — bypasses fingerprint blocks
from bs4 import BeautifulSoup
import csv
import time

URL = "https://www.basketball-reference.com/leagues/NBA_2021_per_game.html"


def scrape_players():
    print(f"Fetching {URL} ...")
    time.sleep(2)  # Be polite — avoid hammering the server

    # impersonate="chrome" replicates Chrome's TLS fingerprint
    response = requests.get(URL, impersonate="chrome", timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "per_game_stats"})

    if not table:
        raise RuntimeError("Could not find the stats table. The page structure may have changed.")

    players = []
    seen = set()  # Deduplicate multi-team rows (TOT row per player traded mid-season)

    for row in table.find("tbody").find_all("tr"):
        # Skip divider / header rows injected mid-table
        if row.get("class") and "thead" in row.get("class"):
            continue

        name_cell = row.find("td", {"data-stat": "name_display"})
        pos_cell  = row.find("td", {"data-stat": "pos"})
        age_cell  = row.find("td", {"data-stat": "age"})

        if not name_cell:
            continue

        name = name_cell.get_text(strip=True)
        pos  = pos_cell.get_text(strip=True)  if pos_cell  else "N/A"
        age  = age_cell.get_text(strip=True)  if age_cell  else "N/A"

        # Keep only the first occurrence (TOT row comes first for traded players)
        if name in seen:
            continue
        seen.add(name)

        players.append({"name": name, "pos": pos, "age": age})

    return players


def save_csv(players, path="nba_2021_players.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "pos", "age"])
        writer.writeheader()
        writer.writerows(players)
    print(f"Saved {len(players)} players to {path}")


if __name__ == "__main__":
    players = scrape_players()

    # Print a preview
    print(f"\n{'NAME':<30} {'POS':<6} {'AGE'}")
    print("-" * 45)
    for p in players[:20]:
        print(f"{p['name']:<30} {p['pos']:<6} {p['age']}")
    print(f"\n... {len(players)} total players found.")

    save_csv(players)