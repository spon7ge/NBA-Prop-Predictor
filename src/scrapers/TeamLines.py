import requests
import json
from datetime import datetime
import os
from pathlib import Path

def fetchTeamLines(api_key, date=None, save_dir=None):
    # Get project root by navigating up from this file's location
    # This file is in src/scrapers/, so go up 2 levels to reach project root
    if save_dir is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        save_dir = os.path.join(str(project_root), "data", "raw", "team_lines")
    # Use today's date if none provided
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    time = datetime.now().strftime("%H%M%S")
    
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None, None

    games_data = []

    for game in data:
        game_info = {
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "commence_time": game["commence_time"],
            "bookmakers": []
        }

        for bookmaker in game["bookmakers"]:
            bookmaker_info = {
                "bookmaker": bookmaker["title"],
                "last_updated": bookmaker["last_update"],
                "markets": []
            }

            for market in bookmaker["markets"]:
                market_info = {
                    "market_key": market["key"],
                    "outcomes": market["outcomes"]
                }
                bookmaker_info["markets"].append(market_info)

            game_info["bookmakers"].append(bookmaker_info)

        games_data.append(game_info)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"NBA_{date}_{time}.json")
    with open(file_path, "w") as f:
        json.dump(games_data, f, indent=4)

    print(f"Saved NBA odds data to {file_path}")
    print(f"Total games fetched: {len(games_data)}")
    
    return games_data, file_path

api_key = "eb76813dfade01480164f36d4da1fc8a"
games, filepath = fetchTeamLines(api_key)