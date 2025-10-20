import requests
import json
from datetime import datetime
import os

def fetchTeamLines(api_key, date=None, save_dir="../DATA/CSV_FILES/PROP_DATA/TEAM_LINES"):
    # Use today's date if none provided
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    
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
    file_path = os.path.join(save_dir, f"NBA_{date}.json")
    with open(file_path, "w") as f:
        json.dump(games_data, f, indent=4)

    print(f"Saved NBA odds data to {file_path}")
    print(f"Total games fetched: {len(games_data)}")
    
    return games_data, file_path

api_key = "eb76813dfade01480164f36d4da1fc8a"
games, filepath = fetchTeamLines(api_key)