from collections import defaultdict
from src.scrapers.SoccerScraper import Soccer_Odds_Scraper
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

class SoccerPropFinder():
    def __init__(self, region='us', leagues=None):
        # Get data only from Odds API
        print("Scraping Odds API (Soccer)...")
        self.region = region
        self.leagues = leagues
        # Store timestamp when data is pulled
        self.pull_timestamp = datetime.now()
        self.odds_data = Soccer_Odds_Scraper(region=region, leagues=leagues)
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        self.save_data()

    def organizeData(self):
        # Create maps for all the different prop types
        self.player_goal_scorer_anytime_map = self.create_map(self.odds_data.player_goal_scorer_anytime)
        self.player_first_goal_scorer_map = self.create_map(self.odds_data.player_first_goal_scorer)
        self.player_last_goal_scorer_map = self.create_map(self.odds_data.player_last_goal_scorer)
        self.player_to_receive_card_map = self.create_map(self.odds_data.player_to_receive_card)
        self.player_to_receive_red_card_map = self.create_map(self.odds_data.player_to_receive_red_card)
        self.player_shots_on_target_map = self.create_map(self.odds_data.player_shots_on_target)
        self.player_shots_map = self.create_map(self.odds_data.player_shots)
        self.player_assists_map = self.create_map(self.odds_data.player_assists)

    def create_map(self, data):
        result = defaultdict(list)
        for game_data in data:
            for prop in game_data:
                if len(prop) >= 9:
                    # From the odds API: (market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update, league)
                    market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update, league = prop
                    key = (market_key, bookmaker, league)
                    result[key].append((player_name, over_under, line_score, price, commence_time, last_update))
        return result

    def getDataFrame(self):
        # List all the maps created in organizeData()
        maps = [
            self.player_goal_scorer_anytime_map,
            self.player_first_goal_scorer_map,
            self.player_last_goal_scorer_map,
            self.player_to_receive_card_map,
            self.player_to_receive_red_card_map,
            self.player_shots_on_target_map,
            self.player_shots_map,
            self.player_assists_map,
        ]

        odds_records = []
        # Format timestamp for display
        pull_timestamp_str = self.pull_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        for market_map in maps:
            for (market_key, bookmaker, league), props in market_map.items():
                for player_name, over_under, line_score, price, commence_time, last_update in props:
                    odds_records.append({
                        'LEAGUE': league,
                        'BOOKMAKER': bookmaker,
                        'CATEGORY': market_key,
                        'NAME': player_name,
                        'OVER/UNDER': over_under,
                        'LINE': line_score,
                        'ODDS': price,
                        'COMMENCE_TIME': commence_time,
                        'LAST_UPDATE': last_update,
                        'DATA_PULLED_AT': pull_timestamp_str
                    })

        # Turn it into a DataFrame and return
        return pd.DataFrame(odds_records)

    def save_data(self):
        # Get project root by navigating up from this file's location
        # This file is in src/scrapers/, so go up 2 levels to reach project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        save_dir = os.path.join(str(project_root), "data", "raw", "soccer_player_lines")
        os.makedirs(save_dir, exist_ok=True)

        # Include both date and time in filename for multiple pulls per day
        date_str = self.pull_timestamp.strftime("%Y%m%d")
        time_str = self.pull_timestamp.strftime("%H%M%S")
        timestamp_full = f"{date_str}_{time_str}"

        filename = f"SOCCER_DFS_{timestamp_full}.csv" if getattr(self, "region", None) == "us_dfs" else f"SOCCER_US_{timestamp_full}.csv"
        filepath = os.path.join(save_dir, filename)

        # Save DataFrame to CSV
        if not self.dataframe.empty:
            self.dataframe.to_csv(filepath, index=False)
            print(f"Soccer prop data saved to: {filepath}")
            print(f"Total records: {len(self.dataframe)}")
            print(f"Data pulled at: {self.pull_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No Soccer data to save (no games or off-season for selected leagues)")
