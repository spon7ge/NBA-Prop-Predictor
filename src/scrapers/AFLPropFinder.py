from collections import defaultdict
from src.scrapers.AFLScraper import AFL_Odds_Scraper
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

class AFLPropFinder():
    def __init__(self, region='au'):
        # Get data only from Odds API
        print("Scraping Odds API (AFL)...")
        self.region = region
        # Store timestamp when data is pulled
        self.pull_timestamp = datetime.now()
        self.odds_data = AFL_Odds_Scraper(region=region)
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        self.save_data()

    def organizeData(self):
        # Create maps for all the different prop types
        self.player_disposals_map = self.create_map(self.odds_data.player_disposals)
        self.player_disposals_over_map = self.create_map(self.odds_data.player_disposals_over)
        self.player_goal_scorer_first_map = self.create_map(self.odds_data.player_goal_scorer_first)
        self.player_goal_scorer_last_map = self.create_map(self.odds_data.player_goal_scorer_last)
        self.player_goal_scorer_anytime_map = self.create_map(self.odds_data.player_goal_scorer_anytime)
        self.player_goals_scored_over_map = self.create_map(self.odds_data.player_goals_scored_over)
        self.player_marks_over_map = self.create_map(self.odds_data.player_marks_over)
        self.player_marks_most_map = self.create_map(self.odds_data.player_marks_most)
        self.player_tackles_over_map = self.create_map(self.odds_data.player_tackles_over)
        self.player_tackles_most_map = self.create_map(self.odds_data.player_tackles_most)
        self.player_afl_fantasy_points_map = self.create_map(self.odds_data.player_afl_fantasy_points)
        self.player_afl_fantasy_points_over_map = self.create_map(self.odds_data.player_afl_fantasy_points_over)
        self.player_afl_fantasy_points_most_map = self.create_map(self.odds_data.player_afl_fantasy_points_most)
        self.player_clearances_over_map = self.create_map(self.odds_data.player_clearances_over)
        self.player_kicks_over_map = self.create_map(self.odds_data.player_kicks_over)
        self.player_handballs_over_map = self.create_map(self.odds_data.player_handballs_over)

    def create_map(self, data):
        result = defaultdict(list)
        for game_data in data:
            for prop in game_data:
                if len(prop) >= 8:
                    # From the odds API: (market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update)
                    market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update = prop
                    key = (market_key, bookmaker)
                    result[key].append((player_name, over_under, line_score, price, commence_time, last_update))
        return result

    def getDataFrame(self):
        # List all the maps created in organizeData()
        maps = [
            self.player_disposals_map,
            self.player_disposals_over_map,
            self.player_goal_scorer_first_map,
            self.player_goal_scorer_last_map,
            self.player_goal_scorer_anytime_map,
            self.player_goals_scored_over_map,
            self.player_marks_over_map,
            self.player_marks_most_map,
            self.player_tackles_over_map,
            self.player_tackles_most_map,
            self.player_afl_fantasy_points_map,
            self.player_afl_fantasy_points_over_map,
            self.player_afl_fantasy_points_most_map,
            self.player_clearances_over_map,
            self.player_kicks_over_map,
            self.player_handballs_over_map,
        ]

        odds_records = []
        # Format timestamp for display
        pull_timestamp_str = self.pull_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        for market_map in maps:
            for (market_key, bookmaker), props in market_map.items():
                for player_name, over_under, line_score, price, commence_time, last_update in props:
                    odds_records.append({
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
        save_dir = os.path.join(str(project_root), "data", "raw", "afl_player_lines")
        os.makedirs(save_dir, exist_ok=True)

        # Include both date and time in filename for multiple pulls per day
        date_str = self.pull_timestamp.strftime("%Y%m%d")
        time_str = self.pull_timestamp.strftime("%H%M%S")
        timestamp_full = f"{date_str}_{time_str}"

        filename = f"AFL_AU_{timestamp_full}.csv"
        filepath = os.path.join(save_dir, filename)

        # Save DataFrame to CSV
        if not self.dataframe.empty:
            self.dataframe.to_csv(filepath, index=False)
            print(f"AFL prop data saved to: {filepath}")
            print(f"Total records: {len(self.dataframe)}")
            print(f"Data pulled at: {self.pull_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No AFL data to save (no games or off-season)")
