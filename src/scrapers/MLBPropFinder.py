from collections import defaultdict
from src.scrapers.MLBScraper import MLB_Odds_Scraper
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

class MLBPropFinder():
    def __init__(self, region='us_dfs'):
        # Get data only from Odds API
        print("Scraping Odds API (MLB)...")
        self.region = region
        # Store timestamp when data is pulled
        self.pull_timestamp = datetime.now()
        self.odds_data = MLB_Odds_Scraper(region=region)
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        self.save_data()

    def organizeData(self):
        # Create maps for all the different prop types
        self.batter_home_runs_map = self.create_map(self.odds_data.batter_home_runs)
        self.batter_first_home_run_map = self.create_map(self.odds_data.batter_first_home_run)
        self.batter_hits_map = self.create_map(self.odds_data.batter_hits)
        self.batter_total_bases_map = self.create_map(self.odds_data.batter_total_bases)
        self.batter_rbis_map = self.create_map(self.odds_data.batter_rbis)
        self.batter_runs_scored_map = self.create_map(self.odds_data.batter_runs_scored)
        self.batter_hits_runs_rbis_map = self.create_map(self.odds_data.batter_hits_runs_rbis)
        self.batter_singles_map = self.create_map(self.odds_data.batter_singles)
        self.batter_doubles_map = self.create_map(self.odds_data.batter_doubles)
        self.batter_triples_map = self.create_map(self.odds_data.batter_triples)
        self.batter_walks_map = self.create_map(self.odds_data.batter_walks)
        self.batter_strikeouts_map = self.create_map(self.odds_data.batter_strikeouts)
        self.batter_stolen_bases_map = self.create_map(self.odds_data.batter_stolen_bases)
        self.batter_fantasy_score_map = self.create_map(self.odds_data.batter_fantasy_score)
        self.pitcher_strikeouts_map = self.create_map(self.odds_data.pitcher_strikeouts)
        self.pitcher_record_a_win_map = self.create_map(self.odds_data.pitcher_record_a_win)
        self.pitcher_hits_allowed_map = self.create_map(self.odds_data.pitcher_hits_allowed)
        self.pitcher_walks_map = self.create_map(self.odds_data.pitcher_walks)
        self.pitcher_earned_runs_map = self.create_map(self.odds_data.pitcher_earned_runs)
        self.pitcher_outs_map = self.create_map(self.odds_data.pitcher_outs)

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
            self.batter_home_runs_map,
            self.batter_first_home_run_map,
            self.batter_hits_map,
            self.batter_total_bases_map,
            self.batter_rbis_map,
            self.batter_runs_scored_map,
            self.batter_hits_runs_rbis_map,
            self.batter_singles_map,
            self.batter_doubles_map,
            self.batter_triples_map,
            self.batter_walks_map,
            self.batter_strikeouts_map,
            self.batter_stolen_bases_map,
            self.batter_fantasy_score_map,
            self.pitcher_strikeouts_map,
            self.pitcher_record_a_win_map,
            self.pitcher_hits_allowed_map,
            self.pitcher_walks_map,
            self.pitcher_earned_runs_map,
            self.pitcher_outs_map,
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
        save_dir = os.path.join(str(project_root), "data", "raw", "mlb_player_lines")
        os.makedirs(save_dir, exist_ok=True)

        # Include both date and time in filename for multiple pulls per day
        date_str = self.pull_timestamp.strftime("%Y%m%d")
        time_str = self.pull_timestamp.strftime("%H%M%S")
        timestamp_full = f"{date_str}_{time_str}"

        filename = f"MLB_DFS_{timestamp_full}.csv" if getattr(self, "region", None) == "us_dfs" else f"MLB_US_{timestamp_full}.csv"
        filepath = os.path.join(save_dir, filename)

        # Save DataFrame to CSV
        if not self.dataframe.empty:
            self.dataframe.to_csv(filepath, index=False)
            print(f"MLB prop data saved to: {filepath}")
            print(f"Total records: {len(self.dataframe)}")
            print(f"Data pulled at: {self.pull_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No MLB data to save (likely off-season)")
