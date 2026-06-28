from collections import defaultdict
from src.scrapers.Odds_Scraper import Odds_Scraper
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

class NBAPropFinder():
    def __init__(self, region='us_dfs', db_upsert: bool = False):
        print("Scraping Odds API...")
        self.region = region
        self.pull_timestamp = datetime.now()
        self.odds_data = Odds_Scraper(region=region)
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        self.save_data()
        if db_upsert:
            self.save_to_db()
        
    def organizeData(self):
        # Create maps for all the different prop types
        self.points_map = self.create_map(self.odds_data.points)
        self.rebounds_map = self.create_map(self.odds_data.rebounds)
        self.assists_map = self.create_map(self.odds_data.assists)
        self.threes_map = self.create_map(self.odds_data.threes)
        self.blocks_map = self.create_map(self.odds_data.blocks)
        self.steals_map = self.create_map(self.odds_data.steals)
        self.fg_map = self.create_map(self.odds_data.fg)
        self.ftm_map = self.create_map(self.odds_data.ftm)
        self.fta_map = self.create_map(self.odds_data.fta)
        self.pra_map = self.create_map(self.odds_data.pra)
        self.pr_map = self.create_map(self.odds_data.pr)
        self.pa_map = self.create_map(self.odds_data.pa)
        self.ra_map = self.create_map(self.odds_data.ra)
        self.to_map = self.create_map(self.odds_data.to)
        self.bs_map = self.create_map(self.odds_data.bs)

    def create_map(self, data):
        result = defaultdict(list)
        for game_data in data:
            for prop in game_data:
                if len(prop) >= 8:  # Updated to check for 8 elements instead of 6
                    # From the odds API: (market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update)
                    market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update = prop
                    key = (market_key, bookmaker)
                    result[key].append((player_name, over_under, line_score, price, commence_time, last_update))
        return result

    def getDataFrame(self):
        # List all the maps created in organizeData()
        maps = [
            self.points_map,
            self.rebounds_map,
            self.assists_map,
            self.threes_map,
            self.blocks_map,
            self.steals_map,
            self.fg_map,
            self.ftm_map,
            self.fta_map,
            self.pra_map,
            self.pr_map,
            self.pa_map,
            self.ra_map,
            self.to_map,
            self.bs_map
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
        save_dir = os.path.join(str(project_root), "data", "raw", "player_lines")
        os.makedirs(save_dir, exist_ok=True)
        
        # Include both date and time in filename for multiple pulls per day
        date_str = self.pull_timestamp.strftime("%Y%m%d")
        time_str = self.pull_timestamp.strftime("%H%M%S")
        timestamp_full = f"{date_str}_{time_str}"

        filename = f"NBA_DFS_{timestamp_full}.csv" if getattr(self, "region", None) == "us_dfs" else f"NBA_US_{timestamp_full}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # Save DataFrame to CSV
        if not self.dataframe.empty:
            self.dataframe.to_csv(filepath, index=False)
            print(f"NBA prop data saved to: {filepath}")
            print(f"Total records: {len(self.dataframe)}")
            print(f"Data pulled at: {self.pull_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No NBA data to save (likely off-season)")

    def save_to_db(self) -> None:
        """Upsert the current prop lines into raw.props_dfs or raw.props_us.

        Table is chosen by region:
            us_dfs          → raw.props_dfs   (DFS books: PrizePicks, Underdog, …)
            anything else   → raw.props_us    (US sportsbooks: DraftKings, FanDuel, …)

        The OVER/UNDER column is renamed to over_under so it is a valid SQL
        identifier. All other column names are lowercased by upsert_df().

        Requires SUPABASE_DB_URL in .env and migration
        scripts/migrations/002_raw_props.sql to have been applied.
        """
        if self.dataframe.empty:
            print("No prop data to upsert (likely off-season)")
            return

        table = "props_dfs" if self.region == "us_dfs" else "props_us"

        df = self.dataframe.rename(columns={"OVER/UNDER": "OVER_UNDER"})

        try:
            from src.utils.db import upsert_df
            upsert_df(table, df)
        except Exception as exc:
            print(f"DB upsert failed for raw.{table}: {exc}")
    