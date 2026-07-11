from collections import defaultdict
from src.scrapers.Odds_Scraper import Odds_Scraper
import pandas as pd
import os
from pathlib import Path
from datetime import datetime


class WNBAPropFinder():
    def __init__(self, region='us_dfs', db_upsert: bool = False):
        print("Scraping Odds API (WNBA)...")
        self.region = region
        self.pull_timestamp = datetime.now()
        self.odds_data = Odds_Scraper(region=region, sport='wnba')
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        self.save_data()
        if db_upsert:
            self.save_to_db()

    def organizeData(self):
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
                if len(prop) >= 8:
                    market_key, bookmaker, player_name, over_under, line_score, price, commence_time, last_update = prop
                    key = (market_key, bookmaker)
                    result[key].append((player_name, over_under, line_score, price, commence_time, last_update))
        return result

    def getDataFrame(self):
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
            self.bs_map,
        ]

        odds_records = []
        pull_timestamp_str = self.pull_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        for market_map in maps:
            for (market_key, bookmaker), props in market_map.items():
                for player_name, over_under, line_score, price, commence_time, last_update in props:
                    odds_records.append({
                        'BOOKMAKER': bookmaker,
                        'CATEGORY': market_key,
                        'NAME': player_name,
                        'OVER_UNDER': over_under,
                        'LINE': line_score,
                        'ODDS': price,
                        'COMMENCE_TIME': commence_time,
                        'LAST_UPDATE': last_update,
                        'DATA_PULLED_AT': pull_timestamp_str,
                    })

        return pd.DataFrame(odds_records)

    def save_data(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        save_dir = os.path.join(str(project_root), "data", "raw", "player_lines")
        os.makedirs(save_dir, exist_ok=True)

        date_str = self.pull_timestamp.strftime("%Y%m%d")
        time_str = self.pull_timestamp.strftime("%H%M%S")
        timestamp_full = f"{date_str}_{time_str}"

        filename = (
            f"WNBA_DFS_{timestamp_full}.csv"
            if getattr(self, "region", None) == "us_dfs"
            else f"WNBA_US_{timestamp_full}.csv"
        )
        filepath = os.path.join(save_dir, filename)

        if not self.dataframe.empty:
            self.dataframe.to_csv(filepath, index=False)
            print(f"WNBA prop data saved to: {filepath}")
            print(f"Total records: {len(self.dataframe)}")
            print(f"Data pulled at: {self.pull_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("No WNBA data to save (likely off-season)")

    def save_to_db(self) -> None:
        """Upsert prop lines into raw.wnba_props_dfs or raw.wnba_props_us.

        Table is chosen by region:
            us_dfs          → raw.wnba_props_dfs
            anything else   → raw.wnba_props_us

        Requires SUPABASE_DB_URL in .env and migration
        db/migrations/013_wnba_raw_props.sql to have been applied.
        """
        if self.dataframe.empty:
            print("No WNBA prop data to upsert (likely off-season)")
            return

        table = "wnba_props_dfs" if self.region == "us_dfs" else "wnba_props_us"

        # Books can emit exact duplicate outcomes in one response; ON CONFLICT
        # cannot update the same PK twice in a single INSERT.
        pk = ["BOOKMAKER", "CATEGORY", "NAME", "OVER_UNDER", "COMMENCE_TIME", "DATA_PULLED_AT", "LINE"]
        df = self.dataframe.drop_duplicates(subset=pk, keep="last")

        try:
            from src.utils.db import upsert_df
            # upsert_df lowercases headers → bookmaker, over_under, … matching 013
            upsert_df(table, df)
        except Exception as exc:
            print(f"DB upsert failed for raw.{table}: {exc}")
