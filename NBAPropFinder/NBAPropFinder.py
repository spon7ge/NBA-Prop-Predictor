from collections import defaultdict
from NBAPropFinder.Odds_Scraper import Odds_Scraper
import pandas as pd
import os
from datetime import datetime

class NBAPropFinder():
    def __init__(self, region='us_dfs'):
        # Get data only from Odds API
        print("Scraping Odds API...")
        self.odds_data = Odds_Scraper(region=region)
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        self.save_data()
        
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
                        'LAST_UPDATE': last_update
                    })

        # Turn it into a DataFrame and return
        return pd.DataFrame(odds_records)
    
    def save_data(self):
        save_dir = "/Users/alexg/Documents/Documents/Prize-Picks-Prop-Predictor/DATA/CSV_FILES/PROP_DATA"
        timestamp = datetime.now().strftime("%Y%m%d")
        

        filename = f"NBA_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # Save DataFrame to CSV
        if not self.dataframe.empty:
            self.dataframe.to_csv(filepath, index=False)
            print(f"NBA prop data saved to: {filepath}")
            print(f"Total records: {len(self.dataframe)}")
        else:
            print("No NBA data to save (likely off-season)")
    