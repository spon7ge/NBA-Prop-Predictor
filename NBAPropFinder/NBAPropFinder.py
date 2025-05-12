from collections import defaultdict
from NBAPropFinder.Odds_Scraper import Odds_Scraper
from NBAPropFinder.PrizePicks_Scraper import PrizePicks_Scraper
import pandas as pd

class NBAPropFinder():
    def __init__(self):
        # Get data from both scrapers
        print("Scraping Odds API...")
        self.odds_data = Odds_Scraper()
        # print("Scraping PrizePicks...")
        self.prizepicks_data = PrizePicks_Scraper().lines
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()

    def organizeData(self):
        temp = set()
        for item in self.prizepicks_data: # (player_name, stat_type, line_score, flash_sale, formatted_date)
            temp.add(item[1])
        self.categories = temp
        self.points_map = self.create_map(self.odds_data.points)
        self.rebounds_map = self.create_map(self.odds_data.rebounds)
        self.assists_map = self.create_map(self.odds_data.assists)
        self.threes_map = self.create_map(self.odds_data.threes)
        self.blocks_map = self.create_map(self.odds_data.blocks)
        self.steals_map = self.create_map(self.odds_data.steals)
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
                if len(prop) >= 6:
                    # From the example: ('player_points', 'PrizePicks', 'Jamal Murray', 'Over', 20.5, -137)
                    market_key, bookmaker, player_name, over_under, line_score, price = prop
                    key = (market_key, bookmaker)
                    result[key].append((player_name, over_under, line_score, price))
        return result

    def getDataFrame(self):
        # list out all of the maps you created in organizeData()
        maps = [
            self.points_map,
            self.rebounds_map,
            self.assists_map,
            self.threes_map,
            self.blocks_map,
            self.steals_map,
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
                for player_name, over_under, line_score, price in props:
                    odds_records.append({
                        'BOOKMAKER':   bookmaker,
                        'CATEGORY':  market_key,
                        'NAME': player_name,
                        'OVER/UNDER':  over_under,
                        'LINE':  line_score,
                        'ODDS':       price
                    })

        # turn it into a DataFrame and return
        return pd.DataFrame(odds_records)

    