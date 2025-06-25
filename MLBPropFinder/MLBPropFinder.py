from collections import defaultdict
from MLBPropFinder.Odds_MLB_Scraper import ODDS_MLB_SCRAPER
from MLBPropFinder.PrizePicks_MLB_Scraper import PRIZEPICKS_MLB_SCRAPER
import pandas as pd

class MLBPropFinder():
    
    def __init__(self, region='us_dfs'):
        # Get data from both scrapers
        print("Scraping Odds API...")
        self.odds_data = ODDS_MLB_SCRAPER(region=region)
        # print("Scraping PrizePicks...")
        self.prizepicks_data = PRIZEPICKS_MLB_SCRAPER().lines
        print("Organizing Data...")
        self.organizeData()
        self.dataframe = self.getDataFrame()
        
    def organizeData(self):
        temp = set()
        for item in self.prizepicks_data: # (player_name, stat_type, line_score, flash_sale, formatted_date)
            temp.add(item[1])
        self.categories = temp
        
        # Batter props maps
        self.batter_home_runs_map = self.create_map(self.odds_data.batter_home_runs)
        self.batter_home_runs_alternate_map = self.create_map(self.odds_data.batter_home_runs_alternate)
        self.batter_first_home_run_map = self.create_map(self.odds_data.batter_first_home_run)
        self.batter_hits_map = self.create_map(self.odds_data.batter_hits)
        self.batter_hits_alternate_map = self.create_map(self.odds_data.batter_hits_alternate)
        self.batter_total_bases_map = self.create_map(self.odds_data.batter_total_bases)
        self.batter_total_bases_alternate_map = self.create_map(self.odds_data.batter_total_bases_alternate)
        self.batter_rbis_map = self.create_map(self.odds_data.batter_rbis)
        self.batter_rbis_alternate_map = self.create_map(self.odds_data.batter_rbis_alternate)
        self.batter_runs_scored_map = self.create_map(self.odds_data.batter_runs_scored)
        self.batter_hits_runs_rbis_map = self.create_map(self.odds_data.batter_hits_runs_rbis)
        self.batter_singles_map = self.create_map(self.odds_data.batter_singles)
        self.batter_doubles_map = self.create_map(self.odds_data.batter_doubles)
        self.batter_triples_map = self.create_map(self.odds_data.batter_triples)
        self.batter_walks_map = self.create_map(self.odds_data.batter_walks)
        self.batter_walks_alternate_map = self.create_map(self.odds_data.batter_walks_alternate)
        self.batter_strikeouts_map = self.create_map(self.odds_data.batter_strikeouts)
        self.batter_stolen_bases_map = self.create_map(self.odds_data.batter_stolen_bases)
        
        # Pitcher props maps
        self.pitcher_strikeouts_map = self.create_map(self.odds_data.pitcher_strikeouts)
        self.pitcher_strikeouts_alternate_map = self.create_map(self.odds_data.pitcher_strikeouts_alternate)
        self.pitcher_record_a_win_map = self.create_map(self.odds_data.pitcher_record_a_win)
        self.pitcher_hits_allowed_map = self.create_map(self.odds_data.pitcher_hits_allowed)
        self.pitcher_hits_allowed_alternate_map = self.create_map(self.odds_data.pitcher_hits_allowed_alternate)
        self.pitcher_walks_map = self.create_map(self.odds_data.pitcher_walks)
        self.pitcher_walks_alternate_map = self.create_map(self.odds_data.pitcher_walks_alternate)
        self.pitcher_earned_runs_map = self.create_map(self.odds_data.pitcher_earned_runs)
        self.pitcher_outs_map = self.create_map(self.odds_data.pitcher_outs)

    def create_map(self, data):
        result = defaultdict(list)
        for game_data in data:
            for prop in game_data:
                if len(prop) >= 6:
                    # From the example: ('batter_hits', 'PrizePicks', 'Mike Trout', 'Over', 1.5, -137)
                    market_key, bookmaker, player_name, over_under, line_score, price = prop
                    key = (market_key, bookmaker)
                    result[key].append((player_name, over_under, line_score, price))
        return result

    def getDataFrame(self):
        # List all maps created in organizeData()
        maps = [
            # Batter props
            self.batter_home_runs_map,
            self.batter_home_runs_alternate_map,
            self.batter_first_home_run_map,
            self.batter_hits_map,
            self.batter_hits_alternate_map,
            self.batter_total_bases_map,
            self.batter_total_bases_alternate_map,
            self.batter_rbis_map,
            self.batter_rbis_alternate_map,
            self.batter_runs_scored_map,
            self.batter_hits_runs_rbis_map,
            self.batter_singles_map,
            self.batter_doubles_map,
            self.batter_triples_map,
            self.batter_walks_map,
            self.batter_walks_alternate_map,
            self.batter_strikeouts_map,
            self.batter_stolen_bases_map,
            
            # Pitcher props
            self.pitcher_strikeouts_map,
            self.pitcher_strikeouts_alternate_map,
            self.pitcher_record_a_win_map,
            self.pitcher_hits_allowed_map,
            self.pitcher_hits_allowed_alternate_map,
            self.pitcher_walks_map,
            self.pitcher_walks_alternate_map,
            self.pitcher_earned_runs_map,
            self.pitcher_outs_map
        ]

        odds_records = []
        for market_map in maps:
            for (market_key, bookmaker), props in market_map.items():
                for player_name, over_under, line_score, price in props:
                    odds_records.append({
                        'BOOKMAKER': bookmaker,
                        'CATEGORY': market_key,
                        'NAME': player_name,
                        'OVER/UNDER': over_under,
                        'LINE': line_score,
                        'ODDS': price
                    })

        # Convert to DataFrame and return
        return pd.DataFrame(odds_records)
