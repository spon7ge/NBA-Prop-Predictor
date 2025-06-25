import requests
from Supplier import Supplier

class ODDS_MLB_SCRAPER:
    def __init__(self, region='us_dfs'):
        self.region = region
        supplier = Supplier()
        self.api_key = supplier.getKey()
        self.base_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/"
        self.ids = self.gameIDs()
        
        # Batter props
        self.batter_home_runs = []
        self.batter_home_runs_alternate = []
        self.batter_first_home_run = []
        self.batter_hits = []
        self.batter_hits_alternate = []
        self.batter_total_bases = []
        self.batter_total_bases_alternate = []
        self.batter_rbis = []
        self.batter_rbis_alternate = []
        self.batter_runs_scored = []
        self.batter_hits_runs_rbis = []
        self.batter_singles = []
        self.batter_doubles = []
        self.batter_triples = []
        self.batter_walks = []
        self.batter_walks_alternate = []
        self.batter_strikeouts = []
        self.batter_stolen_bases = []
        
        # Pitcher props
        self.pitcher_strikeouts = []
        self.pitcher_strikeouts_alternate = []
        self.pitcher_record_a_win = []
        self.pitcher_hits_allowed = []
        self.pitcher_hits_allowed_alternate = []
        self.pitcher_walks = []
        self.pitcher_walks_alternate = []
        self.pitcher_earned_runs = []
        self.pitcher_outs = []
        
        self.collect_all_odds()

    def gameIDs(self):
        url = f"{self.base_url}?apiKey={self.api_key}&regions=us&markets=h2h&oddsFormat=american"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return [game['id'] for game in response.json()]
            else:
                print(f"Failed to retrieve data: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return []
    
    def get_odds(self, id, market_type):
        try:
            response = requests.get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{id}/odds?apiKey={self.api_key}&regions={self.region}&markets={market_type}&oddsFormat=american",
            )
            if response.status_code == 200:
                data = response.json()
                props = []
                for bookmaker in data['bookmakers']:
                    for market in bookmaker['markets']:
                        if market['key'] == market_type:
                            for outcome in market['outcomes']:
                                props.append((
                                    market['key'],
                                    bookmaker['title'],
                                    outcome['description'],
                                    outcome['name'],
                                    outcome['point'],
                                    outcome['price'],
                                ))
                # Save the last response
                self.last_response = response
                return props
            else:
                print(f"Failed to retrieve data: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return []
    
    def collect_all_odds(self):
        for id in self.ids:
            # Collect batter props
            self.batter_home_runs.append(self.get_odds(id, 'batter_home_runs'))
            self.batter_home_runs_alternate.append(self.get_odds(id, 'batter_home_runs_alternate'))
            self.batter_first_home_run.append(self.get_odds(id, 'batter_first_home_run'))
            self.batter_hits.append(self.get_odds(id, 'batter_hits'))
            self.batter_hits_alternate.append(self.get_odds(id, 'batter_hits_alternate'))
            self.batter_total_bases.append(self.get_odds(id, 'batter_total_bases'))
            self.batter_total_bases_alternate.append(self.get_odds(id, 'batter_total_bases_alternate'))
            self.batter_rbis.append(self.get_odds(id, 'batter_rbis'))
            self.batter_rbis_alternate.append(self.get_odds(id, 'batter_rbis_alternate'))
            self.batter_runs_scored.append(self.get_odds(id, 'batter_runs_scored'))
            self.batter_hits_runs_rbis.append(self.get_odds(id, 'batter_hits_runs_rbis'))
            self.batter_singles.append(self.get_odds(id, 'batter_singles'))
            self.batter_doubles.append(self.get_odds(id, 'batter_doubles'))
            self.batter_triples.append(self.get_odds(id, 'batter_triples'))
            self.batter_walks.append(self.get_odds(id, 'batter_walks'))
            self.batter_walks_alternate.append(self.get_odds(id, 'batter_walks_alternate'))
            self.batter_strikeouts.append(self.get_odds(id, 'batter_strikeouts'))
            self.batter_stolen_bases.append(self.get_odds(id, 'batter_stolen_bases'))
            
            # Collect pitcher props
            self.pitcher_strikeouts.append(self.get_odds(id, 'pitcher_strikeouts'))
            self.pitcher_strikeouts_alternate.append(self.get_odds(id, 'pitcher_strikeouts_alternate'))
            self.pitcher_record_a_win.append(self.get_odds(id, 'pitcher_record_a_win'))
            self.pitcher_hits_allowed.append(self.get_odds(id, 'pitcher_hits_allowed'))
            self.pitcher_hits_allowed_alternate.append(self.get_odds(id, 'pitcher_hits_allowed_alternate'))
            self.pitcher_walks.append(self.get_odds(id, 'pitcher_walks'))
            self.pitcher_walks_alternate.append(self.get_odds(id, 'pitcher_walks_alternate'))
            self.pitcher_earned_runs.append(self.get_odds(id, 'pitcher_earned_runs'))
            self.pitcher_outs.append(self.get_odds(id, 'pitcher_outs'))