import requests
from scripts.Supplier import Supplier

# The Odds API sport keys
SPORT_KEYS = {
    'nba': 'basketball_nba',
    'wnba': 'basketball_wnba',
}


class Odds_Scraper():
    def __init__(self, region='us_dfs', sport: str = 'nba'):
        self.region = region
        self.sport = sport.lower()
        if self.sport not in SPORT_KEYS:
            raise ValueError(f"Unknown sport {sport!r}. Choose from: {sorted(SPORT_KEYS)}")
        self.sport_key = SPORT_KEYS[self.sport]

        supplier = Supplier()
        self.api_key = supplier.getKey()
        self.base_url = f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events/"
        self.points = []
        self.rebounds = []
        self.assists = []
        self.threes = []
        self.blocks = []
        self.steals = []
        self.fg = []
        self.ftm = []
        self.fta = []
        self.pra = []
        self.pr = []
        self.pa = []
        self.ra = []
        self.to = []
        self.bs = []
        self.ids = self.gameIDs()
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
                f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/events/{id}/odds"
                f"?apiKey={self.api_key}&regions={self.region}&markets={market_type}&oddsFormat=american",
            )
            if response.status_code == 200:
                data = response.json()
                props = []
                commence_time_raw = data.get('commence_time', '')
                commence_time = commence_time_raw.split('T')[0] if commence_time_raw else ''

                for bookmaker in data['bookmakers']:
                    for market in bookmaker['markets']:
                        if market['key'] == market_type:
                            last_update = market.get('last_update', '')
                            for outcome in market['outcomes']:
                                props.append((
                                    market['key'],
                                    bookmaker['title'],
                                    outcome['description'],
                                    outcome['name'],
                                    outcome['point'],
                                    outcome['price'],
                                    commence_time,
                                    last_update
                                ))
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
            self.points.append(self.get_odds(id, 'player_points'))
            self.rebounds.append(self.get_odds(id, 'player_rebounds'))
            self.assists.append(self.get_odds(id, 'player_assists'))
            self.threes.append(self.get_odds(id, 'player_threes'))
            self.blocks.append(self.get_odds(id, 'player_blocks'))
            self.steals.append(self.get_odds(id, 'player_steals'))
            self.fg.append(self.get_odds(id, 'player_field_goals'))
            self.ftm.append(self.get_odds(id, 'player_frees_made'))
            self.fta.append(self.get_odds(id, 'player_frees_attempts'))
            self.pra.append(self.get_odds(id, 'player_points_rebounds_assists'))
            self.pr.append(self.get_odds(id, 'player_points_rebounds'))
            self.pa.append(self.get_odds(id, 'player_points_assists'))
            self.ra.append(self.get_odds(id, 'player_rebounds_assists'))
            self.to.append(self.get_odds(id, 'player_turnovers'))
            self.bs.append(self.get_odds(id, 'player_blocks_steals'))
