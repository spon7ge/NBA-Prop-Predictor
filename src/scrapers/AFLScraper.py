import requests
from scripts.Supplier import Supplier

class AFL_Odds_Scraper():
    def __init__(self, region='au'):
        # AFL player props are currently available from select AU bookmakers
        # (Sportsbet, Ladbrokes, TAB, Pointsbet, Betr)
        self.region = region
        supplier = Supplier()
        self.api_key = supplier.getKey()
        self.base_url = "https://api.the-odds-api.com/v4/sports/aussierules_afl/events/"
        self.player_disposals = []
        self.player_disposals_over = []
        self.player_goal_scorer_first = []
        self.player_goal_scorer_last = []
        self.player_goal_scorer_anytime = []
        self.player_goals_scored_over = []
        self.player_marks_over = []
        self.player_marks_most = []
        self.player_tackles_over = []
        self.player_tackles_most = []
        self.player_afl_fantasy_points = []
        self.player_afl_fantasy_points_over = []
        self.player_afl_fantasy_points_most = []
        self.player_clearances_over = []
        self.player_kicks_over = []
        self.player_handballs_over = []
        self.ids = self.gameIDs()
        self.collect_all_odds()

    def gameIDs(self):
        url = f"{self.base_url}?apiKey={self.api_key}&regions=au&markets=h2h&oddsFormat=american"
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
                f"{self.base_url}{id}/odds?apiKey={self.api_key}&regions={self.region}&markets={market_type}&oddsFormat=american",
            )
            if response.status_code == 200:
                data = response.json()
                props = []
                # Extract just the date portion (YYYY-MM-DD) from the ISO timestamp
                commence_time_raw = data.get('commence_time', '')
                commence_time = commence_time_raw.split('T')[0] if commence_time_raw else ''

                for bookmaker in data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == market_type:
                            last_update = market.get('last_update', '')
                            for outcome in market.get('outcomes', []):
                                props.append((
                                    market['key'],
                                    bookmaker['title'],
                                    outcome.get('description', ''),
                                    outcome.get('name', ''),
                                    outcome.get('point', None),
                                    outcome.get('price', None),
                                    commence_time,
                                    last_update
                                ))
                self.last_response = response
                return props
            else:
                print(f"Failed to retrieve data ({market_type}): {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return []

    def collect_all_odds(self):
        for id in self.ids:
            self.player_disposals.append(self.get_odds(id, 'player_disposals'))
            self.player_disposals_over.append(self.get_odds(id, 'player_disposals_over'))
            self.player_goal_scorer_first.append(self.get_odds(id, 'player_goal_scorer_first'))
            self.player_goal_scorer_last.append(self.get_odds(id, 'player_goal_scorer_last'))
            self.player_goal_scorer_anytime.append(self.get_odds(id, 'player_goal_scorer_anytime'))
            self.player_goals_scored_over.append(self.get_odds(id, 'player_goals_scored_over'))
            self.player_marks_over.append(self.get_odds(id, 'player_marks_over'))
            self.player_marks_most.append(self.get_odds(id, 'player_marks_most'))
            self.player_tackles_over.append(self.get_odds(id, 'player_tackles_over'))
            self.player_tackles_most.append(self.get_odds(id, 'player_tackles_most'))
            self.player_afl_fantasy_points.append(self.get_odds(id, 'player_afl_fantasy_points'))
            self.player_afl_fantasy_points_over.append(self.get_odds(id, 'player_afl_fantasy_points_over'))
            self.player_afl_fantasy_points_most.append(self.get_odds(id, 'player_afl_fantasy_points_most'))
            self.player_clearances_over.append(self.get_odds(id, 'player_clearances_over'))
            self.player_kicks_over.append(self.get_odds(id, 'player_kicks_over'))
            self.player_handballs_over.append(self.get_odds(id, 'player_handballs_over'))
