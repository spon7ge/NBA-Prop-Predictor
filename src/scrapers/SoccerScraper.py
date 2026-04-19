import requests
from scripts.Supplier import Supplier

class Soccer_Odds_Scraper():
    # Soccer player props are limited to US bookmakers and these leagues
    DEFAULT_LEAGUES = [
        'soccer_epl',
        'soccer_france_ligue_one',
        'soccer_germany_bundesliga',
        'soccer_italy_serie_a',
        'soccer_spain_la_liga',
        'soccer_usa_mls',
    ]

    def __init__(self, region='us', leagues=None):
        # Soccer player props coverage is currently US-only
        self.region = region
        self.leagues = leagues if leagues is not None else list(self.DEFAULT_LEAGUES)
        supplier = Supplier()
        self.api_key = supplier.getKey()
        self.base_url = "https://api.the-odds-api.com/v4/sports/"
        self.player_goal_scorer_anytime = []
        self.player_first_goal_scorer = []
        self.player_last_goal_scorer = []
        self.player_to_receive_card = []
        self.player_to_receive_red_card = []
        self.player_shots_on_target = []
        self.player_shots = []
        self.player_assists = []
        # Track (league, event_id) so each event keeps its league context
        self.events = self.gameIDs()
        self.collect_all_odds()

    def gameIDs(self):
        events = []
        for league in self.leagues:
            url = f"{self.base_url}{league}/events?apiKey={self.api_key}&regions=us&markets=h2h&oddsFormat=american"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    for game in response.json():
                        events.append((league, game['id']))
                else:
                    print(f"Failed to retrieve events for {league}: {response.status_code}")
            except requests.RequestException as e:
                print(f"Request failed for {league}: {e}")
        return events

    def get_odds(self, league, id, market_type):
        try:
            response = requests.get(
                f"{self.base_url}{league}/events/{id}/odds?apiKey={self.api_key}&regions={self.region}&markets={market_type}&oddsFormat=american",
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
                                    last_update,
                                    league,
                                ))
                self.last_response = response
                return props
            else:
                print(f"Failed to retrieve data ({league}/{market_type}): {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return []

    def collect_all_odds(self):
        for league, id in self.events:
            self.player_goal_scorer_anytime.append(self.get_odds(league, id, 'player_goal_scorer_anytime'))
            self.player_first_goal_scorer.append(self.get_odds(league, id, 'player_first_goal_scorer'))
            self.player_last_goal_scorer.append(self.get_odds(league, id, 'player_last_goal_scorer'))
            self.player_to_receive_card.append(self.get_odds(league, id, 'player_to_receive_card'))
            self.player_to_receive_red_card.append(self.get_odds(league, id, 'player_to_receive_red_card'))
            self.player_shots_on_target.append(self.get_odds(league, id, 'player_shots_on_target'))
            self.player_shots.append(self.get_odds(league, id, 'player_shots'))
            self.player_assists.append(self.get_odds(league, id, 'player_assists'))
