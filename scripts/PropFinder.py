from src.scrapers.NBAPropFinder import NBAPropFinder
from src.scrapers.TeamLines import fetchTeamLines
from datetime import datetime

nba_props = NBAPropFinder(region='us')
# nba_team_lines = fetchTeamLines(api_key, date=datetime.now().strftime("%Y%m%d"))
