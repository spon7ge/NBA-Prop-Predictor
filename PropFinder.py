from NBAPropFinder.NBAPropFinder import NBAPropFinder
from WNBAPropFinder.WNBAPropFinder import WNBAPropFinder
from NBAPropFinder.TeamLines import fetchTeamLines
from datetime import datetime
api_key = '39f23451ab7ecb6a361792def17f4f78' 

# recomend using one of these at a time
nba_props = NBAPropFinder(region='us')
# nba_team_lines = fetchTeamLines(api_key, date=datetime.now().strftime("%Y%m%d"))
# wnba_props = WNBAPropFinder(region='us')
