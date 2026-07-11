import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.scrapers.NBAPropFinder import NBAPropFinder
from src.scrapers.WNBAPropFinder import WNBAPropFinder
from src.scrapers.TeamLines import fetchTeamLines
from datetime import datetime

# nba_props = NBAPropFinder(region='us,eu', db_upsert=True)
# nba_props = NBAPropFinder(region='us_dfs', db_upsert=True)
wnba_props = WNBAPropFinder(region='us,eu', db_upsert=True)
wnba_props = WNBAPropFinder(region='us_dfs', db_upsert=True)
# nba_team_lines = fetchTeamLines(api_key, date=datetime.now().strftime("%Y%m%d"))
