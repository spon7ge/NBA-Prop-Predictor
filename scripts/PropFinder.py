import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.scrapers.NBAPropFinder import NBAPropFinder
from src.scrapers.MLBPropFinder import MLBPropFinder
from src.scrapers.SoccerPropFinder import SoccerPropFinder
from src.scrapers.AFLPropFinder import AFLPropFinder
from src.scrapers.TeamLines import fetchTeamLines
from datetime import datetime

nba_props = NBAPropFinder(region='us,eu', db_upsert=True)
nba_props = NBAPropFinder(region='us_dfs', db_upsert=True)
# mlb_props = MLBPropFinder(region='us,eu')
# soccer_props = SoccerPropFinder(region='us')
# afl_props = AFLPropFinder(region='au')
# nba_team_lines = fetchTeamLines(api_key, date=datetime.now().strftime("%Y%m%d"))
