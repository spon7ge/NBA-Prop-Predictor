# PRODUCTION/feature_engine/minutes_features.py
from src.pipeline.pipeline_min import build_min_features
from .base import BaseFeatureBuilder
import pandas as pd
import joblib
from pathlib import Path

# Get project root for model paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

class MinutesFeatureBuilder(BaseFeatureBuilder):
    feature_names = joblib.load(project_root / 'src' / 'models' / 'saved' / 'min_features.pkl')

    def build(self, player_name, data, date, projectedStartingFive=None, mainStartingFive=None, teamStarPlayer=None, league_df=None, findOpp=None, **kwargs):
        """
        Build minutes-based context features (NO predicted values).
        """
        # Extract from kwargs if not provided directly
        if projectedStartingFive is None:
            projectedStartingFive = kwargs.get('projectedStartingFive')
        if mainStartingFive is None:
            mainStartingFive = kwargs.get('mainStartingFive')
        if teamStarPlayer is None:
            teamStarPlayer = kwargs.get('teamStarPlayer')
        if league_df is None:
            league_df = kwargs.get('league_df')
        if findOpp is None:
            findOpp = kwargs.get('findOpp')
        
        # Import league_df if not provided (avoid API calls if already passed)
        if league_df is None:
            from nba_api.stats.endpoints import leaguedashteamstats
            league_df = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable='00',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            if 'TEAM_ID' in league_df.columns:
                league_df = league_df.set_index('TEAM_ID')
        
        # Import defaults if still None
        if projectedStartingFive is None:
            from src.utils.team_info import projectedStartingFive
        if mainStartingFive is None:
            from src.utils.team_info import mainStartingFive
        if teamStarPlayer is None:
            from src.utils.team_info import teamStarPlayer
        if findOpp is None:
            from src.utils.helper_functions import findOpp
    
        feature_dict = build_min_features(
            player_name=player_name,
            data=data,
            current_date=date,
            projectedStartingFive=projectedStartingFive,
            mainStartingFive=mainStartingFive,
            teamStarPlayer=teamStarPlayer,
            league_df=league_df,
            findOpp=findOpp
        )
        
        if feature_dict is None:
            return None
        
        # Convert dictionary to list in the order of feature_names
        feature_list = [feature_dict.get(f, 0.0) for f in self.feature_names]
        
        return feature_list
