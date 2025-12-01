# PRODUCTION/feature_engine/minutes_features.py
from PRODUCTION.pipeline_min import player_min_features
from .base import BaseFeatureBuilder
import pandas as pd
import joblib

class MinutesFeatureBuilder(BaseFeatureBuilder):
    feature_names = joblib.load('../MODELS/SAVED_MODELS/min_features.pkl')

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
            from PRODUCTION.teamInfo import projectedStartingFive
        if mainStartingFive is None:
            from PRODUCTION.teamInfo import mainStartingFive
        if teamStarPlayer is None:
            from PRODUCTION.teamInfo import teamStarPlayer
    
        return player_min_features(player_name, data, date, projectedStartingFive, mainStartingFive, teamStarPlayer, league_df)
