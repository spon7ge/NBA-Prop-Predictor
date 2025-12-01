# PRODUCTION/feature_engine/usage_features.py

from .base import BaseFeatureBuilder
from PRODUCTION.pipeline_usg import player_usg_features
import joblib

class UsageFeatureBuilder(BaseFeatureBuilder):

    def __init__(self):
        # Load base features (without PREDICTED_MIN)
        base_features = joblib.load('../MODELS/SAVED_MODELS/usg_features.pkl')
        # Add PREDICTED_MIN at the end to match training feature order
        self.feature_names = base_features + ['PREDICTED_MIN']

    def build(self, player_name, data, date, predicted_minutes, projectedStartingFive=None, mainStartingFive=None, teamStarPlayer=None, league_df=None, **kwargs):
        """
        Build usage features using historical data + predicted minutes.
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
        
        # Import required dependencies if not provided
        if projectedStartingFive is None:
            from PRODUCTION.teamInfo import projectedStartingFive
        if mainStartingFive is None:
            from PRODUCTION.teamInfo import mainStartingFive
        if teamStarPlayer is None:
            from PRODUCTION.teamInfo import teamStarPlayer
        
        # Get league_df if not provided (avoid API calls if already passed)
        if league_df is None:
            from nba_api.stats.endpoints import leaguedashteamstats
            league_df = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable='00',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            if 'TEAM_ID' in league_df.columns:
                league_df = league_df.set_index('TEAM_ID')
        
        base = player_usg_features(player_name, data, date, projectedStartingFive, mainStartingFive, teamStarPlayer, league_df)
        if base is None:
            return None
        
        # Add PREDICTED_MIN at the end to match training feature order
        # During training: usg_features_with_min = usg_features + ['PREDICTED_MIN']
        support = [predicted_minutes]
        return base + support
