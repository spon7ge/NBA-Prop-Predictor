# PRODUCTION/feature_engine/fga_features.py

from .base import BaseFeatureBuilder
from src.pipeline.pipeline_fga import build_fga_features
import joblib
from pathlib import Path

# Get project root for model paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

class FGAFeatureBuilder(BaseFeatureBuilder):

    def __init__(self):
        # Load base features (without PREDICTED_MIN and PREDICTED_USG_PCT)
        try:
            base_features = joblib.load(project_root / 'src' / 'models' / 'saved' / 'fga_features.pkl')
        except FileNotFoundError:
            # If pickle file doesn't exist, use empty list (will be set during training)
            base_features = []
        # Add PREDICTED_MIN and PREDICTED_USG_PCT at the end to match training feature order
        # During training: fga_features_with_min_usg = fga_features + ['PREDICTED_MIN', 'PREDICTED_USG_PCT']
        self.feature_names = base_features + ['PREDICTED_MIN', 'PREDICTED_USG_PCT']

    def build(self, player_name, data, date, predicted_minutes, predicted_usage, projectedStartingFive=None, mainStartingFive=None, teamStarPlayer=None, league_df=None, findOpp=None, **kwargs):
        """
        Build FGA features using historical data + predicted minutes and usage.
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
        
        # Import required dependencies if not provided
        if projectedStartingFive is None:
            from src.utils.team_info import projectedStartingFive
        if mainStartingFive is None:
            from src.utils.team_info import mainStartingFive
        if teamStarPlayer is None:
            from src.utils.team_info import teamStarPlayer
        if findOpp is None:
            from src.utils.helper_functions import findOpp
        
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
        
        base = build_fga_features(
            player_name=player_name,
            data=data,
            current_date=date,
            projectedStartingFive=projectedStartingFive,
            mainStartingFive=mainStartingFive,
            teamStarPlayer=teamStarPlayer,
            league_df=league_df,
            findOpp=findOpp,
            predicted_minutes=predicted_minutes,
            predicted_usage=predicted_usage
        )
        if base is None:
            return None
        
        # Convert feature dict to list in the order of feature_names
        # Add PREDICTED_MIN and PREDICTED_USG_PCT at the end to match training feature order
        # During training: fga_features_with_min_usg = fga_features + ['PREDICTED_MIN', 'PREDICTED_USG_PCT']
        feature_list = [base.get(f, 0.0) for f in self.feature_names[:-2]]  # All except last 2
        feature_list.append(float(predicted_minutes))  # PREDICTED_MIN
        feature_list.append(float(predicted_usage))     # PREDICTED_USG_PCT
        
        return feature_list

