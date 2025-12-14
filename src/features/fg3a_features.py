# PRODUCTION/feature_engine/fg3a_features.py

from .base import BaseFeatureBuilder
from src.pipeline.pipeline_fg3a import build_fg3a_features
import joblib
from pathlib import Path

# Get project root for model paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

class FG3AFeatureBuilder(BaseFeatureBuilder):

    def __init__(self):
        try:
            base_features = joblib.load(project_root / 'src' / 'models' / 'saved' / 'fg3a_features.pkl')
        except FileNotFoundError:
            base_features = []

        self.feature_names = base_features + ['PREDICTED_MIN', 'PREDICTED_FGA']

    def build(self, player_name, data, date, predicted_minutes, predicted_fga, projectedStartingFive=None, mainStartingFive=None, teamStarPlayer=None, league_df=None, findOpp=None, **kwargs):
        """
        Build FG3A features using historical data + predicted minutes and FGA.
        """
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
        
        if projectedStartingFive is None:
            from src.utils.team_info import projectedStartingFive
        if mainStartingFive is None:
            from src.utils.team_info import mainStartingFive
        if teamStarPlayer is None:
            from src.utils.team_info import teamStarPlayer
        if findOpp is None:
            from src.utils.helper_functions import findOpp
        
        if league_df is None:
            from nba_api.stats.endpoints import leaguedashteamstats
            league_df = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable='00',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            if 'TEAM_ID' in league_df.columns:
                league_df = league_df.set_index('TEAM_ID')
        
        base = build_fg3a_features(
            player_name=player_name,
            data=data,
            current_date=date,
            projectedStartingFive=projectedStartingFive,
            mainStartingFive=mainStartingFive,
            teamStarPlayer=teamStarPlayer,
            league_df=league_df,
            findOpp=findOpp,
            predicted_minutes=predicted_minutes,
            predicted_fga=predicted_fga
        )
        if base is None:
            return None
        
        feature_list = [base.get(f, 0.0) for f in self.feature_names[:-2]]
        
        feature_list.append(float(predicted_minutes))
        feature_list.append(float(predicted_fga))
        
        return feature_list
