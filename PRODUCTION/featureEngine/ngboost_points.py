"""
NGBOOST Points Prediction Module

Uses NGBOOST models (mean and variance) to predict player points.
Requires predicted MIN and USG_PCT from XGBoost models as features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from MODELS.ngboostModel import predict_mean_variance_split
from PRODUCTION.helperFunctions import findOpp


def load_trained_ngboost_models(model_paths=None):
    """
    Load trained NGBOOST mean and variance models.
    
    Args:
        model_paths: Dict with keys 'mean_model', 'variance_model', 'calibration_factor', 
                    'calibration_params' (optional), 'features'
                    If None, uses default paths
    
    Returns:
        Dict with 'mean_model', 'variance_model', 'calibration_factor', 'calibration_params' (optional),
        'features', or None if not found
    """
    if model_paths is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_paths = {
            'mean_model': os.path.join(script_dir, '..', '..', 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_MEAN_MODEL_PRODUCTION.pkl'),
            'variance_model': os.path.join(script_dir, '..', '..', 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_VAR_MODEL_PRODUCTION.pkl'),
            'calibration_factor': os.path.join(script_dir, '..', '..', 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_CALIBRATION_FACTOR_PRODUCTION.pkl'),
            'calibration_params': os.path.join(script_dir, '..', '..', 'MODELS', 'SAVED_MODELS', 'NGBOOST_PTS_CALIBRATION_PARAMS_PRODUCTION.pkl'),
            'features': os.path.join(script_dir, '..', '..', 'MODELS', 'SAVED_MODELS', 'pts_features.pkl')
        }
    
    try:
        # Load models (saved with joblib.dump)
        mean_model = joblib.load(model_paths['mean_model'])
        variance_model = joblib.load(model_paths['variance_model'])
        calibration_factor = joblib.load(model_paths['calibration_factor'])
        features = joblib.load(model_paths['features'])
        
        # Load calibration parameters if available (for score-dependent calibration)
        calibration_params = None
        if 'calibration_params' in model_paths:
            try:
                calibration_params = joblib.load(model_paths['calibration_params'])
            except Exception as e:
                # If file doesn't exist or can't be loaded, use None (fallback to basic calibration)
                pass
        
        # Load star calibration parameters if available (separate file)
        star_calibration_params = None
        if 'star_calibration_params' in model_paths:
            try:
                star_calibration_params = joblib.load(model_paths['star_calibration_params'])
            except Exception as e:
                pass
        
        # If star calibration is in calibration_params, extract it
        if calibration_params is not None and 'star_calibration' in calibration_params:
            star_calibration_params = calibration_params['star_calibration']
        elif star_calibration_params is None:
            # Use default star calibration (from notebook analysis)
            star_calibration_params = {
                'global_star_bias': 1.25,
                'player_specific_bias': {}
            }
        
        return {
            'mean_model': mean_model,
            'variance_model': variance_model,
            'calibration_factor': calibration_factor,
            'calibration_params': calibration_params,
            'star_calibration_params': star_calibration_params,
            'features': features
        }
    except Exception as e:
        print(f"Error loading NGBOOST models: {e}")
        return None


def build_ngboost_points_features(
    player_name,
    data,
    current_date,
    projectedStartingFive,
    mainStartingFive,
    teamStarPlayer,
    league_df,
    findOpp,
    predicted_minutes=None,
    predicted_usage=None
):
    """
    Build feature vector for NGBOOST points prediction.
    Includes PREDICTED_MIN and PREDICTED_USG_PCT from XGBoost models.
    """
    player_df = data[data['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
    if player_df.empty:
        return None
    
    team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    last_row = player_df.iloc[-1]
    
    # Get opponent
    opp, home_flag = findOpp(player_name, data, current_date)
    if opp is None:
        return None
    
    opp_df = data[data['TEAM_ABBREVIATION'] == opp]
    if opp_df.empty:
        return None
    
    matchup_df = player_df[player_df['OPP_ABBREVIATION'] == opp]
    
    # Team datasets
    team_df = data[data['TEAM_ABBREVIATION'] == team].drop_duplicates('GAME_ID').sort_values('GAME_DATE')
    opp_team_df = opp_df.drop_duplicates('GAME_ID').sort_values('GAME_DATE')
    
    # Helper functions
    def safe_mean(series):
        return float(series.mean()) if series.size > 0 else 0.0
    
    def safe_std(series):
        return float(series.std()) if series.size > 0 else 0.0
    
    def safe_delta(series, baseline):
        if series.size == 0:
            return 0.0
        return float(series.mean() - baseline)
    
    # Build features dict
    features = {}
    
    # 1. Starting status
    features['STARTING'] = int(player_name in projectedStartingFive.get(team, []))
    
    # 2. Games played
    features['GAMES_PLAYED_TO_DATE'] = len(player_df)
    
    # 3. PTS ceiling/floor L5
    pts_last_5 = player_df['PTS'].tail(5)
    pts_avg = safe_mean(player_df['PTS'])
    pts_ceiling_l5 = float(pts_last_5.max()) if pts_last_5.size > 0 else pts_avg
    pts_floor_l5 = float(pts_last_5.min()) if pts_last_5.size > 0 else pts_avg
    features['PTS_CEILING_L5_DELTA'] = round(pts_ceiling_l5 - pts_avg, 2)
    features['PTS_FLOOR_L5_DELTA'] = round(pts_floor_l5 - pts_avg, 2)
    
    # 4. PREDICTED_MIN and PREDICTED_USG_PCT (from XGBoost models)
    features['PREDICTED_MIN'] = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    features['PREDICTED_USG_PCT'] = float(predicted_usage) if predicted_usage is not None else safe_mean(player_df['USG_PCT'])
    features['PREDICTED_MIN_x_PREDICTED_USG_PCT'] = features['PREDICTED_MIN'] * features['PREDICTED_USG_PCT']
    
    # 5. Baseline PTS stats
    home_pts = safe_mean(player_df[player_df['HOME_GAME'] == 1]['PTS'])
    away_pts = safe_mean(player_df[player_df['HOME_GAME'] == 0]['PTS'])
    matchup_pts = safe_mean(matchup_df['PTS'])
    
    features['PTS_AVG_TO_DATE'] = pts_avg
    features['PTS_STD_5_TO_DATE'] = safe_std(player_df['PTS'].tail(5))
    features['PTS_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                     (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['PTS']) - pts_avg))
    features['PTS_EXPECTATION_LOCATION'] = (home_flag * (home_pts - pts_avg) + 
                                           (1 - home_flag) * (away_pts - pts_avg))
    
    # 5b. PTS breakdown stats (OFF_TOV, 2ND_CHANCE, FB, PAINT)
    pts_off_tov_avg = safe_mean(player_df['PTS_OFF_TOV']) if 'PTS_OFF_TOV' in player_df.columns else 0.0
    pts_2nd_chance_avg = safe_mean(player_df['PTS_2ND_CHANCE']) if 'PTS_2ND_CHANCE' in player_df.columns else 0.0
    pts_fb_avg = safe_mean(player_df['PTS_FB']) if 'PTS_FB' in player_df.columns else 0.0
    pts_paint_avg = safe_mean(player_df['PTS_PAINT']) if 'PTS_PAINT' in player_df.columns else 0.0
    
    features['PTS_2ND_CHANCE_L5_OVER_BASELINE'] = safe_delta(player_df['PTS_2ND_CHANCE'].tail(5), pts_2nd_chance_avg) if 'PTS_2ND_CHANCE' in player_df.columns else 0.0
    features['PTS_2ND_CHANCE_L10_OVER_BASELINE'] = safe_delta(player_df['PTS_2ND_CHANCE'].tail(10), pts_2nd_chance_avg) if 'PTS_2ND_CHANCE' in player_df.columns else 0.0
    features['PTS_FB_L5_OVER_BASELINE'] = safe_delta(player_df['PTS_FB'].tail(5), pts_fb_avg) if 'PTS_FB' in player_df.columns else 0.0
    features['PTS_FB_L10_OVER_BASELINE'] = safe_delta(player_df['PTS_FB'].tail(10), pts_fb_avg) if 'PTS_FB' in player_df.columns else 0.0
    features['PTS_PAINT_L5_OVER_BASELINE'] = safe_delta(player_df['PTS_PAINT'].tail(5), pts_paint_avg) if 'PTS_PAINT' in player_df.columns else 0.0
    features['PTS_PAINT_L10_OVER_BASELINE'] = safe_delta(player_df['PTS_PAINT'].tail(10), pts_paint_avg) if 'PTS_PAINT' in player_df.columns else 0.0
    
    # 6. CFGA stats
    cfga_avg = safe_mean(player_df['CFGA']) if 'CFGA' in player_df.columns else 0.0
    features['CFGA_AVG_TO_DATE'] = cfga_avg
    
    # 7. UFGA stats
    ufga_avg = safe_mean(player_df['UFGA']) if 'UFGA' in player_df.columns else 0.0
    features['UFGA_AVG_TO_DATE'] = ufga_avg
    features['UFGA_L5_OVER_BASELINE'] = safe_delta(player_df['UFGA'].tail(5), ufga_avg) if 'UFGA' in player_df.columns else 0.0
    
    # 8. FGA stats
    fga_avg = safe_mean(player_df['FGA'])
    home_fga = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FGA'])
    away_fga = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FGA'])
    
    features['FGA_AVG_TO_DATE'] = fga_avg
    features['FGA_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                     (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FGA']) - fga_avg))
    features['FGA_EXPECTATION_LOCATION'] = (home_flag * (home_fga - fga_avg) + 
                                           (1 - home_flag) * (away_fga - fga_avg))
    
    # 9. FTA stats
    fta_avg = safe_mean(player_df['FTA'])
    home_fta = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FTA'])
    away_fta = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FTA'])
    
    features['FTA_AVG_TO_DATE'] = fta_avg
    features['FTA_L5_OVER_BASELINE'] = safe_delta(player_df['FTA'].tail(5), fta_avg)
    features['FTA_L10_OVER_BASELINE'] = safe_delta(player_df['FTA'].tail(10), fta_avg)
    features['FTA_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                      (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FTA']) - fta_avg))
    features['FTA_EXPECTATION_LOCATION'] = (home_flag * (home_fta - fta_avg) + 
                                           (1 - home_flag) * (away_fta - fta_avg))
    
    # 10. FG3A stats
    fg3a_avg = safe_mean(player_df['FG3A'])
    home_fg3a = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FG3A'])
    away_fg3a = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FG3A'])
    
    features['FG3A_AVG_TO_DATE'] = fg3a_avg
    features['FG3A_L5_OVER_BASELINE'] = safe_delta(player_df['FG3A'].tail(5), fg3a_avg)
    features['FG3A_L10_OVER_BASELINE'] = safe_delta(player_df['FG3A'].tail(10), fg3a_avg)
    features['FG3A_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                       (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FG3A']) - fg3a_avg))
    features['FG3A_EXPECTATION_LOCATION'] = (home_flag * (home_fg3a - fg3a_avg) + 
                                            (1 - home_flag) * (away_fg3a - fg3a_avg))
    
    # 9. PLUS_MINUS stats
    pm_avg = safe_mean(player_df['PLUS_MINUS'])
    home_pm = safe_mean(player_df[player_df['HOME_GAME'] == 1]['PLUS_MINUS'])
    away_pm = safe_mean(player_df[player_df['HOME_GAME'] == 0]['PLUS_MINUS'])
    
    features['PLUS_MINUS_AVG_TO_DATE'] = pm_avg
    features['PLUS_MINUS_L5_OVER_BASELINE'] = safe_delta(player_df['PLUS_MINUS'].tail(5), pm_avg)
    features['PLUS_MINUS_L10_OVER_BASELINE'] = safe_delta(player_df['PLUS_MINUS'].tail(10), pm_avg)
    features['PLUS_MINUS_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                             (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['PLUS_MINUS']) - pm_avg))
    features['PLUS_MINUS_EXPECTATION_LOCATION'] = (home_flag * (home_pm - pm_avg) + 
                                                   (1 - home_flag) * (away_pm - pm_avg))
    
    # 10. FG_PCT stats
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    home_fg_pct = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FG_PCT'])
    away_fg_pct = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FG_PCT'])
    matchup_fg_pct = safe_mean(matchup_df['FG_PCT'])
    
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    features['FG_PCT_L5_OVER_BASELINE'] = safe_delta(player_df['FG_PCT'].tail(5), fg_pct_avg)
    features['FG_PCT_EXPECTATION_LOCATION'] = (home_flag * (home_fg_pct - fg_pct_avg) + 
                                               (1 - home_flag) * (away_fg_pct - fg_pct_avg))
    features['MATCHUP_FG_PCT_DELTA'] = matchup_fg_pct - fg_pct_avg if not matchup_df.empty else 0.0
    
    # 11. FG3_PCT stats
    fg3_pct_avg = safe_mean(player_df['FG3_PCT'])
    home_fg3_pct = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FG3_PCT'])
    away_fg3_pct = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FG3_PCT'])
    matchup_fg3_pct = safe_mean(matchup_df['FG3_PCT'])
    
    features['FG3_PCT_AVG_TO_DATE'] = fg3_pct_avg
    features['MATCHUP_FG3_PCT_DELTA'] = matchup_fg3_pct - fg3_pct_avg if not matchup_df.empty else 0.0
    
    # 12. FT_PCT stats
    ft_pct_avg = safe_mean(player_df['FT_PCT'])
    home_ft_pct = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FT_PCT'])
    away_ft_pct = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FT_PCT'])
    matchup_ft_pct = safe_mean(matchup_df['FT_PCT'])
    
    features['FT_PCT_AVG_TO_DATE'] = ft_pct_avg
    features['FT_PCT_L5_OVER_BASELINE'] = safe_delta(player_df['FT_PCT'].tail(5), ft_pct_avg)
    features['FT_PCT_L10_OVER_BASELINE'] = safe_delta(player_df['FT_PCT'].tail(10), ft_pct_avg)
    features['FT_PCT_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                         (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FT_PCT']) - ft_pct_avg))
    features['FT_PCT_EXPECTATION_LOCATION'] = (home_flag * (home_ft_pct - ft_pct_avg) + 
                                               (1 - home_flag) * (away_ft_pct - ft_pct_avg))
    
    # 13. TS_PCT stats
    ts_pct_avg = safe_mean(player_df['TS_PCT'])
    home_ts_pct = safe_mean(player_df[player_df['HOME_GAME'] == 1]['TS_PCT'])
    away_ts_pct = safe_mean(player_df[player_df['HOME_GAME'] == 0]['TS_PCT'])
    matchup_ts_pct = safe_mean(matchup_df['TS_PCT'])
    
    features['TS_PCT_AVG_TO_DATE'] = ts_pct_avg
    features['TS_PCT_L5_OVER_BASELINE'] = safe_delta(player_df['TS_PCT'].tail(5), ts_pct_avg)
    features['TS_PCT_L10_OVER_BASELINE'] = safe_delta(player_df['TS_PCT'].tail(10), ts_pct_avg)
    features['TS_PCT_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                        (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['TS_PCT']) - ts_pct_avg))
    features['TS_PCT_EXPECTATION_LOCATION'] = (home_flag * (home_ts_pct - ts_pct_avg) + 
                                               (1 - home_flag) * (away_ts_pct - ts_pct_avg))
    features['MATCHUP_TS_PCT_DELTA'] = matchup_ts_pct - ts_pct_avg if not matchup_df.empty else 0.0
    
    # 14. Variance Stability (vol_10 / vol_40)
    def calculate_variance_stability(player_df, stat_col):
        """Calculate variance stability: vol_10 / (vol_40 + 0.001)"""
        vol_10_col = f'{stat_col}_VOLATILITY_10_TO_DATE'
        vol_40_col = f'{stat_col}_VOLATILITY_40_TO_DATE'
        
        # Check if columns exist in dataframe
        if vol_10_col in player_df.columns and vol_40_col in player_df.columns:
            vol_10 = player_df[vol_10_col].iloc[-1] if len(player_df) > 0 else 0.0
            vol_40 = player_df[vol_40_col].iloc[-1] if len(player_df) > 0 else 0.0
            if pd.isna(vol_10) or pd.isna(vol_40):
                return 0.0
            if vol_40 == 0:
                return 0.0
            return round(vol_10 / (vol_40 + 0.001), 2)
        
        # If columns don't exist, calculate from raw data
        if stat_col not in player_df.columns:
            return 0.0
        
        # Calculate vol_10 and vol_40 from raw data
        sorted_df = player_df.sort_values('GAME_DATE')
        vol_10 = safe_std(sorted_df[stat_col].tail(10)) if len(sorted_df) >= 10 else 0.0
        vol_40 = safe_std(sorted_df[stat_col].tail(40)) if len(sorted_df) >= 40 else safe_std(sorted_df[stat_col])
        
        if vol_40 == 0:
            return 0.0
        return round(vol_10 / (vol_40 + 0.001), 2)
    
    features['PTS_VARIANCE_STABILITY'] = calculate_variance_stability(player_df, 'PTS')
    features['FGA_VARIANCE_STABILITY'] = calculate_variance_stability(player_df, 'FGA')
    features['FTA_VARIANCE_STABILITY'] = calculate_variance_stability(player_df, 'FTA')
    features['FG3A_VARIANCE_STABILITY'] = calculate_variance_stability(player_df, 'FG3A')
    features['TS_PCT_VARIANCE_STABILITY'] = calculate_variance_stability(player_df, 'TS_PCT')
    
    # 15. Team/Opponent context
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    league_off_avg = safe_mean(league_df['OFF_RATING']) if 'OFF_RATING' in league_df.columns else 110.0
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    team_pace = safe_mean(team_df['TEAM_PACE'])
    team_off = safe_mean(team_df['TEAM_OFF_RATING'])
    opp_pace = safe_mean(opp_team_df['TEAM_PACE'])
    opp_def = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    
    features['TEAM_OFF_RATING_OVER_LEAGUE_AVG'] = team_off - league_off_avg
    features['TEAM_PACE_OVER_LEAGUE_AVG'] = team_pace - league_pace_avg
    features['EXPECTED_PACE'] = (team_pace + opp_pace) / 2
    features['OPP_PACE_OVER_LEAGUE_AVG'] = opp_pace - league_pace_avg
    
    # 16. Positional defense (simplified - would need position data)
    # For now, use team defense as proxy
    features['GUARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    features['FORWARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    return features


def calibrate_star_player_prediction(
    player_name,
    predicted_points,
    team,
    teamStarPlayer=None,
    player_df=None,
    data=None,
    calibration_params=None
):
    """
    Calibrate predictions for star players who tend to underperform.
    
    Args:
        player_name: Player name
        predicted_points: Raw prediction from NGBOOST model
        team: Team abbreviation (can be None, will try to get from data)
        teamStarPlayer: Dict mapping team to star player name
        player_df: Player's historical data (optional)
        data: Full dataset (for getting team if not provided)
        calibration_params: Star calibration parameters dict with 'global_star_bias' and 'player_specific_bias'
    
    Returns:
        Calibrated prediction
    """
    if teamStarPlayer is None:
        try:
            from PRODUCTION.teamInfo import teamStarPlayer
        except ImportError:
            return predicted_points
    
    # Get team if not provided
    if team is None and data is not None:
        player_data = data[data['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
        if not player_data.empty:
            team = player_data['TEAM_ABBREVIATION'].iloc[-1]
    
    if team is None:
        return predicted_points
    
    # Check if player is the team's star player
    star_player = teamStarPlayer.get(team, None)
    if star_player is None or player_name != star_player:
        return predicted_points
    
    # Get calibration parameters
    global_star_bias = 1.25  # Default from analysis
    player_specific_bias = {}
    
    if calibration_params is not None:
        global_star_bias = calibration_params.get('global_star_bias', 1.25)
        player_specific_bias = calibration_params.get('player_specific_bias', {})
    
    # Use player-specific bias if available, otherwise use global
    if player_specific_bias and player_name in player_specific_bias:
        bias = player_specific_bias[player_name]
    else:
        bias = global_star_bias
    
    # Apply calibration (add bias to prediction)
    calibrated = predicted_points + bias
    
    # Ensure non-negative
    return max(0.0, calibrated)


def predict_points_ngboost(
    player_name,
    data,
    date,
    projectedStartingFive=None,
    mainStartingFive=None,
    teamStarPlayer=None,
    league_df=None,
    findOpp=None,
    predicted_minutes=None,
    predicted_usage=None,
    model_wrapper=None,
    model_paths=None
):
    """
    Predict points using NGBOOST models.
    
    Args:
        player_name: Player name
        data: Historical game data
        date: Current date
        predicted_minutes: Predicted minutes from XGBoost MIN model
        predicted_usage: Predicted usage from XGBoost USG model
        model_wrapper: Pre-loaded model wrapper dict
        model_paths: Paths to model files (if model_wrapper not provided)
    
    Returns:
        dict with 'predicted_points', 'mu', 'sigma', 'variance'
    """
    # Load models if not provided
    if model_wrapper is None:
        model_wrapper = load_trained_ngboost_models(model_paths)
    
    if model_wrapper is None:
        raise ValueError("No trained NGBOOST models available. Please train the models first.")
    
    mean_model = model_wrapper['mean_model']
    variance_model = model_wrapper['variance_model']
    calibration_factor = model_wrapper['calibration_factor']
    calibration_params = model_wrapper.get('calibration_params', None)
    star_calibration_params = model_wrapper.get('star_calibration_params', None)
    features_list = model_wrapper['features']
    
    # Build features
    feature_dict = build_ngboost_points_features(
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
    
    if feature_dict is None:
        return None
    
    # Convert feature dict to DataFrame in correct order
    feature_vector = [feature_dict.get(f, 0.0) for f in features_list]
    feature_df = pd.DataFrame([feature_vector], columns=features_list)
    
    # Handle NaN and inf values
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)
    
    # Determine if we should use score-dependent calibration
    use_score_dependent = False
    score_dependent_params = None
    
    if calibration_params is not None and calibration_params.get('use_score_dependent', False):
        use_score_dependent = True
        score_dependent_params = calibration_params.get('score_dependent_params', None)
        # Use base_factor from calibration_params if available, otherwise use calibration_factor
        if 'base_factor' in calibration_params:
            calibration_factor = calibration_params['base_factor']
    
    # Predict using NGBOOST models
    try:
        mu, variance = predict_mean_variance_split(
            mean_model=mean_model,
            variance_model=variance_model,
            df=feature_df,
            features=features_list,
            calibration_factor=calibration_factor,
            prediction_type='mean',
            use_score_dependent_calibration=use_score_dependent,
            score_dependent_params=score_dependent_params
        )
        
        # Convert to scalars if arrays
        mu = float(mu[0] if isinstance(mu, (np.ndarray, pd.Series)) else mu)
        variance = float(variance[0] if isinstance(variance, (np.ndarray, pd.Series)) else variance)
        sigma = np.sqrt(variance)
        
        # Ensure non-negative
        mu = max(0.0, mu)
        sigma = max(0.1, sigma)
        
        # Check if using Negative Binomial distribution
        # Extract n and p parameters for Negative Binomial probability calculations
        n_param = None
        p_param = None
        distribution_type = 'normal'  # Default
        
        # Check the model's distribution type
        if hasattr(mean_model, 'Dist'):
            dist_name = mean_model.Dist.__name__ if hasattr(mean_model.Dist, '__name__') else str(mean_model.Dist)
            if 'NegativeBinomial' in dist_name or 'NegBin' in dist_name:
                distribution_type = 'negative_binomial'
                # For Negative Binomial: mean = n(1-p)/p, variance = n(1-p)/p²
                # Solving: p = mean/variance, n = mean²/(variance - mean)
                if variance > mu and mu > 0:
                    p_param = mu / variance
                    n_param = (mu ** 2) / (variance - mu)
                    # Ensure valid parameters
                    if p_param <= 0 or p_param >= 1 or n_param <= 0:
                        p_param = None
                        n_param = None
        
        # Apply global mean adjustment if model is systematically underpredicting
        # This can be tuned based on validation performance
        mean_adjustment_factor = 1.0
        if calibration_params is not None and 'mean_adjustment_factor' in calibration_params:
            mean_adjustment_factor = calibration_params.get('mean_adjustment_factor', 1.0)
        
        # Apply mean adjustment
        mu = mu * mean_adjustment_factor
        
        # Recalculate n and p if adjusted
        if distribution_type == 'negative_binomial' and n_param is not None and p_param is not None:
            # Recalculate with adjusted mu
            if variance > mu and mu > 0:
                p_param = mu / variance
                n_param = (mu ** 2) / (variance - mu)
                if p_param <= 0 or p_param >= 1 or n_param <= 0:
                    p_param = None
                    n_param = None
        
        # Get team for star player calibration
        player_df = data[data['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
        team = player_df['TEAM_ABBREVIATION'].iloc[-1] if not player_df.empty else None
        
        # Apply star player calibration if available
        calibrated_mu = calibrate_star_player_prediction(
            player_name=player_name,
            predicted_points=mu,
            team=team,
            teamStarPlayer=teamStarPlayer,
            player_df=player_df,
            data=data,
            calibration_params=star_calibration_params
        )
        
        # Recalculate n and p after star calibration if needed
        if distribution_type == 'negative_binomial' and n_param is not None and p_param is not None:
            # Recalculate with calibrated mu
            if variance > calibrated_mu and calibrated_mu > 0:
                p_param = calibrated_mu / variance
                n_param = (calibrated_mu ** 2) / (variance - calibrated_mu)
                if p_param <= 0 or p_param >= 1 or n_param <= 0:
                    p_param = None
                    n_param = None
        
        result = {
            'predicted_points': calibrated_mu,
            'mu': calibrated_mu,
            'sigma': sigma,
            'variance': variance,
            'distribution': distribution_type
        }
        
        # Add Negative Binomial parameters if available
        if n_param is not None and p_param is not None:
            result['n'] = n_param
            result['p'] = p_param
        
        return result
    except Exception as e:
        print(f"Error predicting points for {player_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

