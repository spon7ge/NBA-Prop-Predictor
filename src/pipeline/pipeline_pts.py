import pandas as pd
import numpy as np
from src.utils.helper_functions import findOpp


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
    predicted_usage=None,
    predicted_fga=None
):
    player_df = data[data['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
    if player_df.empty:
        return None
    
    team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    
    opp, home_flag = findOpp(player_name, data, current_date)
    if opp is None:
        return None
    
    opp_df = data[data['TEAM_ABBREVIATION'] == opp]
    if opp_df.empty:
        return None
    
    team_df = data[data['TEAM_ABBREVIATION'] == team].drop_duplicates('GAME_ID').sort_values('GAME_DATE')
    opp_team_df = opp_df.drop_duplicates('GAME_ID').sort_values('GAME_DATE')
    
    def safe_mean(series):
        return float(series.mean()) if series.size > 0 else 0.0
    
    def safe_std(series):
        return float(series.std()) if series.size > 0 else 0.0
    
    def safe_delta(series, baseline):
        if series.size == 0:
            return 0.0
        return float(series.mean() - baseline)
    
    # Calculate values needed for multiple features
    pts_last_5 = player_df['PTS'].tail(5)
    pts_avg = safe_mean(player_df['PTS'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    
    # Predicted upstream values
    predicted_min = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    predicted_usg_pct = float(predicted_usage) if predicted_usage is not None else safe_mean(player_df['USG_PCT'])
    predicted_fga = float(predicted_fga) if predicted_fga is not None else safe_mean(player_df['FGA'])
    
    # Calculate expected_pace early (needed for multiple features)
    expected_pace = (safe_mean(team_df['TEAM_PACE']) + safe_mean(opp_team_df['TEAM_PACE'])) / 2
    
    # Calculate league averages
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    league_off_avg = safe_mean(league_df['OFF_RATING']) if 'OFF_RATING' in league_df.columns else 110.0
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    team_off = safe_mean(team_df['TEAM_OFF_RATING'])
    opp_def = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    
    # Calculate other values needed
    pts_l5_over_baseline = safe_delta(pts_last_5, pts_avg)
    ts_pct_avg = safe_mean(player_df['TS_PCT'])
    opp_def_rating_over_league = opp_def - league_def_avg
    
    # Calculate team PTS rank
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        team_pts_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_pts_avgs[team_player_name] = safe_mean(team_player_df['PTS'])
        
        if team_pts_avgs:
            pts_series = pd.Series(team_pts_avgs)
            pts_ranks = pts_series.rank(method='dense', ascending=False)
            pts_team_rank = float(pts_ranks.get(player_name, len(team_pts_avgs) + 1))
        else:
            pts_team_rank = 1.0
    else:
        pts_team_rank = 1.0
    
    # Calculate various averages and deltas
    pts_ceiling_l5 = float(pts_last_5.max()) if pts_last_5.size > 0 else pts_avg
    pts_std_5 = safe_std(pts_last_5) if len(player_df) >= 5 else 0.0
    
    cfga_avg = safe_mean(player_df['CFGA']) if 'CFGA' in player_df.columns else 0.0
    ufga_avg = safe_mean(player_df['UFGA']) if 'UFGA' in player_df.columns else 0.0
    ufga_last_5 = player_df['UFGA'].tail(5) if 'UFGA' in player_df.columns else pd.Series()
    
    fta_avg = safe_mean(player_df['FTA'])
    fta_last_5 = player_df['FTA'].tail(5)
    fta_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FTA'])
    home_fta = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FTA'])
    away_fta = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FTA'])
    
    fg3a_avg = safe_mean(player_df['FG3A'])
    fg3a_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FG3A'])
    
    pm_avg = safe_mean(player_df['PLUS_MINUS'])
    pm_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['PLUS_MINUS'])
    
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    fg_pct_last_5 = player_df['FG_PCT'].tail(5)
    
    fg3_pct_avg = safe_mean(player_df['FG3_PCT']) if 'FG3_PCT' in player_df.columns else 0.0
    
    ft_pct_avg = safe_mean(player_df['FT_PCT'])
    ft_pct_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FT_PCT'])
    
    ts_pct_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['TS_PCT'])
    
    # Star out calculations
    star_out_flag = int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, []))
    pts_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['PTS'])
    
    # Build features dict in the EXACT order from PTS_features
    features = {}
    
    # 1-4: Predicted upstream values and interaction features (most important)
    features['PREDICTED_MIN'] = predicted_min
    features['PREDICTED_USG_PCT'] = predicted_usg_pct
    features['PREDICTED_FGA'] = predicted_fga
    features['PREDICTED_MIN_x_PREDICTED_USG_PCT'] = predicted_min * predicted_usg_pct
    
    # 5: PTS_PER_MIN
    features['PTS_PER_MIN'] = round(pts_avg / (predicted_min + 1e-8), 3) if predicted_min > 0 else 0.0
    
    # 6: PTS_L5_OVER_BASELINE_x_PREDICTED_MIN
    features['PTS_L5_OVER_BASELINE_x_PREDICTED_MIN'] = pts_l5_over_baseline * predicted_min
    
    # 7-8: EXPECTED_PACE interactions
    features['EXPECTED_PACE_x_PREDICTED_USG_PCT'] = expected_pace * predicted_usg_pct
    features['EXPECTED_PACE_x_PREDICTED_FGA'] = expected_pace * predicted_fga
    
    # 9: PREDICTED_FGA_x_TS_PCT_AVG_TO_DATE
    features['PREDICTED_FGA_x_TS_PCT_AVG_TO_DATE'] = predicted_fga * ts_pct_avg
    
    # 10: OPP_DEF_RATING_OVER_LEAGUE_AVG_x_PTS_AVG_TO_DATE
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG_x_PTS_AVG_TO_DATE'] = opp_def_rating_over_league * pts_avg
    
    # 11: STARTING_X_PTS
    features['STARTING_X_PTS'] = round(starting_flag * pts_avg, 2)
    
    # 12: GAMES_PLAYED_TO_DATE
    features['GAMES_PLAYED_TO_DATE'] = len(player_df)
    
    # 13: PTS_CEILING_L5_DELTA
    features['PTS_CEILING_L5_DELTA'] = round(pts_ceiling_l5 - pts_avg, 2)
    
    # 14: PTS_TEAM_RANK
    features['PTS_TEAM_RANK'] = pts_team_rank
    
    # 15: PTS_AVG_TO_DATE
    features['PTS_AVG_TO_DATE'] = pts_avg
    
    # 16: PTS_STD_5_TO_DATE
    features['PTS_STD_5_TO_DATE'] = pts_std_5
    
    # 17: PTS_BOOST_STAR_OUT
    features['PTS_BOOST_STAR_OUT'] = star_out_flag * (pts_star_out - pts_avg)
    
    # 18: CFGA_AVG_TO_DATE
    features['CFGA_AVG_TO_DATE'] = cfga_avg
    
    # 19-20: UFGA features
    features['UFGA_AVG_TO_DATE'] = ufga_avg
    features['UFGA_L5_OVER_BASELINE'] = safe_delta(ufga_last_5, ufga_avg)
    
    # 21-24: FTA features
    features['FTA_AVG_TO_DATE'] = fta_avg
    features['FTA_L5_OVER_BASELINE'] = safe_delta(fta_last_5, fta_avg)
    features['FTA_BOOST_STAR_OUT'] = star_out_flag * (fta_star_out - fta_avg)
    features['FTA_EXPECTATION_LOCATION'] = (home_flag * (home_fta - fta_avg) + 
                                           (1 - home_flag) * (away_fta - fta_avg))
    
    # 25-27: FG3A features
    features['FG3A_AVG_TO_DATE'] = fg3a_avg
    features['FG3A_L5_OVER_BASELINE'] = safe_delta(player_df['FG3A'].tail(5), fg3a_avg)
    features['FG3A_BOOST_STAR_OUT'] = star_out_flag * (fg3a_star_out - fg3a_avg)
    
    # 28-29: PLUS_MINUS features
    features['PLUS_MINUS_AVG_TO_DATE'] = pm_avg
    features['PLUS_MINUS_BOOST_STAR_OUT'] = star_out_flag * (pm_star_out - pm_avg)
    
    # 30-31: FG_PCT features
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    features['FG_PCT_L5_OVER_BASELINE'] = safe_delta(fg_pct_last_5, fg_pct_avg)
    
    # 32: FG3_PCT_AVG_TO_DATE
    features['FG3_PCT_AVG_TO_DATE'] = fg3_pct_avg
    
    # 33-34: FT_PCT features
    features['FT_PCT_AVG_TO_DATE'] = ft_pct_avg
    features['FT_PCT_BOOST_STAR_OUT'] = star_out_flag * (ft_pct_star_out - ft_pct_avg)
    
    # 35-36: TS_PCT features
    features['TS_PCT_AVG_TO_DATE'] = ts_pct_avg
    features['TS_PCT_BOOST_STAR_OUT'] = star_out_flag * (ts_pct_star_out - ts_pct_avg)
    
    # 37-41: Variance stability features
    if 'PTS_VARIANCE_STABILITY' in player_df.columns:
        features['PTS_VARIANCE_STABILITY'] = float(player_df['PTS_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['PTS_VARIANCE_STABILITY'] = 0.0
    
    if 'FGA_VARIANCE_STABILITY' in player_df.columns:
        features['FGA_VARIANCE_STABILITY'] = float(player_df['FGA_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FGA_VARIANCE_STABILITY'] = 0.0
    
    if 'FTA_VARIANCE_STABILITY' in player_df.columns:
        features['FTA_VARIANCE_STABILITY'] = float(player_df['FTA_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FTA_VARIANCE_STABILITY'] = 0.0
    
    if 'FG3A_VARIANCE_STABILITY' in player_df.columns:
        features['FG3A_VARIANCE_STABILITY'] = float(player_df['FG3A_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FG3A_VARIANCE_STABILITY'] = 0.0
    
    if 'TS_PCT_VARIANCE_STABILITY' in player_df.columns:
        features['TS_PCT_VARIANCE_STABILITY'] = float(player_df['TS_PCT_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['TS_PCT_VARIANCE_STABILITY'] = 0.0
    
    # 42: TEAM_OFF_RATING_OVER_LEAGUE_AVG
    features['TEAM_OFF_RATING_OVER_LEAGUE_AVG'] = team_off - league_off_avg
    
    # 43: EXPECTED_PACE
    features['EXPECTED_PACE'] = expected_pace
    
    # 44-45: Defense rating features
    features['GUARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    features['FORWARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    return features