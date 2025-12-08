import pandas as pd
import numpy as np
from src.utils.helper_functions import findOpp


def build_fga_features(
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
    fga_avg = safe_mean(player_df['FGA'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    
    # Get predicted_minutes for FGA_PER_MIN calculation (used internally but not in base features)
    predicted_min = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    
    # Calculate FGA_PER_MIN (this will be returned in the dict but added separately during training)
    fga_per_min = round(fga_avg / (predicted_min + 1e-8), 3) if predicted_min > 0 else 0.0
    
    # Calculate team FGA rank
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        team_fga_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_fga_avgs[team_player_name] = safe_mean(team_player_df['FGA'])
        
        if team_fga_avgs:
            fga_series = pd.Series(team_fga_avgs)
            fga_ranks = fga_series.rank(method='dense', ascending=False)
            fga_team_rank = float(fga_ranks.get(player_name, len(team_fga_avgs) + 1))
        else:
            fga_team_rank = 1.0
    else:
        fga_team_rank = 1.0
    
    # Calculate various averages and deltas
    fga_std_10 = safe_std(player_df['FGA'].tail(10)) if len(player_df) >= 10 else 0.0
    fga_last_5 = player_df['FGA'].tail(5)
    fga_last_10 = player_df['FGA'].tail(10)
    fga_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FGA'])
    
    cfga_avg = safe_mean(player_df['CFGA']) if 'CFGA' in player_df.columns else 0.0
    ufga_avg = safe_mean(player_df['UFGA']) if 'UFGA' in player_df.columns else 0.0
    ufga_last_5 = player_df['UFGA'].tail(5) if 'UFGA' in player_df.columns else pd.Series()
    
    fg3a_avg = safe_mean(player_df['FG3A'])
    fg3a_last_5 = player_df['FG3A'].tail(5)
    fg3a_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FG3A'])
    
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    fg_pct_last_5 = player_df['FG_PCT'].tail(5)
    
    ts_pct_avg = safe_mean(player_df['TS_PCT'])
    ts_pct_last_5 = player_df['TS_PCT'].tail(5)
    
    pace_avg = safe_mean(player_df['PACE']) if 'PACE' in player_df.columns else safe_mean(team_df['TEAM_PACE'])
    pace_last_5 = player_df['PACE'].tail(5) if 'PACE' in player_df.columns else team_df['TEAM_PACE'].tail(5)
    
    fta_avg = safe_mean(player_df['FTA'])
    fta_last_5 = player_df['FTA'].tail(5)
    
    ast_avg = safe_mean(player_df['AST']) if 'AST' in player_df.columns else 0.0
    tov_avg = safe_mean(player_df['TOV']) if 'TOV' in player_df.columns else 0.0
    
    # Star out flag
    star_out_flag = int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, []))
    
    # Team context calculations
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    league_off_avg = safe_mean(league_df['OFF_RATING']) if 'OFF_RATING' in league_df.columns else 110.0
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    team_pace = safe_mean(team_df['TEAM_PACE'])
    team_off = safe_mean(team_df['TEAM_OFF_RATING'])
    opp_pace = safe_mean(opp_team_df['TEAM_PACE'])
    opp_def = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    
    # Build features dict in the EXACT order from FGA_features
    features = {}
    
    # 1: STARTING_X_FGA
    features['STARTING_X_FGA'] = round(starting_flag * fga_avg, 2)
    
    # 2: GAMES_PLAYED_TO_DATE
    features['GAMES_PLAYED_TO_DATE'] = len(player_df)
    
    # 3: FGA_TEAM_RANK
    features['FGA_TEAM_RANK'] = fga_team_rank
    
    # 4: FGA_AVG_TO_DATE
    features['FGA_AVG_TO_DATE'] = fga_avg
    
    # 5: FGA_STD_10_TO_DATE
    features['FGA_STD_10_TO_DATE'] = fga_std_10
    
    # 6: FGA_L5_OVER_BASELINE
    features['FGA_L5_OVER_BASELINE'] = safe_delta(fga_last_5, fga_avg)
    
    # 7: FGA_L10_OVER_BASELINE
    features['FGA_L10_OVER_BASELINE'] = safe_delta(fga_last_10, fga_avg)
    
    # 8: FGA_BOOST_STAR_OUT
    features['FGA_BOOST_STAR_OUT'] = star_out_flag * (fga_star_out - fga_avg)
    
    # 9: CFGA_AVG_TO_DATE
    features['CFGA_AVG_TO_DATE'] = cfga_avg
    
    # 10: UFGA_AVG_TO_DATE
    features['UFGA_AVG_TO_DATE'] = ufga_avg
    
    # 11: UFGA_L5_OVER_BASELINE
    features['UFGA_L5_OVER_BASELINE'] = safe_delta(ufga_last_5, ufga_avg)
    
    # 12: FG3A_AVG_TO_DATE
    features['FG3A_AVG_TO_DATE'] = fg3a_avg
    
    # 13: FG3A_L5_OVER_BASELINE
    features['FG3A_L5_OVER_BASELINE'] = safe_delta(fg3a_last_5, fg3a_avg)
    
    # 14: FG3A_BOOST_STAR_OUT
    features['FG3A_BOOST_STAR_OUT'] = star_out_flag * (fg3a_star_out - fg3a_avg)
    
    # 15: FG_PCT_AVG_TO_DATE
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    
    # 16: FG_PCT_L5_OVER_BASELINE
    features['FG_PCT_L5_OVER_BASELINE'] = safe_delta(fg_pct_last_5, fg_pct_avg)
    
    # 17: TS_PCT_AVG_TO_DATE
    features['TS_PCT_AVG_TO_DATE'] = ts_pct_avg
    
    # 18: TS_PCT_L5_OVER_BASELINE
    features['TS_PCT_L5_OVER_BASELINE'] = safe_delta(ts_pct_last_5, ts_pct_avg)
    
    # 19: PACE_AVG_TO_DATE
    features['PACE_AVG_TO_DATE'] = pace_avg
    
    # 20: PACE_L5_OVER_BASELINE
    features['PACE_L5_OVER_BASELINE'] = safe_delta(pace_last_5, pace_avg)
    
    # 21: FTA_AVG_TO_DATE
    features['FTA_AVG_TO_DATE'] = fta_avg
    
    # 22: FTA_L5_OVER_BASELINE
    features['FTA_L5_OVER_BASELINE'] = safe_delta(fta_last_5, fta_avg)
    
    # 23: AST_AVG_TO_DATE
    features['AST_AVG_TO_DATE'] = ast_avg
    
    # 24: TOV_AVG_TO_DATE
    features['TOV_AVG_TO_DATE'] = tov_avg
    
    # 25: FGA_VARIANCE_STABILITY
    if 'FGA_VARIANCE_STABILITY' in player_df.columns:
        features['FGA_VARIANCE_STABILITY'] = float(player_df['FGA_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FGA_VARIANCE_STABILITY'] = 0.0
    
    # 26: FG3A_VARIANCE_STABILITY
    if 'FG3A_VARIANCE_STABILITY' in player_df.columns:
        features['FG3A_VARIANCE_STABILITY'] = float(player_df['FG3A_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FG3A_VARIANCE_STABILITY'] = 0.0
    
    # 27: TEAM_OFF_RATING_OVER_LEAGUE_AVG
    features['TEAM_OFF_RATING_OVER_LEAGUE_AVG'] = team_off - league_off_avg
    
    # 28: TEAM_PACE_OVER_LEAGUE_AVG
    features['TEAM_PACE_OVER_LEAGUE_AVG'] = team_pace - league_pace_avg
    
    # 29: EXPECTED_PACE
    expected_pace = (team_pace + opp_pace) / 2
    features['EXPECTED_PACE'] = expected_pace
    
    # 30: OPP_PACE_OVER_LEAGUE_AVG
    features['OPP_PACE_OVER_LEAGUE_AVG'] = opp_pace - league_pace_avg
    
    # 31: OPP_DEF_RATING_OVER_LEAGUE_AVG
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    # 32: GUARD_DEF_RATING_OVER_LEAGUE_AVG
    features['GUARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    # 33: FORWARD_DEF_RATING_OVER_LEAGUE_AVG
    features['FORWARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    features['FGA_PER_MIN'] = fga_per_min
    return features