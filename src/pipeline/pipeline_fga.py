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
    
    matchup_df = player_df[player_df['OPP_ABBREVIATION'] == opp]
    
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
    
    features = {}
    
    # Baseline shot volume (needed for player type classification)
    fga_avg = safe_mean(player_df['FGA'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    
    # Core identifiers
    # 1 — STARTING_X_FGA (new)
    features['STARTING_X_FGA'] = round(starting_flag * fga_avg, 2)
    
    # 2 — GAMES_PLAYED_TO_DATE
    features['GAMES_PLAYED_TO_DATE'] = len(player_df)
    
    # Player type
    # 3 — HIGH_FGA_PLAYER
    features['HIGH_FGA_PLAYER'] = int(fga_avg >= 15)
    
    # 4 — MEDIUM_FGA_PLAYER
    features['MEDIUM_FGA_PLAYER'] = int(fga_avg >= 7 and fga_avg < 15)
    
    # 5 — LOW_FGA_PLAYER
    features['LOW_FGA_PLAYER'] = int(fga_avg < 7)
    
    # 6 — FGA_TEAM_RANK (new)
    # Calculate FGA rank among all players on the team
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        # Calculate FGA_AVG_TO_DATE for each player on the team
        team_fga_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_fga_avgs[team_player_name] = safe_mean(team_player_df['FGA'])
        
        # Create a series and rank (ascending=False means rank 1 = highest FGA)
        if team_fga_avgs:
            fga_series = pd.Series(team_fga_avgs)
            fga_ranks = fga_series.rank(method='dense', ascending=False)
            fga_team_rank = float(fga_ranks.get(player_name, len(team_fga_avgs) + 1))
        else:
            fga_team_rank = 1.0
    else:
        fga_team_rank = 1.0
    features['FGA_TEAM_RANK'] = fga_team_rank
    
    # 7 — PERCENTAGE_OF_TEAM_FGA (new)
    # Calculate team average FGA to date
    team_fga_avg = safe_mean(team_df['TEAM_FGA_AVG_TO_DATE']) if 'TEAM_FGA_AVG_TO_DATE' in team_df.columns else safe_mean(team_df['TEAM_FGA']) if 'TEAM_FGA' in team_df.columns else 85.0
    epsilon = 1e-8
    percentage_of_team_fga = round(fga_avg / (team_fga_avg + epsilon), 4) if team_fga_avg > 0 else 0.0
    features['PERCENTAGE_OF_TEAM_FGA'] = percentage_of_team_fga
    
    # Baseline shot volume (continued)
    fga_std_5 = safe_std(player_df['FGA'].tail(5)) if len(player_df) >= 5 else 0.0
    fga_last_5 = player_df['FGA'].tail(5)
    
    # 8 — FGA_AVG_TO_DATE
    features['FGA_AVG_TO_DATE'] = fga_avg
    
    # 9 — FGA_STD_5_TO_DATE
    features['FGA_STD_5_TO_DATE'] = fga_std_5
    
    # 10 — FGA_L5_OVER_BASELINE
    features['FGA_L5_OVER_BASELINE'] = safe_delta(fga_last_5, fga_avg)
    
    # Situational boosts
    # 11 — FGA_BOOST_STAR_OUT
    fga_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FGA'])
    features['FGA_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                      (fga_star_out - fga_avg))
    
    # 12 — FGA_EXPECTATION_LOCATION
    home_fga = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FGA'])
    away_fga = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FGA'])
    features['FGA_EXPECTATION_LOCATION'] = (home_flag * (home_fga - fga_avg) + 
                                           (1 - home_flag) * (away_fga - fga_avg))
    
    # Shot types
    # 13 — CFGA_AVG_TO_DATE
    cfga_avg = safe_mean(player_df['CFGA']) if 'CFGA' in player_df.columns else 0.0
    features['CFGA_AVG_TO_DATE'] = cfga_avg
    
    # 14 — UFGA_AVG_TO_DATE
    ufga_avg = safe_mean(player_df['UFGA']) if 'UFGA' in player_df.columns else 0.0
    # 15 — UFGA_L5_OVER_BASELINE
    ufga_last_5 = player_df['UFGA'].tail(5) if 'UFGA' in player_df.columns else pd.Series()
    features['UFGA_AVG_TO_DATE'] = ufga_avg
    features['UFGA_L5_OVER_BASELINE'] = safe_delta(ufga_last_5, ufga_avg)
    
    # Three-point volume
    # 16 — FG3A_AVG_TO_DATE
    fg3a_avg = safe_mean(player_df['FG3A'])
    # 17 — FG3A_L5_OVER_BASELINE
    fg3a_last_5 = player_df['FG3A'].tail(5)
    # 18 — FG3A_BOOST_STAR_OUT
    fg3a_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FG3A'])
    
    features['FG3A_AVG_TO_DATE'] = fg3a_avg
    features['FG3A_L5_OVER_BASELINE'] = safe_delta(fg3a_last_5, fg3a_avg)
    features['FG3A_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                       (fg3a_star_out - fg3a_avg))
    
    # Efficiency (affects green light)
    # 19 — FG_PCT_AVG_TO_DATE
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    # 20 — FG_PCT_L5_OVER_BASELINE
    fg_pct_last_5 = player_df['FG_PCT'].tail(5)
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    features['FG_PCT_L5_OVER_BASELINE'] = safe_delta(fg_pct_last_5, fg_pct_avg)
    
    # 21 — TS_PCT_AVG_TO_DATE
    ts_pct_avg = safe_mean(player_df['TS_PCT'])
    # 22 — TS_PCT_L5_OVER_BASELINE
    ts_pct_last_5 = player_df['TS_PCT'].tail(5)
    features['TS_PCT_AVG_TO_DATE'] = ts_pct_avg
    features['TS_PCT_L5_OVER_BASELINE'] = safe_delta(ts_pct_last_5, ts_pct_avg)
    
    # Aggression/playmaking
    # 23 — FTA_AVG_TO_DATE
    fta_avg = safe_mean(player_df['FTA'])
    # 24 — FTA_L5_OVER_BASELINE
    fta_last_5 = player_df['FTA'].tail(5)
    features['FTA_AVG_TO_DATE'] = fta_avg
    features['FTA_L5_OVER_BASELINE'] = safe_delta(fta_last_5, fta_avg)
    
    # 25 — AST_AVG_TO_DATE
    ast_avg = safe_mean(player_df['AST']) if 'AST' in player_df.columns else 0.0
    features['AST_AVG_TO_DATE'] = ast_avg
    
    # 26 — TOV_AVG_TO_DATE
    tov_avg = safe_mean(player_df['TOV']) if 'TOV' in player_df.columns else 0.0
    features['TOV_AVG_TO_DATE'] = tov_avg
    
    # Variance stability (pre-computed features from feature engineering)
    # 27 — FGA_VARIANCE_STABILITY
    # These should already be in the dataframe, but we'll access them safely
    if 'FGA_VARIANCE_STABILITY' in player_df.columns:
        features['FGA_VARIANCE_STABILITY'] = float(player_df['FGA_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FGA_VARIANCE_STABILITY'] = 0.0
    
    # 28 — FG3A_VARIANCE_STABILITY
    if 'FG3A_VARIANCE_STABILITY' in player_df.columns:
        features['FG3A_VARIANCE_STABILITY'] = float(player_df['FG3A_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FG3A_VARIANCE_STABILITY'] = 0.0
    
    # Team context
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    league_off_avg = safe_mean(league_df['OFF_RATING']) if 'OFF_RATING' in league_df.columns else 110.0
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    team_pace = safe_mean(team_df['TEAM_PACE'])
    team_off = safe_mean(team_df['TEAM_OFF_RATING'])
    opp_pace = safe_mean(opp_team_df['TEAM_PACE'])
    opp_def = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    
    # 29 — TEAM_OFF_RATING_OVER_LEAGUE_AVG
    features['TEAM_OFF_RATING_OVER_LEAGUE_AVG'] = team_off - league_off_avg
    
    # 30 — TEAM_PACE_OVER_LEAGUE_AVG
    features['TEAM_PACE_OVER_LEAGUE_AVG'] = team_pace - league_pace_avg
    
    # Game pace
    # 31 — EXPECTED_PACE
    expected_pace = (team_pace + opp_pace) / 2
    features['EXPECTED_PACE'] = expected_pace
    
    # 32 — OPP_PACE_OVER_LEAGUE_AVG
    features['OPP_PACE_OVER_LEAGUE_AVG'] = opp_pace - league_pace_avg
    
    # Opponent defense
    # 33 — OPP_DEF_RATING_OVER_LEAGUE_AVG
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    # 34 — GUARD_DEF_RATING_OVER_LEAGUE_AVG
    # Using team defense rating (simplified approach, matching pipeline_pts.py)
    features['GUARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    # 35 — FORWARD_DEF_RATING_OVER_LEAGUE_AVG
    # Using team defense rating (simplified approach, matching pipeline_pts.py)
    features['FORWARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    # 36 — OPP_DEF_RATING_OVER_LEAGUE_AVG_x_FGA_AVG_TO_DATE
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG_x_FGA_AVG_TO_DATE'] = (
        features['OPP_DEF_RATING_OVER_LEAGUE_AVG'] * features['FGA_AVG_TO_DATE']
    )
    
    # Predicted upstream values (needed for interaction features)
    predicted_min = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    predicted_usg_pct = float(predicted_usage) if predicted_usage is not None else safe_mean(player_df['USG_PCT'])
    
    # Interaction features with predicted values (matching FGA_features list order)
    team_pace_over_league = team_pace - league_pace_avg
    
    # 37 — PREDICTED_MIN_x_PREDICTED_USG_PCT
    features['PREDICTED_MIN_x_PREDICTED_USG_PCT'] = predicted_min * predicted_usg_pct
    
    # 38 — PREDICTED_USG_PCT_x_TEAM_PACE_OVER_LEAGUE_AVG
    features['PREDICTED_USG_PCT_x_TEAM_PACE_OVER_LEAGUE_AVG'] = predicted_usg_pct * team_pace_over_league
    
    # 39 — FGA_BOOST_STAR_OUT_x_PREDICTED_USG_PCT
    features['FGA_BOOST_STAR_OUT_x_PREDICTED_USG_PCT'] = features['FGA_BOOST_STAR_OUT'] * predicted_usg_pct
    
    # 40 — EXPECTED_PACE_x_PREDICTED_MIN
    features['EXPECTED_PACE_x_PREDICTED_MIN'] = expected_pace * predicted_min
    
    return features

