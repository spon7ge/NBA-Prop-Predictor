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
    
    # Calculate pts_avg early (needed for multiple features)
    pts_last_5 = player_df['PTS'].tail(5)
    pts_avg = safe_mean(player_df['PTS'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    
    # 1 — STARTING_X_PTS (new)
    features['STARTING_X_PTS'] = round(starting_flag * pts_avg, 2)
    
    # 2 — GAMES_PLAYED_TO_DATE
    features['GAMES_PLAYED_TO_DATE'] = len(player_df)
    
    # 3 — PTS_CEILING_L5_DELTA
    pts_ceiling_l5 = float(pts_last_5.max()) if pts_last_5.size > 0 else pts_avg
    features['PTS_CEILING_L5_DELTA'] = round(pts_ceiling_l5 - pts_avg, 2)
    
    # Predicted upstream values and interaction features (most important)
    predicted_min = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    predicted_usg_pct = float(predicted_usage) if predicted_usage is not None else safe_mean(player_df['USG_PCT'])
    predicted_fga = float(predicted_fga) if predicted_fga is not None else safe_mean(player_df['FGA'])
    
    # 4 — PREDICTED_MIN
    features['PREDICTED_MIN'] = predicted_min
    
    # 5 — PREDICTED_USG_PCT
    features['PREDICTED_USG_PCT'] = predicted_usg_pct
    
    # 6 — PREDICTED_FGA
    features['PREDICTED_FGA'] = predicted_fga
    
    # 7 — PREDICTED_MIN_x_PREDICTED_USG_PCT
    features['PREDICTED_MIN_x_PREDICTED_USG_PCT'] = predicted_min * predicted_usg_pct
    
    # 8 — PTS_PER_MIN
    # Use PREDICTED_MIN to match training (not historical min_avg)
    features['PTS_PER_MIN'] = round(pts_avg / (predicted_min + 1e-8), 3) if predicted_min > 0 else 0.0
    
    # 9 — PTS_L5_OVER_BASELINE_x_PREDICTED_MIN
    pts_l5_over_baseline = safe_delta(pts_last_5, pts_avg)
    features['PTS_L5_OVER_BASELINE_x_PREDICTED_MIN'] = pts_l5_over_baseline * predicted_min
    
    # 10 — EXPECTED_PACE_x_PREDICTED_USG_PCT
    expected_pace = (safe_mean(team_df['TEAM_PACE']) + safe_mean(opp_team_df['TEAM_PACE'])) / 2
    features['EXPECTED_PACE_x_PREDICTED_USG_PCT'] = expected_pace * predicted_usg_pct
    
    # 11 — EXPECTED_PACE_x_PREDICTED_FGA
    features['EXPECTED_PACE_x_PREDICTED_FGA'] = expected_pace * predicted_fga
    
    # 12 — PREDICTED_FGA_x_TS_PCT_AVG_TO_DATE
    ts_pct_avg = safe_mean(player_df['TS_PCT'])
    features['PREDICTED_FGA_x_TS_PCT_AVG_TO_DATE'] = predicted_fga * ts_pct_avg
    
    # 13 — OPP_DEF_RATING_OVER_LEAGUE_AVG_x_PTS_AVG_TO_DATE
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    opp_def = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    opp_def_rating_over_league = opp_def - league_def_avg
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG_x_PTS_AVG_TO_DATE'] = opp_def_rating_over_league * pts_avg
    
    # 14 — PTS_TEAM_RANK (new)
    # Calculate PTS rank among all players on the team
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        # Calculate PTS_AVG_TO_DATE for each player on the team
        team_pts_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_pts_avgs[team_player_name] = safe_mean(team_player_df['PTS'])
        
        # Create a series and rank (ascending=False means rank 1 = highest PTS)
        if team_pts_avgs:
            pts_series = pd.Series(team_pts_avgs)
            pts_ranks = pts_series.rank(method='dense', ascending=False)
            pts_team_rank = float(pts_ranks.get(player_name, len(team_pts_avgs) + 1))
        else:
            pts_team_rank = 1.0
    else:
        pts_team_rank = 1.0
    features['PTS_TEAM_RANK'] = pts_team_rank
    
    # 15 — PERCENTAGE_OF_TEAM_PTS (new)
    # Calculate team average PTS to date
    team_pts_avg = safe_mean(team_df['TEAM_PTS_AVG_TO_DATE']) if 'TEAM_PTS_AVG_TO_DATE' in team_df.columns else safe_mean(team_df['TEAM_PTS']) if 'TEAM_PTS' in team_df.columns else 110.0
    epsilon = 1e-8
    percentage_of_team_pts = round(pts_avg / (team_pts_avg + epsilon), 4) if team_pts_avg > 0 else 0.0
    features['PERCENTAGE_OF_TEAM_PTS'] = percentage_of_team_pts
    
    # 16 — PTS_AVG_TO_DATE
    features['PTS_AVG_TO_DATE'] = pts_avg
    
    # 17 — PTS_STD_5_TO_DATE
    pts_std_5 = safe_std(pts_last_5) if len(player_df) >= 5 else 0.0
    features['PTS_STD_5_TO_DATE'] = pts_std_5
    
    # 18 — PTS_BOOST_STAR_OUT
    features['PTS_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                     (safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['PTS']) - pts_avg))
    
    # 19 — PTS_EXPECTATION_LOCATION
    home_pts = safe_mean(player_df[player_df['HOME_GAME'] == 1]['PTS'])
    away_pts = safe_mean(player_df[player_df['HOME_GAME'] == 0]['PTS'])
    features['PTS_EXPECTATION_LOCATION'] = (home_flag * (home_pts - pts_avg) + 
                                           (1 - home_flag) * (away_pts - pts_avg))
    
    # 20 — CFGA_AVG_TO_DATE
    cfga_avg = safe_mean(player_df['CFGA']) if 'CFGA' in player_df.columns else 0.0
    features['CFGA_AVG_TO_DATE'] = cfga_avg
    
    # 21 — UFGA_AVG_TO_DATE
    ufga_avg = safe_mean(player_df['UFGA']) if 'UFGA' in player_df.columns else 0.0
    # 22 — UFGA_L5_OVER_BASELINE
    ufga_last_5 = player_df['UFGA'].tail(5) if 'UFGA' in player_df.columns else pd.Series()
    features['UFGA_AVG_TO_DATE'] = ufga_avg
    features['UFGA_L5_OVER_BASELINE'] = safe_delta(ufga_last_5, ufga_avg)
    
    # 23 — FTA_AVG_TO_DATE
    fta_avg = safe_mean(player_df['FTA'])
    # 24 — FTA_L5_OVER_BASELINE
    fta_last_5 = player_df['FTA'].tail(5)
    # 25 — FTA_BOOST_STAR_OUT
    fta_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FTA'])
    # 26 — FTA_EXPECTATION_LOCATION
    home_fta = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FTA'])
    away_fta = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FTA'])
    
    features['FTA_AVG_TO_DATE'] = fta_avg
    features['FTA_L5_OVER_BASELINE'] = safe_delta(fta_last_5, fta_avg)
    features['FTA_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                     (fta_star_out - fta_avg))
    features['FTA_EXPECTATION_LOCATION'] = (home_flag * (home_fta - fta_avg) + 
                                           (1 - home_flag) * (away_fta - fta_avg))
    
    # 27 — FG3A_AVG_TO_DATE
    fg3a_avg = safe_mean(player_df['FG3A'])
    # 28 — FG3A_L5_OVER_BASELINE
    # 29 — FG3A_BOOST_STAR_OUT
    fg3a_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FG3A'])
    
    features['FG3A_AVG_TO_DATE'] = fg3a_avg
    features['FG3A_L5_OVER_BASELINE'] = safe_delta(player_df['FG3A'].tail(5), fg3a_avg)
    features['FG3A_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                      (fg3a_star_out - fg3a_avg))
    
    # 30 — PLUS_MINUS_AVG_TO_DATE
    pm_avg = safe_mean(player_df['PLUS_MINUS'])
    # 31 — PLUS_MINUS_BOOST_STAR_OUT
    pm_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['PLUS_MINUS'])
    
    features['PLUS_MINUS_AVG_TO_DATE'] = pm_avg
    features['PLUS_MINUS_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                             (pm_star_out - pm_avg))
    
    # 32 — FG_PCT_AVG_TO_DATE
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    # 33 — FG_PCT_L5_OVER_BASELINE
    fg_pct_last_5 = player_df['FG_PCT'].tail(5)
    
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    features['FG_PCT_L5_OVER_BASELINE'] = safe_delta(fg_pct_last_5, fg_pct_avg)
    
    # 34 — FG3_PCT_AVG_TO_DATE
    fg3_pct_avg = safe_mean(player_df['FG3_PCT']) if 'FG3_PCT' in player_df.columns else 0.0
    # 35 — MATCHUP_FG3_PCT_DELTA
    matchup_fg3_pct = safe_mean(matchup_df['FG3_PCT']) if 'FG3_PCT' in matchup_df.columns and not matchup_df.empty else fg3_pct_avg
    features['FG3_PCT_AVG_TO_DATE'] = fg3_pct_avg
    features['MATCHUP_FG3_PCT_DELTA'] = matchup_fg3_pct - fg3_pct_avg
    
    # 36 — FT_PCT_AVG_TO_DATE
    ft_pct_avg = safe_mean(player_df['FT_PCT'])
    # 37 — FT_PCT_BOOST_STAR_OUT
    ft_pct_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FT_PCT'])
    
    features['FT_PCT_AVG_TO_DATE'] = ft_pct_avg
    features['FT_PCT_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                        (ft_pct_star_out - ft_pct_avg))
    
    # 38 — TS_PCT_AVG_TO_DATE
    # 39 — TS_PCT_BOOST_STAR_OUT
    ts_pct_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['TS_PCT'])
    
    features['TS_PCT_AVG_TO_DATE'] = ts_pct_avg
    features['TS_PCT_BOOST_STAR_OUT'] = (int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, [])) * 
                                        (ts_pct_star_out - ts_pct_avg))
    
    # Variance stability (pre-computed features from feature engineering)
    # 40 — PTS_VARIANCE_STABILITY
    if 'PTS_VARIANCE_STABILITY' in player_df.columns:
        features['PTS_VARIANCE_STABILITY'] = float(player_df['PTS_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['PTS_VARIANCE_STABILITY'] = 0.0
    
    # 41 — FGA_VARIANCE_STABILITY
    if 'FGA_VARIANCE_STABILITY' in player_df.columns:
        features['FGA_VARIANCE_STABILITY'] = float(player_df['FGA_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FGA_VARIANCE_STABILITY'] = 0.0
    
    # 42 — FTA_VARIANCE_STABILITY
    if 'FTA_VARIANCE_STABILITY' in player_df.columns:
        features['FTA_VARIANCE_STABILITY'] = float(player_df['FTA_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FTA_VARIANCE_STABILITY'] = 0.0
    
    # 43 — FG3A_VARIANCE_STABILITY
    if 'FG3A_VARIANCE_STABILITY' in player_df.columns:
        features['FG3A_VARIANCE_STABILITY'] = float(player_df['FG3A_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['FG3A_VARIANCE_STABILITY'] = 0.0
    
    # 44 — TS_PCT_VARIANCE_STABILITY
    if 'TS_PCT_VARIANCE_STABILITY' in player_df.columns:
        features['TS_PCT_VARIANCE_STABILITY'] = float(player_df['TS_PCT_VARIANCE_STABILITY'].iloc[-1]) if len(player_df) > 0 else 0.0
    else:
        features['TS_PCT_VARIANCE_STABILITY'] = 0.0
    
    # Team context
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    league_off_avg = safe_mean(league_df['OFF_RATING']) if 'OFF_RATING' in league_df.columns else 110.0
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    team_pace = safe_mean(team_df['TEAM_PACE'])
    team_off = safe_mean(team_df['TEAM_OFF_RATING'])
    opp_def = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    
    # 45 — TEAM_OFF_RATING_OVER_LEAGUE_AVG
    features['TEAM_OFF_RATING_OVER_LEAGUE_AVG'] = team_off - league_off_avg
    
    # 46 — EXPECTED_PACE
    features['EXPECTED_PACE'] = expected_pace
    
    # 47 — GUARD_DEF_RATING_OVER_LEAGUE_AVG
    # Using team defense rating (simplified approach, matching pipeline_fga.py)
    features['GUARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    # 48 — FORWARD_DEF_RATING_OVER_LEAGUE_AVG
    # Using team defense rating (simplified approach, matching pipeline_fga.py)
    features['FORWARD_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def - league_def_avg
    
    return features

