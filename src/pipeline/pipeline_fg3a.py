import pandas as pd
import numpy as np
from src.utils.helper_functions import findOpp


def build_fg3a_features(
    player_name,
    data,
    current_date,
    projectedStartingFive,
    mainStartingFive,
    teamStarPlayer,
    league_df,
    findOpp,
    predicted_minutes=None,
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
    
    # Get the most recent row (for features that are already calculated in the dataframe)
    latest_row = player_df.iloc[-1] if len(player_df) > 0 else None

    # Calculate values needed for multiple features
    fg3a_avg = safe_mean(player_df['FG3A'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    star_out_flag = int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, []))
    
    # Calculate FG3A team rank
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        team_fg3a_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_fg3a_avgs[team_player_name] = safe_mean(team_player_df['FG3A'])
        
        if team_fg3a_avgs:
            fg3a_series = pd.Series(team_fg3a_avgs)
            fg3a_ranks = fg3a_series.rank(method='dense', ascending=False)
            fg3a_team_rank = float(fg3a_ranks.get(player_name, len(team_fg3a_avgs) + 1))
        else:
            fg3a_team_rank = 1.0
    else:
        fg3a_team_rank = 1.0
    
    # Calculate various averages and deltas
    fg3a_std_10 = safe_std(player_df['FG3A'].tail(10)) if len(player_df) >= 10 else 0.0
    
    fg3a_l5 = player_df['FG3A'].tail(5)
    fg3a_l10 = player_df['FG3A'].tail(10)
    
    fg3a_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FG3A'])
    
    # FG3M (3-pointers made) calculations
    fg3m_avg = safe_mean(player_df['FG3M']) if 'FG3M' in player_df.columns else 0.0
    fg3m_l5 = player_df['FG3M'].tail(5) if 'FG3M' in player_df.columns else pd.Series()
    
    # FG3_PCT calculations
    fg3_pct_avg = safe_mean(player_df['FG3_PCT']) if 'FG3_PCT' in player_df.columns else 0.0
    
    # FGA calculations
    fga_avg = safe_mean(player_df['FGA'])
    fga_l5 = player_df['FGA'].tail(5)
    
    # USG_PCT calculations
    usg_pct_avg = safe_mean(player_df['USG_PCT'])
    usg_pct_l5 = player_df['USG_PCT'].tail(5)
    
    # MIN calculations
    min_avg = safe_mean(player_df['MIN'])
    
    # Team FG3A calculations
    team_fg3a_avg = safe_mean(team_df['TEAM_FG3A']) if 'TEAM_FG3A' in team_df.columns else 0.0
    team_fg3a_l5 = team_df['TEAM_FG3A'].tail(5) if 'TEAM_FG3A' in team_df.columns else pd.Series()
    
    # Matchup calculations
    matchup_fg3_pct_avg = safe_mean(player_df[player_df['OPP_ABBREVIATION'] == opp]['FG3_PCT']) if 'FG3_PCT' in player_df.columns and len(player_df[player_df['OPP_ABBREVIATION'] == opp]) > 0 else fg3_pct_avg
    
    # Pace calculations
    team_pace_avg = safe_mean(team_df['TEAM_PACE'])
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    opp_pace_avg = safe_mean(opp_team_df['TEAM_PACE'])
    expected_pace = (team_pace_avg + opp_pace_avg) / 2
    
    team_pace_l5 = team_df['TEAM_PACE'].tail(5)
    
    # Opponent defensive rating
    opp_def_avg = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    # Get predicted_minutes for FG3A_PER_MIN calculation
    predicted_min = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    
    # FG3A_PER_MIN calculation
    fg3a_per_min = round(fg3a_avg / (predicted_min + 1e-8), 3) if predicted_min > 0 else 0.0
    
    # Get delta features from latest_row if available
    fg3a_delta_topfg3a_rank_out = 0.0
    if latest_row is not None and 'FG3A_DELTA_TOPFG3A_RANK_OUT' in latest_row:
        fg3a_delta_topfg3a_rank_out = float(latest_row['FG3A_DELTA_TOPFG3A_RANK_OUT']) if not pd.isna(latest_row['FG3A_DELTA_TOPFG3A_RANK_OUT']) else 0.0
    
    fga_delta_topfga_rank_out = 0.0
    if latest_row is not None and 'FGA_DELTA_TOPFGA_RANK_OUT' in latest_row:
        fga_delta_topfga_rank_out = float(latest_row['FGA_DELTA_TOPFGA_RANK_OUT']) if not pd.isna(latest_row['FGA_DELTA_TOPFGA_RANK_OUT']) else 0.0
    
    # Other stats
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    dist_avg = safe_mean(player_df['DIST']) if 'DIST' in player_df.columns else 0.0
    tchs_avg = safe_mean(player_df['TCHS']) if 'TCHS' in player_df.columns else 0.0
    e_off_rating_avg = safe_mean(player_df['E_OFF_RATING']) if 'E_OFF_RATING' in player_df.columns else 0.0
    
    # Build features dict in the EXACT order from fg3a_features
    features = {}
    
    # 1: FG3A_ROLLING_AVG_10
    if latest_row is not None and 'FG3A_ROLLING_AVG_10' in latest_row:
        features['FG3A_ROLLING_AVG_10'] = float(latest_row['FG3A_ROLLING_AVG_10'])
    else:
        features['FG3A_ROLLING_AVG_10'] = safe_mean(fg3a_l10)
    
    # 2: FG3A_PCT_OF_FGA
    features['FG3A_PCT_OF_FGA'] = round(fg3a_avg / (fga_avg + 1e-8), 4) if fga_avg > 0 else 0.0
    
    # 3: TEAM_FG3A_ROLLING_AVG_5
    if len(team_fg3a_l5) > 0:
        features['TEAM_FG3A_ROLLING_AVG_5'] = safe_mean(team_fg3a_l5)
    else:
        features['TEAM_FG3A_ROLLING_AVG_5'] = team_fg3a_avg
    
    # 4: USG_PCT_ROLLING_AVG_5
    if latest_row is not None and 'USG_PCT_ROLLING_AVG_5' in latest_row:
        features['USG_PCT_ROLLING_AVG_5'] = float(latest_row['USG_PCT_ROLLING_AVG_5'])
    else:
        features['USG_PCT_ROLLING_AVG_5'] = safe_mean(usg_pct_l5)
    
    # 5: FG3A_DELTA_TOPFG3A_RANK_OUT
    features['FG3A_DELTA_TOPFG3A_RANK_OUT'] = round(fg3a_delta_topfg3a_rank_out, 2)
    
    # 6: FGA_ROLLING_AVG_5
    if latest_row is not None and 'FGA_ROLLING_AVG_5' in latest_row:
        features['FGA_ROLLING_AVG_5'] = float(latest_row['FGA_ROLLING_AVG_5'])
    else:
        features['FGA_ROLLING_AVG_5'] = safe_mean(fga_l5)
    
    # 7: FG3M_AVG_TO_DATE
    features['FG3M_AVG_TO_DATE'] = fg3m_avg
    
    # 8: TEAM_FG3A_AVG_TO_DATE
    features['TEAM_FG3A_AVG_TO_DATE'] = team_fg3a_avg
    
    # 9: FG3A_TEAM_RANK
    features['FG3A_TEAM_RANK'] = fg3a_team_rank
    
    # 10: FG3A_PER_MIN
    features['FG3A_PER_MIN'] = fg3a_per_min
    
    # 11: FG3A_DELTA_STAR_OUT
    features['FG3A_DELTA_STAR_OUT'] = fg3a_star_out - fg3a_avg
    
    # 12: FG_PCT_AVG_TO_DATE
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    
    # 13: TEAM_PACE_L5_OVER_LEAGUE_AVG
    if len(team_pace_l5) > 0:
        team_pace_l5_avg = safe_mean(team_pace_l5)
        features['TEAM_PACE_L5_OVER_LEAGUE_AVG'] = team_pace_l5_avg - league_pace_avg
    else:
        features['TEAM_PACE_L5_OVER_LEAGUE_AVG'] = team_pace_avg - league_pace_avg
    
    # 14: CENTER_DEF_RATING_OVER_LEAGUE_AVG
    features['CENTER_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def_avg - league_def_avg
    
    # 15: DIST_AVG_TO_DATE
    features['DIST_AVG_TO_DATE'] = dist_avg
    
    # 16: FGA_L5_OVER_BASELINE
    features['FGA_L5_OVER_BASELINE'] = safe_delta(fga_l5, fga_avg)
    
    # 17: STAR_SAT_OUT
    features['STAR_SAT_OUT'] = star_out_flag
    
    # 18: FG3A_BOOST_STAR_OUT
    features['FG3A_BOOST_STAR_OUT'] = star_out_flag * (fg3a_star_out - fg3a_avg)
    
    # 19: FG3M_ROLLING_AVG_5
    if latest_row is not None and 'FG3M_ROLLING_AVG_5' in latest_row:
        features['FG3M_ROLLING_AVG_5'] = float(latest_row['FG3M_ROLLING_AVG_5'])
    else:
        features['FG3M_ROLLING_AVG_5'] = safe_mean(fg3m_l5)
    
    # 20: TCHS_AVG_TO_DATE
    features['TCHS_AVG_TO_DATE'] = tchs_avg
    
    # 21: USG_PCT_L5_OVER_BASELINE
    features['USG_PCT_L5_OVER_BASELINE'] = safe_delta(usg_pct_l5, usg_pct_avg)
    
    # 22: FGA_DELTA_TOPFGA_RANK_OUT
    features['FGA_DELTA_TOPFGA_RANK_OUT'] = round(fga_delta_topfga_rank_out, 2)
    
    # 23: FG3_PCT_AVG_TO_DATE
    features['FG3_PCT_AVG_TO_DATE'] = fg3_pct_avg
    
    # 24: FG3A_STD_10_TO_DATE
    features['FG3A_STD_10_TO_DATE'] = fg3a_std_10
    
    # 25: FG3A_LAG_1
    features['FG3A_LAG_1'] = float(player_df['FG3A'].iloc[-1]) if len(player_df) >= 1 else 0.0
    
    # 26: STARTING_X_FG3A
    features['STARTING_X_FG3A'] = round(starting_flag * fg3a_avg, 2)
    
    # 27: E_OFF_RATING_AVG_TO_DATE
    features['E_OFF_RATING_AVG_TO_DATE'] = e_off_rating_avg
    
    # 28: EXPECTED_PACE
    features['EXPECTED_PACE'] = expected_pace
    
    # 29: OPP_DEF_RATING_OVER_LEAGUE_AVG
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def_avg - league_def_avg
    
    # 30: MATCHUP_AVG_FG3_PCT_TO_DATE
    features['MATCHUP_AVG_FG3_PCT_TO_DATE'] = matchup_fg3_pct_avg
    
    return features
