import pandas as pd
import numpy as np
from src.utils.helper_functions import findOpp


def build_fta_features(
    player_name,
    data,
    current_date,
    projectedStartingFive,
    mainStartingFive,
    teamStarPlayer,
    league_df,
    findOpp,
    predicted_minutes=None,
    predicted_fga=None,
    predicted_fg3a=None
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
    fta_avg = safe_mean(player_df['FTA'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    star_out_flag = int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, []))
    
    # Calculate team FTA rank
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        team_fta_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_fta_avgs[team_player_name] = safe_mean(team_player_df['FTA'])
        
        if team_fta_avgs:
            fta_series = pd.Series(team_fta_avgs)
            fta_ranks = fta_series.rank(method='dense', ascending=False)
            fta_team_rank = float(fta_ranks.get(player_name, len(team_fta_avgs) + 1))
        else:
            fta_team_rank = 1.0
    else:
        fta_team_rank = 1.0
    
    # Calculate various FTA statistics
    fta_std_5 = safe_std(player_df['FTA'].tail(5)) if len(player_df) >= 5 else 0.0
    fta_std_10 = safe_std(player_df['FTA'].tail(10)) if len(player_df) >= 10 else 0.0
    
    fta_l3 = player_df['FTA'].tail(3)
    fta_l10 = player_df['FTA'].tail(10)
    
    fta_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FTA'])
    
    # FGA calculations
    fga_avg = safe_mean(player_df['FGA'])
    fga_l5 = player_df['FGA'].tail(5)
    
    # PTS calculations
    pts_avg = safe_mean(player_df['PTS'])
    
    # USG_PCT calculations
    usg_pct_avg = safe_mean(player_df['USG_PCT'])
    
    # Get predicted_minutes for FTA_PER_MIN calculation
    predicted_min = float(predicted_minutes) if predicted_minutes is not None else safe_mean(player_df['MIN'])
    
    # FTA_PER_MIN calculation
    fta_per_min = round(fta_avg / (predicted_min + 1e-8), 3) if predicted_min > 0 else 0.0
    
    # Get delta features from latest_row if available
    fta_delta_topfta_rank_out = 0.0
    if latest_row is not None and 'FTA_DELTA_TOPFTA_RANK_OUT' in latest_row:
        fta_delta_topfta_rank_out = float(latest_row['FTA_DELTA_TOPFTA_RANK_OUT']) if not pd.isna(latest_row['FTA_DELTA_TOPFTA_RANK_OUT']) else 0.0
    
    fga_delta_topfga_rank_out = 0.0
    if latest_row is not None and 'FGA_DELTA_TOPFGA_RANK_OUT' in latest_row:
        fga_delta_topfga_rank_out = float(latest_row['FGA_DELTA_TOPFGA_RANK_OUT']) if not pd.isna(latest_row['FGA_DELTA_TOPFGA_RANK_OUT']) else 0.0
    
    # Team FTA calculations
    team_fta_avg = safe_mean(team_df['TEAM_FTA']) if 'TEAM_FTA' in team_df.columns else 0.0
    team_fta_l5 = team_df['TEAM_FTA'].tail(5) if 'TEAM_FTA' in team_df.columns else pd.Series()
    
    # Calculate LINEUP_FGA_SHARE_AVG
    projected_starters = projectedStartingFive.get(team, [])
    lineup_fga_shares = []
    team_fga_avg = safe_mean(team_df['TEAM_FGA']) if 'TEAM_FGA' in team_df.columns else 0.0
    
    for starter_name in projected_starters:
        starter_df = data[data['PLAYER_NAME'] == starter_name]
        if not starter_df.empty:
            starter_fga_avg = safe_mean(starter_df['FGA'])
            if team_fga_avg > 0:
                fga_share = (starter_fga_avg / team_fga_avg) * 100
                lineup_fga_shares.append(fga_share)
    
    lineup_fga_share_avg = round(safe_mean(pd.Series(lineup_fga_shares)), 2) if lineup_fga_shares else 0.0
    
    # Home/Away calculations
    home_fta_avg = safe_mean(player_df[player_df['HOME_GAME'] == 1]['FTA']) if 'HOME_GAME' in player_df.columns else fta_avg
    away_fta_avg = safe_mean(player_df[player_df['HOME_GAME'] == 0]['FTA']) if 'HOME_GAME' in player_df.columns else fta_avg
    
    # Pace calculations
    pace_l5 = player_df['PACE'].tail(5) if 'PACE' in player_df.columns else team_df['TEAM_PACE'].tail(5)
    team_pace_avg = safe_mean(team_df['TEAM_PACE'])
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    opp_pace_avg = safe_mean(opp_team_df['TEAM_PACE'])
    expected_pace = (team_pace_avg + opp_pace_avg) / 2
    
    # Opponent defensive rating
    opp_def_avg = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    # Position calculations
    guard = int(latest_row['GUARD']) if latest_row is not None and 'GUARD' in latest_row else 0
    
    # Other stats
    fg_pct_avg = safe_mean(player_df['FG_PCT'])
    ast_avg = safe_mean(player_df['AST']) if 'AST' in player_df.columns else 0.0
    tov_avg = safe_mean(player_df['TOV']) if 'TOV' in player_df.columns else 0.0
    pf_avg = safe_mean(player_df['PF']) if 'PF' in player_df.columns else 0.0
    poss_avg = safe_mean(player_df['POSS']) if 'POSS' in player_df.columns else 0.0
    tchs_avg = safe_mean(player_df['TCHS']) if 'TCHS' in player_df.columns else 0.0
    e_off_rating_avg = safe_mean(player_df['E_OFF_RATING']) if 'E_OFF_RATING' in player_df.columns else 0.0
    net_rating_avg = safe_mean(player_df['NET_RATING']) if 'NET_RATING' in player_df.columns else 0.0
    
    # Build features dict in the EXACT order from fta_features
    features = {}
    
    # 1: TEAM_FTA_ROLLING_AVG_5
    if len(team_fta_l5) > 0:
        features['TEAM_FTA_ROLLING_AVG_5'] = safe_mean(team_fta_l5)
    else:
        features['TEAM_FTA_ROLLING_AVG_5'] = team_fta_avg
    
    # 2: FTA_TEAM_RANK
    features['FTA_TEAM_RANK'] = fta_team_rank
    
    # 3: FTA_ROLLING_AVG_10
    if latest_row is not None and 'FTA_ROLLING_AVG_10' in latest_row:
        features['FTA_ROLLING_AVG_10'] = float(latest_row['FTA_ROLLING_AVG_10'])
    else:
        features['FTA_ROLLING_AVG_10'] = safe_mean(fta_l10)
    
    # 4: FTA_DELTA_TOPFTA_RANK_OUT
    features['FTA_DELTA_TOPFTA_RANK_OUT'] = round(fta_delta_topfta_rank_out, 2)
    
    # 5: FTA_BOOST_STAR_OUT
    features['FTA_BOOST_STAR_OUT'] = star_out_flag * (fta_star_out - fta_avg)
    
    # 6: FTA_PER_FGA
    features['FTA_PER_FGA'] = round(fta_avg / (fga_avg + 1e-8), 4) if fga_avg > 0 else 0.0
    
    # 7: FTA_DELTA_STAR_OUT
    features['FTA_DELTA_STAR_OUT'] = fta_star_out - fta_avg
    
    # 8: FTA_PER_MIN
    features['FTA_PER_MIN'] = fta_per_min
    
    # 9: TEAM_FTA_AVG_TO_DATE
    features['TEAM_FTA_AVG_TO_DATE'] = team_fta_avg
    
    # 10: PLAYER_HOME_AVG_FTA_TO_DATE
    features['PLAYER_HOME_AVG_FTA_TO_DATE'] = home_fta_avg
    
    # 11: POSS_AVG_TO_DATE
    features['POSS_AVG_TO_DATE'] = poss_avg
    
    # 12: FGA_DELTA_TOPFGA_RANK_OUT
    features['FGA_DELTA_TOPFGA_RANK_OUT'] = round(fga_delta_topfga_rank_out, 2)
    
    # 13: FG_PCT_AVG_TO_DATE
    features['FG_PCT_AVG_TO_DATE'] = fg_pct_avg
    
    # 14: CENTER_DEF_RATING_OVER_LEAGUE_AVG
    features['CENTER_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def_avg - league_def_avg
    
    # 15: STARTING
    features['STARTING'] = starting_flag
    
    # 16: LINEUP_FGA_SHARE_AVG
    features['LINEUP_FGA_SHARE_AVG'] = lineup_fga_share_avg
    
    # 17: PTS_AVG_TO_DATE
    features['PTS_AVG_TO_DATE'] = pts_avg
    
    # 18: PF_AVG_TO_DATE
    features['PF_AVG_TO_DATE'] = pf_avg
    
    # 19: FTA_LAG_1
    features['FTA_LAG_1'] = float(player_df['FTA'].iloc[-1]) if len(player_df) >= 1 else 0.0
    
    # 20: FTA_EXPECTATION_LOCATION
    features['FTA_EXPECTATION_LOCATION'] = (home_flag * (home_fta_avg - fta_avg) + 
                                           (1 - home_flag) * (away_fta_avg - fta_avg))
    
    # 21: FTA_STD_10_TO_DATE
    features['FTA_STD_10_TO_DATE'] = fta_std_10
    
    # 22: FTA_L3_OVER_BASELINE
    features['FTA_L3_OVER_BASELINE'] = safe_delta(fta_l3, fta_avg)
    
    # 23: PLAYER_AWAY_FTA_DELTA
    features['PLAYER_AWAY_FTA_DELTA'] = away_fta_avg - fta_avg
    
    # 24: PLAYER_AWAY_AVG_FTA_TO_DATE
    features['PLAYER_AWAY_AVG_FTA_TO_DATE'] = away_fta_avg
    
    # 25: PACE_DIFFERENTIAL
    features['PACE_DIFFERENTIAL'] = team_pace_avg - opp_pace_avg
    
    # 26: EXPECTED_PACE
    features['EXPECTED_PACE'] = expected_pace
    
    # 27: TCHS_AVG_TO_DATE
    features['TCHS_AVG_TO_DATE'] = tchs_avg
    
    # 28: E_OFF_RATING_AVG_TO_DATE
    features['E_OFF_RATING_AVG_TO_DATE'] = e_off_rating_avg
    
    # 29: NET_RATING_AVG_TO_DATE
    features['NET_RATING_AVG_TO_DATE'] = net_rating_avg
    
    # 30: AST_AVG_TO_DATE
    features['AST_AVG_TO_DATE'] = ast_avg
    
    # 31: PACE_ROLLING_AVG_5
    if latest_row is not None and 'PACE_ROLLING_AVG_5' in latest_row:
        features['PACE_ROLLING_AVG_5'] = float(latest_row['PACE_ROLLING_AVG_5'])
    else:
        features['PACE_ROLLING_AVG_5'] = safe_mean(pace_l5)
    
    # 32: USG_PCT_VARIANCE_STABILITY
    if latest_row is not None and 'USG_PCT_VARIANCE_STABILITY' in latest_row:
        features['USG_PCT_VARIANCE_STABILITY'] = float(latest_row['USG_PCT_VARIANCE_STABILITY'])
    else:
        features['USG_PCT_VARIANCE_STABILITY'] = 0.0
    
    # 33: USG_PCT_AVG_TO_DATE
    features['USG_PCT_AVG_TO_DATE'] = usg_pct_avg
    
    # 34: FTA_STD_5_TO_DATE
    features['FTA_STD_5_TO_DATE'] = fta_std_5
    
    # 35: TOV_AVG_TO_DATE
    features['TOV_AVG_TO_DATE'] = tov_avg
    
    # 36: OPP_DEF_RATING_OVER_LEAGUE_AVG
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def_avg - league_def_avg
    
    # 37: PLAYER_HOME_FTA_DELTA
    features['PLAYER_HOME_FTA_DELTA'] = home_fta_avg - fta_avg
    
    # 38: FTA_LAG_2
    features['FTA_LAG_2'] = float(player_df['FTA'].iloc[-2]) if len(player_df) >= 2 else 0.0
    
    # 39: GUARD
    features['GUARD'] = guard
    
    return features

