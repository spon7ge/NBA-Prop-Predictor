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
    predicted_minutes=None
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
    fga_avg = safe_mean(player_df['FGA'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    star_out_flag = int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, []))
    
    
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
    
    # Calculate various FGA statistics
    fga_l3 = player_df['FGA'].tail(3)
    fga_l5 = player_df['FGA'].tail(5)
    fga_l10 = player_df['FGA'].tail(10)
    
    fga_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['FGA'])
    
    # CFGA and UFGA calculations
    cfga_avg = safe_mean(player_df['CFGA']) if 'CFGA' in player_df.columns else 0.0
    ufga_avg = safe_mean(player_df['UFGA']) if 'UFGA' in player_df.columns else 0.0
    
    # FG3A calculations
    fg3a_avg = safe_mean(player_df['FG3A'])
    
    # Calculate FG3A team rank
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
    
    # FTA calculations
    fta_avg = safe_mean(player_df['FTA'])
    
    # AST and TOV calculations
    ast_avg = safe_mean(player_df['AST']) if 'AST' in player_df.columns else 0.0
    tov_avg = safe_mean(player_df['TOV']) if 'TOV' in player_df.columns else 0.0
    
    # USG_PCT calculations
    usg_pct_avg = safe_mean(player_df['USG_PCT'])
    usg_pct_star_out = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['USG_PCT'])
    
    # MIN calculations
    min_avg = safe_mean(player_df['MIN'])
    min_l5 = player_df['MIN'].tail(5)
    
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
    
    # Team FGA calculations
    team_fga_avg_to_date = safe_mean(team_df['TEAM_FGA']) if 'TEAM_FGA' in team_df.columns else 0.0
    
    # Pace calculations
    team_pace_avg = safe_mean(team_df['TEAM_PACE'])
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    opp_pace_avg = safe_mean(opp_team_df['TEAM_PACE'])
    expected_pace = (team_pace_avg + opp_pace_avg) / 2
    
    team_pace_l3 = team_df['TEAM_PACE'].tail(3)
    
    # Team offensive rating
    team_off_avg = safe_mean(team_df['TEAM_OFF_RATING'])
    league_off_avg = safe_mean(league_df['OFF_RATING']) if 'OFF_RATING' in league_df.columns else 110.0
    team_off_l3 = team_df['TEAM_OFF_RATING'].tail(3)
    
    # Opponent defensive rating
    opp_def_avg = safe_mean(opp_team_df['TEAM_DEF_RATING'])
    league_def_avg = safe_mean(league_df['DEF_RATING']) if 'DEF_RATING' in league_df.columns else 110.0
    
    # Other stats
    poss_avg = safe_mean(player_df['POSS']) if 'POSS' in player_df.columns else 0.0
    tchs_avg = safe_mean(player_df['TCHS']) if 'TCHS' in player_df.columns else 0.0
    e_off_rating_avg = safe_mean(player_df['E_OFF_RATING']) if 'E_OFF_RATING' in player_df.columns else 0.0
    
    # Build features dict in the EXACT order from fga_features
    features = {}
    
    # 1: FGA_ROLLING_AVG_10
    if latest_row is not None and 'FGA_ROLLING_AVG_10' in latest_row:
        features['FGA_ROLLING_AVG_10'] = float(latest_row['FGA_ROLLING_AVG_10'])
    else:
        features['FGA_ROLLING_AVG_10'] = safe_mean(fga_l10)
    
    # 2: USG_PCT_AVG_TO_DATE
    features['USG_PCT_AVG_TO_DATE'] = usg_pct_avg
    
    # 3: FGA_TEAM_RANK
    features['FGA_TEAM_RANK'] = fga_team_rank
    
    # 4: STARTING_X_FGA
    features['STARTING_X_FGA'] = round(starting_flag * fga_avg, 2)
    
    # 5: POSS_AVG_TO_DATE
    features['POSS_AVG_TO_DATE'] = poss_avg
    
    # 6: STARTING_X_MIN
    features['STARTING_X_MIN'] = round(starting_flag * min_avg, 2)
    
    # 7: LINEUP_FGA_SHARE_AVG
    features['LINEUP_FGA_SHARE_AVG'] = lineup_fga_share_avg
    
    # 8: FGA_BOOST_STAR_OUT
    features['FGA_BOOST_STAR_OUT'] = star_out_flag * (fga_star_out - fga_avg)
    
    # 9: UFGA_AVG_TO_DATE
    features['UFGA_AVG_TO_DATE'] = ufga_avg
    
    # 10: MIN_L5_OVER_BASELINE
    features['MIN_L5_OVER_BASELINE'] = safe_delta(min_l5, min_avg)
    
    # 11: USG_PCT_BOOST_STAR_OUT
    features['USG_PCT_BOOST_STAR_OUT'] = star_out_flag * (usg_pct_star_out - usg_pct_avg)
    
    # 12: TEAM_PACE_L3_OVER_LEAGUE_AVG
    if len(team_pace_l3) > 0:
        team_pace_l3_avg = safe_mean(team_pace_l3)
        features['TEAM_PACE_L3_OVER_LEAGUE_AVG'] = team_pace_l3_avg - league_pace_avg
    else:
        features['TEAM_PACE_L3_OVER_LEAGUE_AVG'] = team_pace_avg - league_pace_avg
    
    # 13: FG3A_TEAM_RANK
    features['FG3A_TEAM_RANK'] = fg3a_team_rank
    
    # 14: OPP_DEF_RATING_OVER_LEAGUE_AVG
    features['OPP_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def_avg - league_def_avg
    
    # 15: FGA_LAG_1
    features['FGA_LAG_1'] = float(player_df['FGA'].iloc[-1]) if len(player_df) >= 1 else 0.0
    
    # 16: E_OFF_RATING_AVG_TO_DATE
    features['E_OFF_RATING_AVG_TO_DATE'] = e_off_rating_avg
    
    # 17: TCHS_AVG_TO_DATE
    features['TCHS_AVG_TO_DATE'] = tchs_avg
    
    # 18: TEAM_FGA_AVG_TO_DATE
    features['TEAM_FGA_AVG_TO_DATE'] = team_fga_avg_to_date
    
    # 19: PACE_DIFFERENTIAL
    features['PACE_DIFFERENTIAL'] = team_pace_avg - opp_pace_avg
    
    # 20: EXPECTED_PACE
    features['EXPECTED_PACE'] = expected_pace
    
    # 21: FGA_L3_OVER_BASELINE
    features['FGA_L3_OVER_BASELINE'] = safe_delta(fga_l3, fga_avg)
    
    # 22: CENTER_DEF_RATING_OVER_LEAGUE_AVG
    features['CENTER_DEF_RATING_OVER_LEAGUE_AVG'] = opp_def_avg - league_def_avg
    
    # 23: TEAM_OFF_RATING_AVG_TO_DATE
    features['TEAM_OFF_RATING_AVG_TO_DATE'] = team_off_avg
    
    # 24: TEAM_OFF_RATING_L3_OVER_LEAGUE_AVG
    if len(team_off_l3) > 0:
        team_off_l3_avg = safe_mean(team_off_l3)
        features['TEAM_OFF_RATING_L3_OVER_LEAGUE_AVG'] = team_off_l3_avg - league_off_avg
    else:
        features['TEAM_OFF_RATING_L3_OVER_LEAGUE_AVG'] = team_off_avg - league_off_avg
    
    # 25: FG3A_AVG_TO_DATE
    features['FG3A_AVG_TO_DATE'] = fg3a_avg
    
    # 26: CFGA_AVG_TO_DATE
    features['CFGA_AVG_TO_DATE'] = cfga_avg
    
    # 27: AST_AVG_TO_DATE
    features['AST_AVG_TO_DATE'] = ast_avg
    
    # 28: TOV_AVG_TO_DATE
    features['TOV_AVG_TO_DATE'] = tov_avg
    
    # 29: FTA_AVG_TO_DATE
    features['FTA_AVG_TO_DATE'] = fta_avg
    
    return features
