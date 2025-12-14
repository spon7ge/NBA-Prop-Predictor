import pandas as pd
import numpy as np
from src.utils.helper_functions import findOpp


def build_min_features(
    player_name,
    data,
    current_date,
    projectedStartingFive,
    mainStartingFive,
    teamStarPlayer,
    league_df,
    findOpp
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
    
    # Get the most recent row (for features that are already calculated in the dataframe)
    latest_row = player_df.iloc[-1] if len(player_df) > 0 else None
    
    # Calculate values needed for multiple features
    min_avg = safe_mean(player_df['MIN'])
    starting_flag = int(player_name in projectedStartingFive.get(team, []))
    star_out_flag = int(teamStarPlayer.get(team, '') not in projectedStartingFive.get(team, []))
    
    # Calculate days rest
    player_days_rest = float(latest_row['PLAYER_DAYS_REST']) if latest_row is not None and 'PLAYER_DAYS_REST' in latest_row else 3.0
    
    # Calculate usual starters available (actually counts missing starters)
    main_starters = set(mainStartingFive.get(team, []))
    projected_starters = set(projectedStartingFive.get(team, []))
    usual_starters_available = len(main_starters - projected_starters)  # Count of usual starters NOT in projected (missing)
    
    # Check if player missed last game
    player_missed_last = int(latest_row['PLAYER_MISSED_LAST']) if latest_row is not None and 'PLAYER_MISSED_LAST' in latest_row else 0
    
    # Calculate MIN team rank
    team_players_df = data[data['TEAM_ABBREVIATION'] == team].copy()
    if not team_players_df.empty:
        team_min_avgs = {}
        for team_player_name in team_players_df['PLAYER_NAME'].unique():
            team_player_df = data[data['PLAYER_NAME'] == team_player_name].sort_values('GAME_DATE')
            if not team_player_df.empty:
                team_min_avgs[team_player_name] = safe_mean(team_player_df['MIN'])
        
        if team_min_avgs:
            min_series = pd.Series(team_min_avgs)
            min_ranks = min_series.rank(method='dense', ascending=False)
            min_team_rank = float(min_ranks.get(player_name, len(team_min_avgs) + 1))
        else:
            min_team_rank = 1.0
    else:
        min_team_rank = 1.0
    
    # Calculate PTS team rank
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
    
    # Calculate various min statistics
    min_std_5 = safe_std(player_df['MIN'].tail(5)) if len(player_df) >= 5 else 0.0
    
    # Star out calculations
    star_out_min = safe_mean(player_df[player_df.get('STAR_SAT_OUT', pd.Series([0])) == 1]['MIN'])
    min_boost_star_out = star_out_flag * (star_out_min - min_avg)
    
    # Calculate PF_AVG_TO_DATE and FOULS_PER_MIN
    pf_avg = safe_mean(player_df['PF']) if 'PF' in player_df.columns else 0.0
    min_avg_to_date = min_avg
    fouls_per_min = pf_avg / (min_avg_to_date + 1e-8) if min_avg_to_date > 0 else 0.0
    
    # Calculate PLUS_MINUS_AVG_TO_DATE
    plus_minus_avg = safe_mean(player_df['PLUS_MINUS']) if 'PLUS_MINUS' in player_df.columns else 0.0
    
    # STARTER_X_DAYS_REST
    starting_x_min = starting_flag * min_avg
    starter_x_days_rest = starting_x_min * player_days_rest
    
    # Pace calculations
    pace_avg = safe_mean(player_df['PACE']) if 'PACE' in player_df.columns else safe_mean(team_df['TEAM_PACE'])
    team_pace_avg = safe_mean(team_df['TEAM_PACE'])
    league_pace_avg = safe_mean(league_df['PACE']) if 'PACE' in league_df.columns else 100.0
    opp_pace_avg = safe_mean(opp_team_df['TEAM_PACE'])
    expected_pace = (team_pace_avg + opp_pace_avg) / 2
    
    # Build features dict in the EXACT order from min_features
    features = {}
    
    # 1: STARTING_X_MIN
    features['STARTING_X_MIN'] = round(starting_flag * min_avg, 2)
    
    # 2: USUAL_STARTERS_AVAILABLE
    features['USUAL_STARTERS_AVAILABLE'] = usual_starters_available
    
    # 3: MIN_LAG_1
    features['MIN_LAG_1'] = float(player_df['MIN'].iloc[-1]) if len(player_df) >= 1 else 0.0
    
    # 4: MIN_LAG_2
    features['MIN_LAG_2'] = float(player_df['MIN'].iloc[-2]) if len(player_df) >= 2 else 0.0
    
    # 5: MIN_LAG_3
    features['MIN_LAG_3'] = float(player_df['MIN'].iloc[-3]) if len(player_df) >= 3 else 0.0
    
    # 6: MIN_ROLLING_AVG_7
    if latest_row is not None and 'MIN_ROLLING_AVG_7' in latest_row:
        features['MIN_ROLLING_AVG_7'] = float(latest_row['MIN_ROLLING_AVG_7'])
    else:
        features['MIN_ROLLING_AVG_7'] = safe_mean(player_df['MIN'].tail(7))
    
    # 7: MIN_TEAM_RANK
    features['MIN_TEAM_RANK'] = min_team_rank
    
    # 8: PTS_TEAM_RANK
    features['PTS_TEAM_RANK'] = pts_team_rank
    
    # 9: PLAYER_DAYS_REST
    features['PLAYER_DAYS_REST'] = player_days_rest
    
    # 10: STARTER_X_DAYS_REST
    features['STARTER_X_DAYS_REST'] = round(starter_x_days_rest, 2)
    
    # 11: PLAYER_MISSED_LAST_GAME_X_MIN
    features['PLAYER_MISSED_LAST_GAME_X_MIN'] = round(player_missed_last * min_avg, 2)
    
    # 12: MIN_BOOST_STAR_OUT
    features['MIN_BOOST_STAR_OUT'] = min_boost_star_out
    
    # 13: MIN_STD_5_TO_DATE
    features['MIN_STD_5_TO_DATE'] = min_std_5
    
    # 14: MIN_VOLATILITY_10_TO_DATE
    if latest_row is not None and 'MIN_VOLATILITY_10_TO_DATE' in latest_row:
        features['MIN_VOLATILITY_10_TO_DATE'] = float(latest_row['MIN_VOLATILITY_10_TO_DATE'])
    else:
        features['MIN_VOLATILITY_10_TO_DATE'] = safe_std(player_df['MIN'].tail(10))
    
    # 15: TEAM_PACE_OVER_LEAGUE_AVG
    features['TEAM_PACE_OVER_LEAGUE_AVG'] = team_pace_avg - league_pace_avg
    
    # 16: EXPECTED_PACE
    features['EXPECTED_PACE'] = expected_pace
    
    # 17: PACE_AVG_TO_DATE
    features['PACE_AVG_TO_DATE'] = pace_avg
    
    # 18: GAMES_PLAYED_TO_DATE
    features['GAMES_PLAYED_TO_DATE'] = len(player_df)
    
    # 19: FOULS_PER_MIN
    features['FOULS_PER_MIN'] = round(fouls_per_min, 4)
    
    # 20: PF_AVG_TO_DATE
    features['PF_AVG_TO_DATE'] = round(pf_avg, 2)
    
    # 21: PLUS_MINUS_AVG_TO_DATE
    features['PLUS_MINUS_AVG_TO_DATE'] = round(plus_minus_avg, 2)
    
    return features
