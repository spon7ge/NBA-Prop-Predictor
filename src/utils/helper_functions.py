
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from nba_api.stats.endpoints import scheduleleaguev2
from nba_api.stats.endpoints import leaguedashteamstats
from src.features.features_v1 import *
from src.features.features_v1 import teamContext as features_teamContext
from src.utils.player_positions import *

# Get project root by navigating up from this file's location
# This file is in src/utils/, so go up 2 levels to reach project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
project_root_str = str(project_root)

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

today = datetime.today().strftime('%Y-%m-%d')

league_df = leaguedashteamstats.LeagueDashTeamStats(
    league_id_nullable='00',
    per_mode_detailed='PerGame',
    measure_type_detailed_defense='Advanced'
).get_data_frames()[0]
if 'TEAM_ID' in league_df.columns:
    league_df = league_df.set_index('TEAM_ID')

def round_to_2(val):
    """Round non-binary numeric values to 2 decimal places."""
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return round(float(val), 2)
    return val

def calculate_volatility(player_df, stat_col, window=5, use_cv=False):
    if len(player_df) < 2:
        return 0.0
    
    player_df_sorted = player_df.sort_values(by='GAME_DATE')
    
    rolling_std = player_df_sorted[stat_col].rolling(window=window, min_periods=2).std().iloc[-1]
    
    if pd.isna(rolling_std):
        return 0.0
    
    if use_cv:
        rolling_mean = player_df_sorted[stat_col].rolling(window=window, min_periods=2).mean().iloc[-1]
        if pd.isna(rolling_mean) or rolling_mean == 0:
            return 0.0
        cv = rolling_std / rolling_mean
        if pd.isna(cv) or abs(cv) == float('inf'):
            return 0.0
        return cv
    
    return rolling_std


def get_volatility_or_calculate(player_df, volatility_col, stat_col=None, window=5):
    if volatility_col in player_df.columns:
        value = player_df[volatility_col].iloc[-1]
        if not pd.isna(value):
            return value
    
    if stat_col is None:
        stat_col = volatility_col.replace('_VOLATILITY_5_TO_DATE', '').replace('_CV_5_TO_DATE', '')
    
    use_cv = '_CV_' in volatility_col
    
    if stat_col in player_df.columns:
        return calculate_volatility(player_df, stat_col, window, use_cv)
    
    return 0.0

_gameCache = {}

def getUpcomingGamesCached(date):
    if date not in _gameCache:
        schedule = scheduleleaguev2.ScheduleLeagueV2().get_data_frames()[0]
        schedule['gameDate'] = pd.to_datetime(schedule['gameDate']).dt.strftime('%Y-%m-%d')
        _gameCache[date] = schedule
    return _gameCache[date]


def findOpp(playerName, players_df, gameDate, max_days_ahead=3):
    player_team = players_df.loc[
        players_df['PLAYER_NAME'] == playerName, 'TEAM_ABBREVIATION'
    ].iloc[-1]
    
    base_date = datetime.strptime(gameDate, '%Y-%m-%d')
    dates_to_check = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(max_days_ahead + 1)]
    
    for check_date in dates_to_check:
        schedule = getUpcomingGamesCached(check_date)
        schedule_filtered = schedule[schedule['gameDate'] == check_date]
        homeTeams = schedule_filtered['homeTeam_teamTricode'].unique().tolist()
        awayTeams = schedule_filtered['awayTeam_teamTricode'].unique().tolist()
        
        home = 0
        if player_team in homeTeams:
            opp_team = awayTeams[homeTeams.index(player_team)]
            home = 1
            return opp_team, home
        elif player_team in awayTeams:
            opp_team = homeTeams[awayTeams.index(player_team)]
            home = 0
            return opp_team, home
    
    print(f"No game found for {player_team} within {max_days_ahead} days from {gameDate}")
    return None, None

def convert_min_to_float(min_str):
    try:
        if isinstance(min_str, str) and ":" in min_str:
            minutes, seconds = map(int, min_str.split(":"))
            total_minutes = minutes + seconds / 60
            return round(total_minutes, 2)
        elif isinstance(min_str, (int, float)):
            return float(min_str)
        else:
            return 0
    except:
        return 0

def process_season_features(season_df, prop_type, year):
    df = season_df.copy()
    df = sort_data_for_features(df)
    df['STARTING'] = df['START_POSITION'].apply(lambda x: 1 if x in ['G','F','C'] else 0)
    
    # Add position data
    cache_file = os.path.join(project_root_str, 'data', 'processed', 'player_info.csv')
    df = assign_position_with_cache(df, cache_file=cache_file, max_workers=4, delay_between_requests=1.5)
    df = add_rest_day_features(df)  
    df['MIN'] = df['MIN'].apply(convert_min_to_float)
    df = rollingAverages(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[3,5,7,10,20])
    df = statAgainstTeam(df, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION')
    df = HomeAwayAverages(df, player_id_col='PLAYER_ID', date_col='GAME_DATE')
    df = getPlayerAvgToDateVectorized(df) 
    df = features_teamContext(df)
    df = addLagFeatures(df)
    df = assign_opponent_team_stats_dict(df)
    df = add_volatility_features(df, windows=[7,10,20,40])
    df = assign_team_opp_def_by_position(df, min_minutes=1)
    df = process_star_players_data(df, min_minutes=20, min_games=5)
    df = add_performance_without_stars_columns(df, min_games=1)
    df = add_opponent_team_rolling_stats(df, windows=[3,5])
    df = expectedPace(df)
    df = add_usual_starters_availability(df)
    df = add_team_rolling_stats(df, windows=[3, 5, 7,])
    df = calculate_league_avg_team_def_rating(df)
    df = calculate_league_avg_team_pace(df)
    df = calculate_league_avg_team_off_rating(df)
    df = get_standard_deviation(df, windows=[5,10,15])
    df = add_lineup_usage_fga_share(df)
    df = add_interaction_features(df)
    df = test_features(df)
    df = add_team_min_rank(df)
    df = add_team_fga_rank(df)
    df = add_team_pts_rank(df)
    df = add_team_fg3a_rank(df)
    df = add_team_fta_rank(df)
    df = add_performance_without_top_ranked_columns(df, stat='MIN')
    df = add_performance_without_top_ranked_columns(df, stat='PTS')
    df = add_performance_without_top_ranked_columns(df, stat='FGA')
    df = add_performance_without_top_ranked_columns(df, stat='FG3A')
    df = add_performance_without_top_ranked_columns(df, stat='FTA')
    return df


def count_line_hits(player_df, line, category, game_windows=[5, 10, 15]):
    results = {}
    player_df_sorted = player_df.sort_values('GAME_DATE')
    total_games = len(player_df_sorted)

    for window in game_windows:
        # Handle players with fewer games
        if total_games < window:
            last_n_games = player_df_sorted
        else:
            last_n_games = player_df_sorted.tail(window)

        if category == 'player_points':
            hits = (last_n_games['PTS'] > line).sum()
        elif category == 'player_assists':
            hits = (last_n_games['AST'] > line).sum()
        elif category == 'player_rebounds':
            hits = (last_n_games['REB'] > line).sum()
        elif category == 'player_threes':
            hits = (last_n_games['FG3M'] > line).sum()
        elif category == 'player_blocks':
            hits = (last_n_games['BLK'] > line).sum()
        elif category == 'player_steals':
            hits = (last_n_games['STL'] > line).sum()
        elif category == 'player_field_goals':
            hits = (last_n_games['FGM'] > line).sum()
        elif category == 'player_frees_made':
            hits = (last_n_games['FTM'] > line).sum()
        elif category == 'player_points_rebounds_assists':
            hits = (last_n_games['PTS'] + last_n_games['REB'] + last_n_games['AST'] > line).sum()
        elif category == 'player_points_rebounds':
            hits = (last_n_games['PTS'] + last_n_games['REB'] > line).sum()
        elif category == 'player_points_assists':
            hits = (last_n_games['PTS'] + last_n_games['AST'] > line).sum()
        elif category == 'player_rebounds_assists':
            hits = (last_n_games['REB'] + last_n_games['AST'] > line).sum()
        elif category == 'player_turnovers':
            hits = (last_n_games['TOV'] > line).sum()
        elif category == 'player_blocks_steals':
            hits = (last_n_games['BLK'] + last_n_games['STL'] > line).sum()
        else:
            hits = 0

        results['NAME'] = player_df_sorted['PLAYER_NAME'].iloc[0] if total_games > 0 else 'Unknown'
        results['CATEGORY'] = category
        results['LINE'] = line
        results[f'L-{window}'] = round(hits / window, 2)


    return results