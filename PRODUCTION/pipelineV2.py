import requests 
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2
from PRODUCTION.teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive, nameDict
from nba_api.stats.endpoints import leaguedashteamstats

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

def playerContext(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].copy()
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    player_name = player_df['PLAYER_NAME'].iloc[-1]
    
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    
    current_date_dt = pd.to_datetime(current_date)
    current_date_str = current_date_dt.strftime('%Y-%m-%d')
    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    res = []

    # Starting
    if player_name in projectedStartingFive[player_team]:
        res.append(int(1))
    else:
        res.append(int(0))

    # Player Days Rest
    days_rested = (current_date_dt - player_df['GAME_DATE'].max()).days
    res.append(int(days_rested))

    # Team Star Player
    if player_name in teamStarPlayer[player_team]:
        res.append(int(1))
    else:
        res.append(int(0))

    # Usual Starters Available
    main_in_projected = len(set(mainStartingFive[player_team]) & set(projectedStartingFive[player_team]))
    res.append(5 - main_in_projected)
    res.append(len(player_df))

    return res

def playerScoring(player_name, data, current_date, teamStarPlayer, projectedStartingFive):
    player_df = data[data['PLAYER_NAME'] == player_name].copy()
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    team_df = data[data['TEAM_ABBREVIATION'] == player_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')

    res = []

    # Baseline
    res.append(round_to_2(player_df['MIN'].mean()))
    res.append(round_to_2(player_df['PTS'].mean()))
    res.append(round_to_2(player_df['FGA'].mean()))
    res.append(round_to_2(player_df['FG3A'].mean()))
    res.append(round_to_2(player_df['FTA'].mean()))
    res.append(round_to_2(player_df['FG_PCT'].mean()))
    res.append(round_to_2(player_df['FG3_PCT'].mean()))
    res.append(round_to_2(player_df['FT_PCT'].mean()))
    res.append(round_to_2(player_df['USG_PCT'].mean()))
    res.append(round_to_2(player_df['TS_PCT'].mean()))
    res.append(round_to_2(player_df['POSS'].mean()))
    res.append(round_to_2(player_df['PLUS_MINUS'].mean()))
    
    # Rolling Averages
    res.append(round_to_2((player_df['PTS'].tail(5).mean()) / ((player_df['PTS'].mean() + 0.01))))
    res.append(round_to_2((player_df['PTS'].tail(7).mean()) / ((player_df['PTS'].mean() + 0.01))))
    res.append(round_to_2((player_df['PTS'].tail(10).mean()) / ((player_df['PTS'].mean() + 0.01))))
    res.append(round_to_2((player_df['MIN'].tail(5).mean()) / (player_df['MIN'].mean() + 0.01)))
    res.append(round_to_2((player_df['MIN'].tail(7).mean()) / ((player_df['MIN'].mean() + 0.01))))
    res.append(round_to_2((player_df['MIN'].tail(10).mean()) / ((player_df['MIN'].mean() + 0.01))))
    res.append(round_to_2((player_df['FGA'].tail(5).mean()) / (player_df['FGA'].mean() + 0.01)))
    res.append(round_to_2((player_df['FGA'].tail(7).mean()) / ((player_df['FGA'].mean() + 0.01))))
    res.append(round_to_2((player_df['FGA'].tail(10).mean()) / (player_df['FGA'].mean() + 0.01)))
    res.append(round_to_2((player_df['FTA'].tail(5).mean()) / ((player_df['FTA'].mean() + 0.01))))
    res.append(round_to_2((player_df['FTA'].tail(7).mean()) / ((player_df['FTA'].mean() + 0.01))))
    res.append(round_to_2((player_df['FTA'].tail(10).mean()) / ((player_df['FTA'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG3A'].tail(5).mean()) / ((player_df['FG3A'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG3A'].tail(7).mean()) / ((player_df['FG3A'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG3A'].tail(10).mean()) / ((player_df['FG3A'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG_PCT'].tail(5).mean()) / ((player_df['FG_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG_PCT'].tail(7).mean()) / ((player_df['FG_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG_PCT'].tail(10).mean()) / ((player_df['FG_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG3_PCT'].tail(5).mean()) / ((player_df['FG3_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FG3_PCT'].tail(7).mean()) / (player_df['FG3_PCT'].mean() + 0.01)))
    res.append(round_to_2((player_df['FG3_PCT'].tail(10).mean()) / ((player_df['FG3_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FT_PCT'].tail(5).mean()) / ((player_df['FT_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FT_PCT'].tail(7).mean()) / ((player_df['FT_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['FT_PCT'].tail(10).mean()) / ((player_df['FT_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['USG_PCT'].tail(5).mean()) / ((player_df['USG_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['USG_PCT'].tail(7).mean()) / ((player_df['USG_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['USG_PCT'].tail(10).mean()) / ((player_df['USG_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['TS_PCT'].tail(5).mean()) / ((player_df['TS_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['TS_PCT'].tail(7).mean()) / ((player_df['TS_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['TS_PCT'].tail(10).mean()) / ((player_df['TS_PCT'].mean() + 0.01))))
    res.append(round_to_2((player_df['POSS'].tail(5).mean()) / (player_df['POSS'].mean() + 0.01)))
    res.append(round_to_2((player_df['POSS'].tail(7).mean()) / (player_df['POSS'].mean() + 0.01)))
    res.append(round_to_2((player_df['POSS'].tail(10).mean()) / (player_df['POSS'].mean() + 0.01)))
    res.append(round_to_2((player_df['PLUS_MINUS'].tail(5).mean()) / (player_df['PLUS_MINUS'].mean() + 0.01)))
    res.append(round_to_2((player_df['PLUS_MINUS'].tail(7).mean()) / (player_df['PLUS_MINUS'].mean() + 0.01)))
    res.append(round_to_2((player_df['PLUS_MINUS'].tail(10).mean()) / (player_df['PLUS_MINUS'].mean() + 0.01)))
    
    # Variance Stability
    res.append(round_to_2(calculate_volatility(player_df, 'PTS', 10) / (calculate_volatility(player_df, 'PTS', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'FGA', 10) / (calculate_volatility(player_df, 'FGA', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'FG3A', 10) / (calculate_volatility(player_df, 'FG3A', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'FTA', 10) / (calculate_volatility(player_df, 'FTA', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'FT_PCT', 10) / (calculate_volatility(player_df, 'FT_PCT', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'FG_PCT', 10) / (calculate_volatility(player_df, 'FG_PCT', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'FG3_PCT', 10) / (calculate_volatility(player_df, 'FG3_PCT', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'USG_PCT', 10) / (calculate_volatility(player_df, 'USG_PCT', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'TS_PCT', 10) / (calculate_volatility(player_df, 'TS_PCT', 40) + 0.01)))
    res.append(round_to_2(calculate_volatility(player_df, 'POSS', 10) / (calculate_volatility(player_df, 'POSS', 40) + 0.01)))

    #Star Dynamics
    starStatus = 1 if teamStarPlayer[player_team] not in projectedStartingFive[player_team] else 0
    starOut_df = player_df[player_df['STAR_SAT_OUT'] == 1]
    starIn_df = player_df[player_df['STAR_SAT_OUT'] == 0]
    
    # Handle empty DataFrames to avoid NaN
    starOut_pts = starOut_df['PTS'].mean() if not starOut_df.empty else 0.0
    starIn_pts = starIn_df['PTS'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_pts - starIn_pts)))
    
    starOut_fga = starOut_df['FGA'].mean() if not starOut_df.empty else 0.0
    starIn_fga = starIn_df['FGA'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_fga - starIn_fga)))
    
    starOut_fta = starOut_df['FTA'].mean() if not starOut_df.empty else 0.0
    starIn_fta = starIn_df['FTA'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_fta - starIn_fta)))
    
    starOut_usg = starOut_df['USG_PCT'].mean() if not starOut_df.empty else 0.0
    starIn_usg = starIn_df['USG_PCT'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_usg - starIn_usg)))
    
    starOut_ts = starOut_df['TS_PCT'].mean() if not starOut_df.empty else 0.0
    starIn_ts = starIn_df['TS_PCT'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_ts - starIn_ts)))
    
    starOut_fgm = starOut_df['FGM'].mean() if not starOut_df.empty else 0.0
    starIn_fgm = starIn_df['FGM'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_fgm - starIn_fgm)))
    
    starOut_ftm = starOut_df['FTM'].mean() if not starOut_df.empty else 0.0
    starIn_ftm = starIn_df['FTM'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_ftm - starIn_ftm)))
    
    starOut_fg3m = starOut_df['FG3M'].mean() if not starOut_df.empty else 0.0
    starIn_fg3m = starIn_df['FG3M'].mean() if not starIn_df.empty else 0.0
    res.append(round_to_2(starStatus * (starOut_fg3m - starIn_fg3m)))
    res.append(int(len(starOut_df)))  # GAMES_WITHOUT_STAR is an integer count
    
    res.append(round_to_2(player_df['MIN'].tail(5).std()))
    res.append(round_to_2(player_df['MIN'].tail(10).std()))
    res.append(round_to_2(player_df['PTS'].tail(5).std()))
    res.append(round_to_2(player_df['PTS'].tail(10).std()))
    res.append(round_to_2(player_df['FGA'].tail(5).std()))
    res.append(round_to_2(player_df['FGA'].tail(10).std()))
    res.append(round_to_2(player_df['FG3A'].tail(5).std()))
    res.append(round_to_2(player_df['FG3A'].tail(10).std()))
    res.append(round_to_2(player_df['FTA'].tail(5).std()))
    res.append(round_to_2(player_df['FTA'].tail(10).std()))
    res.append(round_to_2(player_df['USG_PCT'].tail(5).std()))
    res.append(round_to_2(player_df['USG_PCT'].tail(10).std()))
    res.append(round_to_2(player_df['TS_PCT'].tail(5).std()))
    res.append(round_to_2(player_df['TS_PCT'].tail(10).std()))

    return res

def teamContext(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    # Get player's team to fetch ALL team games (not just games player played in)
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    team_df = data[data['TEAM_ABBREVIATION'] == player_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')
    res.append(round_to_2(team_df['TEAM_OFF_RATING'].mean() - league_df['OFF_RATING'].mean()))
    res.append(round_to_2(team_df['TEAM_OFF_RATING'].tail(3).mean() - league_df['OFF_RATING'].mean()))
    res.append(round_to_2(team_df['TEAM_OFF_RATING'].tail(5).mean() - league_df['OFF_RATING'].mean()))
    res.append(round_to_2(team_df['TEAM_PACE'].mean() - league_df['PACE'].mean()))
    res.append(round_to_2(team_df['TEAM_PACE'].tail(3).mean() - league_df['PACE'].mean()))
    res.append(round_to_2(team_df['TEAM_PACE'].tail(5).mean() - league_df['PACE'].mean()))
    
    return res


def playerVsOpp(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    all_players_df = data.sort_values(by='GAME_DATE')
    team_id = player_df['TEAM_ID'].iloc[-1]
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []

    opp_team, _ = findOpp(player_name, data, current_date)
    if opp_team is None:
        print(f"No opponent found for {player_name}")
        return None
        
    opp_df = data[data['TEAM_ABBREVIATION'] == opp_team].sort_values(by='GAME_DATE', ascending=True)
    if opp_df.empty:
        print(f"No opponent data found for {opp_team}")
        return None
    opp_team_id = opp_df['TEAM_ID'].iloc[-1]
    opp_team_df = data[data['TEAM_ABBREVIATION'] == opp_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')

    # Safely filter with None handling
    def safe_min_filter(df, position_col, min_threshold=10):
        """Filter dataframe by position and minimum minutes, handling None values"""
        if df.empty or 'MIN' not in df.columns or position_col not in df.columns:
            return df[df[position_col] == 1]  # Return empty or just position filter
        min_means = df.groupby('PLAYER_NAME')['MIN'].transform('mean')
        # Replace None/NaN with 0 for comparison
        min_means = min_means.fillna(0)
        return df[(df[position_col] == 1) & (min_means > min_threshold)]
    
    opp_guard_df = safe_min_filter(opp_df, 'GUARD', 10)
    opp_forward_df = safe_min_filter(opp_df, 'FORWARD', 10)
    opp_center_df = safe_min_filter(opp_df, 'CENTER', 10)
    league_guard_df = safe_min_filter(all_players_df, 'GUARD', 10)
    league_forward_df = safe_min_filter(all_players_df, 'FORWARD', 10)
    league_center_df = safe_min_filter(all_players_df, 'CENTER', 10)

    # Helper function to safely get league stats
    def get_league_stat(team_id, stat_name, default=100.0):
        try:
            if team_id in league_df.index:
                return league_df.at[team_id, stat_name]
            else:
                # Fallback: try to find by TEAM_ID column if index wasn't set properly
                if 'TEAM_ID' in league_df.columns:
                    team_row = league_df[league_df['TEAM_ID'] == team_id]
                    if not team_row.empty:
                        return team_row[stat_name].iloc[0]
                return default
        except (KeyError, IndexError):
            return default
    
    # Team Stats
    opp_def_rating = get_league_stat(opp_team_id, 'DEF_RATING', 100.0)
    opp_pace = get_league_stat(opp_team_id, 'PACE', 100.0)
    opp_off_rating = get_league_stat(opp_team_id, 'OFF_RATING', 100.0)
    league_def_rating = league_df['DEF_RATING'].mean()
    league_pace = league_df['PACE'].mean()
    league_off_rating = league_df['OFF_RATING'].mean()

    res.append(round_to_2(opp_def_rating - league_def_rating))
    res.append(round_to_2(opp_team_df['TEAM_DEF_RATING'].tail(3).mean() - league_df['DEF_RATING'].mean()))
    res.append(round_to_2(opp_off_rating - league_off_rating))
    res.append(round_to_2(opp_team_df['TEAM_OFF_RATING'].tail(3).mean() - league_df['OFF_RATING'].mean()))
    res.append(round_to_2(opp_pace - league_pace))
    res.append(round_to_2(opp_team_df['TEAM_PACE'].tail(3).mean() - league_df['PACE'].mean()))

    # team_pace = get_league_stat(team_id, 'PACE', 100.0)
    # opp_pace = get_league_stat(opp_team_id, 'PACE', 100.0)
    # expected_pace = (team_pace + opp_pace) / 2
    # res.append(expected_pace)

    res.append(player_df['GUARD'].iloc[-1] * float(round_to_2(opp_guard_df['E_DEF_RATING'].mean() - league_guard_df['E_DEF_RATING'].mean())))
    res.append(player_df['FORWARD'].iloc[-1] * float(round_to_2(opp_forward_df['E_DEF_RATING'].mean() - league_forward_df['E_DEF_RATING'].mean())))
    res.append(player_df['CENTER'].iloc[-1] * float(round_to_2(opp_center_df['E_DEF_RATING'].mean() - league_center_df['E_DEF_RATING'].mean())))
    return res

def playerMatchup(player_name, data, current_date_str):
    player_df = data[data['PLAYER_NAME']==player_name].copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    
    opp, home = findOpp(player_name, data, current_date_str)
    if opp is None or home is None:
        print(f"No opponent found for {player_name} in playerMatchup")
        return None
    
    home_df = player_df[player_df['HOME_GAME'] == 1]
    away_df = player_df[player_df['HOME_GAME'] == 0]
    
    # Handle empty DataFrames to avoid NaN
    home_min = home_df['MIN'].mean() if not home_df.empty else player_df['MIN'].mean()
    away_min = away_df['MIN'].mean() if not away_df.empty else player_df['MIN'].mean()
    res = []
    res.append(round_to_2((home * (home_min - player_df['MIN'].mean())) + ((1 - home) * (away_min - player_df['MIN'].mean()))))
    
    home_pts = home_df['PTS'].mean() if not home_df.empty else player_df['PTS'].mean()
    away_pts = away_df['PTS'].mean() if not away_df.empty else player_df['PTS'].mean()
    res.append(round_to_2((home * (home_pts - player_df['PTS'].mean())) + ((1 - home) * (away_pts - player_df['PTS'].mean()))))
    
    home_fga = home_df['FGA'].mean() if not home_df.empty else player_df['FGA'].mean()
    away_fga = away_df['FGA'].mean() if not away_df.empty else player_df['FGA'].mean()
    res.append(round_to_2((home * (home_fga - player_df['FGA'].mean())) + ((1 - home) * (away_fga - player_df['FGA'].mean()))))
    
    home_fg3a = home_df['FG3A'].mean() if not home_df.empty else player_df['FG3A'].mean()
    away_fg3a = away_df['FG3A'].mean() if not away_df.empty else player_df['FG3A'].mean()
    res.append(round_to_2((home * (home_fg3a - player_df['FG3A'].mean())) + ((1 - home) * (away_fg3a - player_df['FG3A'].mean()))))
    
    home_fta = home_df['FTA'].mean() if not home_df.empty else player_df['FTA'].mean()
    away_fta = away_df['FTA'].mean() if not away_df.empty else player_df['FTA'].mean()
    res.append(round_to_2((home * (home_fta - player_df['FTA'].mean())) + ((1 - home) * (away_fta - player_df['FTA'].mean()))))
    
    home_usg = home_df['USG_PCT'].mean() if not home_df.empty else player_df['USG_PCT'].mean()
    away_usg = away_df['USG_PCT'].mean() if not away_df.empty else player_df['USG_PCT'].mean()
    res.append(round_to_2((home * (home_usg - player_df['USG_PCT'].mean())) + ((1 - home) * (away_usg - player_df['USG_PCT'].mean()))))
    
    home_ts = home_df['TS_PCT'].mean() if not home_df.empty else player_df['TS_PCT'].mean()
    away_ts = away_df['TS_PCT'].mean() if not away_df.empty else player_df['TS_PCT'].mean()
    res.append(round_to_2((home * (home_ts - player_df['TS_PCT'].mean())) + ((1 - home) * (away_ts - player_df['TS_PCT'].mean()))))

    # Re-fetch opp and home (already checked above, but keeping for consistency)
    opp, home = findOpp(player_name, data, current_date_str)
    if opp is None:
        res.append(0.0)
        res.append(0.0)
        res.append(0.0)
        res.append(0.0)
    else:
        player_vs_opp_df = player_df[player_df['OPP_ABBREVIATION'] == opp]
        if not player_vs_opp_df.empty:
            res.append(round_to_2(player_vs_opp_df['PTS'].mean() - player_df['PTS'].mean()))
            res.append(round_to_2(player_vs_opp_df['FGA'].mean() - player_df['FGA'].mean()))
            res.append(round_to_2(player_vs_opp_df['USG_PCT'].mean() - player_df['USG_PCT'].mean()))
            res.append(round_to_2(player_vs_opp_df['TS_PCT'].mean() - player_df['TS_PCT'].mean()))
        else:
            res.append(0.0)
            res.append(0.0)
            res.append(0.0)
            res.append(0.0)
    
    return res

def buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    
    # Get results from each function, checking for None returns
    player_context = playerContext(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
    player_scoring = playerScoring(player_name, data, current_date, teamStarPlayer, projectedStartingFive)
    player_matchup = playerMatchup(player_name, data, current_date)
    team_context = teamContext(player_name, data)
    player_vs_opp = playerVsOpp(player_name, data, current_date)
    
    # Check if any function returned None
    if any(x is None for x in [player_context, player_scoring, player_matchup, team_context, player_vs_opp]):
        print(f"One or more feature functions returned None for {player_name}")
        return None
    
    res = [player_context + player_scoring + player_matchup + team_context + player_vs_opp]
    
    return res

def makePrediction(player_name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    # Now build the vector using the actual game date
    vector = buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
    if vector is None:
        print(f"buildVector returned None for {player_name}")
        return None
    
    vector = [item for sublist in vector for item in sublist]
    vector = pd.DataFrame([vector], columns=features)
    
    for col in vector.columns:
        vector[col] = pd.to_numeric(vector[col], errors='coerce')
    
    vector = vector.fillna(0)
    
    # Check if model is a tuple (split model: mean, variance, calibration_factor, opt_params)
    if isinstance(model, tuple):
        mean_model = model[0]
        variance_model = model[1]
        calibration_factor = model[2] if len(model) > 2 else 1.25
        score_dependent_params = model[3] if len(model) > 3 else None
        use_score_dependent = score_dependent_params is not None
        
        from MODELS.ngboostModel import predict_mean_variance_split
        
        # Use the proper prediction function for split models
        pred_mean, _ = predict_mean_variance_split(
            mean_model, variance_model, vector, features, calibration_factor,
            prediction_type='mean',
            use_score_dependent_calibration=use_score_dependent,
            score_dependent_params=score_dependent_params
        )
        
        # Extract scalar prediction
        if isinstance(pred_mean, np.ndarray):
            pred = float(pred_mean.flat[0])
        elif isinstance(pred_mean, pd.Series):
            pred = float(pred_mean.iloc[0])
        else:
            pred = float(pred_mean)
        
        return round(pred, 3)
    else:
        # Fallback for single model (legacy support)
        pred = model.predict(vector)[0]
        return round(float(pred), 3)
    
    
