import requests 
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2
from PRODUCTION.teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive, nameDict


today = datetime.today().strftime('%Y-%m-%d')

def calculate_slope(player_df, stat_col, window=5):
    if len(player_df) < 2:
        return 0.0
    
    # Sort by date to ensure chronological order
    player_df_sorted = player_df.sort_values(by='GAME_DATE')
    
    # Get the last 'window' values
    recent_values = player_df_sorted[stat_col].tail(window).values
    
    # Remove NaN values
    clean_values = recent_values[~pd.isna(recent_values)]
    
    # Need at least 2 points to calculate slope
    if len(clean_values) < 2:
        return 0.0
    
    # Create x-axis (game indices)
    x = np.arange(len(clean_values))
    y = clean_values
    
    n = len(x)
    
    # Check for zero variance (all x values the same - shouldn't happen, but safety check)
    if n < 2 or np.var(x) == 0:
        return 0.0
    
    # Calculate slope using the formula: (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
    numerator = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = n * np.sum(x**2) - np.sum(x)**2
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    
    return round(float(slope), 3)

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


def get_espn_games(date_str=today):  # YYYYMMDD format
    url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    response = requests.get(url)
    data = response.json()
    
    # Define timezone objects
    utc = pytz.UTC
    pst = pytz.timezone('America/Los_Angeles')

    games_list = []
    for event in data['events']:
        # Parse UTC time from ESPN
        utc_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%MZ').replace(tzinfo=utc)
        # Convert to PST
        pst_time = utc_time.astimezone(pst)
        
        game_dict = {
            'game_date': pst_time.strftime('%Y-%m-%d'),
            'home_team': event['competitions'][0]['competitors'][0]['team']['abbreviation'],
            'away_team': event['competitions'][0]['competitors'][1]['team']['abbreviation'],
            'game_time': pst_time.strftime('%I:%M %p'),  # 12-hour format with AM/PM
            'venue': event['competitions'][0]['venue']['fullName']
        }
        games_list.append(game_dict)
    
    return games_list

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
        res.append(1)
    else:
        res.append(0)

    # Team Star Player 
    if player_name == teamStarPlayer[player_team]:
        res.append(1)
    else:
        res.append(0)

    # Usual Starters Available
    main_in_projected = len(set(mainStartingFive[player_team]) & set(projectedStartingFive[player_team]))
    res.append(main_in_projected)

    if player_df.empty:
        res.append(0)
    else:
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

    # Rolling Averages
    res.append(player_df['PTS'].tail(3).mean())
    res.append(player_df['PTS'].tail(5).mean())
    res.append(player_df['PTS'].tail(10).mean())
    res.append(player_df['MIN'].tail(3).mean())
    res.append(player_df['MIN'].tail(5).mean())
    res.append(player_df['USG_PCT'].tail(3).mean())
    res.append(player_df['USG_PCT'].tail(5).mean())
    res.append(player_df['FGA'].tail(3).mean())
    res.append(player_df['FGA'].tail(5).mean())
    res.append(player_df['FGA'].tail(10).mean())
    res.append(player_df['FG3A'].tail(3).mean())
    res.append(player_df['FG3A'].tail(5).mean())
    res.append(player_df['FG3A'].tail(10).mean())
    res.append(player_df['FTA'].tail(3).mean())
    res.append(player_df['FTA'].tail(5).mean())
    res.append(player_df['FTA'].tail(10).mean())
    res.append(player_df['E_OFF_RATING'].tail(3).mean())
    res.append(player_df['E_OFF_RATING'].tail(5).mean())
    res.append(player_df['E_OFF_RATING'].tail(10).mean())
    res.append(player_df['NET_RATING'].tail(3).mean())
    res.append(player_df['NET_RATING'].tail(5).mean())
    res.append(player_df['NET_RATING'].tail(10).mean())
    res.append(player_df['NET_RATING'].tail(20).mean())
    res.append(player_df['PLUS_MINUS'].tail(3).mean())
    res.append(player_df['PLUS_MINUS'].tail(5).mean())
    res.append(player_df['PLUS_MINUS'].tail(10).mean())

    # Volatility
    res.append(calculate_volatility(player_df, 'MIN', 15))
    res.append(calculate_volatility(player_df, 'MIN', 20))
    res.append(calculate_volatility(player_df, 'PTS', 15))
    res.append(calculate_volatility(player_df, 'PTS', 20))
    res.append(calculate_volatility(player_df, 'FGA', 10))
    res.append(calculate_volatility(player_df, 'FGA', 15))
    res.append(calculate_volatility(player_df, 'FGA', 20))
    res.append(calculate_volatility(player_df, 'USG_PCT', 20, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', 20, use_cv=True))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 10))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 20))
    res.append(calculate_volatility(player_df, 'PLUS_MINUS', 10))
    res.append(calculate_volatility(player_df, 'PLUS_MINUS', 15))
    res.append(calculate_volatility(player_df, 'PLUS_MINUS', 20))

    # standard deviation
    res.append(player_df['PTS'].tail(3).std())
    res.append(player_df['PTS'].tail(5).std())
    res.append(player_df['PTS'].tail(7).std())
    res.append(player_df['MIN'].tail(3).std())
    res.append(player_df['MIN'].tail(5).std())
    res.append(player_df['MIN'].tail(7).std())
    res.append(player_df['FGA'].tail(3).std())
    res.append(player_df['FGA'].tail(5).std())
    res.append(player_df['FGA'].tail(7).std())
    res.append(player_df['USG_PCT'].tail(3).std())
    res.append(player_df['USG_PCT'].tail(5).std())
    res.append(player_df['USG_PCT'].tail(7).std())
    res.append(player_df['E_OFF_RATING'].tail(3).std())
    res.append(player_df['E_OFF_RATING'].tail(5).std())
    res.append(player_df['E_OFF_RATING'].tail(7).std())
    res.append(player_df['TS_PCT'].tail(3).std())
    res.append(player_df['TS_PCT'].tail(5).std())
    res.append(player_df['TS_PCT'].tail(7).std())
    res.append(player_df['NET_RATING'].tail(3).std())
    res.append(player_df['NET_RATING'].tail(5).std())
    res.append(player_df['NET_RATING'].tail(7).std())
    res.append(player_df['PLUS_MINUS'].tail(3).std())
    res.append(player_df['PLUS_MINUS'].tail(5).std())
    res.append(player_df['PLUS_MINUS'].tail(7).std())

    # short vs long term divergences
    pts_mean = player_df['PTS'].mean()
    pts_10_mean = player_df['PTS'].tail(10).mean()
    pts_20_mean = player_df['PTS'].tail(20).mean()
    fga_mean = player_df['FGA'].mean()
    fga_10_mean = player_df['FGA'].tail(10).mean()
    fga_20_mean = player_df['FGA'].tail(20).mean()
    
    res.append(pts_10_mean / pts_mean if pts_mean != 0 else 0.0)
    res.append(pts_10_mean / pts_20_mean if pts_20_mean != 0 else 0.0)
    res.append(fga_10_mean / fga_mean if fga_mean != 0 else 0.0)
    res.append(fga_10_mean / fga_20_mean if fga_20_mean != 0 else 0.0)
    
    # variance stability metrics
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0.0
    
    res.append(safe_divide(calculate_volatility(player_df, 'PTS', 10), calculate_volatility(player_df, 'PTS', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'MIN', 10), calculate_volatility(player_df, 'MIN', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'USG_PCT', 10), calculate_volatility(player_df, 'USG_PCT', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'TS_PCT', 10), calculate_volatility(player_df, 'TS_PCT', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'FGA', 10), calculate_volatility(player_df, 'FGA', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'FG3A', 10), calculate_volatility(player_df, 'FG3A', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'PTS_OFF_TOV', 10), calculate_volatility(player_df, 'PTS_OFF_TOV', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'PTS_2ND_CHANCE', 10), calculate_volatility(player_df, 'PTS_2ND_CHANCE', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'PTS_FB', 10), calculate_volatility(player_df, 'PTS_FB', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'PTS_PAINT', 10), calculate_volatility(player_df, 'PTS_PAINT', 40)))
    res.append(safe_divide(calculate_volatility(player_df, 'PLUS_MINUS', 10), calculate_volatility(player_df, 'PLUS_MINUS', 40)))

    #Trends
    res.append(calculate_slope(player_df, 'PTS', 5))
    res.append(calculate_slope(player_df, 'MIN', 5))
    res.append(calculate_slope(player_df, 'PTS', 10))
    res.append(calculate_slope(player_df, 'MIN', 10))
    
    #Interactions
    res.append((player_df['PTS'].mean() / (player_df['MIN'].mean() + 0.01)) * player_df['USG_PCT'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['TS_PCT'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['MIN'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['E_OFF_RATING'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['FGA'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['PTS'].mean())
    res.append(player_df['PTS'].mean() / (player_df['FGA'].mean() + 0.44 * player_df['FTA'].mean() + player_df['TOV'].mean() + 0.01))
    res.append(player_df['FGA'].mean()/(player_df['MIN'].mean() + 0.001))
    res.append(player_df['FTA'].mean()/(player_df['MIN'].mean() + 0.001))
    res.append(player_df['FG3A'].mean()/(player_df['MIN'].mean() + 0.001))
    res.append(player_df['POSS'].mean()/(player_df['MIN'].mean() + 0.001))
    
    #Star Dynamics
    starStatus = 1 if teamStarPlayer[player_team] not in projectedStartingFive[player_team] else 0
    starOut_df = player_df[player_df['STAR_SAT_OUT'] == 1]
    starIn_df = player_df[player_df['STAR_SAT_OUT'] == 0]
    res.append(starStatus * (starOut_df['PTS'].mean() - starIn_df['PTS'].mean()))
    res.append(starStatus * (starOut_df['FGA'].mean() - starIn_df['FGA'].mean()))
    res.append(starStatus * (starOut_df['FGM'].mean() - starIn_df['FGM'].mean()))
    res.append(starStatus * (starOut_df['FTA'].mean() - starIn_df['FTA'].mean()))
    res.append(starStatus * (starOut_df['FG3M'].mean() - starIn_df['FG3M'].mean()))
    res.append(starStatus * (starOut_df['PLUS_MINUS'].mean() - starIn_df['PLUS_MINUS'].mean()))
    res.append(len(starOut_df))

    #Tiers
    res.append(int(player_df['PTS'].mean() < 10) * player_df['MIN'].mean())
    res.append(int(player_df['PTS'].mean() < 10) * player_df['FGA'].mean())

    res.append(int((player_df['PTS'].mean() >= 10) & (player_df['PTS'].mean() <= 20)) * player_df['MIN'].mean())
    res.append(int((player_df['PTS'].mean() >= 10) & (player_df['PTS'].mean() <= 20)) * player_df['FGA'].mean())

    return res


def teamContext(player_name, data, teamStarPlayer, projectedStartingFive):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    # Get player's team to fetch ALL team games (not just games player played in)
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    team_df = data[data['TEAM_ABBREVIATION'] == player_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')
    res.append(team_df['TEAM_PTS'].tail(3).mean())
    res.append(team_df['TEAM_PTS'].tail(5).mean())
    res.append(team_df['TEAM_AST'].tail(3).mean())
    res.append(team_df['TEAM_AST'].tail(5).mean())
    res.append(team_df['TEAM_AST'].tail(10).mean())
    res.append(team_df['TEAM_OFF_RATING'].tail(3).mean())
    res.append(team_df['TEAM_OFF_RATING'].tail(5).mean())
    res.append(team_df['TEAM_OFF_RATING'].tail(20).mean())
    res.append(team_df['TEAM_OFF_RATING'].mean())
    res.append(team_df['TEAM_PACE'].tail(3).mean())
    res.append(team_df['TEAM_DEF_RATING'].tail(3).mean())
    res.append(team_df['TEAM_DEF_RATING'].tail(5).mean())
    res.append(team_df['TEAM_FTA'].tail(3).mean())
    return res

def playerVsOpp(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []

    opp_team, _ = findOpp(player_name, data, current_date)
    if opp_team is None:
        print(f"No opponent found for {player_name}")
        return None
        
    opp_df = data[data['TEAM_ABBREVIATION'] == opp_team].sort_values(by='GAME_DATE', ascending=True)
    opp_team_df = opp_df.drop_duplicates(subset=['GAME_ID'])
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    team_df = data[data['TEAM_ABBREVIATION'] == player_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')

    opp_guard_df = opp_df[(opp_df['GUARD'] == 1) & (opp_df.groupby('PLAYER_NAME')['MIN'].transform('mean') > 10)]
    opp_forward_df = opp_df[(opp_df['FORWARD'] == 1) & (opp_df.groupby('PLAYER_NAME')['MIN'].transform('mean') > 10)]
    opp_center_df = opp_df[(opp_df['CENTER'] == 1) & (opp_df.groupby('PLAYER_NAME')['MIN'].transform('mean') > 10)]
    fga_mean = player_df['FGA'].mean()
    player_3PA_rate = (player_df['FG3A'].mean() / fga_mean if fga_mean != 0 else 0.0) + 0.01
    playerFG_PCT = player_df['FG_PCT'].mean()
    
    # Team Stats
    res.append(opp_team_df['TEAM_DEF_RATING'].mean())
    res.append(opp_team_df['TEAM_PACE'].mean())
    res.append(opp_team_df['TEAM_DEF_RATING'].tail(5).mean())
    res.append(opp_team_df['TEAM_PACE'].tail(5).mean())
    res.append(opp_team_df['TEAM_TOV'].mean())
    res.append(opp_team_df['TEAM_BLK'].mean())
    res.append(opp_team_df['TEAM_STL'].mean())
    
    res.append(player_df['GUARD'] * (opp_guard_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['GUARD'] * player_3PA_rate * (opp_guard_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(player_df['GUARD'] * opp_guard_df['PTS_ALLOWED_PER_MIN'].mean())
    res.append(player_df['GUARD'] * opp_guard_df['DEF_RATING'].mean())

    
    res.append(player_df['FORWARD'] * (opp_forward_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['FORWARD'] * player_3PA_rate * (opp_forward_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(player_df['FORWARD'] * opp_forward_df['PTS_ALLOWED_PER_MIN'].mean())
    res.append(player_df['FORWARD'] * opp_forward_df['DEF_RATING'].mean())

    res.append(team_df['TEAM_OFF_RATING'].tail(3).mean() - opp_df['TEAM_DEF_RATING'].mean())
    expected_pace = (team_df['TEAM_PACE'].mean() + opp_df['TEAM_PACE'].mean()) / 2
    res.append(expected_pace)
    res.append(expected_pace * player_df['PTS'].mean())
    res.append(expected_pace * player_df['MIN'].mean())
    res.append(expected_pace * player_df['FGA'].mean())
    res.append(expected_pace * player_df['FG3A'].mean())
    res.append(expected_pace * player_df['FTA'].mean())
    res.append(expected_pace * player_df['USG_PCT'].mean())
    res.append((team_df['TEAM_PACE'].mean() - opp_df['TEAM_PACE'].mean()))
    return res


def playerMatchup(player_name, data, current_date_str):
    player_df = data[data['PLAYER_NAME']==player_name].copy()
    opp, home = findOpp(player_name, data, current_date_str)
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    home_df = player_df[player_df['HOME_GAME'] == 1]
    away_df = player_df[player_df['HOME_GAME'] == 0]
    
    res = []
    res.append(home * home_df['PTS'].mean() + (1 - home) * away_df['PTS'].mean())
    res.append(home * home_df['MIN'].mean() + (1 - home) * away_df['MIN'].mean())
    res.append(home * home_df['AST'].mean() + (1 - home) * away_df['AST'].mean())
    res.append(home * home_df['FGA'].mean() + (1 - home) * away_df['FGA'].mean())
    res.append(home * home_df['FGM'].mean() + (1 - home) * away_df['FGM'].mean())
    res.append(home * home_df['FTA'].mean() + (1 - home) * away_df['FTA'].mean())
    res.append(home * home_df['FG3A'].mean() + (1 - home) * away_df['FG3A'].mean())
    res.append(home * home_df['FG3M'].mean() + (1 - home) * away_df['FG3M'].mean())
    res.append(home * home_df['E_OFF_RATING'].mean() + (1 - home) * away_df['E_OFF_RATING'].mean())
    res.append(home * home_df['NET_RATING'].mean() + (1 - home) * away_df['NET_RATING'].mean())
    res.append(home * home_df['PLUS_MINUS'].mean() + (1 - home) * away_df['PLUS_MINUS'].mean())
    res.append(home * home_df['PTS_OFF_TOV'].mean() + (1 - home) * away_df['PTS_OFF_TOV'].mean())
    res.append(home * home_df['PTS_2ND_CHANCE'].mean() + (1 - home) * away_df['PTS_2ND_CHANCE'].mean())
    res.append(home * home_df['PTS_FB'].mean() + (1 - home) * away_df['PTS_FB'].mean())
    res.append(home * home_df['PTS_PAINT'].mean() + (1 - home) * away_df['PTS_PAINT'].mean())
    return res


def buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = [playerContext(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer) + 
    playerScoring(player_name, data, current_date, teamStarPlayer, projectedStartingFive) + 
    teamContext(player_name, data, teamStarPlayer, projectedStartingFive) + 
    playerVsOpp(player_name, data, current_date) + 
    playerMatchup(player_name, data, current_date)]
    
    return res

def makePrediction(player_name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE').copy()
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    # Now build the vector using the actual game date
    vector = buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
    vector = [item for sublist in vector for item in sublist]
    vector = pd.DataFrame([vector], columns=features)
    
    for col in vector.columns:
        vector[col] = pd.to_numeric(vector[col], errors='coerce')
    
    vector = vector.fillna(0)
    
    pred = model.predict(vector)[0]
    return round(float(pred), 3)
    
    
