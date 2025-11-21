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
    
    # Home or Away
    opp_abv, home = findOpp(player_name, data, current_date_str)
    res.append(home)

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

    # Team Star Out
    if teamStarPlayer[player_team] not in projectedStartingFive[player_team]:
        res.append(1)
    else:
        res.append(0)

    # Days Rested
    days_rested = (current_date_dt - player_df['GAME_DATE'].iloc[-1]).days
    res.append(days_rested)

    # Games Missed Last 5
    player_df_sorted = player_df.sort_values('GAME_DATE')
    team_games = data[data['TEAM_ABBREVIATION'] == player_team].copy()
    team_games = team_games.sort_values('GAME_DATE')
    last_5_team_games = team_games.tail(5)['GAME_DATE'].values
    games_missed = 0
    for team_game_date in last_5_team_games:
        player_played = player_df_sorted[player_df_sorted['GAME_DATE'] == team_game_date]
        if player_played.empty:
            games_missed += 1
    res.append(games_missed)
  
    # Days Rest After Missed
    res.append(days_rested)

    # Long Rest Indicator
    res.append(int(days_rested > 7))

    # Usual Starters Available
    main_in_projected = len(set(mainStartingFive[player_team]) & set(projectedStartingFive[player_team]))
    res.append(main_in_projected)

    opp_in_projected = len(set(mainStartingFive[opp_abv]) & set(projectedStartingFive[opp_abv]))
    res.append(opp_in_projected)
    opp_star_out = 1 if teamStarPlayer[opp_abv] not in projectedStartingFive[opp_abv] else 0
    res.append(opp_star_out)

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

    # Season Averages
    res.append(player_df['PTS'].mean())
    res.append(player_df['MIN'].mean())
    res.append(player_df['USG_PCT'].mean())
    res.append(player_df['TS_PCT'].mean())
    res.append(player_df['FGA'].mean())
    res.append(player_df['FTA'].mean())
    res.append(player_df['FG3A'].mean())
    res.append(player_df['FG_PCT'].mean())
    res.append(player_df['FT_PCT'].mean())
    res.append(player_df['FG3_PCT'].mean())
    res.append(player_df['TCHS'].mean())
    res.append(player_df['POSS'].mean())

    # Rolling Averages
    res.append(player_df['PTS'].tail(3).mean())
    res.append(player_df['PTS'].tail(5).mean())
    res.append(player_df['PTS'].tail(10).mean())
    res.append(player_df['PTS'].tail(20).mean())
    res.append(player_df['MIN'].tail(3).mean())
    res.append(player_df['MIN'].tail(5).mean())
    res.append(player_df['MIN'].tail(10).mean())
    res.append(player_df['MIN'].tail(20).mean())
    res.append(player_df['USG_PCT'].tail(3).mean())
    res.append(player_df['USG_PCT'].tail(5).mean())
    res.append(player_df['USG_PCT'].tail(10).mean())
    res.append(player_df['USG_PCT'].tail(20).mean())
    res.append(player_df['TS_PCT'].tail(10).mean())
    res.append(player_df['TS_PCT'].tail(20).mean())
    res.append(player_df['FGA'].tail(3).mean())
    res.append(player_df['FGA'].tail(5).mean())
    res.append(player_df['FGA'].tail(10).mean())
    res.append(player_df['FGA'].tail(20).mean())
    res.append(player_df['FG_PCT'].tail(10).mean())
    res.append(player_df['FG_PCT'].tail(20).mean())
    res.append(player_df['FG3A'].tail(5).mean())
    res.append(player_df['FG3A'].tail(10).mean())
    res.append(player_df['FG3A'].tail(20).mean())
    res.append(player_df['FG3_PCT'].tail(10).mean())
    res.append(player_df['FG3_PCT'].tail(20).mean())
    res.append(player_df['FTA'].tail(5).mean())
    res.append(player_df['FTA'].tail(10).mean())
    res.append(player_df['FTA'].tail(20).mean())
    res.append(player_df['FT_PCT'].tail(10).mean())
    res.append(player_df['FT_PCT'].tail(20).mean())
    res.append(player_df['E_OFF_RATING'].tail(5).mean())
    res.append(player_df['E_OFF_RATING'].tail(10).mean())
    res.append(player_df['E_OFF_RATING'].tail(20).mean())
    res.append(player_df['NET_RATING'].tail(5).mean())
    res.append(player_df['NET_RATING'].tail(10).mean())
    res.append(player_df['NET_RATING'].tail(20).mean())
    res.append(player_df['PIE'].tail(20).mean())
    
    # Volatility
    res.append(calculate_volatility(player_df, 'MIN', 5))
    res.append(calculate_volatility(player_df, 'MIN', 10))
    res.append(calculate_volatility(player_df, 'MIN', 20))
    res.append(calculate_volatility(player_df, 'PTS', 5))
    res.append(calculate_volatility(player_df, 'PTS', 10))
    res.append(calculate_volatility(player_df, 'PTS', 20))
    res.append(calculate_volatility(player_df, 'FGA', 5))
    res.append(calculate_volatility(player_df, 'FGA', 10))
    res.append(calculate_volatility(player_df, 'FGA', 20))
    res.append(calculate_volatility(player_df, 'USG_PCT', 5, use_cv=True))
    res.append(calculate_volatility(player_df, 'USG_PCT', 10, use_cv=True))
    res.append(calculate_volatility(player_df, 'USG_PCT', 20, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', 10, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', 20, use_cv=True))
    res.append(calculate_volatility(player_df, 'PACE', 5))
    res.append(calculate_volatility(player_df, 'PACE', 10))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 5))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 10))


    #MIN/MAX
    res.append(player_df['MIN'].tail(10).max())
    res.append(player_df['MIN'].tail(10).min())
    res.append(player_df['PTS'].tail(10).min())
    res.append(player_df['PTS'].tail(10).max())
    res.append(player_df['PTS'].tail(20).min())
    res.append(player_df['PTS'].tail(20).max())
    res.append(player_df['USG_PCT'].tail(10).max())
    res.append(player_df['USG_PCT'].tail(10).min())
    res.append(player_df['TS_PCT'].tail(10).max())
    res.append(player_df['TS_PCT'].tail(10).min())
    res.append((player_df['PTS'].tail(10).max() - player_df['PTS'].tail(10).min()))
    res.append((player_df['USG_PCT'].tail(10).max() - player_df['USG_PCT'].tail(10).min()))
    res.append((player_df['TS_PCT'].tail(10).max() - player_df['TS_PCT'].tail(10).min()))
    
    # short vs long term divergences
    res.append(player_df['PTS'].tail(5).mean() / player_df['PTS'].tail(20).mean())
    res.append(player_df['MIN'].tail(5).mean() / player_df['MIN'].tail(20).mean())
    res.append(player_df['USG_PCT'].tail(5).mean() / player_df['USG_PCT'].tail(20).mean())
    res.append(player_df['PTS'].tail(5).mean() / player_df['PTS'].mean())
    res.append(player_df['MIN'].tail(5).mean() / player_df['MIN'].mean())
    res.append(player_df['USG_PCT'].tail(5).mean() / player_df['USG_PCT'].mean())
    res.append(player_df['TS_PCT'].tail(5).mean() / player_df['TS_PCT'].mean())
    res.append(player_df['FGA'].tail(5).mean() / player_df['FGA'].mean())
    res.append(calculate_volatility(player_df, 'PTS', 5) / calculate_volatility(player_df, 'PTS', len(player_df)) + 0.01)
    res.append(calculate_volatility(player_df, 'MIN', 5) / calculate_volatility(player_df, 'MIN', len(player_df)) + 0.01)
    res.append(calculate_volatility(player_df, 'USG_PCT', 5, use_cv=True) / calculate_volatility(player_df, 'USG_PCT', len(player_df), use_cv=True) + 0.01)
    res.append(calculate_volatility(player_df, 'TS_PCT', 5, use_cv=True) / calculate_volatility(player_df, 'TS_PCT', len(player_df), use_cv=True) + 0.01)
    res.append(calculate_volatility(player_df, 'FGA', 5) / calculate_volatility(player_df, 'FGA', len(player_df)) + 0.01)

    # consistency metrics
    res.append(1 / (calculate_volatility(player_df, 'PTS', 10) + 0.01))
    res.append(1 / (calculate_volatility(player_df, 'MIN', 10) + 0.01))
    res.append(1 / (calculate_volatility(player_df, 'USG_PCT', 10) + 0.01))
    
    # variance stability metrics
    res.append(calculate_volatility(player_df, 'PTS', 10) / calculate_volatility(player_df, 'PTS', 40))
    res.append(calculate_volatility(player_df, 'MIN', 10) / calculate_volatility(player_df, 'MIN', 40))
    res.append(calculate_volatility(player_df, 'USG_PCT', 10) / calculate_volatility(player_df, 'USG_PCT', 40))
    res.append(calculate_volatility(player_df, 'TS_PCT', 10) / calculate_volatility(player_df, 'TS_PCT', 40))
    
    #Trends
    res.append(calculate_slope(player_df, 'PTS', 5))
    res.append(calculate_slope(player_df, 'MIN', 5))
    res.append(calculate_slope(player_df, 'PTS', 10))
    res.append(calculate_slope(player_df, 'MIN', 10))
    
    #Interactions
    pts_avg_to_date = player_df['PTS'].mean()  
    net_rating_avg_to_date = player_df['NET_RATING'].mean()
    is_star = 1 if player_name == teamStarPlayer[player_team] else 0
    res.append(pts_avg_to_date * is_star)
    res.append(net_rating_avg_to_date * is_star)
    res.append((player_df['PTS'].mean() / (player_df['MIN'].mean() + 0.01)) * player_df['USG_PCT'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['TS_PCT'].mean())
    res.append(player_df['USG_PCT'].mean() * player_df['MIN'].mean())
    res.append(player_df['PTS'].mean() / (player_df['FGA'].mean() + 0.44 * player_df['FTA'].mean() + player_df['TOV'].mean() + 0.01))
    current_date_dt = pd.to_datetime(current_date)
    current_date_str = current_date_dt.strftime('%Y-%m-%d')
    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    days_rested = (current_date_dt - player_df['GAME_DATE'].iloc[-1]).days
    b2b = 1 if days_rested == 1 else 0
    res.append(b2b * player_df['MIN'].tail(5).mean())
    starPlayerOut = 1 if teamStarPlayer[player_team] not in projectedStartingFive[player_team] else 0
    res.append(starPlayerOut * player_df['USG_PCT'].mean())
    res.append(player_df['PTS'].mean() / 36 )

    #Shot Profile
    res.append(player_df['percentagePointsMidrange2pt'].mean())
    res.append(player_df['percentagePointsPaint'].mean())

    #Star Dynamics
    res.append(player_df['PTS_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['USG_PCT_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['MIN_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['FGA_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['TS_PCT_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['GAMES_WITH_STAR'].iloc[-1])

    #Streaks & Tiers
    res.append(int(player_df['PTS'].mean() < 10))
    res.append(int((player_df['PTS'].mean() >= 10) & (player_df['PTS'].mean() <= 20)))
    res.append(int(player_df['PTS'].mean() > 20))

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
    res.append(team_df['TEAM_AST'].tail(3).mean())
    res.append(team_df['TEAM_OFF_RATING'].tail(3).mean())
    res.append(team_df['TEAM_PACE'].tail(3).mean())
    res.append(team_df['TEAM_DEF_RATING'].tail(3).mean())
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
    player_3PA_rate = player_df['FG3A'].mean() / player_df['FGA'].mean() + 0.01
    playerFG_PCT = player_df['FG_PCT'].mean()
    
    # Team Stats
    res.append(opp_team_df['TEAM_DEF_RATING'].tail(5).mean())
    res.append(opp_team_df['TEAM_PACE'].tail(5).mean())
    res.append(opp_team_df['TEAM_TOV'].mean())
    res.append(opp_team_df['TEAM_BLK'].mean())
    res.append(opp_team_df['TEAM_STL'].mean())

    # Opponent Player Stats
    res.append(opp_guard_df['E_DEF_RATING'].mean())
    res.append(opp_forward_df['E_DEF_RATING'].mean())
    res.append(opp_center_df['E_DEF_RATING'].mean())
    
    res.append(player_df['GUARD'] * (opp_guard_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['GUARD'] * player_3PA_rate * (opp_guard_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(opp_guard_df['PTS_ALLOWED_PER_MIN'].mean())

    
    res.append(player_df['FORWARD'] * (opp_forward_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['FORWARD'] * player_3PA_rate * (opp_forward_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(opp_forward_df['PTS_ALLOWED_PER_MIN'].mean())

    res.append(player_df['CENTER'] * (opp_center_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['CENTER'] * player_3PA_rate * (opp_center_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(opp_center_df['PTS_ALLOWED_PER_MIN'].mean())

    res.append(team_df['TEAM_OFF_RATING'].tail(3).mean() - opp_df['TEAM_DEF_RATING'].mean())
    res.append((team_df['TEAM_PACE'].mean() + opp_df['TEAM_PACE'].mean()) / 2)
    res.append((team_df['TEAM_PACE'].mean() - opp_df['TEAM_PACE'].mean()))
    return res


def playerMatchup(player_name, data, current_date_str):
    player_df = data[data['PLAYER_NAME']==player_name].copy()
    opp, home = findOpp(player_name, data, current_date_str)
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    home_df = player_df[player_df['HOME_GAME'] == 1]
    
    res = []
    res.append(home_df['PTS'].mean() - player_df['PTS'].mean())
    res.append(home_df['USG_PCT'].mean() - player_df['USG_PCT'].mean())
    res.append(home_df['FGA'].mean() - player_df['FGA'].mean())
    res.append(home_df['FTA'].mean() - player_df['FTA'].mean())
    res.append(home_df['TS_PCT'].mean() - player_df['TS_PCT'].mean())
    
    oppTeam = player_df[player_df['OPP_ABBREVIATION'] == opp]
    if oppTeam.empty:
        res.append(0)
        res.append(0)
        res.append(0)
        res.append(0)
    else:
        res.append(oppTeam['PTS'].mean() - player_df['PTS'].mean())
        res.append(oppTeam['MIN'].mean() - player_df['MIN'].mean())
        res.append(oppTeam['USG_PCT'].mean() - player_df['USG_PCT'].mean())
        res.append(oppTeam['TS_PCT'].mean() - player_df['TS_PCT'].mean())
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
    
    
