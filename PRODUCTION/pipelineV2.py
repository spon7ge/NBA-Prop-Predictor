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
    
    # # Home or Away
    # opp_abv, home = findOpp(player_name, data, current_date_str)
    # res.append(home)

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

    # # Team Star Out
    # if teamStarPlayer[player_team] not in projectedStartingFive[player_team]:
    #     res.append(1)
    # else:
    #     res.append(0)

    # Days Rested
    # days_rested = (current_date_dt - player_df['GAME_DATE'].iloc[-1]).days
    # res.append(days_rested)

    # # Games Missed Last 5
    # player_df_sorted = player_df.sort_values('GAME_DATE')
    # team_games = data[data['TEAM_ABBREVIATION'] == player_team].copy()
    # team_games = team_games.sort_values('GAME_DATE')
    # last_5_team_games = team_games.tail(5)['GAME_DATE'].values
    # games_missed = 0
    # for team_game_date in last_5_team_games:
    #     player_played = player_df_sorted[player_df_sorted['GAME_DATE'] == team_game_date]
    #     if player_played.empty:
    #         games_missed += 1
    # res.append(games_missed)
  
    # # Days Rest After Missed
    # res.append(days_rested)

    # # Long Rest Indicator
    # res.append(int(days_rested > 7))

    # Usual Starters Available
    main_in_projected = len(set(mainStartingFive[player_team]) & set(projectedStartingFive[player_team]))
    res.append(main_in_projected)

    # opp_in_projected = len(set(mainStartingFive[opp_abv]) & set(projectedStartingFive[opp_abv]))
    # res.append(opp_in_projected)
    # opp_star_out = 1 if teamStarPlayer[opp_abv] not in projectedStartingFive[opp_abv] else 0
    # res.append(opp_star_out)

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
    res.append(player_df['FGA'].mean())
    res.append(player_df['FGM'].mean())
    res.append(player_df['AST'].mean())
    res.append(player_df['USG_PCT'].mean())
    res.append(player_df['E_OFF_RATING'].mean())
    res.append(player_df['FTA'].mean())
    res.append(player_df['FTM'].mean())
    res.append(player_df['FG3A'].mean())
    res.append(player_df['FG3M'].mean())
    res.append(player_df['TCHS'].mean())
    res.append(player_df['POSS'].mean())
    res.append(player_df['UFGA'].mean())
    res.append(player_df['UFGM'].mean())
    res.append(player_df['CFGA'].mean())
    res.append(player_df['CFGM'].mean())
    res.append(player_df['DIST'].mean())
    res.append(player_df['SPD'].mean())
    res.append(player_df['DFGA'].mean())
    res.append(player_df['DFGM'].mean())
    res.append(player_df['PIE'].mean())


    # LAGS
    res.append(player_df['STARTING'].iloc[-1])
    res.append(player_df['STARTING'].iloc[-2])
    res.append(player_df['PTS'].iloc[-1])
    res.append(player_df['FGA'].iloc[-1])
    res.append(player_df['POSS'].iloc[-1])
    res.append(player_df['POSS'].iloc[-2])
    res.append(player_df['MIN'].iloc[-1])
    res.append(player_df['USG_PCT'].iloc[-1])
    res.append(player_df['TCHS'].iloc[-1])
    res.append(player_df['CFGA'].iloc[-1])
    res.append(player_df['UFGA'].iloc[-1])

    # Rolling Averages
    res.append(player_df['PTS'].tail(3).mean())
    res.append(player_df['PTS'].tail(5).mean())
    res.append(player_df['PTS'].tail(10).mean())
    res.append(player_df['PTS'].tail(20).mean())
    res.append(player_df['MIN'].tail(3).mean())
    res.append(player_df['MIN'].tail(5).mean())
    res.append(player_df['USG_PCT'].tail(3).mean())
    res.append(player_df['USG_PCT'].tail(5).mean())
    res.append(player_df['USG_PCT'].tail(10).mean())
    res.append(player_df['FGA'].tail(3).mean())
    res.append(player_df['FGA'].tail(5).mean())
    res.append(player_df['FGA'].tail(10).mean())
    res.append(player_df['FGA'].tail(20).mean())
    res.append(player_df['FGM'].tail(10).mean())
    res.append(player_df['FGM'].tail(20).mean())
    res.append(player_df['FG3A'].tail(5).mean())
    res.append(player_df['FG3A'].tail(10).mean())
    res.append(player_df['FG3A'].tail(20).mean())
    res.append(player_df['FTA'].tail(3).mean())
    res.append(player_df['FTA'].tail(5).mean())
    res.append(player_df['FTA'].tail(10).mean())
    res.append(player_df['FTA'].tail(20).mean())
    res.append(player_df['E_OFF_RATING'].tail(5).mean())
    res.append(player_df['E_OFF_RATING'].tail(10).mean())
    res.append(player_df['E_OFF_RATING'].tail(20).mean())
    res.append(player_df['NET_RATING'].tail(5).mean())
    res.append(player_df['NET_RATING'].tail(10).mean())
    res.append(player_df['NET_RATING'].tail(20).mean())
    res.append(player_df['UFGA'].tail(5).mean())
    res.append(player_df['UFGA'].tail(10).mean())
    res.append(player_df['CFGA'].tail(5).mean())
    res.append(player_df['CFGA'].tail(10).mean())
    res.append(player_df['POSS'].tail(3).mean())
    res.append(player_df['POSS'].tail(5).mean())
    res.append(player_df['POSS'].tail(10).mean())
    res.append(player_df['POSS'].tail(20).mean())
    res.append(player_df['TCHS'].tail(3).mean())
    res.append(player_df['TCHS'].tail(5).mean())
    res.append(player_df['TCHS'].tail(10).mean())
    res.append(player_df['TCHS'].tail(20).mean())
    res.append(player_df['DIST'].tail(3).mean())
    res.append(player_df['DIST'].tail(5).mean())
    res.append(player_df['DIST'].tail(10).mean())
    res.append(player_df['DIST'].tail(20).mean())
    res.append(player_df['percentageUnassistedFGM'].tail(10).mean())
    res.append(player_df['percentageUnassistedFGM'].tail(20).mean())
    res.append(player_df['percentageAssistedFGM'].tail(10).mean())
    res.append(player_df['percentageAssistedFGM'].tail(20).mean())

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
    res.append(calculate_volatility(player_df, 'FTA', 5))
    res.append(calculate_volatility(player_df, 'FTA', 10))
    res.append(calculate_volatility(player_df, 'FTA', 20))
    res.append(calculate_volatility(player_df, 'FG3A', 5))
    res.append(calculate_volatility(player_df, 'FG3A', 10))
    res.append(calculate_volatility(player_df, 'FG3A', 20))
    res.append(calculate_volatility(player_df, 'USG_PCT', 5, use_cv=True))
    res.append(calculate_volatility(player_df, 'USG_PCT', 10, use_cv=True))
    res.append(calculate_volatility(player_df, 'USG_PCT', 20, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', 10, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', 20, use_cv=True))
    res.append(calculate_volatility(player_df, 'PACE', 5))
    res.append(calculate_volatility(player_df, 'PACE', 10))
    res.append(calculate_volatility(player_df, 'PACE', 20))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 5))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 10))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', 20))
    res.append(calculate_volatility(player_df, 'POSS', 5))
    res.append(calculate_volatility(player_df, 'POSS', 10))
    res.append(calculate_volatility(player_df, 'POSS', 20))
    res.append(calculate_volatility(player_df, 'TCHS', 5))
    res.append(calculate_volatility(player_df, 'TCHS', 10))
    res.append(calculate_volatility(player_df, 'TCHS', 20))
    res.append(calculate_volatility(player_df, 'DIST', 5))
    res.append(calculate_volatility(player_df, 'DIST', 10))
    res.append(calculate_volatility(player_df, 'DIST', 20))

    #MIN/MAX
    res.append(player_df['MIN'].tail(10).max())
    res.append(player_df['MIN'].tail(10).min())
    res.append(player_df['MIN'].tail(20).max())
    res.append(player_df['PTS'].tail(10).min())
    res.append(player_df['PTS'].tail(10).max())
    res.append(player_df['PTS'].tail(20).min())
    res.append(player_df['PTS'].tail(20).max())
    res.append(player_df['USG_PCT'].tail(10).max())
    res.append(player_df['USG_PCT'].tail(10).min())
    res.append(player_df['TS_PCT'].tail(10).max())
    res.append((player_df['PTS'].tail(10).max() - player_df['PTS'].tail(10).min()))
    res.append((player_df['USG_PCT'].tail(10).max() - player_df['USG_PCT'].tail(10).min()))
    res.append((player_df['TS_PCT'].tail(10).max() - player_df['TS_PCT'].tail(10).min()))
    
    # short vs long term divergences
    res.append(player_df['PTS'].tail(10).mean() / player_df['PTS'].mean())
    res.append(player_df['PTS'].tail(10).mean() / player_df['PTS'].tail(40).mean())
    res.append(player_df['MIN'].tail(10).mean() / player_df['MIN'].mean())
    res.append(player_df['MIN'].tail(10).mean() / player_df['MIN'].tail(40).mean())
    res.append(player_df['USG_PCT'].tail(10).mean() / player_df['USG_PCT'].mean())
    res.append(player_df['USG_PCT'].tail(10).mean() / player_df['USG_PCT'].tail(40).mean())
    res.append(player_df['TS_PCT'].tail(10).mean() / player_df['TS_PCT'].mean())
    res.append(player_df['TS_PCT'].tail(10).mean() / player_df['TS_PCT'].tail(40).mean())
    res.append(player_df['FGA'].tail(10).mean() / player_df['FGA'].mean())
    res.append(player_df['FGA'].tail(10).mean() / player_df['FGA'].tail(40).mean())
    res.append(player_df['TCHS'].tail(5).mean() / player_df['TCHS'].mean())
    res.append(player_df['TCHS'].tail(10).mean() / player_df['TCHS'].tail(40).mean())

    # consistency metrics
    res.append(1 / (calculate_volatility(player_df, 'PTS', 10) + 0.01))
    res.append(1 / (calculate_volatility(player_df, 'MIN', 10) + 0.01))
    res.append(1 / (calculate_volatility(player_df, 'USG_PCT', 10) + 0.01))
    
    # variance stability metrics
    res.append(calculate_volatility(player_df, 'PTS', 10) / calculate_volatility(player_df, 'PTS', 40))
    res.append(calculate_volatility(player_df, 'MIN', 10) / calculate_volatility(player_df, 'MIN', 40))
    res.append(calculate_volatility(player_df, 'USG_PCT', 10) / calculate_volatility(player_df, 'USG_PCT', 40))
    res.append(calculate_volatility(player_df, 'TS_PCT', 10) / calculate_volatility(player_df, 'TS_PCT', 40))
    res.append(calculate_volatility(player_df, 'FTA', 10) / calculate_volatility(player_df, 'FTA', 40))
    res.append(calculate_volatility(player_df, 'FGA', 10) / calculate_volatility(player_df, 'FGA', 40))
    res.append(calculate_volatility(player_df, 'FG3A', 10) / calculate_volatility(player_df, 'FG3A', 40))
    
    #Trends
    res.append(calculate_slope(player_df, 'PTS', 5))
    res.append(calculate_slope(player_df, 'MIN', 5))
    res.append(calculate_slope(player_df, 'PTS', 10))
    res.append(calculate_slope(player_df, 'MIN', 10))
    
    #Interactions
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
    res.append(player_df['TCHS'].mean() / player_df['USG_PCT'].mean())
    res.append(player_df['DIST'].mean() / player_df['MIN'].mean())
    res.append(player_df['TCHS'].mean() / player_df['MIN'].mean())
    res.append(player_df['UFGA'].mean() / player_df['CFGA'].mean())
    res.append(player_df['UFGA'].mean() / player_df['FGA'].mean())
    res.append(player_df['CFGA'].mean() / player_df['FGA'].mean())

    #Shot Profile
    res.append(player_df['percentagePointsMidrange2pt'].mean())
    res.append(player_df['percentagePointsPaint'].mean())
    res.append(player_df['percentagePoints3pt'].mean())
    res.append(player_df['percentagePoints2pt'].mean())
    res.append(player_df['percentageUnassisted3pt'].mean())
    res.append(player_df['percentageAssistedFGM'].mean())
    res.append(player_df['percentageAssisted3pt'].mean())
    res.append(player_df['percentageFieldGoalsAttempted2pt'].mean())
    res.append(player_df['percentageFieldGoalsAttempted3pt'].mean())

    #Star Dynamics(star out performance - star in performance)
    starStatus = 1 if teamStarPlayer[player_team] not in projectedStartingFive[player_team] else 0
    starOut_df = player_df[player_df['STAR_SAT_OUT'] == 1]
    starIn_df = player_df[player_df['STAR_SAT_OUT'] == 0]
    res.append(starStatus * (starOut_df['PTS'].mean() - starIn_df['PTS'].mean()))
    res.append(starStatus * (starOut_df['USG_PCT'].mean() - starIn_df['USG_PCT'].mean()))
    res.append(starStatus * (starOut_df['MIN'].mean() - starIn_df['MIN'].mean()))
    res.append(starStatus * (starOut_df['FGA'].mean() - starIn_df['FGA'].mean()))
    res.append(starStatus * (starOut_df['FTA'].mean() - starIn_df['FTA'].mean()))
    res.append(starStatus * (starOut_df['FG3A'].mean() - starIn_df['FG3A'].mean()))
    res.append(starStatus * (starOut_df['UFGA'].mean() - starIn_df['UFGA'].mean()))
    res.append(starStatus * (starOut_df['UFGM'].mean() - starIn_df['UFGM'].mean()))
    res.append(starStatus * (starOut_df['CFGA'].mean() - starIn_df['CFGA'].mean()))
    res.append(starStatus * (starOut_df['CFGM'].mean() - starIn_df['CFGM'].mean()))
    res.append(starStatus * (starOut_df['DFGA'].mean() - starIn_df['DFGA'].mean()))
    res.append(starStatus * (starOut_df['DFGM'].mean() - starIn_df['DFGM'].mean()))
    res.append(starStatus * (starOut_df['TCHS'].mean() - starIn_df['TCHS'].mean()))
    res.append(starStatus * (starOut_df['POSS'].mean() - starIn_df['POSS'].mean()))
    res.append(starStatus * (starOut_df['EFG_PCT'].mean() - starIn_df['EFG_PCT'].mean()))
    res.append(len(starOut_df))

    #Streaks & Tiers
    res.append(int(player_df['PTS'].mean() < 10))
    res.append(int((player_df['PTS'].mean() >= 10) & (player_df['PTS'].mean() <= 20)))
    res.append(int(player_df['PTS'].mean() > 20))
    res.append(int(player_df['MIN'].mean() < 10))
    res.append(int((player_df['MIN'].mean() >= 10) & (player_df['MIN'].mean() <= 20)))
    res.append(int(player_df['MIN'].mean() > 20))

    #Volume Scorer
    volumeScore = (0.4 * (player_df['USG_PCT'].mean() / 35) + 0.3 * (player_df['FGA'].mean() / 20) + 0.2 * (player_df['MIN'].mean() / 36) + 0.1 * (player_df['PTS'].mean() / 30)).clip(0, 1)
    res.append(int(volumeScore > 0.7))
    res.append(int(volumeScore >= 0.4) & (int(volumeScore <= 0.7)))
    res.append(int(volumeScore < 0.4))

    #Archetypes
    catchAndShootSpecialist = (
        (player_df['percentageAssistedFGM'].mean() > 0.65) & 
        (player_df['percentageAssisted3pt'].mean() > 0.80) & 
        ((player_df['FG3A'].mean()/player_df['FGA'].mean()) > 0.40) & 
        (player_df['FG3A'].mean() > 3)
    )
    res.append(int(catchAndShootSpecialist))

    shotCreationSpecialist = (
        (player_df['percentageUnassistedFGM'].mean() > 0.35) & 
        (player_df['percentageUnassisted2pt'].mean() > 0.30) & 
        (player_df['percentageUnassisted3pt'].mean() > 0.30) & 
        ((player_df['USG_PCT'].mean() > 20) & 
        (player_df['AST'].mean() > 3.5)
    ))
    res.append(int(shotCreationSpecialist))

    rimAttacker = (
        (player_df['percentagePointsPaint'].mean() > 0.40) & 
        (player_df['percentagePoints2pt'].mean() > 0.30) & 
        (player_df['FTA'].mean() > 3.0) & 
        (player_df['percentagePointsFastBreak'].mean() > 0.15) & 
        (player_df['USG_PCT'].mean() > 20) & 
        ((player_df['FG3A'].mean()/player_df['FGA'].mean()) < 0.35)
    )
    res.append(int(rimAttacker))

    purePlaymaker = (
        (player_df['AST'].mean() > 6) & 
        ((player_df['AST'].mean()/(player_df['TOV'].mean() + 0.01)) > 2.5) & 
        (player_df['PTS'].mean() < 18) & 
        (player_df['USG_PCT'].mean() > 20)
    )
    res.append(int(purePlaymaker))

    ultimateTeamPlayer = (
        (player_df['USG_PCT'].mean() > 24) & 
        (player_df['PTS'].mean() > 18) & 
        (player_df['AST'].mean() > 4) & 
        (player_df['REB'].mean() > 5) & 
        (player_df['MIN'].mean() > 24)
    )
    res.append(int(ultimateTeamPlayer))

    floorSpacer = (
        ((player_df['FG3A'].mean()/player_df['FGA'].mean()) > 0.50) & 
        (player_df['FG3A'].mean() > 4) & 
        (player_df['USG_PCT'].mean() < 22) & 
        (player_df['percentageAssisted3pt'].mean() > 0.70)
    )
    res.append(int(floorSpacer))

    rollMan = (
        (player_df['percentagePointsPaint'].mean() > 0.45) & 
        (player_df['percentageAssisted2pt'].mean() > 0.60) & 
        (player_df['FGA'].mean() > 8) & 
        (player_df['USG_PCT'].mean() < 25)
    )
    res.append(int(rollMan))

    energyPlayer = (
        (player_df['percentagePointsPaint'].mean() > 0.45) & 
        (player_df['percentageAssisted2pt'].mean() > 0.60) & 
        (player_df['FGA'].mean() > 8) & 
        (player_df['USG_PCT'].mean() < 25)
    )
    res.append(int(energyPlayer))

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
    player_3PA_rate = player_df['FG3A'].mean() / player_df['FGA'].mean() + 0.01
    playerFG_PCT = player_df['FG_PCT'].mean()
    
    # Team Stats
    res.append(opp_team_df['TEAM_DEF_RATING'].mean())
    res.append(opp_team_df['TEAM_PACE'].mean())
    res.append(opp_team_df['TEAM_DEF_RATING'].tail(5).mean())
    res.append(opp_team_df['TEAM_PACE'].tail(5).mean())
    res.append(opp_team_df['TEAM_TOV'].mean())
    res.append(opp_team_df['TEAM_BLK'].mean())
    res.append(opp_team_df['TEAM_STL'].mean())

    # Opponent Player Stats
    opp_def_rating_avg = opp_team_df['TEAM_DEF_RATING'].mean()
    
    # ===== OVERALL DEFENSIVE DIFFICULTY =====
    # Normalize defensive rating (lower = better defense, so invert)
    # Range: 103-123 (103 = best, 123 = worst)
    oppDefScore = ((123 - opp_def_rating_avg) / 20).clip(0, 1)
    res.append(oppDefScore)
    
    # ===== POSITION-SPECIFIC DEFENSIVE DIFFICULTY =====
    # Check if position-specific defense data is available
    has_pos_def = (
        len(opp_guard_df) > 0 and 
        len(opp_forward_df) > 0 and 
        len(opp_center_df) > 0 and
        'DEF_FG_PCT_ALLOWED' in opp_guard_df.columns
    )
    
    if has_pos_def:
        # Position-specific FG% allowed (lower = better defense)
        # Invert so higher = tougher defense
        guard_def_fg_pct = opp_guard_df['DEF_FG_PCT_ALLOWED'].mean()
        forward_def_fg_pct = opp_forward_df['DEF_FG_PCT_ALLOWED'].mean()
        center_def_fg_pct = opp_center_df['DEF_FG_PCT_ALLOWED'].mean()
        
        guard_def = ((0.50 - guard_def_fg_pct) / 0.20).clip(0, 1)
        forward_def = ((0.50 - forward_def_fg_pct) / 0.20).clip(0, 1)
        center_def = ((0.50 - center_def_fg_pct) / 0.20).clip(0, 1)
        
        # Weight by player position
        opp_position_def_difficulty = (
            player_df['GUARD'].iloc[-1] * guard_def +
            player_df['FORWARD'].iloc[-1] * forward_def +
            player_df['CENTER'].iloc[-1] * center_def
        )
        
        # 3pt defense difficulty
        if 'DEF_3PT_PCT_ALLOWED' in opp_guard_df.columns:
            guard_3pt_def_pct = opp_guard_df['DEF_3PT_PCT_ALLOWED'].mean()
            forward_3pt_def_pct = opp_forward_df['DEF_3PT_PCT_ALLOWED'].mean()
            center_3pt_def_pct = opp_center_df['DEF_3PT_PCT_ALLOWED'].mean()
            
            guard_3pt_def = ((0.40 - guard_3pt_def_pct) / 0.15).clip(0, 1)
            forward_3pt_def = ((0.40 - forward_3pt_def_pct) / 0.15).clip(0, 1)
            center_3pt_def = ((0.40 - center_3pt_def_pct) / 0.15).clip(0, 1)
            
            opp_position_3pt_def_difficulty = (
                player_df['GUARD'].iloc[-1] * guard_3pt_def +
                player_df['FORWARD'].iloc[-1] * forward_3pt_def +
                player_df['CENTER'].iloc[-1] * center_3pt_def
            )
        else:
            opp_position_3pt_def_difficulty = opp_position_def_difficulty
    else:
        # Fallback: use overall defense if position-specific not available
        opp_position_def_difficulty = oppDefScore
        opp_position_3pt_def_difficulty = oppDefScore
    
    
    # ===== COMBINED DEFENSIVE DIFFICULTY SCORE =====
    oppScoreCombined = (
        0.6 * oppDefScore +
        0.4 * opp_position_def_difficulty
    )
    res.append(oppScoreCombined)
    res.append(opp_position_def_difficulty)
    res.append(opp_position_3pt_def_difficulty)

    # ===== DEFENSIVE TIERS =====
    res.append(1 if opp_def_rating_avg < 108 else 0)  # VS_ELITE_DEF
    res.append(1 if (opp_def_rating_avg >= 108 and opp_def_rating_avg < 113) else 0)  # VS_GOOD_DEF
    res.append(1 if (opp_def_rating_avg >= 113 and opp_def_rating_avg < 118) else 0)  # VS_AVERAGE_DEF
    res.append(1 if opp_def_rating_avg >= 118 else 0)  # VS_POOR_DEF
    
    # ===== INTERACTION WITH PLAYER SKILL =====
    # Volume Score
    volumeScore = (0.4 * (player_df['USG_PCT'].mean() / 35) + 
                   0.3 * (player_df['FGA'].mean() / 20) + 
                   0.2 * (player_df['MIN'].mean() / 36) + 
                   0.1 * (player_df['PTS'].mean() / 30)).clip(0, 1)
    
    # DEF_DIFFICULTY_X_VOLUME
    res.append(oppScoreCombined * volumeScore)
    
    # DEF_DIFFICULTY_X_CATCH_SHOOT (use 3pt defense for catch-and-shoot specialists)
    catchAndShootSpecialist = (
        (player_df['percentageAssistedFGM'].mean() > 0.65) & 
        (player_df['percentageAssisted3pt'].mean() > 0.80) & 
        ((player_df['FG3A'].mean()/player_df['FGA'].mean()) > 0.40) & 
        (player_df['FG3A'].mean() > 3)
    )
    if catchAndShootSpecialist:
        res.append(opp_position_3pt_def_difficulty if has_pos_def else oppScoreCombined)
    else:
        res.append(0)
    
    # DEF_DIFFICULTY_X_SHOT_CREATOR
    shotCreationSpecialist = (
        (player_df['percentageUnassistedFGM'].mean() > 0.35) & 
        (player_df['percentageUnassisted2pt'].mean() > 0.30) & 
        (player_df['percentageUnassisted3pt'].mean() > 0.30) & 
        (player_df['USG_PCT'].mean() > 20) & 
        (player_df['AST'].mean() > 3.5)
    )
    res.append(oppScoreCombined if shotCreationSpecialist else 0)

    # Player vs Opponent Matchup
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
    away_df = player_df[player_df['HOME_GAME'] == 0]
    
    res = []
    res.append(home * home_df['PTS'].mean() + (1 - home) * away_df['PTS'].mean())
    res.append(home * home_df['USG_PCT'].mean() + (1 - home) * away_df['USG_PCT'].mean())
    res.append(home * home_df['EFG_PCT'].mean() + (1 - home) * away_df['EFG_PCT'].mean())
    res.append(home * home_df['FGA'].mean() + (1 - home) * away_df['FGA'].mean())
    res.append(home * home_df['FTA'].mean() + (1 - home) * away_df['FTA'].mean())
    res.append(home * home_df['FG3A'].mean() + (1 - home) * away_df['FG3A'].mean())
    res.append(home * home_df['POSS'].mean() + (1 - home) * away_df['POSS'].mean())
    res.append(home * home_df['TCHS'].mean() + (1 - home) * away_df['TCHS'].mean())
    
    oppTeam = player_df[player_df['OPP_ABBREVIATION'] == opp]
    if oppTeam.empty:
        res.append(0 for _ in range(18))
    else:
        res.append(oppTeam['PTS'].mean() )
        res.append(oppTeam['MIN'].mean())
        res.append(oppTeam['USG_PCT'].mean())
        res.append(oppTeam['TS_PCT'].mean())
        res.append(oppTeam['FGA'].mean())
        res.append(oppTeam['FGM'].mean())
        res.append(oppTeam['FG3A'].mean())
        res.append(oppTeam['FG3M'].mean())
        res.append(oppTeam['FTA'].mean())
        res.append(oppTeam['FTM'].mean())
        res.append(oppTeam['UFGA'].mean())
        res.append(oppTeam['UFGM'].mean())
        res.append(oppTeam['CFGA'].mean())
        res.append(oppTeam['CFGM'].mean())
        res.append(oppTeam['DIST'].mean())
        res.append(oppTeam['SPD'].mean())
        res.append(oppTeam['TCHS'].mean())
        res.append(oppTeam['POSS'].mean())
        res.append(len(oppTeam))
    
    # ===== MATCHUP STATS (ACTIVATED) =====
    # Calculate matchup averages (vs this opponent) and overall averages
    epsilon = 1e-8
    games_vs_opp = len(oppTeam) if not oppTeam.empty else 0
    
    # Key metrics for matchup features
    key_metrics = {
        'PTS': 'PTS',
        'FGA': 'FGA',
        'FG3A': 'FG3A',
        'FTA': 'FTA',
        'MIN': 'MIN',
        'USG_PCT': 'USG_PCT',
        'EFG_PCT': 'EFG_PCT'
    }
    
    # Calculate matchup and overall averages
    matchup_avgs = {}
    overall_avgs = {}
    
    for metric_key, metric_col in key_metrics.items():
        if not oppTeam.empty and metric_col in oppTeam.columns:
            matchup_avgs[metric_key] = oppTeam[metric_col].mean()
        else:
            matchup_avgs[metric_key] = 0
        
        if metric_col in player_df.columns:
            overall_avgs[metric_key] = player_df[metric_col].mean()
        else:
            overall_avgs[metric_key] = epsilon
    
    # Calculate ratios (matchup / overall)
    matchup_ratios = {}
    for metric_key in key_metrics.keys():
        if overall_avgs[metric_key] > 0:
            matchup_ratios[metric_key] = matchup_avgs[metric_key] / (overall_avgs[metric_key] + epsilon)
        else:
            matchup_ratios[metric_key] = 1.0
    
    # Calculate deltas (matchup - overall)
    matchup_deltas = {}
    for metric_key in key_metrics.keys():
        matchup_deltas[metric_key] = matchup_avgs[metric_key] - overall_avgs[metric_key]
    
    # Add ratios
    res.append(round(matchup_ratios['PTS'], 3))  # MATCHUP_PTS_RATIO
    res.append(round(matchup_ratios['FGA'], 3))  # MATCHUP_FGA_RATIO
    res.append(round(matchup_ratios['FG3A'], 3))  # MATCHUP_FG3A_RATIO
    res.append(round(matchup_ratios['FTA'], 3))  # MATCHUP_FTA_RATIO
    res.append(round(matchup_ratios['MIN'], 3))  # MATCHUP_MIN_RATIO
    res.append(round(matchup_ratios['USG_PCT'], 3))  # MATCHUP_USG_PCT_RATIO
    res.append(round(matchup_ratios['EFG_PCT'], 3))  # MATCHUP_EFG_PCT_RATIO
    
    # Add deltas
    res.append(round(matchup_deltas['PTS'], 2))  # MATCHUP_PTS_DELTA
    res.append(round(matchup_deltas['FGA'], 2))  # MATCHUP_FGA_DELTA
    res.append(round(matchup_deltas['MIN'], 2))  # MATCHUP_MIN_DELTA
    res.append(round(matchup_deltas['USG_PCT'], 2))  # MATCHUP_USG_PCT_DELTA
    
    # Add advantage flags
    matchup_pts_advantage = 1 if matchup_ratios['PTS'] > 1.05 else 0  # 5% better
    res.append(matchup_pts_advantage)  # MATCHUP_PTS_ADVANTAGE
    
    # MATCHUP_ADVANTAGE_OVERALL (average of key ratios > 1.05)
    key_ratios = [matchup_ratios['PTS'], matchup_ratios['FGA'], matchup_ratios['USG_PCT']]
    matchup_advantage_overall = 1 if (sum(key_ratios) / len(key_ratios)) > 1.05 else 0
    res.append(matchup_advantage_overall)  # MATCHUP_ADVANTAGE_OVERALL
    
    # Calculate volume score for interactions
    volumeScore = (0.4 * (player_df['USG_PCT'].mean() / 35) + 
                   0.3 * (player_df['FGA'].mean() / 20) + 
                   0.2 * (player_df['MIN'].mean() / 36) + 
                   0.1 * (player_df['PTS'].mean() / 30)).clip(0, 1)
    
    # Get defensive difficulty from opponent team data
    opp_team_df = data[data['TEAM_ABBREVIATION'] == opp].drop_duplicates(subset=['GAME_ID'])
    if not opp_team_df.empty and 'TEAM_DEF_RATING' in opp_team_df.columns:
        opp_def_rating_avg = opp_team_df['TEAM_DEF_RATING'].mean()
        opp_def_difficulty_combined = ((123 - opp_def_rating_avg) / 20).clip(0, 1)
    else:
        # Default to average defense if not available
        opp_def_difficulty_combined = 0.5
    
    # Add interactions
    res.append(round(matchup_ratios['PTS'] * volumeScore, 3))  # MATCHUP_PTS_RATIO_X_VOLUME
    res.append(round(matchup_ratios['PTS'] * opp_def_difficulty_combined, 3))  # MATCHUP_PTS_RATIO_X_DEF_DIFFICULTY
    
    # ===== OPPONENT ONE-HOT ENCODING =====
    # All 30 NBA teams in alphabetical order
    nba_teams = [
        'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 
        'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 
        'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 
        'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 
        'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
    ]
    
    # Add one-hot encoding: 1 if this is the opponent, 0 otherwise
    for team in nba_teams:
        res.append(1 if opp == team else 0)  # OPP__ATL, OPP__BKN, etc.
    
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
    
    
