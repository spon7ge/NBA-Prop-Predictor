import requests 
from datetime import datetime, timedelta
import pytz
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2
from MODELS.teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive


today = datetime.today().strftime('%Y-%m-%d')

def calculate_slope():
    pass 

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
        
    # Role Tier
    res.append(player_df['IS_STARTER_TIER'].iloc[-1])
    res.append(player_df['IS_ROLE_TIER'].iloc[-1])
    res.append(player_df['IS_BENCH_TIER'].iloc[-1])
    
    # Role Tier Interactions
    if player_name in projectedStartingFive[player_team]:
        res.append( 1 * player_df['IS_STARTER_TIER'].iloc[-1])
        res.append( 1 * player_df['IS_BENCH_TIER'].iloc[-1])
        res.append( 1 * player_df['IS_ROLE_TIER'].iloc[-1])
    else:
        res.append(0)
        res.append(0)
        res.append(0)

    # Team Star Player
    if player_name == teamStarPlayer[player_team]:
        res.append(1)
        res.append(1 *player_df['PTS'].mean())
        res.append(1 * player_df['NET_RATING'].mean())
    else:
        res.append(0)
        res.append(0 * player_df['PTS'].mean())
        res.append(0 * player_df['NET_RATING'].mean())
 
    res.append(player_df['PTS'].mean() * player_df['IS_STARTER_TIER'].iloc[-1])    

    #Team Star Out
    if teamStarPlayer[player_team] not in projectedStartingFive[player_team]:
        res.append(1)
    else:
        res.append(0)

        
    # Back to Back and Days Rested
    if (current_date_dt - player_df['GAME_DATE'].iloc[-1]).days == 1:
        res.append(1)
    else:
        res.append(0)

    days_rested = (current_date_dt - player_df['GAME_DATE'].iloc[-1]).days
    days_rested = min(days_rested, 3)
    res.append(days_rested)

    # Usual Starters Available
    main_in_projected = len(set(mainStartingFive[player_team]) & set(projectedStartingFive[player_team]))
    res.append(main_in_projected)

    opp_in_projected = len(set(mainStartingFive[opp_abv]) & set(projectedStartingFive[opp_abv]))

    res.append(opp_in_projected)
    if player_df.empty:
        res.append(0)
    else:
        res.append(len(player_df))

    return res

def playerScoring(player_name, data):
    player_df = data[data['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    team_df = data[data['TEAM_ABBREVIATION'] == player_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')

    res = []

    res.append(player_df['MIN'].iloc[-1])
    res.append(player_df['MIN'].tail(10).max())
    res.append(player_df['MIN'].tail(10).min())
    res.append(player_df['MIN'].tail(20).max())
    res.append(player_df['MIN'].tail(20).min())
    res.append(calculate_volatility(player_df, 'MIN', window=5))
    res.append(calculate_volatility(player_df, 'MIN', window=10))
    res.append(calculate_volatility(player_df, 'MIN', window=40))
    res.append(player_df['MIN'].tail(40).mean())
    res.append(player_df['percentagePointsMidrange2pt'].mean())
    res.append(player_df['percentagePointsPaint'].mean())
    res.append(player_df['PTS_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['PTS'].tail(40).mean())
    res.append(player_df['PTS_TREND_LAST_5'].iloc[-1])

    res.append(player_df['MIN_TREND_LAST_5'].iloc[-1])
    res.append((player_df['PTS'].mean() / (player_df['MIN'].mean()) + 0.01) * player_df['USG_PCT'].mean())
    res.append(calculate_volatility(player_df, 'PTS', window=5))
    res.append(player_df['FGA'].iloc[-1])
    res.append(calculate_volatility(player_df, 'FGM', window=10))
    res.append(calculate_volatility(player_df, 'FGM', window=40))
    res.append(calculate_volatility(player_df, 'FGA', window=5))
    res.append(calculate_volatility(player_df, 'FGA', window=10))
    res.append(calculate_volatility(player_df, 'FGA', window=40))

    res.append(player_df['FG3A'].mean() / player_df['FGA'].mean() + 0.01)
    res.append(player_df['USG_PCT'].iloc[-1])
    res.append(calculate_volatility(player_df, 'USG_PCT', window=5, use_cv=True))
    res.append(calculate_volatility(player_df, 'USG_PCT', window=40, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', window=5, use_cv=True))
    res.append(calculate_volatility(player_df, 'TS_PCT', window=10, use_cv=True))
    res.append(player_df['E_OFF_RATING'].mean())
    res.append(player_df['NET_RATING'].mean())
    res.append(player_df['TCHS'].tail(10).mean())

    res.append(player_df['TOV'].tail(10).mean())
    res.append(player_df['PTS'].iloc[-1])
    res.append(player_df['FT_PCT'].tail(10).mean())
    res.append(player_df['FG_PCT'].tail(10).mean())
    res.append(player_df['FTM'].tail(40).mean())
    res.append(player_df['EFG_PCT'].mean())
    res.append(player_df['E_OFF_RATING'].tail(5).mean())
    res.append(player_df['FG3A'].mean() / team_df['TEAM_FG3A'].mean() + 0.01)
    res.append(player_df['USG_PCT_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['USG_PCT'].mean() * player_df['TS_PCT'].mean() + 0.01)

    res.append(player_df['PTS'].mean() / (player_df['FGA'].mean() + 0.44 * player_df['FTA'].mean() + player_df['TOV'].mean()))
    res.append(calculate_volatility(player_df, 'PACE', window=5))
    res.append(calculate_volatility(player_df, 'PACE', window=10))
    res.append(calculate_volatility(player_df, 'PACE', window=40))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', window=5))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', window=10))
    res.append(calculate_volatility(player_df, 'E_OFF_RATING', window=40))
    res.append(calculate_volatility(player_df, 'NET_RATING', window=5))
    res.append(calculate_volatility(player_df, 'NET_RATING', window=10))

    res.append(calculate_volatility(player_df, 'NET_RATING', window=40))
    res.append(calculate_volatility(player_df, 'FG_PCT', window=5, use_cv=True))
    res.append(calculate_volatility(player_df, 'FG_PCT', window=40, use_cv=True))
    res.append(calculate_volatility(player_df, 'FG3A', window=5))
    res.append(calculate_volatility(player_df, 'FG3A', window=40))
    res.append(calculate_volatility(player_df, 'FG3M', window=5))
    res.append(calculate_volatility(player_df, 'FG3M', window=40))
    res.append(calculate_volatility(player_df, 'FG3_PCT', window=5, use_cv=True))
    res.append(calculate_volatility(player_df, 'FG3_PCT', window=10, use_cv=True))
    res.append(calculate_volatility(player_df, 'FG3_PCT', window=20, use_cv=True))

    res.append(calculate_volatility(player_df, 'FTM', window=5))
    res.append(calculate_volatility(player_df, 'FTM', window=40))
    res.append(calculate_volatility(player_df, 'FT_PCT', window=5, use_cv=True))
    res.append(calculate_volatility(player_df, 'FT_PCT', window=10, use_cv=True))
    res.append(calculate_volatility(player_df, 'FT_PCT', window=40, use_cv=True))
    res.append(calculate_volatility(player_df, 'TOV', window=10))
    pts_recent_vs_season = player_df['PTS'].tail(5).mean() / player_df['PTS'].mean()
    res.append(pts_recent_vs_season)
    res.append(1 if pts_recent_vs_season > 1.15 else 0)
    res.append(1 if pts_recent_vs_season < 0.85 else 0)

    res.append(1 if player_df['USG_PCT'].mean() > 28 else 0)
    res.append(1 if player_df['USG_PCT'].mean() > 23 and player_df['USG_PCT'].mean() <= 28 else 0)
    # res.append(player_df['PTS'].tail(10).max())
    res.append(player_df['PTS'].tail(20).max())
    res.append(player_df['PTS'].tail(10).min())
    res.append(1 if (player_df['PLAYER_IS_TEAM_STAR'].iloc[-1] * (player_df['PTS_TREND_LAST_5'].iloc[-1] > 0)) else 0)
    res.append(1 if player_df['PTS'].mean() < 18 else 0)
    res.append(1 if ((player_df['PTS'].mean() > 10) & (player_df['PTS'].mean() < 18)) else 0)


    
    return res


def teamContext(player_name, data, teamStarPlayer, projectedStartingFive): 
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    # Get player's team to fetch ALL team games (not just games player played in)
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    team_df = data[data['TEAM_ABBREVIATION'] == player_team].drop_duplicates(subset=['GAME_ID']).sort_values(by='GAME_DATE')
    res.append(team_df['TEAM_PTS'].tail(3).mean())
    res.append(team_df['TEAM_OFF_RATING'].tail(3).mean())
    res.append(team_df['TEAM_OFF_RATING'].mean())
    res.append(team_df['TEAM_PACE'].mean())
    res.append(team_df['TEAM_DEF_RATING'].tail(3).mean())
    return res

def playerVsOpp(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
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
    
    res.append(opp_team_df['TEAM_DEF_RATING'].mean())
    res.append(opp_team_df['TEAM_PACE'].mean())
    res.append(opp_team_df['TEAM_TOV'].mean())

    # Opponent Player Stats
    res.append(opp_guard_df['E_DEF_RATING'].mean())
    res.append(player_df['GUARD'] * (opp_guard_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['GUARD'] * player_3PA_rate * (opp_guard_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(player_df['GUARD'] * player_df['FG3_PCT'].mean() * (opp_guard_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(opp_guard_df['PTS_ALLOWED_PER_MIN'].mean())

    res.append(opp_forward_df['E_DEF_RATING'].mean())
    res.append(player_df['FORWARD'] * (opp_forward_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['FORWARD'] * player_3PA_rate * (opp_forward_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(player_df['FORWARD'] * player_df['FG3_PCT'].mean() * (opp_forward_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(opp_forward_df['PTS_ALLOWED_PER_MIN'].mean())
    res.append(opp_center_df['E_DEF_RATING'].mean())

    res.append(player_df['CENTER'] * (opp_center_df['DEF_FG_PCT_ALLOWED'].mean() - playerFG_PCT))
    res.append(player_df['CENTER'] * player_3PA_rate * (opp_center_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(player_df['CENTER'] * player_df['FG3_PCT'].mean() * (opp_center_df['DEF_3PT_PCT_ALLOWED'].mean()))
    res.append(opp_center_df['PTS_ALLOWED_PER_MIN'].mean())

    res.append(team_df['TEAM_OFF_RATING'].tail(3).mean() - opp_df['TEAM_DEF_RATING'].mean())
    res.append((team_df['TEAM_PACE'].mean() + opp_df['TEAM_PACE'].mean()) / 2)
    res.append((team_df['TEAM_PACE'].mean() - opp_df['TEAM_PACE'].mean()))
    return res


def playerMatchup(player_name, data, current_date_str):
    player_df = data[data['PLAYER_NAME']==player_name]
    opp, home = findOpp(player_name, data, current_date_str)
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    home_df = player_df[player_df['HOME_GAME'] == 1]
    away_df = player_df[player_df['HOME_GAME'] == 0]
    res = []
    res.append(home_df['PTS'].mean() - away_df['PTS'].mean())
    res.append(home_df['USG_PCT'].mean() - away_df['USG_PCT'].mean())
    res.append(home_df['FGA'].mean() - away_df['FGA'].mean())
    res.append(home_df['FTA'].mean() - away_df['FTA'].mean())
    oppTeam = player_df[player_df['OPP_ABBREVIATION'] == opp]
    if oppTeam.empty:
        res.append(player_df['PTS'].mean())
    else:
        res.append(oppTeam['PTS'].mean())
    res.append(len(oppTeam))
    return res


def buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = [playerContext(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer) + 
    playerScoring(player_name, data) + 
    teamContext(player_name, data, teamStarPlayer, projectedStartingFive) + 
    playerVsOpp(player_name, data, current_date) + 
    playerMatchup(player_name, data, current_date)]
    
    return res

def makePrediction(player_name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
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
    
    
