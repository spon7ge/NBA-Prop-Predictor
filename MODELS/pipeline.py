import requests 
from datetime import datetime, timedelta
import pytz
import pandas as pd
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2
from MODELS.teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive


today = datetime.today().strftime('%Y-%m-%d')


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

def playerContext(player_name, data, current_date, projectedStartingFive, teamStarPlayer):
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
    _ , home = findOpp(player_name, data, current_date_str)
    res.append(home)

    # Starting
    if player_name in projectedStartingFive[player_team]:
        res.append(1)
    else:
        res.append(0)
    
    # Positions
    res.append(player_df['GUARD'].iloc[-1])
    res.append(player_df['FORWARD'].iloc[-1])
    res.append(player_df['CENTER'].iloc[-1])

    # Team Star Player
    if player_name ==teamStarPlayer[player_team]:
        res.append(1)
        res.append(1 *player_df['PTS_AVG_TO_DATE'].iloc[-1])
        res.append(1 * player_df['NET_RATING_AVG_TO_DATE'].iloc[-1])
    else:
        res.append(0)
        res.append(0 * player_df['PTS_AVG_TO_DATE'].iloc[-1])
        res.append(0 * player_df['NET_RATING_AVG_TO_DATE'].iloc[-1])
    
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
    return res

def playerScoring(player_name, data):
    player_df = data[data['PLAYER_NAME'] == player_name]
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    res = []

    res.append(player_df['MIN'].iloc[-1])
    res.append(player_df['MIN_VOLATILITY_5_TO_DATE'].iloc[-1])
    res.append(player_df['MIN_VOLATILITY_10_TO_DATE'].iloc[-1])
    res.append(player_df['percentagePointsMidrange2pt_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['percentagePointsPaint_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['PTS_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['PTS_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['PTS_PER_MIN_X_USG'].iloc[-1])
    res.append(player_df['PTS'].iloc[-2])
    res.append(player_df['PTS_VOLATILITY_25_TO_DATE'].iloc[-1])
    res.append(player_df['PTS_EXTREME_VOLATILITY'].iloc[-1])
    res.append(player_df['PTS_RECENT_HIGH_VOLATILITY'].iloc[-1])
    res.append(player_df['FGA'].iloc[-1])
    res.append(player_df['FGA_VOLATILITY_10_TO_DATE'].iloc[-1])
    res.append(player_df['MATCHUP_AVG_FGA_LAST_3_TO_DATE'].iloc[-1])
    res.append(player_df['FT_RATE_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['3PA_SHARE'].iloc[-1])
    res.append(player_df['USG_PCT'].iloc[-1])
    res.append(player_df['USG_PCT_VOLATILITY_5_TO_DATE'].iloc[-1])
    res.append(player_df['TS_PCT_ROLLING_AVG_25'].iloc[-1])
    res.append(player_df['TS_PCT_VOLATILITY_10_TO_DATE'].iloc[-1])
    res.append(player_df['PIE_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['E_OFF_RATING_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['NET_RATING_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['TCHS_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['TOV_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['FG_PCT_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['CFGA_ROLLING_AVG_25'].iloc[-1])
    res.append(player_df['PTS'].iloc[-1])
    res.append(player_df['FT_PCT_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['FG_PCT_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['FTM_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['FGM_STD_LAST_5'].iloc[-1])
    res.append(player_df['FGM_STD_LAST_10'].iloc[-1])
    res.append(player_df['FGM_STD_LAST_15'].iloc[-1])
    res.append(player_df['FTM_STD_LAST_5'].iloc[-1])
    res.append(player_df['FTM_STD_LAST_15'].iloc[-1])
    res.append(player_df['USG_PCT_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['EFG_PCT_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['E_OFF_RATING_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['USG_PCT_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['PTS_PER_MIN'].iloc[-1])
    res.append(player_df['PLAYER_FG3A_SHARE'].iloc[-1])
    res.append(player_df['USG_PCT_DELTA_STAR_OUT'].iloc[-1])
    res.append(player_df['PTS_PER_MIN'].iloc[-1] * player_df['MIN'].iloc[-1])
    res.append(player_df['PTS_X_OPP_DEF_RATING'].iloc[-1])

    return res


def teamContext(player_name, data, teamStarPlayer, projectedStartingFive): 
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    res.append(player_df['TEAM_OFF_RATING_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['TEAM_DEF_RATING_ROLLING_AVG_3'].iloc[-1])
    res.append(player_df['TEAM_PACE_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['TEAM_PTS_ROLLING_AVG_3'].iloc[-1])
    res.append(player_df['TEAM_AST_ROLLING_AVG_3'].iloc[-1])


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

    opp_guard_df = opp_df[(opp_df['GUARD'] == 1) & (opp_df.groupby('PLAYER_NAME')['MIN'].transform('mean') > 10)]
    opp_forward_df = opp_df[(opp_df['FORWARD'] == 1) & (opp_df.groupby('PLAYER_NAME')['MIN'].transform('mean') > 10)]
    opp_center_df = opp_df[(opp_df['CENTER'] == 1) & (opp_df.groupby('PLAYER_NAME')['MIN'].transform('mean') > 10)]

    # Opponent Team Stats
    res.append(opp_df['TEAM_DEF_RATING_AVG_TO_DATE'].iloc[-1])
    res.append(opp_df['TEAM_PACE_AVG_TO_DATE'].iloc[-1])
    res.append(opp_df['TEAM_BLK'].mean())
    res.append(opp_df['TEAM_TOV_AVG_TO_DATE'].iloc[-1])

    # Opponent Player Stats
    res.append(opp_guard_df['E_DEF_RATING'].mean())
    res.append(opp_guard_df['DEF_FG_PCT_ALLOWED'].mean())
    res.append(opp_guard_df['DEF_3PT_PCT_ALLOWED'].mean())
    res.append(opp_guard_df['PTS_ALLOWED_PER_MIN'].mean())
    res.append(opp_forward_df['E_DEF_RATING'].mean())
    res.append(opp_forward_df['DEF_FG_PCT_ALLOWED'].mean())
    res.append(opp_forward_df['DEF_3PT_PCT_ALLOWED'].mean())
    res.append(opp_forward_df['PTS_ALLOWED_PER_MIN'].mean())
    res.append(opp_center_df['E_DEF_RATING'].mean())
    res.append(opp_center_df['DEF_FG_PCT_ALLOWED'].mean())
    res.append(opp_center_df['DEF_3PT_PCT_ALLOWED'].mean())
    res.append(opp_center_df['PTS_ALLOWED_PER_MIN'].mean())
    res.append(player_df['TEAM_OFF_RATING_ROLLING_AVG_3'].iloc[-1] - opp_df['OPP_DEF_RATING_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['TEAM_PACE_AVG_TO_DATE'].iloc[-1] * opp_df['TEAM_PACE_AVG_TO_DATE'].iloc[-1])

    return res


def playerMatchup(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name]
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    res.append(player_df['PLAYER_HOME_PTS_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_PTS_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_HOME_USG_PCT_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_USG_PCT_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_HOME_FGA_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_FGA_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_HOME_FTA_DELTA'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_FTA_DELTA'].iloc[-1])

    return res


def buildVector(player_name, data, current_date, projectedStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = [playerContext(player_name, data, current_date, projectedStartingFive, teamStarPlayer) + 
    playerScoring(player_name, data) + 
    teamContext(player_name, data, teamStarPlayer, projectedStartingFive) + 
    playerVsOpp(player_name, data, current_date) + 
    playerMatchup(player_name, data)]
    
    return res

def makePrediction(player_name, data, model, features, current_date, projectedStartingFive, teamStarPlayer):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    # Now build the vector using the actual game date
    vector = buildVector(player_name, data, current_date, projectedStartingFive, teamStarPlayer)
    vector = [item for sublist in vector for item in sublist]
    vector = pd.DataFrame([vector], columns=features)
    
    for col in vector.columns:
        vector[col] = pd.to_numeric(vector[col], errors='coerce')
    
    vector = vector.fillna(0)
    
    pred = model.predict(vector)[0]
    return round(float(pred), 3)
    
    
