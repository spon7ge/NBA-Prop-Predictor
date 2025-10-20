import requests 
from datetime import datetime
import pytz
import pandas as pd
from catboost import Pool
from nba_api.stats.endpoints import scoreboardv2, scheduleleaguev2
from teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive


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

def getUpcomingGames(date):
    schedule = scheduleleaguev2.ScheduleLeagueV2().get_data_frames()[0]
    schedule['gameDate'] = pd.to_datetime(schedule['gameDate']).dt.strftime('%Y-%m-%d')
    schedule = schedule[schedule['gameDate'] == date]
    homeTeams = schedule['homeTeam_teamTricode'].unique().tolist()
    awayTeams = schedule['awayTeam_teamTricode'].unique().tolist()
    return homeTeams, awayTeams

def findOpp(playerName, players_df, gameDate):
    player_team = players_df.loc[
        players_df['PLAYER_NAME'] == playerName, 'TEAM_ABBREVIATION'
    ].iloc[-1]
    
    homeTeams, awayTeams = getUpcomingGames(gameDate)
    
    # Convert to lists for easier handling
    homeTeams = list(homeTeams)
    awayTeams = list(awayTeams)
    
    home = 0
    if player_team in homeTeams:
        # Player's team is at home, opponent is away
        opp_team = awayTeams[homeTeams.index(player_team)]
        home = 1
    elif player_team in awayTeams:
        # Player's team is away, opponent is home
        opp_team = homeTeams[awayTeams.index(player_team)]
        home = 0
    else:
        # No game found for this team on this date
        print(f"No game found for {player_team} on {gameDate}")
        return None, None
    
    return opp_team, home

def playerContext(player_name, data, current_date, projectedStartingFive, teamStarPlayer, mainStartingFive):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
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
        
    # Team Star Player
    if player_name ==teamStarPlayer[player_team]:
        res.append(1)
    else:
        res.append(0)
    
    #Team Star Out
    if teamStarPlayer[player_team] not in projectedStartingFive[player_team]:
        res.append(1)
    else:
        res.append(0)
        
    # Number of Usual Starters Present
    mainFive = set(mainStartingFive[player_team])
    projectedFive = set(projectedStartingFive[player_team])
    res.append(len(mainFive & projectedFive))
    
    # Back to Back and Days Rested
    if (current_date_dt - player_df['GAME_DATE'].iloc[-1]).days == 1:
        res.append(1)
    else:
        res.append(0)
    res.append((current_date_dt - player_df['GAME_DATE'].iloc[-1]).days)

    return res

def playerScoring(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    res = []
    
    # Features from your new list
    res.append(player_df['PTS_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['PTS_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['FGA_ROLLING_AVG_7'].iloc[-1])
    res.append(player_df['EXPECTED_USAGE_MIN'].iloc[-1])
    res.append(player_df['PTS_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['PTS_ROLLING_AVG_25'].iloc[-1])
    res.append(player_df['USG_X_POSS'].iloc[-1])
    res.append(player_df['FGA_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['FGA_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['FG3A_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['PTS_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['FG3A_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['PTS_PER_36_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['E_USG_PCT_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['UFGA_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['FGA_LAG_1'].iloc[-1])
    res.append(player_df['USG_X_TEAM_OFF'].iloc[-1])
    res.append(player_df['E_USG_PCT_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['USG_PCT_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['USG_PCT_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['FGA_ROLLING_AVG_25'].iloc[-1])

    return res

def teamContext(player_name, data, teamStarPlayer, projectedStartingFive): 
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    player_team = player_df['TEAM_ABBREVIATION'].iloc[-1]
    res = []
    
    res.append(player_df['TEAM_PTS_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['TEAM_PACE_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['TEAM_OFF_RATING_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['TEAM_DEF_RATING_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['TEAM_AST_ROLLING_AVG_5'].iloc[-1])

    #Team star out( for now just using last game)
    if teamStarPlayer[player_team] not in projectedStartingFive[player_team]:
        res.append(1 * player_df['USG_PCT_AVG_TO_DATE'].iloc[-1])
        res.append(1 * player_df['PTS_AVG_TO_DATE'].iloc[-1])
    else:
        res.append(0 * player_df['USG_PCT_AVG_TO_DATE'].iloc[-1])
        res.append(0 * player_df['PTS_AVG_TO_DATE'].iloc[-1])
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
        
    opp_df = data[data['TEAM_ABBREVIATION'] == opp_team].sort_values(by='GAME_DATE')
    
    # Opponent Defense/Pace
    res.append(opp_df['OPP_DEF_RATING_AVG_TO_DATE'].iloc[-1])
    res.append(opp_df['OPP_PACE_AVG_TO_DATE'].iloc[-1])

    res.append(player_df['PLAYER_PAINT_X_OPP_PAINT_DEF'].iloc[-1])
    res.append(player_df['PTS_PER_MIN'].iloc[-1] * opp_df['OPP_DEF_RATING_AVG_TO_DATE'].iloc[-1])  # Get last value first
    res.append(player_df['USG_PCT_AVG_TO_DATE'].iloc[-1] * opp_df['OPP_DEF_RATING_AVG_TO_DATE'].iloc[-1])  # Get last value first
    
    return res

def playerZones(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    # KEEP - these are in your new feature list
    res.append(player_df['percentagePointsPaint_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['percentagePointsPaint_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['FTM_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['PIE_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['percentagePointsPaint_STD_LAST_40'].iloc[-1])
    res.append(player_df['percentagePoints2pt_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['FTM_AVG_TO_DATE'].iloc[-1])
    
    return res

def playerHomeAway(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    # KEEP - this is in your new feature list
    res.append(player_df['PLAYER_AWAY_AVG_percentagePointsPaint_TO_DATE'].iloc[-1])
    
    return res

def playerMatchup(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    
    # Features from your new list
    res.append(player_df['PFD_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['RBC_ROLLING_AVG_40'].iloc[-1])
    
    return res

def buildVector(player_name, data, current_date):
    # Import here to avoid circular imports
    from teamInfo import mainStartingFive, teamStarPlayer, projectedStartingFive
    
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    res.append(playerContext(player_name, data, current_date, projectedStartingFive, teamStarPlayer, mainStartingFive))
    res.append(playerScoring(player_name, data))
    res.append(teamContext(player_name, data, teamStarPlayer, projectedStartingFive))
    res.append(playerVsOpp(player_name, data, current_date))
    res.append(playerZones(player_name, data))
    res.append(playerHomeAway(player_name, data))
    res.append(playerMatchup(player_name, data, current_date))
    return res

def makePrediction(player_name, data, model, features, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    # Now build the vector using the actual game date
    vector = buildVector(player_name, data, current_date)
    vector = [item for sublist in vector for item in sublist]
    vector = pd.DataFrame([vector], columns=features)
    
    for col in vector.columns:
        vector[col] = pd.to_numeric(vector[col], errors='coerce')
    
    vector = vector.fillna(0)
    
    pred = model.predict(vector)[0]
    return round(float(pred), 3)
    
    
