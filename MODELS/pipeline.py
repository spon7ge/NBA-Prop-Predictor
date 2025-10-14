import requests 
from datetime import datetime
import pytz
import pandas as pd
from catboost import Pool
from nba_api.stats.endpoints import scoreboardv2


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

def findNextGame(player_name, players_df, start_date):
    """
    Find the next game for a player starting from start_date.
    Returns the game date and opponent info.
    """
    player_team_id = players_df.loc[
        players_df['PLAYER_NAME'] == player_name, 'TEAM_ID'
    ].iloc[-1]
    
    # Check today and the next 7 days for a game
    from datetime import timedelta
    start_date = pd.to_datetime(start_date)
    
    for days_ahead in range(3):  # Check up to 7 days ahead
        check_date = (start_date + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        try:
            scoreboard = getScoreboard(check_date)
            row = scoreboard[
                (scoreboard['HOME_TEAM_ID'] == player_team_id) |
                (scoreboard['VISITOR_TEAM_ID'] == player_team_id)
            ]
            
            if not row.empty:
                row = row.iloc[0]
                if player_team_id == row['HOME_TEAM_ID']:
                    opp_team_id = row['VISITOR_TEAM_ID']
                    home = 1
                else:
                    opp_team_id = row['HOME_TEAM_ID']
                    home = 0
                
                return int(opp_team_id), int(player_team_id), home, check_date
        except Exception as e:
            continue
    
    print(f"No game found for {player_name} in the next 7 days from {start_date}")
    return None, None, None, None


scores = {}

def getScoreboard(gameDate):
    if gameDate in scores:
        return scores[gameDate]
    else:
        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=gameDate,
            league_id='00').get_data_frames()[2]
        scores[gameDate] = scoreboard
        return scoreboard

def findOpp(playerName, players_df, gameDate):
    player_team_id = players_df.loc[
        players_df['PLAYER_NAME'] == playerName, 'TEAM_ID'
    ].iloc[-1]
    
    scoreboard = getScoreboard(gameDate)
    
    row = scoreboard[
        (scoreboard['HOME_TEAM_ID'] == player_team_id) |
        (scoreboard['VISITOR_TEAM_ID'] == player_team_id)]
    
    if row.empty:
        print(f"No game found for {playerName} on {gameDate}")
        return None
    
    row = row.iloc[0]
    home = 0
    
    if player_team_id == row['HOME_TEAM_ID']:
        opp_team_id = row['VISITOR_TEAM_ID']
        player_team_id = row['HOME_TEAM_ID']
        home = 1
    else:
        opp_team_id = row['HOME_TEAM_ID']
        player_team_id = row['VISITOR_TEAM_ID']
        home = 0
    return int(opp_team_id), int(player_team_id), home

def playerContext(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')

    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    
    current_date = pd.to_datetime(current_date)
    player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'])
    res = []

    res.append(player_df['GUARD'].iloc[-1])
    res.append(player_df['FORWARD'].iloc[-1])
    res.append(player_df['CENTER'].iloc[-1])
    res.append(player_df['STARTING'].iloc[-1]) # im going to change this once i create a scraper to grab lineups
    res.append(player_df['PLAYER_IS_TEAM_STAR'].iloc[-1])
    res.append(player_df['TEAM_STAR_OUT'].iloc[-1])
    res.append(player_df['NUM_USUAL_STARTERS_PRESENT'].iloc[-1])

    result = findOpp(player_name, data, current_date)
    if result is None:
        res.append(0)  # Default home value
    else:
        _, _, home = result
        res.append(home)

    # Back to Back and Days Rested
    if (current_date - player_df['GAME_DATE'].iloc[-1]).days == 1:
        res.append(1)
    else:
        res.append(0)
    res.append((current_date - player_df['GAME_DATE'].iloc[-1]).days)

    return res

def playerScoring(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    res = []
    
    res.append(player_df['PTS_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['PTS_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['PTS_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['PTS_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['PTS_PER_MIN_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['PTS_PER_MIN'].iloc[-1])
    res.append(player_df['PTS_PER_36_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['PTS_PER_36'].iloc[-1])
    res.append(player_df['PTS_PAINT_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['PTS_PAINT_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['EXPECTED_USAGE_MIN'].iloc[-1])
    res.append(player_df['E_USG_PCT_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['E_USG_PCT_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['E_USG_PCT_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['E_USG_PCT_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['E_USG_PCT_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['E_USG_PCT_LAG_1'].iloc[-1])
    res.append(player_df['USG_PCT_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['USG_PCT_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['USG_X_TEAM_OFF'].iloc[-1])
    res.append(player_df['FGA_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['FGA_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['FGA_LAG_1'].iloc[-1])
    res.append(player_df['FGA_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['FGA_EXTREME_VOLATILITY'].iloc[-1])
    res.append(player_df['UFGA_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['UFGA_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['CFGA_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['CFGM_STD_LAST_40'].iloc[-1])
    res.append(player_df['CFGM_LAG_1'].iloc[-1])
    res.append(player_df['DFGA_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['DFGA_STD_LAST_40'].iloc[-1]) 
    res.append(player_df['DFGM_STD_LAST_40'].iloc[-1])
    res.append(player_df['FG3A_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['FG3A_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['FG3A_WITHOUT_STAR'].iloc[-1])

    return res

def teamContext(player_name, data): 
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None

    res = []
    
    res.append(player_df['TEAM_PTS_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['TEAM_OFF_RATING_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['TEAM_DEF_RATING_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['TEAM_AST_ROLLING_AVG_5'].iloc[-1])

    #Team star out( for now just using last game)
    res.append(player_df['TEAM_STAR_OUT_X_USG'].iloc[-1])
    res.append(player_df['TEAM_STAR_OUT_X_PTS'].iloc[-1])
    res.append(player_df['TEAM_STAR_OUT_X_MIN'].iloc[-1])

    return res

def playerVsOpp(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []

    opp_team_id, _, _ = findOpp(player_name, player_df, current_date)
    opp_df = data[data['TEAM_ID'] == opp_team_id].sort_values(by='GAME_DATE')
    guardDef = opp_df[opp_df['GUARD'] == 1]['DEF_3PT_PCT_ALLOWED'].mean()
    forwardDef = opp_df[opp_df['FORWARD'] == 1]['DEF_3PT_PCT_ALLOWED'].mean()
    centerDef = opp_df[opp_df['CENTER'] == 1]['DEF_3PT_PCT_ALLOWED'].mean()
    forwardFGDef = opp_df[opp_df['FORWARD'] == 1]['DEF_FG_PCT_ALLOWED'].mean()

    res.append(player_df['PLAYER_PAINT_X_OPP_PAINT_DEF'].iloc[-1])
    res.append(player_df['PLAYER_PAINT_X_OPP_PAINT_DEF_RECENT'].iloc[-1])
    res.append(player_df['percentagePointsMidrange2pt_AVG_TO_DATE'].iloc[-1] * forwardFGDef)

    res.append(player_df['percentageFieldGoalsAttempted3pt_AVG_TO_DATE'].iloc[-1] * forwardDef)
    res.append(player_df['percentageFieldGoalsAttempted3pt_AVG_TO_DATE'].iloc[-1] * centerDef)
    res.append(player_df['percentageFieldGoalsAttempted3pt_AVG_TO_DATE'].iloc[-1] * guardDef)
    res.append(player_df['percentageFieldGoalsAttempted3pt_ROLLING_AVG_5'].iloc[-1] * guardDef)
    res.append(player_df['OPP_PTS_PAINT_STD_LAST_40'].iloc[-1])
    res.append(opp_df['TEAM_PACE'].tail(10).mean())

    res.append(player_df['percentageFieldGoalsAttempted3pt_ROLLING_AVG_5'].iloc[-1] * forwardDef)
    res.append(player_df['percentageFieldGoalsAttempted3pt_ROLLING_AVG_5'].iloc[-1] * centerDef)
    return res

def playerMatchup(player_name, data, current_date): # Needs Work
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    result = findOpp(player_name, player_df, current_date)
    if result is None:
        print(f"No game found for {player_name} on {current_date}")
        return [0] * 4
    opp_team_id, _, _ = result
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    player_df = player_df[player_df['OPP_TEAM_ID'] == opp_team_id]
    res.append(player_df['PTS'].tail(3).mean()) 
    res.append(player_df['FGA'].tail(3).mean())
    res.append(player_df['USG_PCT'].tail(3).mean())
    res.append(player_df['USG_PCT'].tail(5).mean())
    return res

def playerZones(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    res.append(player_df['percentagePointsPaint_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['percentagePointsPaint_EXPANDING_VOLATILITY_TO_DATE'].iloc[-1])
    res.append(player_df['percentagePointsPaint_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['percentagePointsPaint_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['percentagePointsPaint_ROLLING_AVG_10'].iloc[-1])
    res.append(player_df['percentagePointsPaint_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['percentagePointsMidrange2pt_STD_LAST_40'].iloc[-1])
    res.append(player_df['percentagePoints2pt_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['percentagePoints2pt_AVG_TO_DATE'].iloc[-1])
    res.append(player_df['percentagePoints3pt_ROLLING_AVG_5'].iloc[-1])
    res.append(player_df['percentageFieldGoalsAttempted2pt_STD_LAST_40'].iloc[-1])

    res.append(player_df['percentageAssisted2pt_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['percentageAssisted3pt_LAG_1'].iloc[-1])
    res.append(player_df['percentageUnassistedFGM_AVG_TO_DATE'].iloc[-1])

    res.append(player_df['FTM_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['FTM_STD_LAST_40'].iloc[-1])
    res.append(player_df['percentagePointsFreeThrow_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['POSS_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['SPD_AVG_TO_DATE'].iloc[-1])

    res.append(player_df['FANTASY_PTS_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['FANTASY_PTS_STD_LAST_40'].iloc[-1])
    res.append(player_df['PIE_WITHOUT_STAR'].iloc[-1])
    res.append(player_df['PIE_LAG_2'].iloc[-1])
    res.append(player_df['NET_RATING_ROLLING_AVG_15'].iloc[-1])
    res.append(player_df['TS_PCT_ROLLING_AVG_40'].iloc[-1])
    res.append(player_df['EFG_X_MIN'].iloc[-1])
    res.append(player_df['PF_AVG_TO_DATE'].iloc[-1])

    return res

def playerHomeAway(player_name, data):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    res.append(player_df['PLAYER_HOME_AVG_percentagePointsPaint_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_HOME_AVG_E_USG_PCT_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_HOME_AVG_BLK_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_HOME_AVG_OPP_PTS_PAINT_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_HOME_AVG_FANTASY_PTS_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_HOME_AVG_PFD_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_HOME_AVG_percentagePointsFreeThrow_TO_DATE'].iloc[-1])

    res.append(player_df['PLAYER_AWAY_AVG_percentagePointsPaint_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_AVG_percentagePointsMidrange2pt_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_AVG_percentagePoints2pt_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_AVG_PTS_TO_DATE'].iloc[-1])
    res.append(player_df['PLAYER_AWAY_AVG_DFGA_TO_DATE'].iloc[-1])

    return res

def buildVector(player_name, data, current_date):
    player_df = data[data['PLAYER_NAME']==player_name].sort_values(by='GAME_DATE')
    if player_df.empty:
        print(f"No data found for {player_name}")
        return None
    res = []
    res.append(playerContext(player_name, data, current_date))
    res.append(playerScoring(player_name, data))
    res.append(teamContext(player_name, data))
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

    # Find the player's next game
    opp_team_id, player_team_id, home, actual_game_date = findNextGame(
        player_name, data, current_date
    )
    
    if actual_game_date is None:
        print(f"No upcoming game found for {player_name}")
        return None

    # Now build the vector using the actual game date
    vector = buildVector(player_name, data, actual_game_date)
    vector = [item for sublist in vector for item in sublist]
    vector = pd.DataFrame([vector], columns=features)
    
    for col in vector.columns:
        vector[col] = pd.to_numeric(vector[col], errors='coerce')
    
    vector = vector.fillna(0)
    
    pred = model.predict(vector)[0]
    return round(float(pred), 3)
    
    
