import requests 
from datetime import datetime
import pytz
import pandas as pd
import joblib
from MODELS.model import *
from nba_api.stats.endpoints import leaguegamelog, teamgamelogs
import re


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

def findOppID(player, data):
    player_id = findPlayerID(player, data)
    player = data[data['PLAYER_ID'] == player_id].sort_values(by='GAME_DATE')
    games = get_espn_games()
    opponent = None
    
    for game in games:
        if game['home_team'] == player['TEAM_ABBREVIATION'].iloc[-1]:
            opponent = game['away_team']
            break
        elif game['away_team'] == player['TEAM_ABBREVIATION'].iloc[-1]:
            opponent = game['home_team']
            break
    if not opponent:
        return "No opponent found"
    
    opponent_id = findTeamID(opponent, data)
    return opponent_id

def findPlayerID(player_name, data):
    result = data[data['PLAYER_NAME'] == player_name]['PLAYER_ID']
    return result.iloc[0] if len(result) > 0 else None

def findTeamID(team_abv, data):
    result = data[data['TEAM_ABBREVIATION'] == team_abv]['TEAM_ID']
    return result.iloc[0] if len(result) > 0 else None

def getPlayerSpecificFeatures(player, data, games, today=today):
    player_id = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    res = []
    
    # Position features  
    res.append(player_id)
    res.append(player_data['TEAM_ID'].iloc[-1])
    res.append(findOppID(player, data))
    res.append(player_data['STARTING'].iloc[-1]) 
    
    # Home game
    home_game = 0
    for game in games:
        if game['home_team'] == player_data['TEAM_ABBREVIATION'].iloc[-1]:
            home_game = 1
            break
        if game['away_team'] == player_data['TEAM_ABBREVIATION'].iloc[-1]:
            home_game = 0
            break
    res.append(home_game)
    res.append(today - player_data['GAME_DATE'].iloc[-1]) # gets how long since last game
    if today - player_data['GAME_DATE'].iloc[-1] == 1: # checks for back to back
        res.append(1)
    else:
        res.append(0)
    return res

def getStarters(game_id, team, data):
    # Filter by game_id and team
    team_game_data = data[
        (data['GAME_ID'] == game_id) & 
        (data['TEAM_ABBREVIATION'] == team)
    ]
    
    # Get starters (players with STARTING == 1)
    starters = team_game_data[
        team_game_data['STARTING'] == 1
    ]['PLAYER_NAME'].tolist()
    
    return starters

def getPlayerStarInformation(player, starters,data):
    player_id = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    res = []
    if player in starters:
        res.append(1)
    else:
        res.append(0)
    cols = [ 'PTS_WITHOUT_STAR', 'MIN_WITHOUT_STAR', 'USG_PCT_WITHOUT_STAR', 'FGA_WITHOUT_STAR', 'FG3A_WITHOUT_STAR', 'FTA_WITHOUT_STAR', 
    'EFG_PCT_WITHOUT_STAR', 'TS_PCT_WITHOUT_STAR', 'AST_WITHOUT_STAR', 'REB_WITHOUT_STAR', 'PTS_PER_36_WITHOUT_STAR']
    for col in cols:
        res.append(player_data[col].iloc[-1])
    return res

def getPlayerSeasonAverages(player, data):
    player_id = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    cols = ['MIN', 'PTS', 'FGA', 'FG3A', 'FTA', 'USG_PCT', 'TS_PCT', 
            'EFG_PCT', 'AST', 'REB', 'TOV']
    res = []
    for col in cols:
        res.append(player_data[col].mean())
    return res

def getPlayerLags(player, data):
    player_id = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    lags = []
    cols = ['PTS', 'FGA', 'MIN', 'USG_PCT']
    for col in cols:
        lags.append(player_data[col].iloc[-1])
        lags.append(player_data[col].iloc[-2])
    return lags

def getPlayerRollingAVG(player, data):
    player_id = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    res = []

    include = [
    'MIN_ROLLING_AVG_5', 'PTS_ROLLING_AVG_5', 'FGA_ROLLING_AVG_5',
    'FG3A_ROLLING_AVG_5', 'FTA_ROLLING_AVG_5', 'USG_PCT_ROLLING_AVG_5',
    'TS_PCT_ROLLING_AVG_5', 'EFG_PCT_ROLLING_AVG_5', 'AST_ROLLING_AVG_5', 
    'REB_ROLLING_AVG_5', 'TOV_ROLLING_AVG_5',
    
    # Medium-term form (15-game rolling averages)
    'MIN_ROLLING_AVG_15', 'PTS_ROLLING_AVG_15', 'FGA_ROLLING_AVG_15',
    'FG3A_ROLLING_AVG_15', 'FTA_ROLLING_AVG_15', 'USG_PCT_ROLLING_AVG_15',
    'TS_PCT_ROLLING_AVG_15', 'EFG_PCT_ROLLING_AVG_15', 'AST_ROLLING_AVG_15', 
    'REB_ROLLING_AVG_15', 'TOV_ROLLING_AVG_15',
    
    # Long-term form (40-game rolling averages)
    'MIN_ROLLING_AVG_40', 'PTS_ROLLING_AVG_40', 'FGA_ROLLING_AVG_40', 'FG3A_ROLLING_AVG_40', 'FTA_ROLLING_AVG_40',
    'USG_PCT_ROLLING_AVG_40', 'TS_PCT_ROLLING_AVG_40', 'EFG_PCT_ROLLING_AVG_40', 'AST_ROLLING_AVG_40', 
    'REB_ROLLING_AVG_40', 'TOV_ROLLING_AVG_40',
    ]

    for col in include:
        try:
            value = player[col].dropna().iloc[-1] if not player[col].dropna().empty else 0
        except:
            print(f"Error: {col} not found in player data")
            value = 0
        res.append(value)
    return res

def getOppStats(oppTeamAbv, data):
    oppTeamID = findTeamID(oppTeamAbv, data)
    
    if oppTeamID is None:
        return [0] * 7
    
    # Filter by opponent team ID
    opp_data = data[data['OPP_TEAM_ID'] == oppTeamID]
    
    if opp_data.empty:
        return [0] * 7
    
    # Get unique games to avoid duplicates
    unique_games = opp_data.drop_duplicates(subset=['GAME_ID'])
    
    stats = [
        unique_games['TEAM_DEF_RATING'].mean(),
        unique_games['TEAM_PACE'].mean(),
        unique_games['TEAM_OFF_RATING'].mean(),  
        unique_games['TEAM_PTS'].mean(),
        unique_games['TEAM_FGA'].mean(),
        unique_games['TEAM_REB'].mean(),
        unique_games['TEAM_AST'].mean(),
        unique_games['TEAM_TOV'].mean(),
        unique_games['TEAM_BLK'].mean(),
        unique_games['TEAM_STL'].mean()
    ]
    return [round(stat, 2) for stat in stats]

def getTeamStats(teamAbv, data):
    team_data = data[data['TEAM_ABBREVIATION'] == teamAbv]
    if team_data.empty:
        return [0] * 7
    unique_games = team_data.drop_duplicates(subset=['GAME_ID'])
    
    stats = [
        unique_games['TEAM_OFF_RATING'].mean(),
        unique_games['TEAM_DEF_RATING'].mean(), 
        unique_games['TEAM_PACE'].mean(),
        unique_games['TEAM_FGA'].mean(),
        unique_games['TEAM_PTS'].mean(),
        unique_games['TEAM_REB'].mean(),
        unique_games['TEAM_AST'].mean(),
        unique_games['TEAM_TOV'].mean()
    ]
    return [round(stat, 2) for stat in stats]

def getMatchupStats(player, opp, data, n_games=3):
    player_id = findPlayerID(player, data)
    opp_id = findTeamID(opp, data)
    include = ['MIN', 'FGA', 'FG3A', 'FTA', 'PTS', 'USG_PCT',
            'EFG_PCT', 'TS_PCT', 'AST', 'REB', 'TOV']

    df = data[(data['PLAYER_ID'] == player_id) & (data['OPP_TEAM_ID'] == opp_id)].copy()
    if df.empty:
        return [None] * len(include)

    if 'GAME_DATE' in df.columns:
        # ensure proper sorting by date
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.sort_values('GAME_DATE')
    elif 'GAME_ID' in df.columns:
        df = df.sort_values('GAME_ID')

    last = df.tail(n_games)
    means = round(last[include].mean(numeric_only=True), 1)

    return [means.get(col, None) for col in include]

def getTeamOdds(player, data, game_id, teamabv):
    player_id = findPlayerID(player, data)
    game_data = data[
        (data['GAME_ID'] == game_id) & 
        (data['TEAM_ABBREVIATION'] == teamabv)
    ]
    
    if game_data.empty:
        return [0, 0, 0, 0, 0]  # Return zeros if no data found
    
    # Get the first row since all players from same team-game should have same odds
    row = game_data.iloc[0]
    odds_features = [
        row.get('team_spread', 0),
        row.get('total', 0), 
        row.get('team_is_favored', 0),
        row.get('TEAM_IMPLIED_PTS_FAV', 0),
        row.get('TEAM_IMPLIED_PTS_UND', 0),
        row.get('BLOWOUT_RISK', 0)
    ]
    
    return odds_features
    
#--------------------------------------------------------------------------------------------------------------------------------
def buildFeatureVector(player, teamabv, opponent, data, gamesSchedule, starters, game_id,n_games=3):
    features = (findPlayerID(player, data) +
                findTeamID(teamabv, data) +
                findTeamID(opponent, data) +
                getPlayerSpecificFeatures(player, data, gamesSchedule) +
                getPlayerStarInformation(player, starters, data) + 
                getPlayerSeasonAverages(player, data) +
                getPlayerLags(player, data) +
                getPlayerRollingAVG(player, data) +
                getTeamStats(opponent, data) +
                getTeamStats(teamabv, data) +
                getMatchupStats(player, opponent, data, n_games=n_games) +
                getTeamOdds(player, data, game_id, teamabv))
    return features

def make_prediction(player_name, bookmakers, opponent, model, data, games, is_playoff, stat_line='PTS'):
    features = buildFeatureVector(player_name, opponent, data, games, is_playoff, stat_line)
    X_pred = pd.DataFrame([features], columns=model.feature_names_in_)
    prediction = model.predict(X_pred)[0]
    prop_line = bookmakers[bookmakers['NAME'] == player_name]['LINE'].values[0]
    return {
        'player': player_name,
        'opponent': opponent,
        'predicted_stat': round(prediction),
        'raw_prediction': prediction,
        'prop_line': prop_line,
        'edge': round(prediction - prop_line, 1),
        'recommendation': 'OVER' if prediction > prop_line else 'UNDER'
    }
    