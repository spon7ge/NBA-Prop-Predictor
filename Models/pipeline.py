import requests 
from datetime import datetime
import pytz
import pandas as pd
import joblib
from Models.model import *
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

def findOPP(player, data):
    player = data[data['PLAYER_NAME'] == player].sort_values(by='GAME_DATE')
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
        return None
    return opponent

def getPlayerSpecificFeatures(player, data, games, stat_type='PTS'):
    player_data = data[data['PLAYER_NAME'] == player].copy()
    res = []
    
    # Position features
    res.append(player_data['GUARD'].iloc[-1])      
    res.append(player_data['FORWARD'].iloc[-1])    
    res.append(player_data['CENTER'].iloc[-1])     
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
    return res

def getPlayerRest(player,data):
    player_data = data[data['PLAYER_NAME'] == player].copy()
    res = []
    res.append(today - player_data['GAME_DATE'].iloc[-1]) # gets how long since last game
    if today - player_data['GAME_DATE'].iloc[-1] == 1: # checks for back to back
        res.append(1)
    else:
        res.append(0)
    return res

def getPlayerSeasonAverages(player, data):
    player_data = data[data['PLAYER_NAME'] == player].copy()
    cols = ['MIN', 'PTS', 'FGA', 'FG3A', 'FTA', 'USG_PCT', 'TS_PCT', 'OFF_RATING', 'POINT_PER_SHOT', 
            'TCHS', 'POSS', 'PACE']
    res = []
    for col in cols:
        res.append(player_data[col].mean())
    return res

def getPlayerRollingAVG(player, data, stat_type='PTS'):
    player = data[data['PLAYER_NAME'] == player].copy()
    player.sort_values(by='GAME_DATE', inplace=True)
    res = []
    feature_sets = {'PTS': [ 'MIN_ROLLING_AVG_5', 'PTS_ROLLING_AVG_5', 'FGA_ROLLING_AVG_5',
    'FG3A_ROLLING_AVG_5', 'FTA_ROLLING_AVG_5', 'USG_PCT_ROLLING_AVG_5',
    'TS_PCT_ROLLING_AVG_5', 'OFF_RATING_ROLLING_AVG_5', 'PTS_LAG_1', 'PTS_LAG_2']
    }
    include = feature_sets[stat_type]

    for col in include:
        try:
            value = player[col].dropna().iloc[-1] if not player[col].dropna().empty else 0
        except:
            print(f"Error: {col} not found in player data")
            value = 0
        res.append(value)
    return res


def getPlayerStarInformation(player, data):
    pass

def get_opponent_defense_category(opp_team, data, current_date=None):
    """
    Determine if opponent is currently a strong (1) or weak (0) defense
    """
    if current_date:
        recent_data = data[data['GAME_DATE'] <= current_date]
    else:
        recent_data = data
    opp_def_rating = recent_data[recent_data['OPP_ABBREVIATION'] == opp_team]['OPP_DEF_RATING'].mean()
    all_team_ratings = recent_data.groupby('OPP_ABBREVIATION')['OPP_DEF_RATING'].mean()
    team_rank = (all_team_ratings <= opp_def_rating).sum()
    return 1 if team_rank <= 10 else 0

def get_most_recent_starters(historical_data, team_id):
    """
    Get starters from the team's most recent game
    """
    # Get team's data and sort by date
    team_data = historical_data[historical_data['TEAM_ID'] == team_id].copy()
    team_data = team_data.sort_values('GAME_DATE', ascending=False)
    
    # Get most recent game date
    most_recent_date = team_data['GAME_DATE'].iloc[0]
    
    # Get starters from that game
    recent_starters = team_data[
        (team_data['GAME_DATE'] == most_recent_date) & 
        (team_data['STARTING'] == 1)
    ]
    
    return recent_starters

def get_team_starter_features(historical_data, team_id):
    """
    Calculate starter features using most recent game's starters, focusing only on statistical averages
    """
    # Get starters from most recent game
    current_starters_df = get_most_recent_starters(historical_data, team_id)
    
    # Calculate all starter averages
    starter_features = {
        'TEAM_STARTER_OFF_RATING_AVG': current_starters_df['OFF_RATING'].mean(),
        'TEAM_STARTER_DEF_RATING_AVG': current_starters_df['DEF_RATING'].mean(),
        'TEAM_STARTER_USG_PCT_AVG': current_starters_df['USG_PCT'].mean(),
        'TEAM_STARTER_SPACING_METRIC': current_starters_df['FG3_PCT'].mean(),
        'TEAM_STARTER_PACE': current_starters_df['PACE'].mean()
    }
    
    return starter_features, starter_features['TEAM_STARTER_PACE']  # Return pace separately for PACE_EXPECTATION

def get_all_starter_features(historical_data, home_team_id, away_team_id):
    """
    Get starter features for both teams, including PACE_EXPECTATION
    """
    home_features, home_pace = get_team_starter_features(historical_data, home_team_id)
    away_features, away_pace = get_team_starter_features(historical_data, away_team_id)
    
    # Calculate PACE_EXPECTATION
    pace_expectation = (home_pace + away_pace) / 2
    
    # Add PACE_EXPECTATION to both feature sets
    home_features['PACE_EXPECTATION'] = pace_expectation
    away_features['PACE_EXPECTATION'] = pace_expectation
    
    return home_features, away_features



#--------------------------------------------------------------------------------------------------------------------------------
def buildFeatureVector(player, opponent, data, games, is_playoff, stat_line='PTS'):
    features = (   getPlayerRollingAVG(player, data, stat_line) + 
                   getPlayerTeam(player, data) +
                   getOppPlayerTeam(opponent) +
                   getPlayerVsDefense(player, data, opponent, stat_line) +
                   getPlayerSpecificFeatures(player, data, games) +
                   get_all_starter_features(data, games) +
                   getPlayoffFeatures(player, data, is_playoff))
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
    