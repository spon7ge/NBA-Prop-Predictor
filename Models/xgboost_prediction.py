import requests 
from datetime import datetime
import pytz
import pandas as pd
import joblib
today = datetime.today().strftime('%Y-%m-%d')

def load_xgboost_model(stat_line='PTS'):
    model = joblib.load(f'Models/{stat_line}_xgboost_model.pkl')
    return model

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

def getPlayerAVG(player, data, stat_type='PTS'):
    player_data = data[data['PLAYER_NAME'] == player]
    feature_sets = {
        'PTS': [
                'MIN','FGA', 'FTA', 'FG3A','FG_PCT', 'FT_PCT', 'FG3_PCT', 'REB','OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                'OFF_RATING','E_OFF_RATING', 'DEF_RATING', 'E_DEF_RATING', 'NET_RATING', 'PointsPerShot', 'EFG_PCT',
                'AST_PCT', 'AST_TOV','USG_PCT', 'TS_PCT','PACE', 'PIE', 'POSS', 'E_USG_PCT', 'PLUS_MINUS',
                'TEAM_FGA', 'TEAM_FG3A','TEAM_FG_PCT','TEAM_FG3_PCT','TEAM_AST', 'TEAM_REB', 'TEAM_STL', 'TEAM_BLK', 
                'TEAM_OFF_RATING', 'TEAM_PACE', 'TEAM_PTS'
                ],
        'AST': [
                'MIN','FGA', 'FTA', 'FG3A','FG_PCT', 'FT_PCT', 'FG3_PCT', 'REB','OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF',
                'OFF_RATING','E_OFF_RATING', 'DEF_RATING', 'E_DEF_RATING', 'NET_RATING', 'PointsPerShot', 'EFG_PCT',
                'AST_PCT', 'AST_TOV','USG_PCT', 'TS_PCT','PACE', 'PIE', 'POSS', 'E_USG_PCT', 'PLUS_MINUS',
                'TEAM_FGA', 'TEAM_FG3A','TEAM_FG_PCT','TEAM_FG3_PCT','TEAM_AST', 'TEAM_REB', 'TEAM_STL', 'TEAM_BLK', 
                'TEAM_OFF_RATING', 'TEAM_PACE', 'TEAM_PTS'
                ],
        'REB': [
                'MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'TOV', 'PF', 'BLKS', 'PointsPerShot', 'USG_PCT', 'TS_PCT', 'EFG_PCT', 'PIE', 'POSS',
                'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'E_PACE',
                'TEAM_PACE', 'TEAM_REB', 'TEAM_OREB', 'TEAM_DREB', 'TEAM_BLKS', 'TEAM_OFF_RATING', 'TEAM_FGA', 'TEAM_FG_PCT', 'TEAM_FG3A', 'TEAM_FG3_PCT'
                ]
    }
    include = feature_sets[stat_type]

    res = [round(player_data[col].mean(), 2) for col in include]
    return res

def findOPP(player, data, games):
    player = data[data['PLAYER_NAME'] == player].sort_values(by='GAME_DATE')
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

def getOppAVG(team, data):
    team_data = data[data['OPP_ABBREVIATION'] == team]
    include = ['OPP_PACE', 'OPP_DEF_RATING','OPP_STL', 'OPP_BLK', 'OPP_REB', 'OPP_FG_PCT']
    team_stats = team_data.groupby('GAME_DATE')[include].mean().reset_index()

    return [round(team_stats[col].mean(), 2) for col in include]

def getPlayerRollingAVG(player, data, stat_type='PTS'):
    player = data[data['PLAYER_NAME'] == player].copy()
    player.sort_values(by='GAME_DATE', inplace=True)
    res = []
    feature_sets = {
        'PTS': [
                'MIN_ROLL_AVG_2', 'PTS_ROLL_AVG_2', 'PTS_STD_AVG_2', 'FGA_ROLL_AVG_2',
                'FGM_ROLL_AVG_2', 'FG_PCT_ROLL_AVG_2', 'FG3A_ROLL_AVG_2', 'FG3M_ROLL_AVG_2',
                'FG3_PCT_ROLL_AVG_2', 'FTM_ROLL_AVG_2', 'FTA_ROLL_AVG_2', 'FT_PCT_ROLL_AVG_2',
                'USG_PCT_ROLL_AVG_2', 'TS_PCT_ROLL_AVG_2', 'EFG_PCT_ROLL_AVG_2',
                'OREB_ROLL_AVG_2', 'DREB_ROLL_AVG_2', 'REB_ROLL_AVG_2',
                'PLUS_MINUS_ROLL_AVG_2', 'PIE_ROLL_AVG_2', 'TEAM_FGA_ROLL_AVG_2',
                'TEAM_FG_PCT_ROLL_AVG_2', 'TEAM_FG3A_ROLL_AVG_2', 'TEAM_FG3_PCT_ROLL_AVG_2',
                'TEAM_FTM_ROLL_AVG_2', 'TEAM_FTA_ROLL_AVG_2', 'TEAM_FT_PCT_ROLL_AVG_2',
                'TEAM_PTS_ROLL_AVG_2', 'TEAM_PACE_ROLL_AVG_2', 'TEAM_OFF_RATING_ROLL_AVG_2',
                'OPP_DEF_RATING_ROLL_AVG_2', 'OPP_PACE_ROLL_AVG_2', 'OPP_FG_PCT_ROLL_AVG_2',
                'MIN_ROLL_AVG_4', 'PTS_ROLL_AVG_4', 'PTS_STD_AVG_4', 'FGA_ROLL_AVG_4',
                'FGM_ROLL_AVG_4', 'FG_PCT_ROLL_AVG_4', 'FG3A_ROLL_AVG_4', 'FG3M_ROLL_AVG_4',
                'FG3_PCT_ROLL_AVG_4', 'FTM_ROLL_AVG_4', 'FTA_ROLL_AVG_4', 'FT_PCT_ROLL_AVG_4',
                'USG_PCT_ROLL_AVG_4', 'TS_PCT_ROLL_AVG_4', 'EFG_PCT_ROLL_AVG_4',
                'OREB_ROLL_AVG_4', 'DREB_ROLL_AVG_4', 'REB_ROLL_AVG_4',
                'PLUS_MINUS_ROLL_AVG_4', 'PIE_ROLL_AVG_4', 'TEAM_FGA_ROLL_AVG_4',
                'TEAM_FG_PCT_ROLL_AVG_4', 'TEAM_FG3A_ROLL_AVG_4', 'TEAM_FG3_PCT_ROLL_AVG_4',
                'TEAM_FTM_ROLL_AVG_4', 'TEAM_FTA_ROLL_AVG_4', 'TEAM_FT_PCT_ROLL_AVG_4',
                'TEAM_PTS_ROLL_AVG_4', 'TEAM_PACE_ROLL_AVG_4', 'TEAM_OFF_RATING_ROLL_AVG_4',
                'OPP_DEF_RATING_ROLL_AVG_4', 'OPP_PACE_ROLL_AVG_4', 'OPP_FG_PCT_ROLL_AVG_4',
                'MIN_ROLL_AVG_6', 'PTS_ROLL_AVG_6', 'PTS_STD_AVG_6', 'FGA_ROLL_AVG_6',
                'FGM_ROLL_AVG_6', 'FG_PCT_ROLL_AVG_6', 'FG3A_ROLL_AVG_6', 'FG3M_ROLL_AVG_6',
                'FG3_PCT_ROLL_AVG_6', 'FTM_ROLL_AVG_6', 'FTA_ROLL_AVG_6', 'FT_PCT_ROLL_AVG_6',
                'USG_PCT_ROLL_AVG_6', 'TS_PCT_ROLL_AVG_6', 'EFG_PCT_ROLL_AVG_6',
                'OREB_ROLL_AVG_6', 'DREB_ROLL_AVG_6', 'REB_ROLL_AVG_6',
                'PLUS_MINUS_ROLL_AVG_6', 'PIE_ROLL_AVG_6', 'TEAM_FGA_ROLL_AVG_6',
                'TEAM_FG_PCT_ROLL_AVG_6', 'TEAM_FG3A_ROLL_AVG_6', 'TEAM_FG3_PCT_ROLL_AVG_6',
                'TEAM_FTM_ROLL_AVG_6', 'TEAM_FTA_ROLL_AVG_6', 'TEAM_FT_PCT_ROLL_AVG_6',
                'TEAM_PTS_ROLL_AVG_6', 'TEAM_PACE_ROLL_AVG_6', 'TEAM_OFF_RATING_ROLL_AVG_6',
                'OPP_DEF_RATING_ROLL_AVG_6', 'OPP_PACE_ROLL_AVG_6', 'OPP_FG_PCT_ROLL_AVG_6',
                'PTS_LAG_1', 'PTS_LAG_2', 'PTS_LAG_3', 'PTS_LAG_4',
                'PLAYER_HOME_AVG_PTS', 'PLAYER_AWAY_AVG_PTS', 'MATCHUP_AVG_PTS_LAST_3'
],
        'AST': ['MIN_ROLL_AVG_2', 'AST_ROLL_AVG_2', 'FGA_ROLL_AVG_2', 'FGM_ROLL_AVG_2',
                'FG_PCT_ROLL_AVG_2', 'FG3A_ROLL_AVG_2', 'FG3M_ROLL_AVG_2', 'FG3_PCT_ROLL_AVG_2',
                'FTM_ROLL_AVG_2', 'FTA_ROLL_AVG_2', 'FT_PCT_ROLL_AVG_2', 'USG_PCT_ROLL_AVG_2',
                'AST_PCT_ROLL_AVG_2', 'AST_TOV_ROLL_AVG_2', 'TS_PCT_ROLL_AVG_2',
                'EFG_PCT_ROLL_AVG_2', 'PIE_ROLL_AVG_2', 'PLUS_MINUS_ROLL_AVG_2',
                'TEAM_FG_PCT_ROLL_AVG_2', 'TEAM_FGM_ROLL_AVG_2', 'TEAM_AST_ROLL_AVG_2',
                'TEAM_TOV_ROLL_AVG_2', 'TEAM_PACE_ROLL_AVG_2', 'TEAM_PTS_ROLL_AVG_2',
                'OPP_DEF_RATING_ROLL_AVG_2', 'OPP_STL_ROLL_AVG_2', 'OPP_PACE_ROLL_AVG_2',
                'MIN_ROLL_AVG_4', 'AST_ROLL_AVG_4', 'FGA_ROLL_AVG_4', 'FGM_ROLL_AVG_4',
                'FG_PCT_ROLL_AVG_4', 'FG3A_ROLL_AVG_4', 'FG3M_ROLL_AVG_4', 'FG3_PCT_ROLL_AVG_4',
                'FTM_ROLL_AVG_4', 'FTA_ROLL_AVG_4', 'FT_PCT_ROLL_AVG_4', 'USG_PCT_ROLL_AVG_4',
                'AST_PCT_ROLL_AVG_4', 'AST_TOV_ROLL_AVG_4', 'TS_PCT_ROLL_AVG_4',
                'EFG_PCT_ROLL_AVG_4', 'PIE_ROLL_AVG_4', 'PLUS_MINUS_ROLL_AVG_4',
                'TEAM_FG_PCT_ROLL_AVG_4', 'TEAM_FGM_ROLL_AVG_4', 'TEAM_AST_ROLL_AVG_4',
                'TEAM_TOV_ROLL_AVG_4', 'TEAM_PACE_ROLL_AVG_4', 'TEAM_PTS_ROLL_AVG_4',
                'OPP_DEF_RATING_ROLL_AVG_4', 'OPP_STL_ROLL_AVG_4', 'OPP_PACE_ROLL_AVG_4',
                'MIN_ROLL_AVG_6', 'AST_ROLL_AVG_6', 'FGA_ROLL_AVG_6', 'FGM_ROLL_AVG_6',
                'FG_PCT_ROLL_AVG_6', 'FG3A_ROLL_AVG_6', 'FG3M_ROLL_AVG_6', 'FG3_PCT_ROLL_AVG_6',
                'FTM_ROLL_AVG_6', 'FTA_ROLL_AVG_6', 'FT_PCT_ROLL_AVG_6', 'USG_PCT_ROLL_AVG_6',
                'AST_PCT_ROLL_AVG_6', 'AST_TOV_ROLL_AVG_6', 'TS_PCT_ROLL_AVG_6',
                'EFG_PCT_ROLL_AVG_6', 'PIE_ROLL_AVG_6', 'PLUS_MINUS_ROLL_AVG_6',
                'TEAM_FG_PCT_ROLL_AVG_6', 'TEAM_FGM_ROLL_AVG_6', 'TEAM_AST_ROLL_AVG_6',
                'TEAM_TOV_ROLL_AVG_6', 'TEAM_PACE_ROLL_AVG_6', 'TEAM_PTS_ROLL_AVG_6',
                'OPP_DEF_RATING_ROLL_AVG_6', 'OPP_STL_ROLL_AVG_6', 'OPP_PACE_ROLL_AVG_6',
                'AST_LAG_1', 'AST_LAG_2', 'AST_LAG_3', 'AST_LAG_4',
                'PLAYER_HOME_AVG_AST', 'PLAYER_AWAY_AVG_AST', 'MATCHUP_AVG_AST_LAST_3'
],
        'REB': ['MIN_ROLL_AVG_2', 'OREB_ROLL_AVG_2', 'DREB_ROLL_AVG_2', 'REB_ROLL_AVG_2',
                'FGA_ROLL_AVG_2', 'FGM_ROLL_AVG_2', 'FG_PCT_ROLL_AVG_2', 'FG3A_ROLL_AVG_2',
                'FG3M_ROLL_AVG_2', 'FG3_PCT_ROLL_AVG_2', 'FTM_ROLL_AVG_2', 'FTA_ROLL_AVG_2',
                'FT_PCT_ROLL_AVG_2', 'OREB_PCT_ROLL_AVG_2', 'DREB_PCT_ROLL_AVG_2',
                'REB_PCT_ROLL_AVG_2', 'PIE_ROLL_AVG_2', 'PLUS_MINUS_ROLL_AVG_2',
                'USG_PCT_ROLL_AVG_2', 'TS_PCT_ROLL_AVG_2', 'EFG_PCT_ROLL_AVG_2',
                'PACE_ROLL_AVG_2', 'POSS_ROLL_AVG_2', 'TEAM_FG_PCT_ROLL_AVG_2',
                'TEAM_FG3_PCT_ROLL_AVG_2', 'TEAM_FGA_ROLL_AVG_2', 'TEAM_FG3A_ROLL_AVG_2',
                'OPP_REB_ROLL_AVG_2', 'OPP_FG_PCT_ROLL_AVG_2', 'OPP_DEF_RATING_ROLL_AVG_2',
                'OPP_PACE_ROLL_AVG_2', 'MIN_ROLL_AVG_4', 'OREB_ROLL_AVG_4', 'DREB_ROLL_AVG_4',
                'REB_ROLL_AVG_4', 'FGA_ROLL_AVG_4', 'FGM_ROLL_AVG_4', 'FG_PCT_ROLL_AVG_4',
                'FG3A_ROLL_AVG_4', 'FG3M_ROLL_AVG_4', 'FG3_PCT_ROLL_AVG_4', 'FTM_ROLL_AVG_4',
                'FTA_ROLL_AVG_4', 'FT_PCT_ROLL_AVG_4', 'OREB_PCT_ROLL_AVG_4',
                'DREB_PCT_ROLL_AVG_4', 'REB_PCT_ROLL_AVG_4', 'PIE_ROLL_AVG_4',
                'PLUS_MINUS_ROLL_AVG_4', 'USG_PCT_ROLL_AVG_4', 'TS_PCT_ROLL_AVG_4',
                'EFG_PCT_ROLL_AVG_4', 'PACE_ROLL_AVG_4', 'POSS_ROLL_AVG_4',
                'TEAM_FG_PCT_ROLL_AVG_4', 'TEAM_FG3_PCT_ROLL_AVG_4', 'TEAM_FGA_ROLL_AVG_4',
                'TEAM_FG3A_ROLL_AVG_4', 'OPP_REB_ROLL_AVG_4', 'OPP_FG_PCT_ROLL_AVG_4',
                'OPP_DEF_RATING_ROLL_AVG_4', 'OPP_PACE_ROLL_AVG_4', 'MIN_ROLL_AVG_6',
                'OREB_ROLL_AVG_6', 'DREB_ROLL_AVG_6', 'REB_ROLL_AVG_6', 'FGA_ROLL_AVG_6',
                'FGM_ROLL_AVG_6', 'FG_PCT_ROLL_AVG_6', 'FG3A_ROLL_AVG_6', 'FG3M_ROLL_AVG_6',
                'FG3_PCT_ROLL_AVG_6', 'FTM_ROLL_AVG_6', 'FTA_ROLL_AVG_6', 'FT_PCT_ROLL_AVG_6',
                'OREB_PCT_ROLL_AVG_6', 'DREB_PCT_ROLL_AVG_6', 'REB_PCT_ROLL_AVG_6',
                'PIE_ROLL_AVG_6', 'PLUS_MINUS_ROLL_AVG_6', 'USG_PCT_ROLL_AVG_6',
                'TS_PCT_ROLL_AVG_6', 'EFG_PCT_ROLL_AVG_6', 'PACE_ROLL_AVG_6',
                'POSS_ROLL_AVG_6', 'TEAM_FG_PCT_ROLL_AVG_6', 'TEAM_FG3_PCT_ROLL_AVG_6',
                'TEAM_FGA_ROLL_AVG_6', 'TEAM_FG3A_ROLL_AVG_6', 'OPP_REB_ROLL_AVG_6',
                'OPP_FG_PCT_ROLL_AVG_6', 'OPP_DEF_RATING_ROLL_AVG_6', 'OPP_PACE_ROLL_AVG_6',
                'REB_LAG_1', 'REB_LAG_2', 'REB_LAG_3', 'REB_LAG_4',
                'PLAYER_HOME_AVG_REB', 'PLAYER_AWAY_AVG_REB', 'MATCHUP_AVG_REB_LAST_3']
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

def otherFeatures(player, data, games, is_playoff=0):
    player = data[data['PLAYER_NAME'] == player].copy()
    player.sort_values(by='GAME_DATE', inplace=True)
    
    res = []
    res.append(player['GUARD'].iloc[-1])
    res.append(player['FORWARD'].iloc[-1])
    res.append(player['CENTER'].iloc[-1])
    res.append(player['STARTING'].iloc[-1])
    res.append(player['DAYS_OF_REST'].iloc[-1])
    for game in games:
        if game['home_team'] == player['TEAM_ABBREVIATION'].iloc[-1]:
            res.append(1)
        else:
            res.append(0)
    res.append(is_playoff)
    if is_playoff == 0:
        series = 0
        gameInSeries = 0
    elif is_playoff == 1:
        series = 1
        gameInSeries = 1
    else:
        series = 0
        gameInSeries = 0
    res.append(series)
    res.append(gameInSeries)
    return res

def buildFeatureVector(player, opponent, data, games, is_playoff, series, game_in_series, stat_line='PTS'):
    features = (getPlayerAVG(player, data, stat_line) + 
                   getOppAVG(opponent, data) + 
                   getPlayerRollingAVG(player, data, stat_line) + 
                   otherFeatures(player, data, games, is_playoff, series, game_in_series))
    return features

def loadPrizePicksProps(date_str=today, prop_type=None):
    propsData = pd.read_csv(f'PROPS_DATA/Playoffs_DFS({date_str}).csv')
    prizePicksProps = propsData[(propsData['BOOKMAKER'] == 'PrizePicks') & (propsData['CATEGORY'] == prop_type)]
    prizePicksProps = prizePicksProps[['NAME', 'LINE']].drop_duplicates()
    return prizePicksProps

def make_prediction(player_name, opponent, model, data, prizePicksProps, games, is_playoff, series, game_in_series, stat_line='PTS'):
    features = buildFeatureVector(player_name, opponent, data, games, is_playoff, series, game_in_series, stat_line)
    X_pred = pd.DataFrame([features], columns=model.feature_names_in_)
    prediction = model.predict(X_pred)[0]
    prop_line = prizePicksProps[prizePicksProps['NAME'] == player_name]['LINE'].values[0]
    return {
        'player': player_name,
        'opponent': opponent,
        'predicted_stat': round(prediction),
        'raw_prediction': prediction,
        'prop_line': prop_line,
        'edge': round(prediction - prop_line, 1),
        'recommendation': 'OVER' if prediction > prop_line else 'UNDER'
    }
    