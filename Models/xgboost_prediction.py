import requests 
from datetime import datetime
import pytz
import pandas as pd
import joblib
from Models.xgboost_model import *
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

def getPlayerAVG(player, data, stat_type='PTS'):
    player_data = data[data['PLAYER_NAME'] == player]
    feature_sets = {
        'PTS': [
                'MIN', 'FGA', 'FTA', 'FG3A', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'REB', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF',
                'OFF_RATING', 'DEF_RATING', 'PointsPerShot', 'EFG_PCT', 'AST_PCT', 'AST_TOV', 'USG_PCT', 'TS_PCT', 'PACE', 'POSS',
                'TEAM_FGA', 'TEAM_FG3A', 'TEAM_FG_PCT', 'TEAM_FG3_PCT', 'TEAM_AST', 'TEAM_REB', 'TEAM_STL', 'TEAM_BLK', 'TEAM_PACE', 'TEAM_PTS'
                ],
        'AST': [
                'MIN', 'FGA', 'USG_PCT', 'AST_PCT', 'AST_TOV', 'TOV', 'PACE', 'POSS',
                'OFF_RATING', 'NET_RATING', 'PIE', 'PLUS_MINUS',
                'TEAM_AST', 'TEAM_PACE', 'TEAM_OFF_RATING', 'TEAM_FG_PCT',
            ],
        'REB': [
                'MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'USG_PCT', 'BLK', 'DEF_RATING', 'PACE', 'POSS',
                'OREB_PCT', 'DREB_PCT', 'REB_PCT', 
                'TEAM_FGA', 'TEAM_FG_PCT', 'TEAM_FG3A', 'TEAM_FG3_PCT', 'TEAM_PACE', 'TEAM_REB', 'TEAM_OREB', 'TEAM_DREB'
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
            'MIN_ROLL_AVG_2', 'FGA_ROLL_AVG_2', 'FGM_ROLL_AVG_2', 'FG_PCT_ROLL_AVG_2',
            'FG3A_ROLL_AVG_2', 'FG3M_ROLL_AVG_2', 'FG3_PCT_ROLL_AVG_2', 'FTM_ROLL_AVG_2',
            'FTA_ROLL_AVG_2', 'FT_PCT_ROLL_AVG_2', 'USG_PCT_ROLL_AVG_2', 'TS_PCT_ROLL_AVG_2',
            'EFG_PCT_ROLL_AVG_2', 'TEAM_PACE_ROLL_AVG_2', 'TEAM_OFF_RATING_ROLL_AVG_2',
            'OPP_DEF_RATING_ROLL_AVG_2', 'MIN_ROLL_AVG_4', 'FGA_ROLL_AVG_4', 'FGM_ROLL_AVG_4',
            'FG_PCT_ROLL_AVG_4', 'FG3A_ROLL_AVG_4', 'FG3M_ROLL_AVG_4', 'FG3_PCT_ROLL_AVG_4',
            'FTM_ROLL_AVG_4', 'FTA_ROLL_AVG_4', 'FT_PCT_ROLL_AVG_4', 'USG_PCT_ROLL_AVG_4',
            'TS_PCT_ROLL_AVG_4', 'EFG_PCT_ROLL_AVG_4', 'TEAM_PACE_ROLL_AVG_4',
            'TEAM_OFF_RATING_ROLL_AVG_4', 'OPP_DEF_RATING_ROLL_AVG_4', 'MIN_ROLL_AVG_6',
            'FGA_ROLL_AVG_6', 'FGM_ROLL_AVG_6', 'FG_PCT_ROLL_AVG_6', 'FG3A_ROLL_AVG_6',
            'FG3M_ROLL_AVG_6', 'FG3_PCT_ROLL_AVG_6', 'FTM_ROLL_AVG_6', 'FTA_ROLL_AVG_6',
            'FT_PCT_ROLL_AVG_6', 'USG_PCT_ROLL_AVG_6', 'TS_PCT_ROLL_AVG_6', 'EFG_PCT_ROLL_AVG_6',
            'TEAM_PACE_ROLL_AVG_6', 'TEAM_OFF_RATING_ROLL_AVG_6', 'OPP_DEF_RATING_ROLL_AVG_6',
            'PLAYER_HOME_AVG_PTS', 'PLAYER_AWAY_AVG_PTS',
],
        'AST': [    
            'MIN_ROLL_AVG_2', 'USG_PCT_ROLL_AVG_2', 'AST_PCT_ROLL_AVG_2', 'AST_TOV_ROLL_AVG_2',
            'TEAM_FGM_ROLL_AVG_2', 'TEAM_AST_ROLL_AVG_2', 'TEAM_PACE_ROLL_AVG_2',
            'OPP_DEF_RATING_ROLL_AVG_2', 'OPP_STL_ROLL_AVG_2',
            'MIN_ROLL_AVG_4', 'USG_PCT_ROLL_AVG_4', 'AST_PCT_ROLL_AVG_4', 'AST_TOV_ROLL_AVG_4',
            'TEAM_FGM_ROLL_AVG_4', 'TEAM_AST_ROLL_AVG_4', 'TEAM_PACE_ROLL_AVG_4',
            'OPP_DEF_RATING_ROLL_AVG_4', 'OPP_STL_ROLL_AVG_4',
            'MIN_ROLL_AVG_6', 'USG_PCT_ROLL_AVG_6', 'AST_PCT_ROLL_AVG_6', 'AST_TOV_ROLL_AVG_6',
            'TEAM_FGM_ROLL_AVG_6', 'TEAM_AST_ROLL_AVG_6', 'TEAM_PACE_ROLL_AVG_6',
            'OPP_DEF_RATING_ROLL_AVG_6', 'OPP_STL_ROLL_AVG_6',
            'PLAYER_HOME_AVG_AST', 'PLAYER_AWAY_AVG_AST',
],
        'REB': [
            'MIN_ROLL_AVG_2', 'FGA_ROLL_AVG_2', 'FGM_ROLL_AVG_2',
            'OREB_PCT_ROLL_AVG_2', 'DREB_PCT_ROLL_AVG_2', 'REB_PCT_ROLL_AVG_2',
            'USG_PCT_ROLL_AVG_2', 'POSS_ROLL_AVG_2',
            'TEAM_FGA_ROLL_AVG_2', 'TEAM_FG3A_ROLL_AVG_2', 'TEAM_FG_PCT_ROLL_AVG_2', 'TEAM_FG3_PCT_ROLL_AVG_2',
            'OPP_REB_ROLL_AVG_2', 'OPP_FG_PCT_ROLL_AVG_2', 'OPP_PACE_ROLL_AVG_2',
            'MIN_ROLL_AVG_4', 'FGA_ROLL_AVG_4', 'FGM_ROLL_AVG_4',
            'OREB_PCT_ROLL_AVG_4', 'DREB_PCT_ROLL_AVG_4', 'REB_PCT_ROLL_AVG_4',
            'USG_PCT_ROLL_AVG_4', 'POSS_ROLL_AVG_4',
            'TEAM_FGA_ROLL_AVG_4', 'TEAM_FG3A_ROLL_AVG_4', 'TEAM_FG_PCT_ROLL_AVG_4', 'TEAM_FG3_PCT_ROLL_AVG_4',
            'OPP_REB_ROLL_AVG_4', 'OPP_FG_PCT_ROLL_AVG_4', 'OPP_PACE_ROLL_AVG_4',
            'MIN_ROLL_AVG_6', 'FGA_ROLL_AVG_6', 'FGM_ROLL_AVG_6',
            'OREB_PCT_ROLL_AVG_6', 'DREB_PCT_ROLL_AVG_6', 'REB_PCT_ROLL_AVG_6',
            'USG_PCT_ROLL_AVG_6', 'POSS_ROLL_AVG_6',
            'TEAM_FGA_ROLL_AVG_6', 'TEAM_FG3A_ROLL_AVG_6', 'TEAM_FG_PCT_ROLL_AVG_6', 'TEAM_FG3_PCT_ROLL_AVG_6',
            'OPP_REB_ROLL_AVG_6', 'OPP_FG_PCT_ROLL_AVG_6', 'OPP_PACE_ROLL_AVG_6',
            'PLAYER_HOME_AVG_REB', 'PLAYER_AWAY_AVG_REB',]
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

def getPlayerTeam(player, data):
    teams = ['TEAM__ATL', 'TEAM__BKN', 'TEAM__BOS', 'TEAM__CHA', 'TEAM__CHI',
    'TEAM__CLE', 'TEAM__DAL', 'TEAM__DEN', 'TEAM__DET', 'TEAM__GSW',
    'TEAM__HOU', 'TEAM__IND', 'TEAM__LAC', 'TEAM__LAL', 'TEAM__MEM',
    'TEAM__MIA', 'TEAM__MIL', 'TEAM__MIN', 'TEAM__NOP', 'TEAM__NYK',
    'TEAM__OKC', 'TEAM__ORL', 'TEAM__PHI', 'TEAM__PHX', 'TEAM__POR',
    'TEAM__SAC', 'TEAM__SAS', 'TEAM__TOR', 'TEAM__UTA', 'TEAM__WAS']
    res = []
    player_team = data[data['PLAYER_NAME'] == player]['TEAM_ABBREVIATION'].iloc[-1]
    for team in teams:
        team_abbv = team.split('__')[1]
        if team_abbv in player_team:
            res.append(1)
        else:
            res.append(0)
    return res

def getOppPlayerTeam(opp_team):
    teams = ['OPP__ATL', 'OPP__BKN', 'OPP__BOS', 'OPP__CHA', 'OPP__CHI',
    'OPP__CLE', 'OPP__DAL', 'OPP__DEN', 'OPP__DET', 'OPP__GSW',
    'OPP__HOU', 'OPP__IND', 'OPP__LAC', 'OPP__LAL', 'OPP__MEM',
    'OPP__MIA', 'OPP__MIL', 'OPP__MIN', 'OPP__NOP', 'OPP__NYK',
    'OPP__OKC', 'OPP__ORL', 'OPP__PHI', 'OPP__PHX', 'OPP__POR',
    'OPP__SAC', 'OPP__SAS', 'OPP__TOR', 'OPP__UTA', 'OPP__WAS']
    res = []
    for team in teams:
        team_abbv = team.split('__')[1]
        if team_abbv in opp_team:
            res.append(1)
        else:
            res.append(0)
    return res

def getPlayerVsDefense(player, data, Opp, stat_type='PTS'):
    '''
    DEF_CATEGORY,
    PTS_VS_DEF_STRONG, PTS_VS_DEF_WEAK, PTS_VS_DEF_DIFF,
    FGA_VS_DEF_STRONG, FGA_VS_DEF_WEAK, FGA_VS_DEF_DIFF,
    FTA_VS_DEF_STRONG, FTA_VS_DEF_WEAK, FTA_VS_DEF_DIFF,
    FG3A_VS_DEF_STRONG, FG3A_VS_DEF_WEAK, FG3A_VS_DEF_DIFF,
    '''
    player_data = data[data['PLAYER_NAME'] == player].copy()
    res = []
    def get_defense_averages(df, stat=stat_type):
        strong = df[df['DEF_CATEGORY'] == 1][stat].mean()
        weak = df[df['DEF_CATEGORY'] == 0][stat].mean()
        diff = strong - weak
        return [strong, weak, diff]
    res.extend(get_defense_averages(player_data, Opp))
    




def getPlayerSpecificFeatures(player, data, games, stat_type='PTS'):
    currStars = ['Shai Gilgeous-Alexander', 'Nikola Jokić', 'Giannis Antetokounmpo', 'Jayson Tatum', 'Donovan Mitchell',
        'Anthony Edwards', 'LeBron James', 'Stephen Curry', 'Evan Mobley', 'Jalen Brunson',
        'Cade Cunningham', 'Karl-Anthony Towns', 'Tyrese Haliburton', 'Jalen Williams', 'James Harden',
        'Darius Garland', 'Kevin Durant', 'Anthony Davis', 'Kyrie Irving', 'Jaylen Brown', 'Tyler Herro', 'Jaren Jackson Jr.', 
        'Pascal Siakam', 'Victor Wembanyama', 'Alperen Sengun', 'Trae Young']
    
    player_data = data[data['PLAYER_NAME'] == player].copy()
    res = []
    
    # Calculate matchup average against opponent
    opponent_team = None
    for game in games:
        if game['home_team'] == player_data['TEAM_ABBREVIATION'].iloc[-1]:
            opponent_team = game['away_team']
            break
        if game['away_team'] == player_data['TEAM_ABBREVIATION'].iloc[-1]:
            opponent_team = game['home_team']
            break
    
    if opponent_team:
        # Get last 3 games against this opponent
        games_vs_opp = player_data[player_data['OPP_ABBREVIATION'] == opponent_team].tail(3)
        matchup_avg = games_vs_opp[stat_type].mean() if not games_vs_opp.empty else 0
        res.append(matchup_avg)
    else:
        res.append(0)  # No matchup found
    
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
    
    # Rest features
    days_rested = (datetime.today() - player_data['GAME_DATE'].iloc[-1]).days
    res.append(days_rested)
    if days_rested == 0:
        res.append(1)
    else:
        res.append(0)
    res.append(player_data['MISSED_LAST'].iloc[-1])
    
    # Star features
    if player in currStars:
        res.append(1)
    else:
        res.append(0)
    return res






#--------------------------------------------------------------------------------------------------------------------------------
def buildFeatureVector(player, opponent, data, games, is_playoff, stat_line='PTS'):
    features = (getPlayerAVG(player, data, stat_line) + 
                   getOppAVG(opponent, data) + 
                   getPlayerRollingAVG(player, data, stat_line) + 
                   otherFeatures(player, data, games, is_playoff))
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
    