import requests 
from datetime import datetime
import pytz
import pandas as pd
from MODELS.model import *
from catboost import Pool


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

def findOppTeam(player, data, games): # call get_espn_games to get games before and enter the correct date
    player_id, player_team = findPlayerID(player, data)
    player_team = normalize_team_abbreviation(player_team)
    opponent_team = None
    homeGame = 0
    for game in games:
        home_team = normalize_team_abbreviation(game['home_team'])
        away_team = normalize_team_abbreviation(game['away_team'])
        
        if home_team == player_team:
            opponent_team = away_team
            homeGame = 1
            break
        elif away_team == player_team:
            opponent_team = home_team
            homeGame = 0
            break
    return opponent_team, homeGame

def findPlayerID(player_name, data):
    result = data[data['PLAYER_NAME'] == player_name]['PLAYER_ID']
    player_id = result.iloc[0] if len(result) > 0 else None
    player_team = data[data['PLAYER_ID'] == player_id]['TEAM_ABBREVIATION'].iloc[0]
    return player_id, player_team

def findTeamID(team_abv, data):
    result = data[data['TEAM_ABBREVIATION'] == team_abv]['TEAM_ID']
    return result.iloc[0] if len(result) > 0 else None
    
ESPN_MAPPING = {
    'NO': 'NOP',    
    'UTAH': 'UTA',    
    'GS': 'GSW',    
    'NY': 'NYK',    
    'SA': 'SAS',    
}

def normalize_team_abbreviation(espn_abbrev):
    return ESPN_MAPPING.get(espn_abbrev, espn_abbrev)

def getPlayerSpecificFeatures(player, data, starters, games, todayDate):
    player_id, player_team = findPlayerID(player, data)
    opponent_team, homeGame = findOppTeam(player, data, games)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    player_data.sort_values(by='GAME_DATE', inplace=True, ascending=False)
    res = []
    
    # player ID and team ID
    res.append(player_id)
    res.append(findTeamID(player_team, data))
    
    # opponent ID
    res.append(findTeamID(opponent_team, data))
    
    # checking if player is a starter
    if player in starters:
        res.append(1)
    else:
        res.append(0)
    
    # checking if player is playing at home
    res.append(homeGame)
    
    # get how long since last game
    todayDate = str(todayDate)
    lastDate = str(player_data['GAME_DATE'].iloc[-1])
    date_obj = datetime.strptime(lastDate, "%Y-%m-%d")
    formatted_date = date_obj.strftime("%Y%m%d")
    
    date1 = datetime.strptime(formatted_date, "%Y%m%d").date()
    date2 = datetime.strptime(todayDate, "%Y%m%d").date()
    res.append(date2 - date1) # gets how long since last game
    if date2 - date1 == 1: # checks for back to back
        res.append(1)
    else:
        res.append(0)
    return res

def getStarters(game_id, team, data): # temporary until i find a better way to get starters
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

# teamStars = [ 'Shai Gilgeous-Alexander', 'Nikola Jokić', 'Giannis Antetokounmpo', 'Jayson Tatum', 'Donovan Mitchell',
#         'Anthony Edwards', 'LeBron James', 'Stephen Curry', 'Evan Mobley', 'Jalen Brunson', 'Kevin Durant', 
#         'Cade Cunningham', 'Karl-Anthony Towns', 'Tyrese Haliburton', 'Jalen Williams', 'James Harden',
#         'Darius Garland', 'Damian Lillard', 'Anthony Davis', 'Kyrie Irving', 'Jaylen Brown', 'Tyler Herro', 'Jaren Jackson Jr.', 
#         'Pascal Siakam', 'Victor Wembanyama', 'Alperen Sengun', 'Trae Young', 'LaMelo Ball', 'Devin Booker', 'Joel Embiid', 'Luka Doncic'
# ]

# def getAllNBAPlayers(player, data, teamStars, starters, game_id):   
#     player_id, player_team = findPlayerID(player, data)
#     player_data = data[data['PLAYER_ID'] == player_id].copy()
#     player_data = player_data[player_data['GAME_ID'] == game_id].copy()
#     res = []
#     starCount = 0
    
#     if player_data['PLAYER_NAME'].isin(teamStars).any():
#         res.append(1)
#     else:
#         res.append(0)
#     starters = data[data['GAME_ID'] == game_id][data['STARTING'] == 1]['PLAYER_NAME'].tolist()
    
#     for starter in starters:
#         if starter in teamStars:
#             starCount += 1
#     res.append(starCount)
#     return res
    

def getPlayerSeasonAverages(player, data):
    player_id, player_team = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    cols = ['MIN', 'PTS', 'FGA', 'FG3A', 'FTA', 'USG_PCT', 'TS_PCT', 
            'EFG_PCT', 'AST', 'REB', 'TOV']
    res = []
    for col in cols:
        res.append(round(player_data[col].mean(), 2))
    return res

def getPlayerLags(player, data):
    player_id, player_team = findPlayerID(player, data)
    player_data = data[data['PLAYER_ID'] == player_id].copy()
    lags = []
    cols = ['PTS', 'FGA', 'MIN', 'USG_PCT']
    for col in cols:
        lags.append(player_data[col].iloc[-1])
        lags.append(player_data[col].iloc[-2])
    return lags

def getPlayerRollingAVG(player, data):
    player_id, player_team = findPlayerID(player, data)
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
        value = player_data[col].dropna().iloc[-1] if not player_data[col].dropna().empty else 0
        res.append(value)
    return res

def getOppStats(player, data, games):
    opponent_team, homeGame = findOppTeam(player, data, games)
    oppTeamID = findTeamID(opponent_team, data)
    
    if oppTeamID is None:
        return [0] * 7
    
    # Filter by opponent team ID
    opp_data = data[data['OPP_TEAM_ID'] == oppTeamID]
    
    if opp_data.empty:
        return [0] * 9
    
    # Get unique games to avoid duplicates
    unique_games = opp_data.drop_duplicates(subset=['GAME_ID'])
    
    stats = [
        unique_games['TEAM_DEF_RATING'].mean(),
        unique_games['TEAM_PACE'].mean(),
        unique_games['TEAM_PTS'].mean(),
        unique_games['TEAM_FGA'].mean(),
        unique_games['TEAM_REB'].mean(),
        unique_games['TEAM_AST'].mean(),
        unique_games['TEAM_TOV'].mean(),
        unique_games['TEAM_BLK'].mean(),
        unique_games['TEAM_STL'].mean()
    ]
    return [round(stat, 2) for stat in stats]

def getTeamStats(player, data):
    player_id, player_team = findPlayerID(player, data)
    team_data = data[data['TEAM_ABBREVIATION'] == player_team]
    if team_data.empty:
        return [0] * 8  # Should return 8 zeros to match the 8 stats below
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

def getMatchupStats(player, data, games,n_games=3): #still needs fixing
    player_id, player_team = findPlayerID(player, data)
    opponent_team, homeGame = findOppTeam(player, data, games)
    opp_id = findTeamID(opponent_team, data)
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

def getTeamOdds(player, data, game_id):
    player_id, player_team = findPlayerID(player, data)
    game_data = data[
        (data['GAME_ID'] == game_id) & 
        (data['TEAM_ABBREVIATION'] == player_team)
    ]
    
    if game_data.empty:
        return [0, 0, 0, 0, 0, 0]  
    
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
def buildFeatureVector(player, data, games, todayDate, starters, game_id, n_games=3):
    features = (getPlayerSpecificFeatures(player, data, starters, games, todayDate) +
                getPlayerSeasonAverages(player, data) +
                getPlayerLags(player, data) +
                getPlayerRollingAVG(player, data) +
                getOppStats(player, data, games) +
                getTeamStats(player, data) +
                getMatchupStats(player, data, games, n_games=n_games) +
                getTeamOdds(player, data, game_id))
    return features

def makePredictionCatBoost(player_name, data, model, bookmakers, games, todayDate, starters, game_id, features, n_games=3):
    from catboost import Pool
    import pandas as pd
    
    # Get feature vector
    feature_vector = buildFeatureVector(player_name, data, games, todayDate, starters, game_id, n_games=n_games)
    
    # Convert to DataFrame with proper feature names
    X = pd.DataFrame([feature_vector], columns=features)
    
    # Define categorical features (same as used during training)
    categorical_cols = ['PLAYER_ID', 'TEAM_ID', 'OPP_TEAM_ID']
    cat_cols = [c for c in categorical_cols if c in features]
    cat_idx = [features.index(c) for c in cat_cols]
    
    # Data type cleanup (matching training preprocessing)
    for c in X.columns:
        if c not in cat_cols:
            if X[c].dtype == 'bool':
                X[c] = X[c].astype(int)
            elif X[c].dtype == 'object':
                X[c] = pd.to_numeric(X[c], errors='coerce')
    
    # Create CatBoost Pool with categorical features
    pool = Pool(X, cat_features=cat_idx)
    
    # Make prediction
    prediction = model.predict(pool)[0]
    
    prop_line = bookmakers[bookmakers['player'] == player_name]['line'].values[0]
    
    # Get opponent team for display
    player_id, player_team = findPlayerID(player_name, data)
    opponent, homeGame = findOppTeam(player_name, data, games)
    
    return {
        'player': player_name,
        'opponent': opponent,
        'predicted_stat': round(prediction, 2),
        'raw_prediction': prediction,
        'prop_line': prop_line,
        'edge': round(prediction - prop_line, 2),
        'recommendation': 'OVER' if prediction > prop_line else 'UNDER'
    }