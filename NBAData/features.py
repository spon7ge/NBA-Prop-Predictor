import pandas as pd
import numpy as np

#grabbing play by play data
import re
from nba_api.stats.endpoints import PlayByPlayV2
def PlayByPlayOrangized(game_id):
    df = PlayByPlayV2(game_id=game_id).get_data_frames()[0]
    df['DESCRIPTION'] = df['HOMEDESCRIPTION'].fillna(df['VISITORDESCRIPTION'])
    df['DESCRIPTION'] = df['DESCRIPTION'].fillna(df['NEUTRALDESCRIPTION'])
    
    scores = df[['GAME_ID','PERIOD','PCTIMESTRING','DESCRIPTION','SCORE', 'PLAYER1_NAME', 'PLAYER2_NAME', 'PLAYER3_NAME','PLAYER1_TEAM_ABBREVIATION','PLAYER2_TEAM_ABBREVIATION','PLAYER3_TEAM_ABBREVIATION','PLAYER1_ID','PLAYER2_ID','PLAYER3_ID']].reset_index(drop=True)
    scores['SECONDS_REMAINING'] = scores['PCTIMESTRING'].apply(lambda x: 
    int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    scores['HOME_SCORE'] = scores['SCORE'].str.split('-').str[0].astype(float)
    scores['AWAY_SCORE'] = scores['SCORE'].str.split('-').str[1].astype(float)

    scores['HOME_SCORE'] = scores['HOME_SCORE'].ffill().fillna(0).astype(int)
    scores['AWAY_SCORE'] = scores['AWAY_SCORE'].ffill().fillna(0).astype(int)

    scores = scores.drop(columns=['SCORE'])
    scores = scores[['GAME_ID', 'PERIOD', 'PCTIMESTRING', 'SECONDS_REMAINING', 'DESCRIPTION', 'HOME_SCORE', 'AWAY_SCORE', 'PLAYER1_NAME', 'PLAYER2_NAME', 'PLAYER3_NAME','PLAYER1_TEAM_ABBREVIATION','PLAYER2_TEAM_ABBREVIATION','PLAYER3_TEAM_ABBREVIATION']]
    return scores

#grabbing action types from description
def parseDescription(pbp_df):
    # Initialize columns - only PLAYER1 gets shot-related columns
    pbp_df['PLAYER1_NAME'] = None
    pbp_df['PLAYER1_ACTION'] = None
    pbp_df['PLAYER1_SHOT_TYPE'] = None
    pbp_df['PLAYER1_SHOT_OUTCOME'] = None
    pbp_df['PLAYER1_DISTANCE'] = None
    
    # Other players just need name and action
    pbp_df['PLAYER2_NAME'] = None
    pbp_df['PLAYER2_ACTION'] = None
    
    pbp_df['PLAYER3_NAME'] = None
    pbp_df['PLAYER3_ACTION'] = None
    
    for idx, description in enumerate(pbp_df['DESCRIPTION']):
        if pd.isna(description):
            continue
            
        # Primary action (shots)
        shot_match = re.search(r"(\w+)\s+(\d+)'\s+([\w\s]+)(?=\s+\()", str(description))
        if shot_match:
            pbp_df.loc[idx, 'PLAYER1_NAME'] = shot_match.group(1)  # Player name
            pbp_df.loc[idx, 'PLAYER1_DISTANCE'] = int(shot_match.group(2))  # Shot distance
            pbp_df.loc[idx, 'PLAYER1_SHOT_TYPE'] = shot_match.group(3)  # Shot type
            pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'SHOT'
            pbp_df.loc[idx, 'PLAYER1_SHOT_OUTCOME'] = 1 if re.search(r"\(\d+ PTS\)", description) else 0
            
            # Check for assist
            assist_match = re.search(r"\((\w+)\s+(\d+)\s+AST\)", str(description))
            if assist_match:
                pbp_df.loc[idx, 'PLAYER2_NAME'] = assist_match.group(1)
                pbp_df.loc[idx, 'PLAYER2_ACTION'] = 'ASSIST'
    
        
        # Handle non-shot actions
        elif 'REBOUND' in str(description):
            rebound_match = re.search(r"(\w+)\s+REBOUND", str(description))
            if rebound_match:
                pbp_df.loc[idx, 'PLAYER1_NAME'] = rebound_match.group(1)
                pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'REBOUND'
                
        elif 'STEAL' in str(description):
            steal_match = re.search(r"(\w+)\s+STEAL", str(description))
            if steal_match:
                pbp_df.loc[idx, 'PLAYER1_NAME'] = steal_match.group(1)
                pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'STEAL'
                
        elif 'BLOCK' in str(description):
            block_match = re.search(r"(\w+)\s+BLOCK", str(description))
            if block_match:
                pbp_df.loc[idx, 'PLAYER1_NAME'] = block_match.group(1)
                pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'BLOCK'
            
    return pbp_df

#grabs players rest days between games
def calculateDaysOfRest(df, player_id_col='PLAYER_ID', game_date_col='GAME_DATE'):
    df[game_date_col] = pd.to_datetime(df[game_date_col], format='%Y-%m-%d')
    df = df.sort_values(by=[player_id_col, game_date_col])
    df['DAYS_OF_REST'] = df.groupby(player_id_col)[game_date_col].diff().dt.days
    return df

def convert_height_to_inches(height_str):
    if pd.isna(height_str):
        return np.nan
    # Split the string into feet and inches
    feet, inches = map(int, height_str.split('-'))
    # Convert to total inches
    return (feet * 12) + inches
    
# only for the playoffs
def assign_playoff_series_info(df):
    df = df.copy()
    # Ensure GAME_DATE is in datetime format
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Create a consistent matchup key regardless of home/away
    df['MATCHUP_KEY'] = df.apply(
        lambda x: '-'.join(sorted([x['TEAM_ABBREVIATION'], x['OPP_ABBREVIATION']])), axis=1
    )

    # Get unique games to avoid player duplicates
    unique_games = df.drop_duplicates(subset=['GAME_ID'])
    
    # Sort by matchup and date
    unique_games = unique_games.sort_values(by=['GAME_DATE', 'MATCHUP_KEY'])
    
    # Initialize team tracking
    team_series = {}
    team_opponents = {}
    
    # First pass: Track all opponents for each team
    for _, game in unique_games.iterrows():
        team1, team2 = game['TEAM_ABBREVIATION'], game['OPP_ABBREVIATION']
        date = game['GAME_DATE']
        
        # Initialize if not exists
        if team1 not in team_opponents:
            team_opponents[team1] = []
        if team2 not in team_opponents:
            team_opponents[team2] = []
            
        # Add opponent to list if it's a new matchup
        if team2 not in team_opponents[team1]:
            team_opponents[team1].append(team2)
        if team1 not in team_opponents[team2]:
            team_opponents[team2].append(team1)
    
    # Determine series number for each team
    for team in team_opponents:
        team_series[team] = len(team_opponents[team])
    
    # Apply series numbers to the full dataset
    def get_series_number(row):
        team = row['TEAM_ABBREVIATION']
        opp = row['OPP_ABBREVIATION']
        
        # Get the opponent's index in the team's opponent list (1-based)
        try:
            series_num = team_opponents[team].index(opp) + 1
        except ValueError:
            series_num = 0
            
        return series_num
    
    # Apply series number and game number within series
    df['Series'] = df.apply(get_series_number, axis=1)
    df['GameInSeries'] = df.groupby(['MATCHUP_KEY'])['GAME_DATE'].rank(method='dense').astype(int)
    
    # Add series name for better context
    def get_series_name(series_num):
        series_names = {
            1: 'First Round',
            2: 'Conference Semifinals',
            3: 'Conference Finals',
            4: 'NBA Finals'
        }
        return series_names.get(series_num, 'Unknown')
    
    df['SeriesName'] = df['Series'].map(get_series_name)
    
    return df

#rolling averages for points against each team
def statAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, 'GAME_DATE'])
    
    # Calculate recent average points (last 3 games)
    player_data[f'MATCHUP_AVG_{stat_line}_LAST_3'] = (
        player_data.groupby([player_id_col, opp_col])[stat_line]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().round(2))
    )
    
    # Count number of games against this team
    player_data['GAMES_VS_OPP'] = player_data.groupby([player_id_col, opp_col]).cumcount() + 1
    
    return player_data

########################################################################################
#rolling averages
########################################################################################

def rollingAverages(player_data, rolling_windows, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, date_col]).copy()
    rolling_features = {
        'PTS': [
            'MIN', 'PTS', 'FGA', 'FGM', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'EFG_PCT',
            'OREB', 'DREB', 'REB', 'PLUS_MINUS', 'PIE',
            'TEAM_FGA', 'TEAM_FG_PCT', 'TEAM_FG3A', 'TEAM_FG3_PCT',
            'TEAM_FTM', 'TEAM_FTA', 'TEAM_FT_PCT',
            'TEAM_PTS', 'TEAM_PACE', 'TEAM_OFF_RATING',
            'OPP_DEF_RATING', 'OPP_PACE', 'OPP_FG_PCT'
        ],
        'AST': [
            'MIN', 'AST', 'FGA', 'FGM', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'USG_PCT', 'AST_PCT', 'AST_TOV', 'TS_PCT', 'EFG_PCT', 'PIE', 'PLUS_MINUS',
            'TEAM_FG_PCT', 'TEAM_FGM', 'TEAM_AST', 'TEAM_TOV', 'TEAM_PACE', 'TEAM_PTS',
            'OPP_DEF_RATING', 'OPP_STL', 'OPP_PACE'
        ],
        'REB': [
            'MIN', 'OREB', 'DREB', 'REB', 'FGA', 'FGM', 'FG_PCT',
            'FG3A', 'FG3M', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'PIE', 'PLUS_MINUS',
            'USG_PCT', 'TS_PCT', 'EFG_PCT', 'PACE', 'POSS',
            'TEAM_FG_PCT', 'TEAM_FG3_PCT', 'TEAM_FGA', 'TEAM_FG3A',
            'OPP_REB', 'OPP_FG_PCT', 'OPP_DEF_RATING', 'OPP_PACE'
        ]
    }

    if stat_line not in rolling_features:
        raise ValueError(f"Invalid stat_line: {stat_line}. Must be one of {list(rolling_features.keys())}")

    for window in rolling_windows:
        for feature in rolling_features[stat_line]:
            roll_avg_col = f'{feature}_ROLL_AVG_{window}'

            # Compute rolling average (no leakage)
            player_data[roll_avg_col] = (
                player_data
                .groupby(player_id_col)[feature]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2))
            )

            # Fill missing values: bfill → ffill → global mean
            global_mean = player_data[roll_avg_col].mean()
            player_data[roll_avg_col] = (
                player_data[roll_avg_col]
                .bfill()
                .ffill()
                .fillna(global_mean)
            )

            # Add rolling std for PTS only
            if stat_line == 'PTS' and feature == 'PTS':
                std_col = f'{feature}_STD_AVG_{window}'
                player_data[std_col] = (
                    player_data
                    .groupby(player_id_col)[feature]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().round(2))
                )
                global_std = player_data[std_col].mean()
                player_data[std_col] = (
                    player_data[std_col]
                    .bfill()
                    .ffill()
                    .fillna(global_std)
                )

    return player_data

def HomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, date_col]).copy()
    
    for home_away in ['HOME', 'AWAY']:
        avg_column_name = f'PLAYER_{home_away}_AVG_{stat_line}'

        def compute_expanding_avg(group):
            mask = group['HOME_GAME'] == (1 if home_away == 'HOME' else 0)
            result = pd.Series(index=group.index, dtype='float64')
            result[mask] = group.loc[mask, stat_line].expanding().mean().round(2)
            if home_away == 'AWAY':
                result = result.bfill().ffill()
            elif home_away == 'HOME':
                result = result.bfill().ffill()
            return result

        player_data[avg_column_name] = (
            player_data.groupby(player_id_col)
            .apply(compute_expanding_avg)
            .reset_index(level=0, drop=True)
        )

    return player_data



def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for lag in range(1,5):
        player_data[f'{stat_line}_LAG_{lag}'] = player_data.groupby(player_id_col)[stat_line].shift(lag)
    return player_data

def preprocessGamesData(player_data):   
    player_data['GAME_DATE'] = pd.to_datetime(
        player_data['GAME_DATE'], format='%b %d, %Y')
    today = pd.Timestamp.today().normalize()
    player_data['DAYS_AGO'] = (today - player_data['GAME_DATE']).dt.days
    return player_data

def add_opponent_metrics(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION'):
    """
    Add opponent-related metrics using rolling windows to track opponent defensive strength
    """
    # First sort by date to ensure chronological calculations
    player_data = player_data.sort_values(['GAME_DATE'])
    
    # Calculate opponent rolling averages (last 5 games) - defensive stats only
    opp_metrics = ['OPP_DEF_RATING', 'OPP_STL', 'OPP_BLK']
    window = 5
    
    for metric in opp_metrics:
        # Calculate opponent's rolling average for each metric
        player_data[f'OPP_ROLL_{metric}_{window}'] = (
            player_data.groupby(opp_col)[metric]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean().round(2))
        )
    
    # Normalize defensive rating (higher rating is worse defense, so multiply by -1)
    player_data['NORMALIZED_DEF_RATING'] = -1 * (
        (player_data[f'OPP_ROLL_OPP_DEF_RATING_{window}'] - 
         player_data[f'OPP_ROLL_OPP_DEF_RATING_{window}'].mean()) / 
        player_data[f'OPP_ROLL_OPP_DEF_RATING_{window}'].std()
    )

    # Calculate defensive strength using normalized values
    player_data['OPP_DEF_STRENGTH'] = (
        (player_data[f'OPP_ROLL_OPP_STL_{window}'] / player_data[f'OPP_ROLL_OPP_STL_{window}'].mean()) +
        (player_data[f'OPP_ROLL_OPP_BLK_{window}'] / player_data[f'OPP_ROLL_OPP_BLK_{window}'].mean()) +
        player_data['NORMALIZED_DEF_RATING']
    ) / 3

    return player_data

def categorize_opponent_defense(player_data):
    """
    Categorize opponents into strong (1) and weak (0) defenses
    """
    # Calculate median defensive strength
    median_defense = player_data['OPP_DEF_STRENGTH'].median()
    
    # Categorize defenses
    player_data['DEF_CATEGORY'] = (
        player_data['OPP_DEF_STRENGTH'] >= median_defense).astype(int)
    
    return player_data

def CalculatePlayerVsDefense(player_data, player_id_col='PLAYER_ID', stat_line='PTS'):
    """
    Calculate how players perform against different defensive categories
    """
    # Calculate average performance against each defensive category
    metrics = {
        'PTS': ['PTS','FGA', 'FTA', 'FG3A', 'USG_PCT'],
        'AST': ['AST','AST_PCT', 'AST_TOV', 'USG_PCT', 'PACE', 'POSS', 'OFF_RATING'],
        'REB': ['REB','OREB', 'DREB', 'REB_PCT', 'USG_PCT', 'GAME_PACE']
    }
    
    for metric in metrics[stat_line]:
        # Calculate average against strong and weak defenses
        avg_by_def = (
            player_data.groupby([player_id_col, 'DEF_CATEGORY'])[metric]
            .transform('mean')
            .round(2)
        )
        player_data[f'{metric}_VS_DEF'] = (
            (player_data[metric] - avg_by_def) / avg_by_def
        ).round(3)

    return player_data

def add_all_opponent_features(player_data, stat_line='PTS'):
    """
    Wrapper function to add all opponent-related features
    """
    player_data = add_opponent_metrics(player_data)
    player_data = categorize_opponent_defense(player_data)
    player_data = CalculatePlayerVsDefense(player_data, stat_line=stat_line)
    return player_data