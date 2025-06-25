import pandas as pd
import numpy as np

#grabs players rest days between games
def calculate_days_of_rest(df, player_id_col='PLAYER_ID', game_date_col='GAME_DATE'):
    df[game_date_col] = pd.to_datetime(df[game_date_col], format='%Y-%m-%d')
    df = df.sort_values(by=[player_id_col, game_date_col])
    df['DAYS_OF_REST'] = df.groupby(player_id_col)[game_date_col].diff().dt.days
    return df
    
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
def add_matchup_points(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION'):
    player_data = player_data.sort_values([player_id_col, 'GAME_DATE'])
    
    # Calculate recent average points (last 3 games)
    player_data['MATCHUP_AVG_PTS_LAST_3'] = (
        player_data.groupby([player_id_col, opp_col])['PTS']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().round(2))
    )
    
    # Count number of games against this team
    player_data['GAMES_VS_OPP'] = player_data.groupby([player_id_col, opp_col]).cumcount() + 1
    
    return player_data

########################################################################################
#rolling averages for players points
########################################################################################

def calculate_rolling_averages(player_data, rolling_windows, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate rolling averages and standard deviations for key features per player over multiple window sizes,
    using only *past* games to avoid target leakage.
    """
    player_data = player_data.sort_values([player_id_col, date_col])
    rolling_features = ['PTS', 'MIN', 'FG_PCT', 'FGM', 'FGA', 'FG3M', 'FG3A',
                        'FG3_PCT', 'FTM', 'FT_PCT', 'REB', 'AST', 'USG_PCT']
    for window in rolling_windows:
        for feature in rolling_features:
            roll_avg_col = f'{feature}_ROLL_AVG_{window}'
            player_data[roll_avg_col] = (
                player_data
                .groupby(player_id_col)[feature]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2))
            )

            if feature == 'PTS':
                std_col = f'{feature}_STD_AVG_{window}'
                player_data[std_col] = (
                    player_data
                    .groupby(player_id_col)[feature]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().round(2))
                )
    return player_data

def home_away_averages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for home_away in ['HOME', 'AWAY']:
        avg_column_name = f'PLAYER_{home_away}_AVG_PTS'
        is_home = player_data['HOME_GAME'] == (1 if home_away == 'HOME' else 0)
        player_data[avg_column_name] = (
        player_data.groupby(player_id_col)
        .apply(lambda group: group.loc[is_home, 'PTS'].expanding().mean().round(2), include_groups=False)
        .reset_index(level=0, drop=True)
    )
    return player_data

def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for lag in range(1,5):
        player_data[f'PTS_LAG_{lag}'] = player_data.groupby(player_id_col)['PTS'].shift(lag)
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

def calculate_player_vs_defense(player_data, player_id_col='PLAYER_ID'):
    """
    Calculate how players perform against different defensive categories
    """
    # Calculate average performance against each defensive category
    metrics = ['PTS','FGA', 'FTA', 'FG3A', 'USG_PCT']
    
    for metric in metrics:
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

def add_all_opponent_features(player_data):
    """
    Wrapper function to add all opponent-related features
    """
    player_data = add_opponent_metrics(player_data)
    player_data = categorize_opponent_defense(player_data)
    player_data = calculate_player_vs_defense(player_data)
    return player_data

########################################################################################
#rolling averages for assists
########################################################################################    

def calculate_rolling_averages_ast(player_data, rolling_windows, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate rolling averages and standard deviations for key features per player over multiple window sizes,
    using only *past* games to avoid target leakage.
    """
    player_data = player_data.sort_values([player_id_col, date_col])
    rolling_features = ['MIN', 'AST', 'TOV', 'AST_TOV', 'USG_PCT', 'AST_PCT', 'PACE', 'POSS', 'OFF_RATING']
    for window in rolling_windows:
        for feature in rolling_features:
            roll_avg_col = f'{feature}_ROLL_AVG_{window}'
            player_data[roll_avg_col] = (
                player_data
                .groupby(player_id_col)[feature]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2))
            )

            if feature == 'AST':
                std_col = f'{feature}_STD_AVG_{window}'
                player_data[std_col] = (
                    player_data
                    .groupby(player_id_col)[feature]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().round(2))
                )
    return player_data

def addLagFeatures_Ast(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for lag in range(1,5):
        player_data[f'AST_LAG_{lag}'] = player_data.groupby(player_id_col)['AST'].shift(lag)
    return player_data

def home_away_ast_averages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for home_away in ['HOME', 'AWAY']:
        avg_column_name = f'PLAYER_{home_away}_AVG_AST'
        is_home = player_data['HOME_GAME'] == (1 if home_away == 'HOME' else 0)
        player_data[avg_column_name] = (
        player_data.groupby(player_id_col)
        .apply(lambda group: group.loc[is_home, 'AST'].expanding().mean().round(2), include_groups=False)
        .reset_index(level=0, drop=True)
    )
    return player_data

def calculate_player_ast_vs_defense(player_data, player_id_col='PLAYER_ID'):
    """
    Calculate how players perform against different defensive categories
    """
    # Calculate average performance against each defensive category
    metrics = ['AST','AST_PCT', 'AST_TOV', 'USG_PCT', 'PACE', 'POSS', 'OFF_RATING']
    
    for metric in metrics:
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

def add_all_opponent_features_ast(player_data):
    """
    Wrapper function to add all opponent-related features
    """
    player_data = add_opponent_metrics(player_data)
    player_data = categorize_opponent_defense(player_data)
    player_data = calculate_player_ast_vs_defense(player_data)
    return player_data

########################################################################################
#rolling averages for rebounds
########################################################################################   

def calculate_rolling_averages_reb(player_data, rolling_windows, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate rolling averages and standard deviations for key features per player over multiple window sizes,
    using only *past* games to avoid target leakage.
    """
    player_data = player_data.sort_values([player_id_col, date_col])
    rolling_features = ['MIN', 'REB', 'OREB', 'DREB', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'USG_PCT', 'GAME_PACE']
    for window in rolling_windows:
        for feature in rolling_features:
            roll_avg_col = f'{feature}_ROLL_REB_{window}'
            player_data[roll_avg_col] = (
                player_data
                .groupby(player_id_col)[feature]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2))
            )

            if feature == 'REB':
                std_col = f'{feature}_STD_AVG_{window}'
                player_data[std_col] = (
                    player_data
                    .groupby(player_id_col)[feature]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().round(2))
                )
    return player_data

def addLagFeatures_Reb(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for lag in range(1,5):
        player_data[f'REB_LAG_{lag}'] = player_data.groupby(player_id_col)['REB'].shift(lag)
    return player_data

def home_away_reb_averages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for home_away in ['HOME', 'AWAY']:
        avg_column_name = f'PLAYER_{home_away}_AVG_REB'
        is_home = player_data['HOME_GAME'] == (1 if home_away == 'HOME' else 0)
        player_data[avg_column_name] = (
        player_data.groupby(player_id_col)
        .apply(lambda group: group.loc[is_home, 'REB'].expanding().mean().round(2), include_groups=False)
        .reset_index(level=0, drop=True)
    )
    return player_data

def calculate_player_reb_vs_defense(player_data, player_id_col='PLAYER_ID'):
    """
    Calculate how players perform against different defensive categories
    """
    # Calculate average performance against each defensive category
    metrics = ['REB','OREB_PCT', 'DREB_PCT', 'REB_PCT', 'USG_PCT', 'GAME_PACE']
    
    for metric in metrics:
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

def add_all_opponent_features_reb(player_data):
    """
    Wrapper function to add all opponent-related features
    """
    player_data = add_opponent_metrics(player_data)
    player_data = categorize_opponent_defense(player_data)
    player_data = calculate_player_reb_vs_defense(player_data)
    return player_data