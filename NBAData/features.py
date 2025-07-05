import pandas as pd
import numpy as np


#grabs players rest days between games
def add_rest_day_features(df):
    '''
    Add rest day features for both teams and individual players.
    '''
    df = df.copy()
    
    # Convert GAME_DATE to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['GAME_DATE']):
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Sort by date
    df = df.sort_values(['TEAM_ID', 'GAME_DATE', 'PLAYER_ID'])
    
    # Calculate team rest days
    df['TEAM_DAYS_REST'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    df['TEAM_DAYS_REST'] = df['TEAM_DAYS_REST'].fillna(3)  # First game of season
    df['TEAM_B2B'] = (df['TEAM_DAYS_REST'] <= 1).astype(int)
    
    # Calculate player rest days
    df['PLAYER_DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
    df['PLAYER_DAYS_REST'] = df['PLAYER_DAYS_REST'].fillna(3)  # First game of season
    df['PLAYER_B2B'] = (df['PLAYER_DAYS_REST'] <= 1).astype(int)
    
    # Calculate if player missed team's last game
    # First, get the previous game date for each team
    df['PREV_TEAM_GAME'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    
    # Then, get the previous game date for each player
    df['PREV_PLAYER_GAME'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    
    # Player missed last game if:
    # 1. It's not the team's first game (PREV_TEAM_GAME is not null)
    # 2. Either:
    #    a. Player has no previous game (PREV_PLAYER_GAME is null), or
    #    b. Player's last game was before team's last game
    df['PLAYER_MISSED_LAST'] = ((~df['PREV_TEAM_GAME'].isna()) & 
                               (df['PREV_PLAYER_GAME'].isna() | 
                                (df['PREV_PLAYER_GAME'] != df['PREV_TEAM_GAME']))).astype(int)
    
    # Drop helper columns
    df = df.drop(['PREV_TEAM_GAME', 'PREV_PLAYER_GAME'], axis=1)
    
    # Ensure all numeric columns are properly typed
    numeric_columns = ['TEAM_DAYS_REST', 'TEAM_B2B', 'PLAYER_DAYS_REST', 'PLAYER_B2B', 'PLAYER_MISSED_LAST']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
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
    df['SERIES'] = df.apply(get_series_number, axis=1)
    df['GAME_IN_SERIES'] = df.groupby(['MATCHUP_KEY'])['GAME_DATE'].rank(method='dense').astype(int)
    
    # Add series name for better context
    def get_series_name(series_num):
        series_names = {
            1: 'First Round',
            2: 'Conference Semifinals',
            3: 'Conference Finals',
            4: 'NBA Finals'
        }
        return series_names.get(series_num, 'Unknown')
    
    df['SERIES_NAME'] = df['SERIES'].map(get_series_name)
    
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
            'MIN', 'FGA', 'FGM', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'EFG_PCT', 
            'TEAM_PACE', 'TEAM_OFF_RATING', 'OPP_DEF_RATING'
        ],
        'AST': [
            'MIN', 'USG_PCT', 'AST_PCT', 'AST_TOV', 'TEAM_FGM', 'TEAM_AST', 'TEAM_PACE', 'OPP_DEF_RATING', 'OPP_STL', 'TEAM_FGA',
            'TEAM_FG3A'
        ],
        'REB': [
            'MIN', 'FGA', 'FGM', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'USG_PCT', 'POSS', 
            'TEAM_FGA', 'TEAM_FG3A', 'TEAM_FG_PCT', 'TEAM_FG3_PCT', 'OPP_REB', 'OPP_FG_PCT', 'OPP_PACE',
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

            # Add rolling std for stat_line only
            if feature == stat_line:
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

def dynamic_defense_ranking(df, game_date_col='GAME_DATE'):
    """Rank defenses based only on games played before each game date"""
    df_ranked = []
    
    for date in df[game_date_col].unique():
        # Only use games before this date for ranking
        historical_data = df[df[game_date_col] < date]
        
        if len(historical_data) > 0:
            # Calculate defensive strength up to this date
            team_strength = historical_data.groupby('OPP_ABBREVIATION')['OPP_DEF_RATING'].mean()
            rankings = team_strength.rank(ascending=True, method='min')
            
            # Apply rankings to games on this date
            games_today = df[df[game_date_col] == date].copy()
            games_today['DEF_CATEGORY'] = games_today['OPP_ABBREVIATION'].map(
                lambda x: 1 if rankings.get(x, 999) <= 10 else 0
            )
            df_ranked.append(games_today)
    
    return pd.concat(df_ranked, ignore_index=True)

# If you want absolute averages instead of relative performance
def calculate_absolute_vs_defense(player_data, player_id_col='PLAYER_ID', stat_line='PTS'):
    """
    Calculate absolute averages against strong vs weak defenses for different stat categories
    """
    
    # Define metrics for each stat line
    metrics = {
        'PTS': ['PTS', 'FGA', 'FTA', 'FG3A', 'USG_PCT', 'TOV'],
        'AST': ['AST', 'AST_PCT', 'AST_TOV', 'USG_PCT', 'PACE', 'POSS', 'OFF_RATING', 'TOV'],
        'REB': ['REB', 'OREB', 'DREB', 'REB_PCT', 'USG_PCT', 'GAME_PACE', 'TOV']
    }
    
    # Validate stat_line parameter
    if stat_line not in metrics:
        raise ValueError(f"stat_line must be one of: {list(metrics.keys())}")
    
    # Filter data by defense category
    strong_def = player_data[player_data['DEF_CATEGORY'] == 1]
    weak_def = player_data[player_data['DEF_CATEGORY'] == 0]
    
    # Calculate absolute averages for each metric
    for metric in metrics[stat_line]:
        # Check if metric exists in the dataframe
        if metric not in player_data.columns:
            print(f"Warning: Column '{metric}' not found in dataframe, skipping...")
            continue
            
        # Calculate averages vs strong defenses
        strong_avg = strong_def.groupby(player_id_col)[metric].mean()
        player_data[f'{metric}_VS_DEF_STRONG'] = (
            player_data[player_id_col].map(strong_avg).fillna(0).round(2)
        )
        
        # Calculate averages vs weak defenses  
        weak_avg = weak_def.groupby(player_id_col)[metric].mean()
        player_data[f'{metric}_VS_DEF_WEAK'] = (
            player_data[player_id_col].map(weak_avg).fillna(0).round(2)
        )
        
        # Optional: Calculate difference (performance gap)
        player_data[f'{metric}_VS_DEF_DIFF'] = (
            player_data[f'{metric}_VS_DEF_WEAK'] - player_data[f'{metric}_VS_DEF_STRONG']
        ).round(2)
    
    return player_data

def add_all_opponent_features(player_data, stat_line='PTS'):
    """
    Wrapper function to add all opponent-related features
    """
    player_data = dynamic_defense_ranking(player_data)
    player_data = calculate_absolute_vs_defense(player_data, stat_line=stat_line)
    return player_data


########################################################################################
#lineup composition features
########################################################################################
def teamUsualStarters(df):
    """
    Adds to df:
    - TEAM_STARTER_OFF_RATING_AVG, TEAM_STARTER_DEF_RATING_AVG, TEAM_STARTER_USG_PCT_AVG
    - NUM_USUAL_STARTERS_PRESENT: number of usual starters present for own team
    """
    # 1) Compute usual starters: top 5 most frequent starters per team
    player_starts = (
        df[df['STARTING'] == 1]
        .groupby(['TEAM_ID', 'PLAYER_ID'])
        .size()
        .reset_index(name='NUM_STARTS')
    )
    usual_starters = (
        player_starts
        .sort_values(['TEAM_ID', 'NUM_STARTS'], ascending=[True, False])
        .groupby('TEAM_ID')
        .head(5)
    )
    usual_starters_dict = (
        usual_starters
        .groupby('TEAM_ID')['PLAYER_ID']
        .apply(set)
        .to_dict()
    )

    # 2) Compute NUM_USUAL_STARTERS_PRESENT for each game-team
    starters_per_game = (
        df[df['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PLAYER_ID': list})
        .reset_index()
    )
    def count_usual_starters(row):
        team_id = row['TEAM_ID']
        actual_starters = set(row['PLAYER_ID'])
        usual_starters = usual_starters_dict.get(team_id, set())
        return len(actual_starters & usual_starters)
    
    starters_per_game['NUM_USUAL_STARTERS_PRESENT'] = starters_per_game.apply(count_usual_starters, axis=1)
    
    # 3) Compute team starter averages (OFF/DEF/USG)
    starters_df = df[df['STARTING'] == 1].copy()
    team_starter_features = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({
            'OFF_RATING': 'mean',
            'DEF_RATING': 'mean',
            'USG_PCT': 'mean'
        })
        .rename(columns={
            'OFF_RATING': 'TEAM_STARTER_OFF_RATING_AVG',
            'DEF_RATING': 'TEAM_STARTER_DEF_RATING_AVG',
            'USG_PCT': 'TEAM_STARTER_USG_PCT_AVG'
        })
        .reset_index()
    )

    # 4) Merge NUM_USUAL_STARTERS_PRESENT into team starter features
    team_starter_features = team_starter_features.merge(
        starters_per_game[['GAME_ID', 'TEAM_ID', 'NUM_USUAL_STARTERS_PRESENT']],
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    # 5) Merge combined features into main df
    df = df.merge(
        team_starter_features,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    return df

def oppTeamUsualStarters(df):
    """
    Adds opponent-side starter features:
    - NUM_USUAL_STARTERS_PRESENT_OPP: count of opponent usual starters present
    - OPP_STARTER_AVG_DEF_RATING: average DEF_RATING of opponent starters
    - OPP_GUARDS_AVG_DEF_RATING_OPP: average DEF_RATING of opponent starting guards
    - OPP_FORWARDS_AVG_DEF_RATING_OPP: average DEF_RATING of opponent starting forwards
    """

    # 1) Compute usual starters: top 5 most frequent starters per team
    player_starts = (
        df[df['STARTING'] == 1]
        .groupby(['TEAM_ID', 'PLAYER_ID'])
        .size()
        .reset_index(name='NUM_STARTS')
    )
    usual_starters = (
        player_starts
        .sort_values(['TEAM_ID', 'NUM_STARTS'], ascending=[True, False])
        .groupby('TEAM_ID')
        .head(5)
    )
    usual_starters_dict = (
        usual_starters
        .groupby('TEAM_ID')['PLAYER_ID']
        .apply(set)
        .to_dict()
    )

    # 2) Compute NUM_USUAL_STARTERS_PRESENT for opponent team
    opp_actual_starters_per_game = (
        df[df['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PLAYER_ID': list})
        .reset_index()
    )
    def count_opp_usual_starters(row):
        team_id = row['TEAM_ID']
        actual_starters = set(row['PLAYER_ID'])
        usual_starters = usual_starters_dict.get(team_id, set())
        return len(actual_starters & usual_starters)
    
    opp_actual_starters_per_game['NUM_USUAL_STARTERS_PRESENT'] = opp_actual_starters_per_game.apply(count_opp_usual_starters, axis=1)
    df = df.merge(
        opp_actual_starters_per_game[['GAME_ID', 'TEAM_ID', 'NUM_USUAL_STARTERS_PRESENT']],
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    # 3) Compute overall opponent starter average DEF_RATING
    starters_df = df[df['STARTING'] == 1].copy()
    opp_starter_def_rating = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'DEF_RATING': 'mean'})
        .rename(columns={'DEF_RATING': 'OPP_STARTER_AVG_DEF_RATING'})
        .reset_index()
    )
    df = df.merge(
        opp_starter_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    # 4) Compute average DEF_RATING of opponent starting guards and forwards directly using existing categories
    guards_df = starters_df[starters_df['GUARD'] == 1].copy()
    opp_guards_def_rating = (
        guards_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'DEF_RATING': 'mean'})
        .rename(columns={'DEF_RATING': 'OPP_GUARDS_AVG_DEF_RATING_OPP'})
        .reset_index()
    )
    df = df.merge(
        opp_guards_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    forwards_df = starters_df[starters_df['FORWARD'] == 1].copy()
    opp_forwards_def_rating = (
        forwards_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'DEF_RATING': 'mean'})
        .rename(columns={'DEF_RATING': 'OPP_FORWARDS_AVG_DEF_RATING_OPP'})
        .reset_index()
    )
    df = df.merge(
        opp_forwards_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    centers_df = starters_df[starters_df['CENTER'] == 1].copy()
    opp_centers_def_rating = (
    centers_df
    .groupby(['GAME_ID', 'TEAM_ID'])
    .agg({'DEF_RATING': 'mean'})
    .rename(columns={'DEF_RATING': 'OPP_CENTERS_AVG_DEF_RATING_OPP'})
    .reset_index()
    )
    df = df.merge(
        opp_centers_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )
    return df

def team_starter_spacing(df):
    starters_df = df[df['STARTING'] == 1].copy()
    
    team_spacing = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'FG3_PCT': 'mean'})
        .rename(columns={'FG3_PCT': 'TEAM_STARTER_SPACING_METRIC'})
        .reset_index()
    )
    
    df = df.merge(
        team_spacing,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    return df

def pace_expectation(df):
    starters_df = df[df['STARTING'] == 1].copy()
    
    # Team starter average pace
    team_pace = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PACE': 'mean'})
        .rename(columns={'PACE': 'TEAM_STARTER_PACE'})
        .reset_index()
    )
    df = df.merge(
        team_pace,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    # Opponent starter average pace
    opp_pace = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PACE': 'mean'})
        .rename(columns={'PACE': 'OPP_STARTER_PACE'})
        .reset_index()
    )
    df = df.merge(
        opp_pace,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )
    
    # Calculate expected pace as average of team + opponent starters
    df['PACE_EXPECTATION'] = (df['TEAM_STARTER_PACE'] + df['OPP_STARTER_PACE']) / 2
    
    return df

def process_star_players_data(season_df, star_players):
    """
    Process star players data for a single season and return simple star metrics.
    """
    # Add IS_STAR column
    season_df['IS_STAR'] = season_df['PLAYER_NAME'].isin(star_players).astype(int)
    
    # Get starters per game
    starters_per_game = (
        season_df[season_df['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PLAYER_NAME': list})
        .reset_index()
    )
    
    # Count stars for each team-game
    starters_per_game['NUM_STARS_ON_TEAM'] = starters_per_game['PLAYER_NAME'].apply(
        lambda players: sum(1 for player in players if player in star_players)
    )
    
    # Merge the star count back to main DataFrame
    merge_cols = ['GAME_ID', 'TEAM_ID', 'NUM_STARS_ON_TEAM']
    merged_df = season_df.merge(starters_per_game[merge_cols], on=['GAME_ID', 'TEAM_ID'], how='left')
    
    # Fill any NaN values with 0
    merged_df['NUM_STARS_ON_TEAM'] = merged_df['NUM_STARS_ON_TEAM'].fillna(0)
    
    return merged_df

def allLineupFeatures(df, star_players):
    df = teamUsualStarters(df)
    df = oppTeamUsualStarters(df)
    df = team_starter_spacing(df)
    df = pace_expectation(df)
    df = process_star_players_data(df, star_players)
    return df

def encode_teams(df):
    # One-hot encode player team and opponent team
    df_teams = pd.get_dummies(df['TEAM_ABBREVIATION'], prefix='TEAM_').astype(int)
    df_opps = pd.get_dummies(df['OPP_ABBREVIATION'], prefix='OPP_').astype(int)
    df_encoded = pd.concat([df, df_teams, df_opps], axis=1)
    return df_encoded

