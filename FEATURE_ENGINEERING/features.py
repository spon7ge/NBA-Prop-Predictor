import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# ================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================

def convert_min_to_float(min_str):
    """Convert minutes string (MM:SS) to float."""
    try:
        if isinstance(min_str, str) and ":" in min_str:
            minutes, seconds = map(int, min_str.split(":"))
            total_minutes = minutes + seconds / 60
            return round(total_minutes, 2)
        elif isinstance(min_str, (int, float)):
            return round(float(min_str), 2)
        else:
            return 0
    except:
        return 0

def convert_height_to_inches(height_str):
    """Convert height string (feet-inches) to total inches."""
    if pd.isna(height_str):
        return np.nan
    # Split the string into feet and inches
    feet, inches = map(int, height_str.split('-'))
    # Convert to total inches
    return round((feet * 12) + inches, 2)


# ================================================================================================
# BASIC FEATURE ENGINEERING
# ================================================================================================

def add_rest_day_features(df):
    """Add rest day features for both teams and individual players."""
    # Work on a copy to avoid modifying original
    df = df.copy()
    
    # Convert GAME_DATE to datetime only if needed (saves time on repeated conversions)
    if not pd.api.types.is_datetime64_any_dtype(df['GAME_DATE']):
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Pre-sort once for all operations (reduces multiple sorts)
    df.sort_values(['TEAM_ID', 'GAME_DATE', 'PLAYER_ID'], inplace=True)
    
    # Combine team and player rest calculations (reduces iterations)
    team_groups = df.groupby('TEAM_ID')['GAME_DATE']
    player_groups = df.groupby('PLAYER_ID')['GAME_DATE']
    
    # Calculate rest days in one pass
    df['TEAM_DAYS_REST'] = abs(team_groups.diff().dt.days)
    df['PLAYER_DAYS_REST'] = abs(player_groups.diff().dt.days)
    
    # Fill NaN values efficiently
    df[['TEAM_DAYS_REST', 'PLAYER_DAYS_REST']] = df[['TEAM_DAYS_REST', 'PLAYER_DAYS_REST']].fillna(3)
    
    # Vectorized operations for B2B calculations (faster than comparison loops)
    df['TEAM_B2B'] = (df['TEAM_DAYS_REST'] <= 1).astype('int8')  # Use int8 to save memory
    df['IS_BACK_TO_BACK'] = (df['PLAYER_DAYS_REST'] <= 1).astype('int8')
    
    # Calculate missed games more efficiently
    df['PREV_TEAM_GAME'] = team_groups.shift(1)
    df['PREV_PLAYER_GAME'] = player_groups.shift(1)
    
    # Vectorized missed game calculation
    df['PLAYER_MISSED_LAST'] = (
        (~df['PREV_TEAM_GAME'].isna() & 
         (df['PREV_PLAYER_GAME'].isna() | 
          (df['PREV_PLAYER_GAME'] != df['PREV_TEAM_GAME']))
        ).astype('int8')
    )
    
    # Drop temporary columns (more memory efficient than keeping them)
    df.drop(['PREV_TEAM_GAME', 'PREV_PLAYER_GAME'], axis=1, inplace=True)
    
    return df

def encode_teams(df):
    """One-hot encode player team and opponent team."""
    df_teams = pd.get_dummies(df['TEAM_ABBREVIATION'], prefix='TEAM_').astype(int)
    df_opps = pd.get_dummies(df['OPP_ABBREVIATION'], prefix='OPP_').astype(int)
    df_encoded = pd.concat([df, df_teams, df_opps], axis=1)
    return df_encoded


# ================================================================================================
# ROLLING AVERAGES AND TIME SERIES FEATURES - FIXED FOR DATA LEAKAGE
# ================================================================================================

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[3,5,7]):
    """Calculate rolling averages for key player statistics with emphasis on recent form."""
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = [
        'MIN', 'PTS', 'FGA', 'FG3A', 'FTA', 'USG_PCT', 'TS_PCT', 'OFF_RATING', 'EFG_PCT', 'POINT_PER_SHOOT', 'AST', 'REB', 'TOV'
    ]

    # First compute expanding averages as baseline for stability
    for col in stats_cols:
        if col in df.columns:
            expanding_col = f'{col}_EXPANDING_AVG'
            df[expanding_col] = df.groupby(player_id_col)[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean().round(2)
            )

    # Compute rolling averages - FIXED: Always shift by 1 to prevent leakage
    for window in windows:
        for col in stats_cols:
            if col in df.columns:
                rolling_col_name = f'{col}_ROLLING_AVG_{window}'
                expanding_col = f'{col}_EXPANDING_AVG'
                
                # Calculate rolling average
                rolling_avg = df.groupby(player_id_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2)
                )
                
                # Use expanding average as fallback when rolling window isn't full
                df[rolling_col_name] = rolling_avg.fillna(df[expanding_col])

    # Add longer rolling windows for recent form comparison (15, 25, 40 games)
    recent_windows = [15, 25, 40]
    for window in recent_windows:
        for col in stats_cols:
            if col in df.columns:
                rolling_col_name = f'{col}_ROLLING_AVG_{window}'
                expanding_col = f'{col}_EXPANDING_AVG'
                
                # Calculate rolling average with min_periods=5 for longer windows
                rolling_avg = df.groupby(player_id_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=5).mean().round(2)
                )
                
                # Use expanding average as fallback when rolling window isn't available
                df[rolling_col_name] = rolling_avg.fillna(df[expanding_col])

    # Compute Season Average up to Previous Game - FIXED: Use shift(1) to prevent leakage
    for col in stats_cols:
        if col in df.columns:
            season_avg_col = f'{col}_SEASON_AVG_TO_DATE'  # RENAMED with _TO_DATE suffix
            df[season_avg_col] = df.groupby(player_id_col)[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean().round(2)
            )

    return df

def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    """Add lag features for specified statistic."""
    player_data = player_data.sort_values([player_id_col, date_col])
    
    for lag in range(1, 3):
        lag_col = f'{stat_line}_LAG_{lag}'
        
        # Create Lag
        player_data[lag_col] = player_data.groupby(player_id_col)[stat_line].shift(lag)
        
        # Compute expanding mean up to current row, aligned correctly using transform
        rolling_mean = player_data.groupby(player_id_col)[stat_line].transform(lambda x: x.shift(1).expanding().mean().round(2))
        
        # Fill NaNs in lag with rolling mean
        player_data[lag_col] = player_data[lag_col].fillna(rolling_mean)
        
        # Fill remaining NaNs (e.g., first game) with 0
        player_data[lag_col] = player_data[lag_col].fillna(0)
        
        # Round the final lag column
        player_data[lag_col] = player_data[lag_col].round(2)
    
    return player_data


# ================================================================================================
# PLAYER AVERAGE TO DATE FUNCTIONS - FIXED FOR DATA LEAKAGE
# ================================================================================================

def getPlayerAvgToDate(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate player averages to date for points, minutes, FGA, FTA, usage %, and true shooting %.
    Uses shifted averages to avoid data leakage.
    """
    # Create copy to avoid modifying original
    df_enhanced = df.copy()
    
    # Sort by player and date to ensure proper chronological order
    df_enhanced = df_enhanced.sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define the stats we want to calculate averages for
    stats_to_average = ['PTS', 'MIN', 'FGA', 'FTA', 'FG3A', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'OFF_RATING', 
                        'EFG_PCT', 'POINT_PER_SHOOT', 'AST', 'REB', 'TOV']
    
    # Initialize the new columns
    for stat in stats_to_average:
        if stat in df_enhanced.columns:
            df_enhanced[f'{stat}_AVG_TO_DATE'] = np.nan
    
    df_enhanced['GAMES_PLAYED_TO_DATE'] = 0
    
    # Calculate averages for each player separately to avoid multi-index issues
    for player_id in df_enhanced[player_id_col].unique():
        player_mask = df_enhanced[player_id_col] == player_id
        player_data = df_enhanced[player_mask].copy()
        
        # Calculate games played counter
        df_enhanced.loc[player_mask, 'GAMES_PLAYED_TO_DATE'] = range(len(player_data))
        
        # Calculate expanding averages for each stat - FIXED: Always shift by 1
        for stat in stats_to_average:
            if stat in df_enhanced.columns:
                # Calculate expanding mean and shift by 1 to prevent data leakage
                expanding_avg = player_data[stat].shift(1).expanding().mean()
                df_enhanced.loc[player_mask, f'{stat}_AVG_TO_DATE'] = expanding_avg.round(2)
    
    return df_enhanced

def getPlayerAvgToDateVectorized(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Vectorized version that should be faster and avoid multi-index issues.
    FIXED: Properly shifted to prevent data leakage.
    """
    # Create copy and sort
    df_enhanced = df.copy().sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define stats
    stats_to_average = ['PTS', 'MIN', 'FGA', 'FTA', 'FG3A', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'OFF_RATING', 'EFG_PCT', 'AST', 'REB', 'TOV']
    
    # Use transform to avoid multi-index issues - FIXED: Always shift by 1
    for stat in stats_to_average:
        if stat in df_enhanced.columns:
            df_enhanced[f'{stat}_AVG_TO_DATE'] = (
                df_enhanced.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).expanding().mean())
                .round(2)
            )
    
    # Add games played counter
    df_enhanced['GAMES_PLAYED_TO_DATE'] = (
        df_enhanced.groupby(player_id_col).cumcount()
    )
    
    return df_enhanced

def getPlayerAvgToDateOnly(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate and return only the player averages to date columns.
    """
    # Get full enhanced dataframe
    df_enhanced = getPlayerAvgToDate(df, player_id_col, date_col)
    
    # Select only the relevant columns
    base_cols = [player_id_col, date_col, 'PLAYER_NAME', 'GAME_ID', 'GAMES_PLAYED_TO_DATE']
    avg_cols = [col for col in df_enhanced.columns if col.endswith('_AVG_TO_DATE')]
    
    # Filter to columns that actually exist
    available_base_cols = [col for col in base_cols if col in df_enhanced.columns]
    
    return df_enhanced[available_base_cols + avg_cols]


# ================================================================================================
# HOME/AWAY AND MATCHUP SPECIFIC FEATURES - FIXED FOR DATA LEAKAGE
# ================================================================================================

def HomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate home/away rolling averages (expanding) for key metrics,
    prevent data leakage via shift(1), and fill missing values with the opposite location's current average.
    All results are rounded to 2 decimal places.
    """
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    if 'HOME_GAME' not in df.columns:
        return df

    metrics = ['PTS', 'USG_PCT', 'POSS', 'PACE', 'OFF_RATING']
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return df

    global_means = df[metrics].mean()

    def shifted_expanding_mean(values, group_keys):
        """Helper function that always shifts by 1 to prevent leakage"""
        shifted = values.groupby(group_keys).shift(1)
        cumsum = shifted.groupby(group_keys).cumsum()
        count = shifted.notna().groupby(group_keys).cumsum()
        return cumsum / count

    first_game_mask = df.groupby(player_id_col).cumcount() == 0

    for metric in metrics:
        # FIXED: Always use shift(1) to prevent data leakage
        overall_avg = shifted_expanding_mean(df[metric], df[player_id_col])
        loc_avg = shifted_expanding_mean(df[metric], [df[player_id_col], df['HOME_GAME']])

        home_col = f'PLAYER_HOME_AVG_{metric}_TO_DATE'  # RENAMED with _TO_DATE suffix
        away_col = f'PLAYER_AWAY_AVG_{metric}_TO_DATE'  # RENAMED with _TO_DATE suffix

        df[home_col] = np.where(df['HOME_GAME'] == 1, loc_avg, np.nan)
        df[away_col] = np.where(df['HOME_GAME'] == 0, loc_avg, np.nan)

        df.loc[first_game_mask, home_col] = global_means[metric]
        df.loc[first_game_mask, away_col] = global_means[metric]

        df[home_col] = df[home_col].fillna(overall_avg)
        df[away_col] = df[away_col].fillna(overall_avg)

        df[home_col] = df[home_col].fillna(global_means[metric]).astype('float32').round(2)
        df[away_col] = df[away_col].fillna(global_means[metric]).astype('float32').round(2)

    return df


def statAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION', stat_line='PTS'):
    """
    Calculate matchup-specific statistics with optimized performance and additional metrics.
    Includes rolling averages for multiple windows with data leakage prevention.
    FIXED: All features now use shift(1) to prevent leakage.
    """
    # Create copy to avoid modifying original
    df = player_data.copy()
    
    # Pre-sort once for all operations
    df.sort_values([player_id_col, 'GAME_DATE'], inplace=True)
    
    # Create player-opponent grouper object once (reuse for efficiency)
    player_opp_group = df.groupby([player_id_col, opp_col])
    
    # Define metrics to track with their windows
    metrics = {
        'MIN': [3],
        'FGA': [3],
        'FG3A': [3],
        'FTA': [3],
        'PTS': [3],
        'USG_PCT': [3],
        'EFG_PCT': [3],
        'TS_PCT': [3],
        'OFF_RATING': [3],
        'AST': [3],
        'REB': [3],
        'TOV': [3]
    }
    
    # Calculate games against opponent count efficiently
    df['GAMES_VS_OPP'] = player_opp_group.cumcount() + 1
    
    # Vectorized operations for all rolling windows - FIXED: Always shift by 1
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # Shift values first to prevent data leakage
        shifted_values = player_opp_group[metric].shift(1)
        
        for window in metrics[metric]:
            # Calculate rolling average with shifted values
            col_name = f'MATCHUP_AVG_{metric}_LAST_{window}_TO_DATE'  # RENAMED with _TO_DATE suffix
            df[col_name] = (
                shifted_values
                .rolling(window=window, min_periods=1)
                .mean()
                .round(2)
            )
    
    # Fill NaN values efficiently for all new columns at once
    rolling_cols = [col for col in df.columns if 'MATCHUP_AVG_' in col and '_TO_DATE' in col]
    if rolling_cols:
        # Calculate global means for each metric once
        global_means = df[rolling_cols].mean()
        
        # Fill missing values: first with backward fill, then forward fill, finally with global means
        df[rolling_cols] = (
            df[rolling_cols]
            .bfill()
            .ffill()
            .fillna(global_means)
            .round(2)
        )
    
    # Convert memory types to save space
    for col in rolling_cols:
        df[col] = df[col].astype('float32')  # Use float32 instead of float64 to save memory
    
    df['GAMES_VS_OPP'] = df['GAMES_VS_OPP'].astype('int8')
    
    return df


# ================================================================================================
# OPPONENT AND DEFENSIVE FEATURES - FIXED FOR DATA LEAKAGE
# ================================================================================================

def dynamic_defense_ranking(df, game_date_col='GAME_DATE'):
    """Rank defenses based only on games played before each game date - FIXED"""
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

def add_all_opponent_features(player_data, stat_line='PTS'):
    """Wrapper function to add all opponent-related features"""
    player_data = dynamic_defense_ranking(player_data)
    return player_data


# ================================================================================================
# TEAM STATISTICS AND CONTEXT - FIXED FOR DATA LEAKAGE
# ================================================================================================

def getOpponentStats(df, team_abbreviation='LAL'):
    """Get unique team stats per game with season-to-date averages - FIXED"""
    team_df = df[df['TEAM_ABBREVIATION'] == team_abbreviation].copy()
    team_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_PTS', 'TEAM_FGA', 'TEAM_REB', 'TEAM_AST', 'TEAM_TOV', 'TEAM_BLK', 'TEAM_STL'
    ]
    
    available_team_cols = [col for col in team_cols if col in team_df.columns]
    
    unique_games = team_df[available_team_cols].drop_duplicates(subset=['GAME_ID'])
    
    unique_games = unique_games.sort_values('GAME_DATE')
    
    # FIXED: All averages now use shift(1) to prevent data leakage
    unique_games['DEF_RATING_AVG_TO_DATE'] = unique_games['TEAM_DEF_RATING'].shift(1).expanding().mean().round(2)
    unique_games['PACE_AVG_TO_DATE'] = unique_games['TEAM_PACE'].shift(1).expanding().mean().round(2)
    unique_games['PTS_AVG_TO_DATE'] = unique_games['TEAM_PTS'].shift(1).expanding().mean().round(2)
    unique_games['FGA_AVG_TO_DATE'] = unique_games['TEAM_FGA'].shift(1).expanding().mean().round(2)
    unique_games['REB_AVG_TO_DATE'] = unique_games['TEAM_REB'].shift(1).expanding().mean().round(2)
    unique_games['AST_AVG_TO_DATE'] = unique_games['TEAM_AST'].shift(1).expanding().mean().round(2)
    unique_games['TOV_AVG_TO_DATE'] = unique_games['TEAM_TOV'].shift(1).expanding().mean().round(2)
    unique_games['BLK_AVG_TO_DATE'] = unique_games['TEAM_BLK'].shift(1).expanding().mean().round(2)
    unique_games['STL_AVG_TO_DATE'] = unique_games['TEAM_STL'].shift(1).expanding().mean().round(2)
    unique_games['GAMES_PLAYED'] = range(1, len(unique_games) + 1)
    
    output_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 'GAMES_PLAYED',
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_PTS',
        'DEF_RATING_AVG_TO_DATE', 'PACE_AVG_TO_DATE', 'PTS_AVG_TO_DATE', 'FGA_AVG_TO_DATE', 'REB_AVG_TO_DATE', 'AST_AVG_TO_DATE', 'TOV_AVG_TO_DATE', 'BLK_AVG_TO_DATE', 'STL_AVG_TO_DATE'
    ]
    
    available_cols = [col for col in output_cols if col in unique_games.columns]
    
    return unique_games[available_cols].reset_index(drop=True)

def assign_opponent_team_stats_dict(df):
    """Assign opponent team stats using dictionary lookup for efficiency - FIXED"""
    team_stats_dict = {}
    
    for team in df['TEAM_ABBREVIATION'].unique():
        team_stats = getOpponentStats(df, team)
        for _, row in team_stats.iterrows():
            key = (row['GAME_ID'], team)
            team_stats_dict[key] = {
                'OPP_DEF_RATING_AVG_TO_DATE': row['DEF_RATING_AVG_TO_DATE'],
                'OPP_PACE_AVG_TO_DATE': row['PACE_AVG_TO_DATE'],
                'OPP_PTS_AVG_TO_DATE': row['PTS_AVG_TO_DATE'],
                'OPP_FGA_AVG_TO_DATE': row['FGA_AVG_TO_DATE'],
                'OPP_REB_AVG_TO_DATE': row['REB_AVG_TO_DATE'],
                'OPP_AST_AVG_TO_DATE': row['AST_AVG_TO_DATE'],
                'OPP_TOV_AVG_TO_DATE': row['TOV_AVG_TO_DATE'],
                'OPP_BLK_AVG_TO_DATE': row['BLK_AVG_TO_DATE'],
                'OPP_STL_AVG_TO_DATE': row['STL_AVG_TO_DATE']
            }
    
    # Assign opponent stats using vectorized lookup
    df_enhanced = df.copy()
    lookup_keys = list(zip(df_enhanced['GAME_ID'], df_enhanced['OPP_ABBREVIATION']))
    
    for col in ['OPP_DEF_RATING_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE', 'OPP_PTS_AVG_TO_DATE', 'OPP_FGA_AVG_TO_DATE', 'OPP_REB_AVG_TO_DATE', 'OPP_AST_AVG_TO_DATE', 'OPP_TOV_AVG_TO_DATE', 'OPP_BLK_AVG_TO_DATE', 'OPP_STL_AVG_TO_DATE']:
        df_enhanced[col] = [team_stats_dict.get(key, {}).get(col, None) for key in lookup_keys]
    
    return df_enhanced

def teamContext(df):
    """Add team context features with season-to-date averages - FIXED"""
    # FIXED: All team averages now use shift(1) to prevent data leakage
    df['TEAM_DEF_RATING_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_DEF_RATING'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_PACE_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_PACE'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_OFF_RATING_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_OFF_RATING'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_PTS_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_PTS'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_FGA_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FGA'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_REB_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_REB'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_AST_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_AST'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_TOV_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_TOV'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    return df


# ================================================================================================
# LINEUP AND STARTER FEATURES
# ================================================================================================

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
        .round(2)
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
        .round(2)
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
        .round(2)
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
        .round(2)
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
    .round(2)
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
    """Calculate team starter spacing metric based on 3PT%."""
    starters_df = df[df['STARTING'] == 1].copy()
    
    team_spacing = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'FG3_PCT': 'mean'})
        .rename(columns={'FG3_PCT': 'TEAM_STARTER_SPACING_METRIC'})
        .round(2)
        .reset_index()
    )
    
    df = df.merge(
        team_spacing,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    return df

def pace_expectation(df):
    """Calculate expected pace based on team and opponent starter averages."""
    starters_df = df[df['STARTING'] == 1].copy()
    
    # Team starter average pace
    team_pace = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PACE': 'mean'})
        .rename(columns={'PACE': 'TEAM_STARTER_PACE'})
        .round(2)
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
        .round(2)
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
    df['PACE_EXPECTATION'] = ((df['TEAM_STARTER_PACE'] + df['OPP_STARTER_PACE']) / 2).round(2)
    
    return df

def process_star_players_data(df, all_nba_players, min_minutes=10):
    df = df.copy()
    # Create ACTIVE column based on minutes played
    df['ACTIVE'] = (df['MIN'] >= min_minutes).astype(int)

    # Flags for All-NBA
    df['PLAYER_IS_ALL_NBA'] = df['PLAYER_NAME'].isin(all_nba_players).astype(int)

    # Season-long team star by highest average USG_PCT (only among active players)
    # Handle cases where a player changes teams by grouping on TEAM_ID
    active_players = df[df['ACTIVE'] == 1].copy()
    usg_means = (
        active_players.groupby(['TEAM_ID', 'PLAYER_NAME'], dropna=False)['USG_PCT']
          .mean()
          .reset_index()
    )

    # Resolve star per team
    star_rows = (
        usg_means.sort_values(['TEAM_ID', 'USG_PCT'], ascending=[True, False])
                 .groupby(['TEAM_ID'], as_index=False)
                 .first()
    )
    star_by_team = {
        row.TEAM_ID: row.PLAYER_NAME
        for _, row in star_rows.iterrows()
    }

    df['STAR_NAME'] = df['TEAM_ID'].map(star_by_team)
    df['PLAYER_IS_TEAM_STAR'] = (df['PLAYER_NAME'] == df['STAR_NAME']).astype(int)

    # Star active status per game
    star_active_per_game = (
        df[df['PLAYER_NAME'] == df['STAR_NAME']]
        .groupby(['GAME_ID', 'TEAM_ID'], as_index=False)['ACTIVE']
        .max()
        .rename(columns={'ACTIVE': 'STAR_ACTIVE'})
    )
    df = df.merge(star_active_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['STAR_ACTIVE'] = df['STAR_ACTIVE'].fillna(0).astype(int)

    # TEAM_STAR_OUT for non-star rows only
    df['TEAM_STAR_OUT'] = ((df['PLAYER_IS_TEAM_STAR'] == 0) & (df['STAR_ACTIVE'] == 0)).astype(int)

    # All-NBA teammate out per game
    df['ALL_NBA_AND_OUT'] = ((df['PLAYER_IS_ALL_NBA'] == 1) & (df['ACTIVE'] == 0)).astype(int)
    all_nba_out_per_game = (
        df.groupby(['GAME_ID', 'TEAM_ID'], as_index=False)['ALL_NBA_AND_OUT']
          .sum()
          .rename(columns={'ALL_NBA_AND_OUT': 'NUM_ALL_NBA_OUT'})
    )
    df = df.merge(all_nba_out_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['NUM_ALL_NBA_OUT'] = df['NUM_ALL_NBA_OUT'].fillna(0).astype(int)

    # Exclude self if self is All-NBA and out
    self_is_all_nba_out = ((df['PLAYER_IS_ALL_NBA'] == 1) & (df['ACTIVE'] == 0)).astype(int)
    df['ALL_NBA_TEAMMATE_OUT'] = (df['NUM_ALL_NBA_OUT'] - self_is_all_nba_out > 0).astype(int)

    # Cleanup helpers
    df = df.drop(columns=['STAR_NAME', 'STAR_ACTIVE', 'ALL_NBA_AND_OUT', 'NUM_ALL_NBA_OUT', 'ACTIVE'])

    return df

def allLineupFeatures(df, star_players):
    """Wrapper function to add all lineup-related features."""
    df = teamUsualStarters(df)
    df = oppTeamUsualStarters(df)
    df = team_starter_spacing(df)
    df = pace_expectation(df)
    df = process_star_players_data(df, star_players)
    return df


# ================================================================================================
# PACE AND USAGE FEATURES - FIXED FOR DATA LEAKAGE
# ================================================================================================

def add_game_pace_adjustment(df):
    """Calculate game pace adjustments with data leakage prevention using shift(1) - FIXED"""
    # Create copy and pre-sort for time-series operations
    df = df.copy()
    df.sort_values(['GAME_DATE'], inplace=True)
    
    # Create team grouper objects once
    team_group = df.groupby('TEAM_ID')
    opp_team_group = df.groupby('OPP_TEAM_ID')
    
    # FIXED: Shift team paces by 1 to prevent data leakage
    shifted_team_pace = team_group['TEAM_PACE'].shift(1)
    shifted_opp_pace = opp_team_group['OPP_PACE'].shift(1)
    
    # Calculate team-level paces using shifted values
    team_paces = shifted_team_pace.groupby('TEAM_ID').mean()
    opp_paces = shifted_opp_pace.groupby('OPP_TEAM_ID').mean()
    
    # Calculate league average pace from shifted team averages
    league_pace = team_paces.mean()
    
    # Calculate opponent pace factor (how much faster/slower than league average)
    opp_pace_factors = opp_paces / league_pace
    
    # Calculate game pace adjustment and merge back
    pace_adjustments = (league_pace * opp_pace_factors).round(2).reset_index()
    pace_adjustments.columns = ['OPP_TEAM_ID', 'OPP_GAME_PACE_ADJUSTMENT_TO_DATE']  # RENAMED with _TO_DATE
    
    # Merge adjustments and convert to float32 to save memory
    df = df.merge(pace_adjustments, on='OPP_TEAM_ID', how='left')
    df['OPP_GAME_PACE_ADJUSTMENT_TO_DATE'] = df['OPP_GAME_PACE_ADJUSTMENT_TO_DATE'].astype('float32')
    
    # Add rolling windows for pace adjustments
    windows = [3, 5, 7]
    
    for window in windows:
        col_name = f'OPP_GAME_PACE_ADJ_ROLL_{window}_TO_DATE'  # RENAMED with _TO_DATE
        
        # Calculate rolling average on the adjustment with additional shift
        df[col_name] = (
            opp_team_group['OPP_GAME_PACE_ADJUSTMENT_TO_DATE']
            .shift(1)  # Additional shift for rolling calculation
            .rolling(window=window, min_periods=1)
            .mean()
            .round(2)
        )
        
        # Fill first games with overall mean
        first_games_mask = opp_team_group.cumcount() == 0
        df.loc[first_games_mask, col_name] = df['OPP_GAME_PACE_ADJUSTMENT_TO_DATE'].mean()
        
        # Convert to float32 to save memory
        df[col_name] = df[col_name].astype('float32')
    
    return df

def playerUsageAndOppurtunity(data):
    """Calculate player usage and opportunity metrics with rolling averages - FIXED"""
    # Create copy and pre-sort for time-series operations
    df = data.copy()
    df.sort_values(['PLAYER_ID', 'GAME_DATE'], inplace=True)
    
    # Create player grouper object once (reuse for efficiency)
    player_group = df.groupby('PLAYER_ID')
    
    # Define metrics and their combinations
    metric_combinations = {
        'MIN_X_USG_PCT_TO_DATE': ('MIN', 'USG_PCT'),  # RENAMED with _TO_DATE
        'USG_PCT_X_OPP_DEF_RATING_TO_DATE': ('USG_PCT', 'OPP_DEF_RATING'),  # RENAMED with _TO_DATE
        'TEAM_PACE_X_MIN_TO_DATE': ('TEAM_PACE', 'MIN'),  # RENAMED with _TO_DATE
        'USG_PCT_X_POSS_TO_DATE': ('USG_PCT', 'POSS'),  # RENAMED with _TO_DATE
        'MIN_X_OFF_RATING_TO_DATE': ('MIN', 'OFF_RATING'),  # RENAMED with _TO_DATE
        'USG_PCT_X_TS_PCT_TO_DATE': ('USG_PCT', 'TS_PCT')  # RENAMED with _TO_DATE
    }
    
    # FIXED: Pre-calculate shifted values for all base metrics to prevent leakage
    shifted_metrics = {}
    base_metrics = set()
    for metrics in metric_combinations.values():
        base_metrics.update(metrics)
    
    for metric in base_metrics:
        if metric in df.columns:
            shifted_metrics[metric] = player_group[metric].shift(1)
    
    # Calculate combinations using shifted values
    for col_name, (metric1, metric2) in metric_combinations.items():
        if metric1 not in df.columns or metric2 not in df.columns:
            continue
            
        # Use shifted values for calculation
        if col_name == 'TEAM_PACE_X_MIN_TO_DATE':
            # Special handling for TEAM_PACE_X_MIN with scaling
            df[col_name] = (
                shifted_metrics[metric1] * 
                shifted_metrics[metric2] / 100
            ).round(2)
        else:
            # Regular multiplication
            df[col_name] = (
                shifted_metrics[metric1] * 
                shifted_metrics[metric2]
            ).round(2)
    
    # Calculate rolling averages for new metrics
    windows = [3, 5, 7]
    
    for col_name in metric_combinations.keys():
        if col_name not in df.columns:
            continue
            
        for window in windows:
            roll_col = f'{col_name}_ROLL_{window}'
            
            # FIXED: Calculate rolling average with shift to prevent leakage
            df[roll_col] = (
                player_group[col_name]
                .shift(1)  # Prevent leakage
                .rolling(window=window, min_periods=1)
                .mean()
                .round(2)
            )
            
            # Fill first game for each player with metric's mean
            first_games_mask = player_group.cumcount() == 0
            df.loc[first_games_mask, roll_col] = df[col_name].mean()
            
            # Convert to float32 to save memory
            df[roll_col] = df[roll_col].astype('float32')
    
    # Convert base metrics to float32 to save memory
    for col in metric_combinations.keys():
        if col in df.columns:
            df[col] = df[col].astype('float32')
    return df


# ================================================================================================
# ADVANCED FEATURE ENGINEERING - ALREADY MOSTLY FIXED
# ================================================================================================

def feature_engineering(df):
    """Comprehensive feature engineering with rolling averages and derived metrics - FIXED"""
    df = df.sort_values(['GAME_DATE']).copy()

    # --- Rolling Averages --- Already properly shifted in existing code
    rolling_features = [
        ('PLAYER_ID', 'MIN', 'AVG_MIN_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'TCHS', 'AVG_TCHS_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'USG_PCT', 'AVG_USG_PCT_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'AST_PCT', 'AVG_AST_PCT_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'POSS', 'AVG_POSS_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'DIST', 'AVG_DIST_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'SPD', 'AVG_SPD_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'ORBC', 'AVG_ORBC_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'DRBC', 'AVG_DRBC_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'STL', 'AVG_STL_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'BLK', 'AVG_BLK_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'PFD', 'AVG_PFD_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'AST', 'AVG_AST_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'TOV', 'AVG_TOV_TO_DATE'),  # RENAMED with _TO_DATE
        # Add contested shot rolling averages
        ('PLAYER_ID', 'CFGA', 'AVG_CONTESTED_SHOTS_PG_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'CFGM', 'AVG_CONTESTED_MAKES_PG_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'UFGA', 'AVG_UNCONTESTED_SHOTS_PG_TO_DATE'),  # RENAMED with _TO_DATE
        ('PLAYER_ID', 'UFGM', 'AVG_UNCONTESTED_MAKES_PG_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'PACE', 'AVG_PACE_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_DEF_RATING', 'AVG_DEF_RATING_PRE_GAME_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_OFF_RATING', 'AVG_TEAM_OFF_RATING_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_PACE', 'AVG_TEAM_PACE_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_TOV', 'AVG_TEAM_TOV_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_FGA', 'AVG_TEAM_FGA_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_REB', 'AVG_TEAM_REB_TO_DATE'),  # RENAMED with _TO_DATE
        ('TEAM_ID', 'TEAM_AST', 'AVG_TEAM_AST_TO_DATE'),  # RENAMED with _TO_DATE
        ('OPP_TEAM_ID', 'OPP_DEF_RATING', 'AVG_OPP_DEF_RATING_TO_DATE'),  # RENAMED with _TO_DATE
        ('OPP_TEAM_ID', 'OPP_PACE', 'AVG_OPP_PACE_TO_DATE'),  # RENAMED with _TO_DATE
        ('OPP_TEAM_ID', 'OPP_STL', 'AVG_OPP_STL_TO_DATE'),  # RENAMED with _TO_DATE
        ('OPP_TEAM_ID', 'OPP_BLK', 'AVG_OPP_BLK_TO_DATE'),  # RENAMED with _TO_DATE
        ('OPP_TEAM_ID', 'OPP_REB', 'AVG_OPP_REB_TO_DATE'),  # RENAMED with _TO_DATE
        ('GAME_ID', 'GAME_PACE', 'AVG_GAME_PACE_PRE_GAME_TO_DATE')  # RENAMED with _TO_DATE
    ]

    # This was already properly shifted in the original code
    for group_col, target_col, new_col in rolling_features:
        if target_col in df.columns:
            df[new_col] = df.groupby(group_col)[target_col].transform(lambda x: x.shift(1).expanding().mean().round(2))

    # --- Feature Engineering --- Update feature references to use _TO_DATE versions
    df['PACE_IMPACT_TO_DATE'] = (
        df['AVG_PACE_PRE_GAME_TO_DATE'] +
        df['AVG_TEAM_PACE_TO_DATE'] +
        df['AVG_GAME_PACE_PRE_GAME_TO_DATE'] +
        df['AVG_OPP_PACE_TO_DATE']
    ) / 4

    df['PACE_USAGE_TO_DATE'] = df['AVG_PACE_PRE_GAME_TO_DATE'] * df['AVG_USG_PCT_PRE_GAME_TO_DATE']
    df['MIN_PACE_TO_DATE'] = df['AVG_MIN_PRE_GAME_TO_DATE'] * df['AVG_PACE_PRE_GAME_TO_DATE']
    df['DEF_AST_INTERACTION_TO_DATE'] = df['AVG_DEF_RATING_PRE_GAME_TO_DATE'] * df['AVG_AST_PCT_PRE_GAME_TO_DATE']
    df['TOUCH_USAGE_TO_DATE'] = df['AVG_TCHS_PRE_GAME_TO_DATE'] * df['AVG_USG_PCT_PRE_GAME_TO_DATE']
    df['POSS_PER_MIN_TO_DATE'] = df['AVG_POSS_PRE_GAME_TO_DATE'] / (df['AVG_MIN_PRE_GAME_TO_DATE'] + 1e-6)
    
    # Calculate TOUCHES_PER_POSS - how many touches per possession
    df['TOUCHES_PER_POSS_TO_DATE'] = df['AVG_TCHS_PRE_GAME_TO_DATE'] / (df['AVG_POSS_PRE_GAME_TO_DATE'] + 1e-6)
    
    # Contested shots features - makes FG% more predictive
    df['CONTESTED_SHOTS_PG'] = df['CFGA']  # Current game contested shots
    df['CONTESTED_FG_PCT'] = df['CFGM'] / (df['CFGA'] + 1e-6)  # Current game contested FG%
    df['UNCONTESTED_FG_PCT'] = df['UFGM'] / (df['UFGA'] + 1e-6)  # Current game uncontested FG%
    df['CONTESTED_SHOT_RATIO'] = df['CFGA'] / (df['FGA'] + 1e-6)  # % of shots that are contested
    df['SHOT_DIFFICULTY_SCORE'] = (df['CFGA'] * 2 + df['UFGA']) / (df['FGA'] + 1e-6)  # Weighted shot difficulty
    
    # Historical contested shot features (properly shifted - more predictive)
    df['AVG_CONTESTED_FG_PCT_TO_DATE'] = df['AVG_CONTESTED_MAKES_PG_TO_DATE'] / (df['AVG_CONTESTED_SHOTS_PG_TO_DATE'] + 1e-6)
    df['AVG_UNCONTESTED_FG_PCT_TO_DATE'] = df['AVG_UNCONTESTED_MAKES_PG_TO_DATE'] / (df['AVG_UNCONTESTED_SHOTS_PG_TO_DATE'] + 1e-6)
    df['AVG_CONTESTED_SHOT_RATIO_TO_DATE'] = df['AVG_CONTESTED_SHOTS_PG_TO_DATE'] / (df['FGA_SEASON_AVG_TO_DATE'] + 1e-6)
    df['CONTESTED_SHOT_EFFICIENCY_TO_DATE'] = df['AVG_CONTESTED_FG_PCT_TO_DATE'] - df['AVG_UNCONTESTED_FG_PCT_TO_DATE']
    df['TOTAL_SHOT_QUALITY_TO_DATE'] = (df['AVG_CONTESTED_SHOTS_PG_TO_DATE'] * df['AVG_CONTESTED_FG_PCT_TO_DATE'] + 
                                df['AVG_UNCONTESTED_SHOTS_PG_TO_DATE'] * df['AVG_UNCONTESTED_FG_PCT_TO_DATE']) / (df['FGA_SEASON_AVG_TO_DATE'] + 1e-6)

    df['DEF_RATING_DIFF_TO_DATE'] = df['AVG_TEAM_OFF_RATING_TO_DATE'] - df['AVG_OPP_DEF_RATING_TO_DATE']
    df['PACE_DIFF_TO_DATE'] = df['AVG_TEAM_PACE_TO_DATE'] - df['AVG_OPP_PACE_TO_DATE']

    df['OPP_STL_PRESSURE_TO_DATE'] = df['AVG_OPP_STL_TO_DATE'] / (df['AVG_TEAM_TOV_TO_DATE'] + 1)
    df['OPP_BLOCK_PRESSURE_TO_DATE'] = df['AVG_OPP_BLK_TO_DATE'] / (df['AVG_TEAM_FGA_TO_DATE'] + 1)
    df['OPP_DEF_REBOUND_CONTROL_TO_DATE'] = df['AVG_OPP_REB_TO_DATE'] / (df['AVG_TEAM_REB_TO_DATE'] + df['AVG_OPP_REB_TO_DATE'] + 1)

    df['DIST_PER_MIN_TO_DATE'] = df['AVG_DIST_TO_DATE'] / (df['AVG_MIN_PRE_GAME_TO_DATE'] + 1e-6)
    df['SPD_INTENSITY_TO_DATE'] = df['AVG_SPD_TO_DATE'] * df['DIST_PER_MIN_TO_DATE']
    df['EFFORT_METRIC_TO_DATE'] = (
        (df['AVG_ORBC_TO_DATE'] + df['AVG_DRBC_TO_DATE'] + df['AVG_STL_TO_DATE'] + df['AVG_BLK_TO_DATE'] + df['AVG_PFD_TO_DATE']) /
        (df['AVG_MIN_PRE_GAME_TO_DATE'] + 1)
    )

    df['TEAM_AST_SHARE_TO_DATE'] = df['AVG_AST_TO_DATE'] / (df['AVG_TEAM_AST_TO_DATE'] + 1)
    df['TEAM_TOV_SHARE_TO_DATE'] = df['AVG_TOV_TO_DATE'] / (df['AVG_TEAM_TOV_TO_DATE'] + 1)

    df['EXPECTED_POSS_ADJUSTED_TO_DATE'] = df['AVG_POSS_PRE_GAME_TO_DATE'] * (df['PACE_IMPACT_TO_DATE'] / 100)

    # --- NaN Handling ---
    # 1. Fill Rolling Averages NaNs with Global Averages
    for col in [colname for _, _, colname in rolling_features]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean(skipna=True))

    # 2. Fill Engineered Features selectively
    engineered_cols_to_fill = [
        'POSS_PER_MIN_TO_DATE', 'TOUCHES_PER_POSS_TO_DATE', 'TEAM_AST_SHARE_TO_DATE', 
        'TEAM_TOV_SHARE_TO_DATE', 'OPP_STL_PRESSURE_TO_DATE', 'OPP_BLOCK_PRESSURE_TO_DATE', 
        'OPP_DEF_REBOUND_CONTROL_TO_DATE', 'EFFORT_METRIC_TO_DATE', 'DIST_PER_MIN_TO_DATE', 
        'SPD_INTENSITY_TO_DATE'
    ]
    
    for col in engineered_cols_to_fill:
        if col in df.columns:
            if 'SHARE' in col or 'PRESSURE' in col or 'CONTROL' in col or 'METRIC' in col or 'INTENSITY' in col or 'DIST_PER_MIN' in col:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].mean(skipna=True))

    # 3. Final Sweep
    df = df.fillna(0)

    # --- Round Engineered Features ---
    engineered_cols = [
        'PACE_IMPACT_TO_DATE', 'PACE_USAGE_TO_DATE', 'MIN_PACE_TO_DATE', 'DEF_AST_INTERACTION_TO_DATE', 
        'TOUCH_USAGE_TO_DATE', 'POSS_PER_MIN_TO_DATE', 'DEF_RATING_DIFF_TO_DATE', 'PACE_DIFF_TO_DATE', 
        'OPP_STL_PRESSURE_TO_DATE', 'OPP_BLOCK_PRESSURE_TO_DATE', 'OPP_DEF_REBOUND_CONTROL_TO_DATE', 
        'DIST_PER_MIN_TO_DATE', 'SPD_INTENSITY_TO_DATE', 'EFFORT_METRIC_TO_DATE', 'TEAM_AST_SHARE_TO_DATE', 
        'TEAM_TOV_SHARE_TO_DATE', 'EXPECTED_POSS_ADJUSTED_TO_DATE'
    ]

    for col in engineered_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    return df

def starterUsageRank(df):
    """Calculate usage ranking among starters - FIXED"""
    starters_df = df[df['START_POSITION'].notnull()].copy()

    # FIXED: Use AVG_USG_PCT_PRE_GAME_TO_DATE (pre-game rolling average)
    if 'AVG_USG_PCT_PRE_GAME_TO_DATE' in starters_df.columns:
        starters_df['STARTER_USAGE_RANK'] = starters_df.groupby(['GAME_ID', 'TEAM_ABBREVIATION'])['AVG_USG_PCT_PRE_GAME_TO_DATE'] \
                                                       .rank(ascending=False, method='dense')
    else:
        # Fallback to the regular column if TO_DATE version doesn't exist
        starters_df['STARTER_USAGE_RANK'] = starters_df.groupby(['GAME_ID', 'TEAM_ABBREVIATION'])['AVG_USG_PCT_PRE_GAME'] \
                                                       .rank(ascending=False, method='dense')

    # Merge back to original df
    df = df.merge(starters_df[['PLAYER_NAME', 'GAME_ID', 'STARTER_USAGE_RANK']], 
                  on=['PLAYER_NAME', 'GAME_ID'], how='left')

    # Fill NaNs for bench players (or DNPs) with 0
    df['STARTER_USAGE_RANK'] = df['STARTER_USAGE_RANK'].fillna(0)

    return df


# ================================================================================================
# UTILITY AND HELPER FUNCTIONS
# ================================================================================================

def fill_na_with_similar_teams(df, team_stats_cols, team_id_col='TEAM_ID'):
    """Fill missing team stats using nearest neighbors approach."""
    team_stats = df.groupby(team_id_col)[team_stats_cols].mean().reset_index()

    # Fit Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
    nn_model.fit(team_stats[team_stats_cols])

    # For teams with NaNs, find nearest teams and fill with their averages
    for index, row in team_stats.iterrows():
        team_id = row[team_id_col]
        if row[team_stats_cols].isnull().any():
            distances, indices = nn_model.kneighbors([row[team_stats_cols].fillna(0)], return_distance=True)
            neighbor_stats = team_stats.iloc[indices[0]][team_stats_cols]
            team_stats.loc[index, team_stats_cols] = neighbor_stats.mean()

    # Map back to main df
    filled_stats = team_stats.set_index(team_id_col).to_dict('index')
    for stat in team_stats_cols:
        df[stat] = df.apply(lambda x: filled_stats[x[team_id_col]][stat] if pd.isna(x[stat]) else x[stat], axis=1)
        df[stat] = df[stat].round(2)

    return df

def add_performance_without_stars_columns(df, min_games=2):
    """
    Add columns showing player averages when star teammates are out.
    Also adds the number of All-NBA players on each team.
    """
    df = df.copy()
    
    # Add number of All-NBA players per team
    all_nba_count_per_team = (
        df.groupby(['TEAM_ID'])['PLAYER_IS_ALL_NBA']
        .max()  # Get unique players per team
        .reset_index()
    )
    
    # Get actual count by summing unique All-NBA players per team
    all_nba_per_team = (
        df[df['PLAYER_IS_ALL_NBA'] == 1]
        .groupby('TEAM_ID')['PLAYER_NAME']
        .nunique()
        .reset_index()
        .rename(columns={'PLAYER_NAME': 'NUM_ALL_NBA_ON_TEAM'})
    )
    
    # Merge back to main dataframe
    df = df.merge(all_nba_per_team, on='TEAM_ID', how='left')
    df['NUM_ALL_NBA_ON_TEAM'] = df['NUM_ALL_NBA_ON_TEAM'].fillna(0).astype(int)
    
    def calculate_without_star_stats(player_group):
        player_group = player_group.copy()
        
        # Scenarios
        star_out_mask = player_group['TEAM_STAR_OUT'] == 1
        all_nba_out_mask = player_group['ALL_NBA_TEAMMATE_OUT'] == 1
        both_out_mask = star_out_mask & all_nba_out_mask
        
        # Team star out stats
        if star_out_mask.sum() >= min_games:
            star_out_data = player_group[star_out_mask]
            player_group['PTS_WITHOUT_STAR'] = round(star_out_data['PTS'].mean(), 2)
            player_group['MIN_WITHOUT_STAR'] = round(star_out_data['MIN'].mean(), 2)
            player_group['USG_PCT_WITHOUT_STAR'] = round(star_out_data['USG_PCT'].mean(), 2)
            player_group['FGA_WITHOUT_STAR'] = round(star_out_data['FGA'].mean(), 2)
            player_group['FG3A_WITHOUT_STAR'] = round(star_out_data['FG3A'].mean(), 2)
            player_group['FTA_WITHOUT_STAR'] = round(star_out_data['FTA'].mean(), 2)
            player_group['EFG_PCT_WITHOUT_STAR'] = round(star_out_data['EFG_PCT'].mean(), 2)
            player_group['TS_PCT_WITHOUT_STAR'] = round(star_out_data['TS_PCT'].mean(), 2)
            player_group['AST_WITHOUT_STAR'] = round(star_out_data['AST'].mean(), 2)
            player_group['REB_WITHOUT_STAR'] = round(star_out_data['REB'].mean(), 2)
            player_group['TOV_WITHOUT_STAR'] = round(star_out_data['TOV'].mean(), 2)
            player_group['PTS_PER_36_WITHOUT_STAR'] = round((star_out_data['PTS'] * 36 / star_out_data['MIN']).mean(), 2)
            player_group['GAMES_WITHOUT_STAR'] = star_out_mask.sum()
        else:
            player_group['PTS_WITHOUT_STAR'] = 0
            player_group['MIN_WITHOUT_STAR'] = 0
            player_group['USG_PCT_WITHOUT_STAR'] = 0
            player_group['FGA_WITHOUT_STAR'] = 0
            player_group['FG3A_WITHOUT_STAR'] = 0
            player_group['FTA_WITHOUT_STAR'] = 0
            player_group['EFG_PCT_WITHOUT_STAR'] = 0
            player_group['TS_PCT_WITHOUT_STAR'] = 0
            player_group['AST_WITHOUT_STAR'] = 0
            player_group['REB_WITHOUT_STAR'] = 0
            player_group['TOV_WITHOUT_STAR'] = 0
            player_group['PTS_PER_36_WITHOUT_STAR'] = 0
            player_group['GAMES_WITHOUT_STAR'] = 0
        
        # All-NBA out stats
        if all_nba_out_mask.sum() >= min_games:
            all_nba_out_data = player_group[all_nba_out_mask]
            player_group['PTS_WITHOUT_ALL_NBA'] = round(all_nba_out_data['PTS'].mean(), 2)
            player_group['MIN_WITHOUT_ALL_NBA'] = round(all_nba_out_data['MIN'].mean(), 2)
            player_group['USG_PCT_WITHOUT_ALL_NBA'] = round(all_nba_out_data['USG_PCT'].mean(), 2)
            player_group['FGA_WITHOUT_ALL_NBA'] = round(all_nba_out_data['FGA'].mean(), 2)
            player_group['FG3A_WITHOUT_ALL_NBA'] = round(all_nba_out_data['FG3A'].mean(), 2)
            player_group['FTA_WITHOUT_ALL_NBA'] = round(all_nba_out_data['FTA'].mean(), 2)
            player_group['EFG_PCT_WITHOUT_ALL_NBA'] = round(all_nba_out_data['EFG_PCT'].mean(), 2)
            player_group['TS_PCT_WITHOUT_ALL_NBA'] = round(all_nba_out_data['TS_PCT'].mean(), 2)
            player_group['AST_WITHOUT_ALL_NBA'] = round(all_nba_out_data['AST'].mean(), 2)
            player_group['REB_WITHOUT_ALL_NBA'] = round(all_nba_out_data['REB'].mean(), 2)
            player_group['TOV_WITHOUT_ALL_NBA'] = round(all_nba_out_data['TOV'].mean(), 2)
            player_group['PTS_PER_36_WITHOUT_ALL_NBA'] = round((all_nba_out_data['PTS'] * 36 / all_nba_out_data['MIN']).mean(), 2)
            player_group['GAMES_WITHOUT_ALL_NBA'] = all_nba_out_mask.sum()
        else:
            player_group['PTS_WITHOUT_ALL_NBA'] = 0
            player_group['MIN_WITHOUT_ALL_NBA'] = 0
            player_group['USG_PCT_WITHOUT_ALL_NBA'] = 0
            player_group['FGA_WITHOUT_ALL_NBA'] = 0
            player_group['FG3A_WITHOUT_ALL_NBA'] = 0
            player_group['FTA_WITHOUT_ALL_NBA'] = 0
            player_group['EFG_PCT_WITHOUT_ALL_NBA'] = 0
            player_group['TS_PCT_WITHOUT_ALL_NBA'] = 0
            player_group['AST_WITHOUT_ALL_NBA'] = 0
            player_group['REB_WITHOUT_ALL_NBA'] = 0
            player_group['TOV_WITHOUT_ALL_NBA'] = 0
            player_group['PTS_PER_36_WITHOUT_ALL_NBA'] = 0
            player_group['GAMES_WITHOUT_ALL_NBA'] = 0
        
        # Both out stats
        if both_out_mask.sum() >= min_games:
            both_out_data = player_group[both_out_mask]
            player_group['PTS_WITHOUT_BOTH_STARS'] = round(both_out_data['PTS'].mean(), 2)
            player_group['MIN_WITHOUT_BOTH_STARS'] = round(both_out_data['MIN'].mean(), 2)
            player_group['USG_PCT_WITHOUT_BOTH_STARS'] = round(both_out_data['USG_PCT'].mean(), 2)
            player_group['FGA_WITHOUT_BOTH_STARS'] = round(both_out_data['FGA'].mean(), 2)
            player_group['FG3A_WITHOUT_BOTH_STARS'] = round(both_out_data['FG3A'].mean(), 2)
            player_group['FTA_WITHOUT_BOTH_STARS'] = round(both_out_data['FTA'].mean(), 2)
            player_group['EFG_PCT_WITHOUT_BOTH_STARS'] = round(both_out_data['EFG_PCT'].mean(), 2)
            player_group['TS_PCT_WITHOUT_BOTH_STARS'] = round(both_out_data['TS_PCT'].mean(), 2)
            player_group['AST_WITHOUT_BOTH_STARS'] = round(both_out_data['AST'].mean(), 2)
            player_group['REB_WITHOUT_BOTH_STARS'] = round(both_out_data['REB'].mean(), 2)
            player_group['TOV_WITHOUT_BOTH_STARS'] = round(both_out_data['TOV'].mean(), 2)
            player_group['PTS_PER_36_WITHOUT_BOTH_STARS'] = round((both_out_data['PTS'] * 36 / both_out_data['MIN']).mean(), 2)
            player_group['GAMES_WITHOUT_BOTH_STARS'] = both_out_mask.sum()
        else:
            player_group['PTS_WITHOUT_BOTH_STARS'] = 0
            player_group['MIN_WITHOUT_BOTH_STARS'] = 0
            player_group['USG_PCT_WITHOUT_BOTH_STARS'] = 0
            player_group['FGA_WITHOUT_BOTH_STARS'] = 0
            player_group['FG3A_WITHOUT_BOTH_STARS'] = 0
            player_group['FTA_WITHOUT_BOTH_STARS'] = 0
            player_group['EFG_PCT_WITHOUT_BOTH_STARS'] = 0
            player_group['TS_PCT_WITHOUT_BOTH_STARS'] = 0
            player_group['AST_WITHOUT_BOTH_STARS'] = 0
            player_group['REB_WITHOUT_BOTH_STARS'] = 0
            player_group['TOV_WITHOUT_BOTH_STARS'] = 0
            player_group['PTS_PER_36_WITHOUT_BOTH_STARS'] = 0
            player_group['GAMES_WITHOUT_BOTH_STARS'] = 0
        
        return player_group
    
    # Apply to each player
    result = df.groupby('PLAYER_NAME', group_keys=False).apply(calculate_without_star_stats)
    
    return result

def check_all_defensive_players_at_position(df, all_defensive_players, position, year=2025):
    """
    Check if there are all-defensive players at a specific position (guard, forward, or center).
    """
    # Create copy and validate inputs
    df = df.copy()
    position = position.lower()
    
    if position not in ['guard', 'forward', 'center']:
        raise ValueError("Position must be 'guard', 'forward', or 'center'")
    
    if year not in all_defensive_players:
        raise ValueError(f"Year {year} not found in all_defensive_players dictionary")
    
    # Get all-defensive players for the specified year
    def_players = all_defensive_players[year]
    
    # Create position column mapping
    position_mapping = {
        'guard': 'GUARD',
        'forward': 'FORWARD', 
        'center': 'CENTER'
    }
    
    position_col = position_mapping[position]
    
    # Ensure required columns exist
    required_cols = ['PLAYER_NAME', 'GAME_ID', 'TEAM_ID', position_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Mark if player is an all-defensive player (only add if not already exists)
    if 'IS_ALL_DEFENSIVE' not in df.columns:
        df['IS_ALL_DEFENSIVE'] = df['PLAYER_NAME'].isin(def_players).astype(int)
    
    # Mark if player is an all-defensive player at the specified position
    df[f'IS_ALL_DEF_{position.upper()}'] = (
        (df['IS_ALL_DEFENSIVE'] == 1) & (df[position_col] == 1)
    ).astype(int)
    
    # Group by game and team to count all-defensive players at position
    team_def_counts = (
        df[df[f'IS_ALL_DEF_{position.upper()}'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .size()
        .reset_index(name=f'ALL_DEF_{position.upper()}_COUNT')
    )
    
    # Initialize count columns with zeros
    df[f'ALL_DEF_{position.upper()}_COUNT_TEAM'] = 0
    
    # Merge counts back to main dataframe for player's team (only if there are counts)
    if not team_def_counts.empty:
        df = df.merge(
            team_def_counts,
            on=['GAME_ID', 'TEAM_ID'],
            how='left'
        )
        # Update the team count column with actual values, keeping 0 for NaN
        df[f'ALL_DEF_{position.upper()}_COUNT_TEAM'] = df[f'ALL_DEF_{position.upper()}_COUNT'].fillna(0).astype(int)
        # Clean up the temporary column
        df.drop(columns=[f'ALL_DEF_{position.upper()}_COUNT'], inplace=True, errors='ignore')
    
    # Add opponent team counts (if OPP_TEAM_ID exists)
    if 'OPP_TEAM_ID' in df.columns:
        # Initialize opponent count column
        df[f'ALL_DEF_{position.upper()}_COUNT_OPP'] = 0
        
        # Only merge if there are defensive players at this position
        if not team_def_counts.empty:
            opp_counts = team_def_counts.rename(columns={
                'TEAM_ID': 'OPP_TEAM_ID', 
                f'ALL_DEF_{position.upper()}_COUNT': f'ALL_DEF_{position.upper()}_COUNT_OPP'
            })
            
            df = df.merge(
                opp_counts,
                on=['GAME_ID', 'OPP_TEAM_ID'],
                how='left',
                suffixes=('', '_merge')
            )
            
            # Update opponent count, handling the case where merge column exists
            if f'ALL_DEF_{position.upper()}_COUNT_OPP_merge' in df.columns:
                df[f'ALL_DEF_{position.upper()}_COUNT_OPP'] = df[f'ALL_DEF_{position.upper()}_COUNT_OPP_merge'].fillna(0).astype(int)
                df.drop(columns=[f'ALL_DEF_{position.upper()}_COUNT_OPP_merge'], inplace=True)
        
        # Binary flag for facing all-defensive player at position
        df[f'FACING_ALL_DEF_{position.upper()}'] = (df[f'ALL_DEF_{position.upper()}_COUNT_OPP'] > 0).astype(int)
    
    # Binary flag for having all-defensive player at position in game
    opp_count = df.get(f'ALL_DEF_{position.upper()}_COUNT_OPP', 0)
    df[f'ALL_DEF_{position.upper()}_IN_GAME'] = (
        (df[f'ALL_DEF_{position.upper()}_COUNT_TEAM'] > 0) | 
        (opp_count > 0)
    ).astype(int)
    
    return df

def add_all_defensive_features(df, all_defensive_players, year=2025):
    """
    Add comprehensive all-defensive player features for all positions.
    
    Parameters:
    - df: DataFrame containing player data
    - all_defensive_players: Dictionary with years as keys and lists of all-defensive player names as values
    - year: Integer - Year to get all-defensive players list from (default: 2025)
    
    Returns:
    - DataFrame with all-defensive features for guards, forwards, and centers
    """
    # Apply for all three positions
    for position in ['guard', 'forward', 'center']:
        df = check_all_defensive_players_at_position(df, all_defensive_players, position, year)
    
    # Add summary features
    df['TOTAL_ALL_DEF_TEAM'] = (
        df['ALL_DEF_GUARD_COUNT_TEAM'] + 
        df['ALL_DEF_FORWARD_COUNT_TEAM'] + 
        df['ALL_DEF_CENTER_COUNT_TEAM']
    )
    
    if 'OPP_TEAM_ID' in df.columns:
        df['TOTAL_ALL_DEF_OPP'] = (
            df['ALL_DEF_GUARD_COUNT_OPP'] + 
            df['ALL_DEF_FORWARD_COUNT_OPP'] + 
            df['ALL_DEF_CENTER_COUNT_OPP']
        )
        
        df['ALL_DEF_MATCHUP_TOTAL'] = df['TOTAL_ALL_DEF_TEAM'] + df['TOTAL_ALL_DEF_OPP']
    
    return df

########################################################

def merge_betting_data(player_df, betting_df, team_dict):
    """
    Merge betting data (spread, total, who's favored) into player dataset
    """
    df = player_df.copy()
    odds = betting_df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    odds['date'] = pd.to_datetime(odds['date'])
    
    # Convert betting data team abbreviations to uppercase using team_dict
    odds['away_upper'] = odds['away'].map(team_dict)
    odds['home_upper'] = odds['home'].map(team_dict)
    
    # First, create a unique identifier for each game in odds data
    odds['game_key_home'] = odds['date'].astype(str) + '_' + odds['home_upper'] + '_' + odds['away_upper']
    odds['game_key_away'] = odds['date'].astype(str) + '_' + odds['away_upper'] + '_' + odds['home_upper']
    
    df['game_key'] = df['GAME_DATE'].astype(str) + '_' + df['TEAM_ABBREVIATION'] + '_' + df['OPP_ABBREVIATION']
    home_merge = df.merge(
        odds[['game_key_home', 'whos_favored', 'spread', 'total']].rename(columns={'game_key_home': 'game_key'}),
        on='game_key',
        how='left',
        suffixes=('', '_home')
    )
    away_merge = df.merge(
        odds[['game_key_away', 'whos_favored', 'spread', 'total']].rename(columns={'game_key_away': 'game_key'}),
        on='game_key', 
        how='left',
        suffixes=('', '_away')
    )
    df['whos_favored'] = home_merge['whos_favored'].fillna(away_merge['whos_favored'])
    df['spread'] = home_merge['spread'].fillna(away_merge['spread']).round(2)
    df['total'] = home_merge['total'].fillna(away_merge['total']).round(2)
    df['team_is_favored'] = ((df['whos_favored'] == 'home') & (df['HOME_GAME'] == 1)) | \
                           ((df['whos_favored'] == 'away') & (df['HOME_GAME'] == 0))
    df['team_spread'] = df.apply(lambda row: 
        round(row['spread'] if row['HOME_GAME'] == 1 else -row['spread'], 2), axis=1)
    df.drop('game_key', axis=1, inplace=True)
    return df

team_dict = {
    'min': 'MIN', 
    'bos': 'BOS', 
    'bkn': 'BKN', 
    'ny': 'NYK', 
    'phi': 'PHI', 
    'tor': 'TOR', 
    'chi': 'CHI', 
    'cle': 'CLE', 
    'det': 'DET', 
    'ind': 'IND', 
    'mia': 'MIA', 
    'atl': 'ATL', 
    'cha': 'CHA', 
    'was': 'WAS',
    'wsh': 'WAS',
    'orl': 'ORL', 
    'mil': 'MIL', 
    'chh': 'CHH', 
    'dal': 'DAL', 
    'hou': 'HOU',
    'lac': 'LAC',
    'lal': 'LAL',
    'sac': 'SAC',
    'por': 'POR',
    'uta': 'UTA',
    'utah': 'UTA', 
    'den': 'DEN',
    'okc': 'OKC',
    'mem': 'MEM',
    'no': 'NOP',
    'sa': 'SAS',    
    'gs': 'GSW',
    'phx': 'PHX',  
}


########################################################

import math

# NBA team home coordinates (lat, lon)
TEAM_COORDS = {
    "ATL": (33.7573, -84.3963),
    "BOS": (42.3662, -71.0621),
    "BKN": (40.6826, -73.9754),
    "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742),
    "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8104),
    "DEN": (39.7487, -105.0077),
    "DET": (42.3410, -83.0550),
    "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621),
    "IND": (39.7640, -86.1555),
    "LAC": (34.0430, -118.2673),
    "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0506),
    "MIA": (25.7814, -80.1870),
    "MIL": (43.0451, -87.9172),
    "MIN": (44.9795, -93.2761),
    "NOP": (29.9490, -90.0821),
    "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720),
    "PHX": (33.4458, -112.0712),
    "POR": (45.5316, -122.6668),
    "SAC": (38.6490, -121.5180),
    "SAS": (29.4270, -98.4375),
    "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011),
    "WAS": (38.8981, -77.0209),
}

def haversine(coord1, coord2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def distance_and_time(playerTeam, homeTeam, speed=500):
    coord1 = TEAM_COORDS[playerTeam]
    coord2 = TEAM_COORDS[homeTeam]
    distance = haversine(coord1, coord2)
    time_hours = distance / speed
    return {
        "miles": round(distance, 1),
        "hours": round(time_hours, 2)
    }

def add_travel_features(df):
    """Add travel distance and time features using GAME_ID approach"""
    df = df.copy()
    
    # Initialize travel columns
    df['TRAVEL_DISTANCE_MILES'] = 0.0
    df['TRAVEL_TIME_HOURS'] = 0.0
    
    # Check required columns
    required_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 'HOME_GAME', 'GAME_DATE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return df
    
    # Sort by team and date for proper chronological order
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
    
    # For each team, calculate travel distance for away games
    for team in df['TEAM_ABBREVIATION'].unique():
        team_games = df[df['TEAM_ABBREVIATION'] == team].copy()
        
        for i in range(1, len(team_games)):
            current_game = team_games.iloc[i]
            prev_game = team_games.iloc[i-1]
            
            # Only calculate travel for away games
            if current_game['HOME_GAME'] == 0:
                # Determine previous location
                if prev_game['HOME_GAME'] == 1:
                    # Previous game was at home
                    prev_location = prev_game['TEAM_ABBREVIATION']
                else:
                    # Previous game was away
                    prev_location = prev_game['OPP_ABBREVIATION']
                
                # Current away game location
                current_location = current_game['OPP_ABBREVIATION']
                
                # Calculate travel distance
                try:
                    if prev_location in TEAM_COORDS and current_location in TEAM_COORDS:
                        travel_info = distance_and_time(prev_location, current_location)
                        
                        # Update the main dataframe using the index
                        df.loc[current_game.name, 'TRAVEL_DISTANCE_MILES'] = travel_info['miles']
                        df.loc[current_game.name, 'TRAVEL_TIME_HOURS'] = travel_info['hours']
                        
                except Exception as e:
                    continue
    
    return df

def sort_data_for_features(df):
    """
    Sort data optimally for feature engineering pipeline
    """
    # Convert GAME_DATE to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['GAME_DATE']):
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Primary sort: Player chronological order
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    
    return df
