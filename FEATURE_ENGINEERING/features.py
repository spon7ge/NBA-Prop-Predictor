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

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[3,5,10]):
    """Calculate rolling averages for key player statistics with emphasis on recent form."""
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = ['PTS', 'MIN', 'FGA', 'FTA', 'FG3A', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 
                  'USG_PCT', 'TS_PCT', 'OFF_RATING', 'EFG_PCT','POSS', 'TCHS','AST', 'REB', 'TOV'
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

def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_lines=['PTS']):
    player_data = player_data.sort_values([player_id_col, date_col])
    if isinstance(stat_lines, str):
        stat_lines = [stat_lines]
    
    for stat_line in stat_lines:
        if stat_line not in player_data.columns:
            continue
            
        for lag in range(1, 3):
            lag_col = f'{stat_line}_LAG_{lag}'
            player_data[lag_col] = player_data.groupby(player_id_col)[stat_line].shift(lag)
            rolling_mean = player_data.groupby(player_id_col)[stat_line].transform(
                lambda x: x.shift(1).expanding().mean().round(2)
            )
            player_data[lag_col] = player_data[lag_col].fillna(rolling_mean)
            
            # Fill remaining NaNs (e.g., first game) with NaN
            player_data[lag_col] = player_data[lag_col].fillna(np.nan)
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
                        'EFG_PCT','POSS', 'TCHS','AST', 'REB', 'TOV', 'PASS', 'SAST', 'FTAST', 'PLUS_MINUS', 'DIST', 'SPD']
    
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
    stats_to_average = ['PTS', 'MIN', 'FGA', 'FTA', 'FG3A', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'OFF_RATING', 
                        'EFG_PCT','POSS', 'TCHS','AST', 'REB', 'TOV']
    
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

    metrics = ['PTS', 'MIN', 'FGA', 'FTA', 'FG3A', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'OFF_RATING', 
                'EFG_PCT','POSS', 'TCHS','AST', 'REB', 'TOV']
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

def expectedPace(df):
    df = df.copy()
    required_cols = ['TEAM_PACE_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Setting EXPECTED_PACE to 0.")
        df['EXPECTED_PACE'] = np.nan
        return df
    
    # Calculate expected pace by multiplying team and opponent pace averages
    df['EXPECTED_PACE'] = ((df['TEAM_PACE_AVG_TO_DATE'] + df['OPP_PACE_AVG_TO_DATE']) / 2).round(2)
    
    # Handle any NaN values that might result from missing data
    df['EXPECTED_PACE'] = df['EXPECTED_PACE'].fillna(np.nan)
    return df

def teamUsualStarters(df):
    """
    Adds NUM_USUAL_STARTERS_PRESENT: number of usual starters present for own team
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
    
    # 3) Merge into main df
    df = df.merge(
        starters_per_game[['GAME_ID', 'TEAM_ID', 'NUM_USUAL_STARTERS_PRESENT']],
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    return df

def oppTeamUsualStarters(df):
    """
    Adds NUM_USUAL_STARTERS_PRESENT_OPP: count of opponent usual starters present
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

    return df
########################################################
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
    
# ================================================================================================
# LINEUP AND STARTER FEATURES
# ================================================================================================



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
# ================================================================================================
# UTILITY AND HELPER FUNCTIONS
# ================================================================================================

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
        
        # Only team star out scenario
        star_out_mask = player_group['TEAM_STAR_OUT'] == 1
        
        # Team star out stats
        if star_out_mask.sum() >= min_games:
            star_out_data = player_group[star_out_mask]
            player_group['PTS_WITHOUT_STAR'] = round(star_out_data['PTS'].mean(), 2)
            player_group['MIN_WITHOUT_STAR'] = round(star_out_data['MIN'].mean(), 2)
            player_group['USG_PCT_WITHOUT_STAR'] = round(star_out_data['USG_PCT'].mean(), 2)
            player_group['FGA_WITHOUT_STAR'] = round(star_out_data['FGA'].mean(), 2)
            player_group['FG3A_WITHOUT_STAR'] = round(star_out_data['FG3A'].mean(), 2)
            player_group['FTA_WITHOUT_STAR'] = round(star_out_data['FTA'].mean(), 2)
            player_group['FG_PCT_WITHOUT_STAR'] = round(star_out_data['FG_PCT'].mean(), 2)
            player_group['FG3_PCT_WITHOUT_STAR'] = round(star_out_data['FG3_PCT'].mean(), 2)
            player_group['FT_PCT_WITHOUT_STAR'] = round(star_out_data['FT_PCT'].mean(), 2)
            player_group['EFG_PCT_WITHOUT_STAR'] = round(star_out_data['EFG_PCT'].mean(), 2)
            player_group['TS_PCT_WITHOUT_STAR'] = round(star_out_data['TS_PCT'].mean(), 2)
            player_group['AST_WITHOUT_STAR'] = round(star_out_data['AST'].mean(), 2)
            player_group['POSS_WITHOUT_STAR'] = round(star_out_data['POSS'].mean(), 2)
            player_group['TCHS_WITHOUT_STAR'] = round(star_out_data['TCHS'].mean(), 2)
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
            player_group['FG_PCT_WITHOUT_STAR'] = 0
            player_group['FG3_PCT_WITHOUT_STAR'] = 0
            player_group['FT_PCT_WITHOUT_STAR'] = 0
            player_group['EFG_PCT_WITHOUT_STAR'] = 0
            player_group['TS_PCT_WITHOUT_STAR'] = 0
            player_group['AST_WITHOUT_STAR'] = 0
            player_group['POSS_WITHOUT_STAR'] = 0
            player_group['TCHS_WITHOUT_STAR'] = 0
            player_group['REB_WITHOUT_STAR'] = 0
            player_group['TOV_WITHOUT_STAR'] = 0
            player_group['PTS_PER_36_WITHOUT_STAR'] = 0
            player_group['GAMES_WITHOUT_STAR'] = 0
        
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

##############################################################################################################
def add_volatility_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[5, 10, 15]):
    """
    Calculate volatility features for player performance metrics.
    Uses standard deviation, coefficient of variation, and consistency metrics.
    """
    # Create copy and sort data
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    # Define stats to calculate volatility for
    volatility_stats = ['PTS', 'MIN', 'FGA', 'FTA', 'FG3A', 'USG_PCT', 'TS_PCT', 
                       'EFG_PCT', 'OFF_RATING', 'AST', 'REB', 'TOV']
    
    # Filter to only available columns
    available_stats = [stat for stat in volatility_stats if stat in df.columns]
    
    if not available_stats:
        print("Warning: No volatility stats found in dataframe")
        return df
    
    # Calculate volatility metrics for each window
    for window in windows:
        for stat in available_stats:
            # Rolling standard deviation (shifted to prevent leakage)
            volatility_col = f'{stat}_VOLATILITY_{window}_TO_DATE'
            df[volatility_col] = (
                df.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
                .round(3)
            )
            
            # Coefficient of variation (volatility relative to mean)
            cv_col = f'{stat}_CV_{window}_TO_DATE'
            rolling_mean = (
                df.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).mean())
            )
            rolling_std = (
                df.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
            )
            
            # Calculate CV, handling division by zero
            df[cv_col] = np.where(
                rolling_mean != 0,
                (rolling_std / rolling_mean).round(3),
                0
            )
            
            # Consistency score (inverse of CV, capped at reasonable values)
            consistency_col = f'{stat}_CONSISTENCY_{window}_TO_DATE'
            df[consistency_col] = np.where(
                df[cv_col] > 0,
                np.minimum(1 / df[cv_col], 10),  # Cap at 10 for extreme consistency
                10  # Perfect consistency when CV is 0
            ).round(3)
    
    # Add expanding volatility metrics (season-long volatility)
    for stat in available_stats:
        # Expanding standard deviation
        expanding_vol_col = f'{stat}_EXPANDING_VOLATILITY_TO_DATE'
        df[expanding_vol_col] = (
            df.groupby(player_id_col)[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).std())
            .round(3)
        )
        
        # Expanding coefficient of variation
        expanding_cv_col = f'{stat}_EXPANDING_CV_TO_DATE'
        expanding_mean = (
            df.groupby(player_id_col)[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).mean())
        )
        expanding_std = (
            df.groupby(player_id_col)[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).std())
        )
        
        df[expanding_cv_col] = np.where(
            expanding_mean != 0,
            (expanding_std / expanding_mean).round(3),
            0
        )
    
    # Add volatility trend features (is player becoming more/less volatile?)
    for stat in available_stats:
        if len(windows) >= 2:
            short_window = min(windows)
            long_window = max(windows)
            
            short_vol_col = f'{stat}_VOLATILITY_{short_window}_TO_DATE'
            long_vol_col = f'{stat}_VOLATILITY_{long_window}_TO_DATE'
            
            # Volatility trend (positive = becoming more volatile)
            trend_col = f'{stat}_VOLATILITY_TREND_TO_DATE'
            df[trend_col] = (
                df[short_vol_col] - df[long_vol_col]
            ).round(3)
            
            # Volatility ratio (short-term vs long-term volatility)
            ratio_col = f'{stat}_VOLATILITY_RATIO_TO_DATE'
            df[ratio_col] = np.where(
                df[long_vol_col] != 0,
                (df[short_vol_col] / df[long_vol_col]).round(3),
                1.0  # Default ratio when long-term volatility is 0
            )
    
    # Add game-to-game change features
    for stat in available_stats:
        # Absolute change from previous game
        change_col = f'{stat}_GAME_CHANGE_TO_DATE'
        df[change_col] = (
            df.groupby(player_id_col)[stat].diff().abs().round(2)
        )
        
        # Percentage change from previous game
        pct_change_col = f'{stat}_GAME_PCT_CHANGE_TO_DATE'
        prev_value = df.groupby(player_id_col)[stat].shift(1)
        df[pct_change_col] = np.where(
            prev_value != 0,
            ((df[stat] - prev_value) / prev_value * 100).round(2),
            0
        )
    
    # Add streak-based volatility features
    for stat in available_stats:
        # Count consecutive games above/below season average
        season_avg_col = f'{stat}_SEASON_AVG_TO_DATE'
        if season_avg_col in df.columns:
            # Above average streak
            above_avg = (df[stat] > df[season_avg_col]).astype(int)
            above_streak_col = f'{stat}_ABOVE_AVG_STREAK_TO_DATE'
            df[above_streak_col] = (
                above_avg.groupby([df[player_id_col], (above_avg != above_avg.shift()).cumsum()])
                .cumsum()
                .where(above_avg == 1, 0)
            )
            
            # Below average streak
            below_avg = (df[stat] < df[season_avg_col]).astype(int)
            below_streak_col = f'{stat}_BELOW_AVG_STREAK_TO_DATE'
            df[below_streak_col] = (
                below_avg.groupby([df[player_id_col], (below_avg != below_avg.shift()).cumsum()])
                .cumsum()
                .where(below_avg == 1, 0)
            )
    
    # Fill NaN values with appropriate defaults
    volatility_cols = [col for col in df.columns if any(x in col for x in ['VOLATILITY', 'CV', 'CONSISTENCY', 'CHANGE', 'STREAK'])]
    
    for col in volatility_cols:
        if 'CV' in col or 'VOLATILITY' in col:
            df[col] = df[col].fillna(0)  # No volatility for first games
        elif 'CONSISTENCY' in col:
            df[col] = df[col].fillna(10)  # Perfect consistency for first games
        elif 'CHANGE' in col:
            df[col] = df[col].fillna(0)  # No change for first games
        elif 'STREAK' in col:
            df[col] = df[col].fillna(0)  # No streak for first games
    
    # Convert to appropriate data types to save memory
    for col in volatility_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df


def add_performance_volatility_categories(df, player_id_col='PLAYER_ID'):
    """
    Add categorical volatility features based on percentiles.
    Categorizes players as Low, Medium, or High volatility.
    """
    df = df.copy()
    
    # Define key volatility metrics to categorize
    key_metrics = ['PTS_EXPANDING_CV_TO_DATE', 'MIN_EXPANDING_CV_TO_DATE', 'USG_PCT_EXPANDING_CV_TO_DATE']
    
    for metric in key_metrics:
        if metric in df.columns:
            # Calculate percentiles for categorization
            p33 = df[metric].quantile(0.33)
            p67 = df[metric].quantile(0.67)
            
            # Create categorical column
            category_col = metric.replace('_EXPANDING_CV_TO_DATE', '_VOLATILITY_CATEGORY')
            df[category_col] = pd.cut(
                df[metric],
                bins=[-np.inf, p33, p67, np.inf],
                labels=['Low', 'Medium', 'High']
            )
            
            # Create binary flags for each category
            for category in ['Low', 'Medium', 'High']:
                flag_col = f"{metric.replace('_EXPANDING_CV_TO_DATE', '')}_{category.upper()}_VOLATILITY"
                df[flag_col] = (df[category_col] == category).astype(int)
    
    return df


def add_recent_form_volatility(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', lookback_games=5):
    """
    Add features that capture recent form and hot/cold streaks.
    Focuses on whether a player is in a volatile period recently.
    """
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    key_stats = ['PTS', 'FGA', 'USG_PCT', 'TS_PCT']
    available_stats = [stat for stat in key_stats if stat in df.columns]
    
    for stat in available_stats:
        # Recent volatility vs season volatility
        recent_vol_col = f'{stat}_VOLATILITY_{lookback_games}_TO_DATE'
        season_vol_col = f'{stat}_EXPANDING_VOLATILITY_TO_DATE'
        
        if recent_vol_col in df.columns and season_vol_col in df.columns:
            # Is player more volatile recently than usual?
            hot_cold_col = f'{stat}_RECENT_HIGH_VOLATILITY'
            df[hot_cold_col] = (
                df[recent_vol_col] > df[season_vol_col] * 1.2  # 20% more volatile than season average
            ).astype(int)
            
            # Extreme volatility flag
            extreme_vol_col = f'{stat}_EXTREME_VOLATILITY'
            volatility_95th = df[recent_vol_col].quantile(0.95)
            df[extreme_vol_col] = (df[recent_vol_col] > volatility_95th).astype(int)
    
    return df