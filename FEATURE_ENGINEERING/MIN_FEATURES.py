import pandas as pd
import numpy as np

def convert_min_to_float(min_str):
    """Convert minutes string (MM:SS) to float."""
    try:
        if isinstance(min_str, str) and ":" in min_str:
            minutes, seconds = map(int, min_str.split(":"))
            total_minutes = minutes + seconds / 60
            return round(total_minutes, 2)
        elif isinstance(min_str, (int, float)):
            return float(min_str)
        else:
            return 0
    except:
        return 0

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
    df['TEAM_DAYS_REST'] = team_groups.diff().dt.days
    df['PLAYER_DAYS_REST'] = player_groups.diff().dt.days
    
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

def MINrollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[3,5,7]):
    """Calculate rolling averages for key player statistics."""
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = [
        'MIN'
    ]

    # Compute rolling averages (leave NaNs untouched)
    for window in windows:
        for col in stats_cols:
            rolling_col_name = f'{col}_ROLLING_AVG_{window}'
            df[rolling_col_name] = df.groupby(player_id_col)[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

    # Compute Season Average up to Previous Game (leave NaNs untouched)
    for col in stats_cols:
        season_avg_col = f'{col}_SEASON_AVG'
        df[season_avg_col] = df.groupby(player_id_col)[col].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    # Compute Delta between Rolling Avg (window=3) and Season Avg
    for col in stats_cols:
        delta_col_name = f'{col}_DELTA_3_vs_SEASON'
        rolling_col_name = f'{col}_ROLLING_AVG_3'
        season_avg_col = f'{col}_SEASON_AVG'
        df[delta_col_name] = df[rolling_col_name] - df[season_avg_col]

    return df

def MINLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='MIN'):
    """Add lag features for specified statistic."""
    player_data = player_data.sort_values([player_id_col, date_col])
    
    for lag in range(1, 5):
        lag_col = f'MIN_LAG_{lag}'
        # Create Lag
        player_data[lag_col] = player_data.groupby(player_id_col)[stat_line].shift(lag)
        # Compute expanding mean up to current row, aligned correctly using transform
        rolling_mean = player_data.groupby(player_id_col)[stat_line].transform(lambda x: x.shift(1).expanding().mean())
        # Fill NaNs in lag with rolling mean
        player_data[lag_col] = player_data[lag_col].fillna(rolling_mean)
        # Fill remaining NaNs (e.g., first game) with 0
        player_data[lag_col] = player_data[lag_col].fillna(0)
    return player_data


def getPlayerMINAvgToDate(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Vectorized version that should be faster and avoid multi-index issues.
    """
    # Create copy and sort
    df_enhanced = df.copy().sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define stats
    stats_to_average = ['MIN']
    
    # Use transform to avoid multi-index issues
    for stat in stats_to_average:
        if stat in df_enhanced.columns:
            df_enhanced[f'MIN_AVG_TO_DATE'] = (
                df_enhanced.groupby(player_id_col)[stat]
                .transform(lambda x: x.expanding().mean().shift(1))
                .round(2)
            )
    return df_enhanced

def MINHomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate home/away rolling averages (expanding) for key metrics,
    prevent data leakage via shift(1), and fill missing values with the opposite location's current average.
    All results are rounded to 2 decimal places.
    """
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    metrics = ['MIN']
    global_means = df[metrics].mean()
    player_group = df.groupby(player_id_col)

    for location in ['HOME', 'AWAY']:
        loc_mask = df['HOME_GAME'] == (1 if location == 'HOME' else 0)

        for metric in metrics:
            if metric not in df.columns:
                continue

            col_name = f'PLAYER_{location}_AVG_{metric}'
            df[col_name] = np.nan

            for player_id in df[player_id_col].unique():
                player_mask = df[player_id_col] == player_id
                combined_mask = player_mask & loc_mask

                shifted = df.loc[combined_mask, metric].shift(1)
                expanding_mean = shifted.expanding().mean()

                df.loc[combined_mask, col_name] = expanding_mean

            # Fill first games with global mean
            first_games_mask = player_group.cumcount() == 0
            df.loc[first_games_mask, col_name] = global_means[metric]

    # Fill NaNs with the opposite-location average
    for metric in metrics:
        home_col = f'PLAYER_HOME_AVG_{metric}'
        away_col = f'PLAYER_AWAY_AVG_{metric}'

        df[home_col] = df[home_col].fillna(df[away_col])
        df[away_col] = df[away_col].fillna(df[home_col])

        df[home_col] = df[home_col].fillna(global_means[metric])
        df[away_col] = df[away_col].fillna(global_means[metric])

        # Cast to float32 then apply real rounding
        df[home_col] = df[home_col].astype('float32')
        df[home_col] = df[home_col].apply(lambda x: round(x, 2))

        df[away_col] = df[away_col].astype('float32')
        df[away_col] = df[away_col].apply(lambda x: round(x, 2))
    return df

def MINAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION', stat_line='MIN'):
    """
    Calculate matchup-specific statistics with optimized performance and additional metrics.
    Includes rolling averages for multiple windows with data leakage prevention.
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
    }
    
    # Calculate games against opponent count efficiently
    df['GAMES_VS_OPP'] = player_opp_group.cumcount() + 1
    
    # Vectorized operations for all rolling windows
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # Shift values first to prevent data leakage
        shifted_values = player_opp_group[metric].shift(1)
        
        for window in metrics[metric]:
            # Calculate rolling average with shifted values
            col_name = f'MATCHUP_AVG_{metric}_LAST_{window}'
            df[col_name] = (
                shifted_values
                .rolling(window=window, min_periods=1)
                .mean()
                .round(2)
            )
    
    # Fill NaN values efficiently for all new columns at once
    rolling_cols = [col for col in df.columns if 'MATCHUP_AVG_' in col]
    if rolling_cols:
        # Calculate global means for each metric once
        global_means = df[rolling_cols].mean()
        
        # Fill missing values: first with backward fill, then forward fill, finally with global means
        df[rolling_cols] = (
            df[rolling_cols]
            .bfill()
            .ffill()
            .fillna(global_means)
        )
    
    # Convert memory types to save space
    for col in rolling_cols:
        df[col] = df[col].astype('float32')  # Use float32 instead of float64 to save memory
    df['GAMES_VS_OPP'] = df['GAMES_VS_OPP'].astype('int8')
    return df

def getOpponentPace(df, team_abbreviation='LAL'):
    """Get unique team stats per game with season-to-date averages."""
    team_df = df[df['TEAM_ABBREVIATION'] == team_abbreviation].copy()
    team_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 
        'TEAM_PACE'
    ]
    
    available_team_cols = [col for col in team_cols if col in team_df.columns]
    unique_games = team_df[available_team_cols].drop_duplicates(subset=['GAME_ID'])
    unique_games = unique_games.sort_values('GAME_DATE')
    unique_games['PACE_AVG_TO_DATE'] = unique_games['TEAM_PACE'].expanding().mean().shift(1).round(2)
    output_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION',
        'TEAM_PACE',
        'PACE_AVG_TO_DATE'
    ]
    
    available_cols = [col for col in output_cols if col in unique_games.columns]
    
    return unique_games[available_cols].reset_index(drop=True)

def assign_opponent_team_stats_dict(df):
    """Assign opponent team stats using dictionary lookup for efficiency."""
    team_stats_dict = {}
    
    for team in df['TEAM_ABBREVIATION'].unique():
        team_stats = getOpponentPace(df, team)
        for _, row in team_stats.iterrows():
            key = (row['GAME_ID'], team)
            team_stats_dict[key] = {
                'OPP_PACE_AVG_TO_DATE': row['PACE_AVG_TO_DATE'],
            }
    
    # Assign opponent stats using vectorized lookup
    df_enhanced = df.copy()
    lookup_keys = list(zip(df_enhanced['GAME_ID'], df_enhanced['OPP_ABBREVIATION']))
    
    for col in ['OPP_PACE_AVG_TO_DATE']:
        df_enhanced[col] = [team_stats_dict.get(key, {}).get(col, None) for key in lookup_keys]
    return df_enhanced