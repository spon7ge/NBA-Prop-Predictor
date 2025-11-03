import pandas as pd
import numpy as np
import warnings

# Suppress performance and future warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================

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

def encode_teams_label(df):
    """Label encode teams for XGBoost - RECOMMENDED"""
    df_encoded = df.copy()
    
    # Create consistent encoding across all data
    all_teams = sorted(set(df['TEAM_ABBREVIATION'].unique()) | set(df['OPP_ABBREVIATION'].unique()))
    
    # Create mapping dictionaries
    team_to_label = {team: idx for idx, team in enumerate(all_teams)}
    
    # Apply encoding
    df_encoded['TEAM_ENCODED'] = df_encoded['TEAM_ABBREVIATION'].map(team_to_label)
    df_encoded['OPP_ENCODED'] = df_encoded['OPP_ABBREVIATION'].map(team_to_label)
    
    # Drop original columns
    df_encoded = df_encoded.drop(['TEAM_ABBREVIATION', 'OPP_ABBREVIATION'], axis=1)
    
    return df_encoded, team_to_label
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

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[3,5,7,10,15,25,40]):
    """Calculate rolling averages for key player statistics only."""
    df = player_data.copy()
    
    # Reset index to ensure unique indices
    df.reset_index(drop=True, inplace=True)
    
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = [ 'PTS', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT', 'TOV', 'TS_PCT', 'USG_PCT','MIN', 'PACE', 'PIE', 
                'E_OFF_RATING', 'NET_RATING', 'TCHS', 'POSS', 'EFG_PCT',
                'percentagePointsPaint', 'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFreeThrow', 'CFGA', 'UFGA' ]

    for window in windows:
        for col in stats_cols:
            if col in df.columns:
                rolling_col_name = f'{col}_ROLLING_AVG_{window}'
                
                # Calculate rolling average
                df[rolling_col_name] = df.groupby(player_id_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2)
                )

    return df

def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    # Reset index to ensure unique indices
    player_data.reset_index(drop=True, inplace=True)
    
    player_data = player_data.sort_values([player_id_col, date_col])
    stats_lines = [ 'PTS', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT', 'TOV', 'TS_PCT', 'USG_PCT','MIN', 'PACE', 'PIE', 'E_OFF_RATING', 'NET_RATING', 'TCHS', 'POSS', 'EFG_PCT',
                'percentagePointsPaint', 'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFreeThrow', 'CFGA', 'UFGA' ]
    for stat_line in stats_lines:
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

def add_trend_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', window=5):
    """Add trend features based on linear regression slope of last N games.
    
    Calculates the slope (trend) of various metrics over the last N games.
    Positive slope = trending up, negative slope = trending down.
    
    Args:
        df: DataFrame with player data
        player_id_col: Column name for player identifier
        date_col: Column name for date
        window: Number of games to use for trend calculation
        
    Returns:
        DataFrame with trend features added
    """
    df = df.copy()
    df = df.sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Metrics to calculate trends for
    trend_metrics = ['PTS', 'MIN']
    
    # Only process metrics that exist in the dataframe
    available_metrics = [m for m in trend_metrics if m in df.columns]
    
    for metric in available_metrics:
        trend_col = f'{metric}_TREND_LAST_{window}'
        
        # Calculate rolling slope for each player group
        def calculate_player_trend(group_metric):
            """Calculate slope for rolling window of last N games."""
            # Create results array same length as input
            results = np.full(len(group_metric), np.nan)
            
            for i in range(window, len(group_metric)):
                # Get last window values
                window_values = group_metric.iloc[i-window:i].values
                
                # Remove NaN
                clean_values = window_values[~np.isnan(window_values)]
                
                if len(clean_values) < 2:
                    results[i] = 0
                    continue
                
                # Create x axis
                x = np.arange(len(clean_values))
                y = clean_values
                
                # Calculate slope: (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
                n = len(x)
                if n < 2 or np.var(x) == 0:
                    results[i] = 0
                    continue
                
                slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
                results[i] = slope
            
            return pd.Series(results, index=group_metric.index)
        
        # Apply trend calculation grouped by player with shifted values
        df[trend_col] = (df.groupby(player_id_col)[metric]
                        .shift(1)
                        .groupby(df[player_id_col])
                        .apply(calculate_player_trend)
                        .reset_index(level=0, drop=True))
        
        # Fill NaN values with 0 (no trend)
        df[trend_col] = df[trend_col].fillna(0).round(3)
        
    return df


def add_player_role_tier_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    df = df.copy()
    df = df.sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Check if required columns exist
    if 'PTS_AVG_TO_DATE' not in df.columns:
        print("Warning: PTS_AVG_TO_DATE not found, skipping role tier features")
        return df
    
    starter_threshold = 20.0
    role_threshold = 10.0
    
    # Create categorical role tier
    def assign_role_tier(pts_avg):
        if pd.isna(pts_avg):
            return 'Unknown'
        elif pts_avg >= starter_threshold:
            return 'Starter'
        elif pts_avg >= role_threshold and pts_avg < starter_threshold:
            return 'Role'
        else:
            return 'Bench'
    
    df['PLAYER_ROLE_TIER'] = df['PTS_AVG_TO_DATE'].apply(assign_role_tier)
    
    # Create binary flags for each tier (useful for interactions)
    df['IS_STARTER_TIER'] = (df['PLAYER_ROLE_TIER'] == 'Starter').astype(int)
    df['IS_ROLE_TIER'] = (df['PLAYER_ROLE_TIER'] == 'Role').astype(int)
    df['IS_BENCH_TIER'] = (df['PLAYER_ROLE_TIER'] == 'Bench').astype(int)
    
    return df


def getPlayerAvgToDateVectorized(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Vectorized version that should be faster and avoid multi-index issues.
    FIXED: Properly shifted to prevent data leakage.
    """
    # Create copy and sort
    df_enhanced = df.copy().sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define stats
    stats_cols = [ 'PTS', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT', 'TOV', 'TS_PCT', 'USG_PCT','MIN', 'PACE', 'PIE', 'E_OFF_RATING', 'NET_RATING', 'TCHS', 'POSS', 'EFG_PCT',
                'percentagePointsPaint', 'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFreeThrow', 'CFGA', 'UFGA' ]
    for stat in stats_cols:
        if stat in df_enhanced.columns:
            df_enhanced[f'{stat}_AVG_TO_DATE'] = (
                df_enhanced.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).expanding().mean())
                .round(2)
            )
        else:
            print(f"Column {stat} not found in dataframe")
    
    # Add games played counter
    df_enhanced['GAMES_PLAYED_TO_DATE'] = (
        df_enhanced.groupby(player_id_col).cumcount()
    )
    return df_enhanced

# ================================================================================================
# HOME/AWAY AND MATCHUP SPECIFIC FEATURES - FIXED FOR DATA LEAKAGE
# ================================================================================================

def HomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    df = player_data.copy()
    
    # Reset index to ensure unique indices and avoid reindexing issues
    df.reset_index(drop=True, inplace=True)
    
    df.sort_values([player_id_col, date_col], inplace=True)
    
    if 'HOME_GAME' not in df.columns:
        return df

    metrics = [ 'PTS', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT', 'TOV', 'TS_PCT', 'USG_PCT','MIN', 'PACE', 'PIE', 'E_OFF_RATING', 'NET_RATING', 'TCHS', 'POSS', 'EFG_PCT',
                'percentagePointsPaint', 'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFreeThrow', 'CFGA', 'UFGA' ]
    
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

        home_col = f'PLAYER_HOME_AVG_{metric}_TO_DATE'
        away_col = f'PLAYER_AWAY_AVG_{metric}_TO_DATE'
        
        # Calculate deltas
        home_delta_col = f'PLAYER_HOME_{metric}_DELTA'
        away_delta_col = f'PLAYER_AWAY_{metric}_DELTA'

        df[home_col] = np.where(df['HOME_GAME'] == 1, loc_avg, np.nan)
        df[away_col] = np.where(df['HOME_GAME'] == 0, loc_avg, np.nan)

        df.loc[first_game_mask, home_col] = global_means[metric]
        df.loc[first_game_mask, away_col] = global_means[metric]

        df[home_col] = df[home_col].fillna(overall_avg)
        df[away_col] = df[away_col].fillna(overall_avg)

        df[home_col] = df[home_col].fillna(global_means[metric]).astype('float32').round(2)
        df[away_col] = df[away_col].fillna(global_means[metric]).astype('float32').round(2)
        
        # Calculate deltas (home/away performance - overall performance)
        df[home_delta_col] = (df[home_col] - overall_avg).round(2)
        df[away_delta_col] = (df[away_col] - overall_avg).round(2)
        
        # Fill delta NaN values with 0 (no difference from average)
        df[home_delta_col] = df[home_delta_col].fillna(0)
        df[away_delta_col] = df[away_delta_col].fillna(0)
    
    # Add interaction features for PTS only (can extend to other metrics if needed)
    if 'PTS' in metrics:
        # Calculate baseline using shifted expanding mean to prevent leakage
        pts_overall_avg = shifted_expanding_mean(df['PTS'], df[player_id_col])
        df['PTS_BASELINE'] = pts_overall_avg.fillna(global_means['PTS']).astype('float32')
        
        # Calculate home/away multipliers (ratio of home to away performance)
        # Add epsilon to avoid division by zero
        eplison = 1e-8
        df['HOME_PTS_MULTIPLIER'] = (df['PLAYER_HOME_AVG_PTS_TO_DATE'] / (df['PLAYER_AWAY_AVG_PTS_TO_DATE'] + eplison)).fillna(1.0).astype('float32')
        
        # Create interaction term: baseline * home_game * multiplier
        df['PTS_BASELINE_x_HOME'] = (df['PTS_BASELINE'] * df['HOME_GAME'] * df['HOME_PTS_MULTIPLIER']).astype('float32')

    return df



def statAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION', stat_line='PTS'):
    # Create copy to avoid modifying original
    df = player_data.copy()
    
    # Reset index to ensure unique indices
    df.reset_index(drop=True, inplace=True)
    
    # Pre-sort once for all operations
    df.sort_values([player_id_col, 'GAME_DATE'], inplace=True)
    
    # Create player-opponent grouper object once (reuse for efficiency)
    player_opp_group = df.groupby([player_id_col, opp_col])
    
    # Define metrics to track with their windows
    metrics = {
        'MIN': [3,5],
        'FGA': [3,5],
        'FG3A': [3,5],
        'FTA': [3,5],
        'PTS': [3,5],
        'USG_PCT': [3,5],
        'EFG_PCT': [3,5],
        'TS_PCT': [3,5],
        'TCHS': [3,5],
        'PF': [3,5],
    }
    
    # Calculate games against opponent count efficiently
    df['GAMES_VS_OPP'] = player_opp_group.cumcount() + 1
    
    # First, calculate overall averages for each player
    for metric in metrics:
        if metric in df.columns:
            df[f'{metric}_AVG_TO_DATE'] = (
                df.groupby(player_id_col)[metric]
                .transform(lambda x: x.shift(1).expanding().mean())
                .round(2)
            )
    
    # Vectorized operations for all rolling windows - FIXED: Always shift by 1
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # Shift values first to prevent data leakage
        shifted_values = player_opp_group[metric].shift(1)
        
        for window in metrics[metric]:
            # Calculate rolling average with shifted values
            matchup_avg_col = f'MATCHUP_AVG_{metric}_LAST_{window}_TO_DATE'
            df[matchup_avg_col] = (
                shifted_values
                .rolling(window=window, min_periods=1)
                .mean()
                .round(2)
            )
            
            # Calculate delta (matchup performance - overall performance)
            delta_col = f'MATCHUP_{metric}_DELTA_LAST_{window}'
            df[delta_col] = (
                df[matchup_avg_col] - df[f'{metric}_AVG_TO_DATE']
            ).round(2)
    
    # Fill NaN values efficiently for all new columns at once
    rolling_cols = [col for col in df.columns if 'MATCHUP_AVG_' in col and '_TO_DATE' in col]
    delta_cols = [col for col in df.columns if 'MATCHUP_' in col and '_DELTA_' in col]
    
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
    
    # Fill delta columns with 0 (no difference from average)
    if delta_cols:
        df[delta_cols] = df[delta_cols].fillna(0).round(2)
    
    # Convert memory types to save space
    for col in rolling_cols + delta_cols:
        df[col] = df[col].astype('float32')
    
    df['GAMES_VS_OPP'] = df['GAMES_VS_OPP'].astype('int8')
    
    return df
def assign_team_opp_def_by_position(df, min_minutes=15):
    # Define defensive columns that actually exist in your dataset
    def_cols = [
        'E_DEF_RATING',
        'DEF_FG_PCT_ALLOWED', 
        'DEF_3PT_PCT_ALLOWED', 
        'PTS_ALLOWED_PER_MIN'
    ]
    
    # Check which columns actually exist
    available_def_cols = [col for col in def_cols if col in df.columns]
    
    if not available_def_cols:
        print("Warning: No defensive columns found in dataset")
        return df
    
    positions = ['GUARD', 'FORWARD', 'CENTER']
    team_def_list = []

    # Filter for players with sufficient minutes
    df_filtered = df[df['MIN'] >= min_minutes].copy()
    df_filtered = df_filtered.sort_values(['TEAM_ID', 'GAME_DATE'])

    for pos in positions:
        # Get players at this position
        pos_data = df_filtered[df_filtered[pos] == 1].copy()
        
        if pos_data.empty:
            continue
            
        # Shift defensive stats to prevent data leakage
        pos_data[available_def_cols] = pos_data.groupby('TEAM_ID')[available_def_cols].shift(1)
        
        # Calculate team defensive averages by position for each game
        tmp = (
            pos_data
            .groupby(['TEAM_ID', 'GAME_ID'])[available_def_cols]
            .mean()
            .round(3)
            .reset_index()
        )
        
        # Rename columns dynamically based on what's available
        rename_dict = {}
        for col in available_def_cols:
            if col == 'E_DEF_RATING':
                rename_dict[col] = f'TEAM_{pos}_DEF_RATING'
            elif col == 'DEF_FG_PCT_ALLOWED':
                rename_dict[col] = f'TEAM_{pos}_DEF_FG_PCT_ALLOWED'
            elif col == 'DEF_3PT_PCT_ALLOWED':
                rename_dict[col] = f'TEAM_{pos}_DEF_3PT_PCT_ALLOWED'
            elif col == 'PTS_ALLOWED_PER_MIN':
                rename_dict[col] = f'TEAM_{pos}_PTS_ALLOWED_PER_MIN'
        
        tmp = tmp.rename(columns=rename_dict)
        team_def_list.append(tmp)

    if not team_def_list:
        return df

    # Merge all position-based team stats
    team_def = team_def_list[0]
    for tmp in team_def_list[1:]:
        team_def = team_def.merge(tmp, on=['TEAM_ID', 'GAME_ID'], how='outer')
    
    # Merge with main dataframe
    df = df.merge(team_def, on=['TEAM_ID', 'GAME_ID'], how='left')
    
    # Create opponent versions
    opp_def = team_def.rename(columns={
        'TEAM_ID': 'OPP_TEAM_ID',
        **{col: col.replace('TEAM_', 'OPP_') for col in team_def.columns if col not in ['TEAM_ID', 'GAME_ID']}
    })
    df = df.merge(opp_def, on=['OPP_TEAM_ID', 'GAME_ID'], how='left')
    
    return df

def get_rolling_opp_def_by_position(df, min_minutes=15, windows=[5, 10, 15]):
    """
    Calculate rolling opponent defensive ratings by position with proper shifting.
    """
    df_enhanced = df.copy()
    
    # Filter for players with sufficient minutes
    df_filtered = df[df['MIN'] >= min_minutes].copy()
    df_filtered = df_filtered.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    # Define defensive columns
    def_cols = [
        'E_DEF_RATING',
        'DEF_FG_PCT_ALLOWED', 
        'DEF_3PT_PCT_ALLOWED', 
        'PTS_ALLOWED_PER_MIN',
        'DEF_TOV_FORCED_PER_MIN',
        'DEF_BLOCKS_PER_MIN',
        'DEF_SHOOTING_FOULS_PER_MIN',
        'DEF_AST_ALLOWED_PER_MIN'
    ]
    
    for pos in ['GUARD', 'FORWARD', 'CENTER']:
        pos_data = df_filtered[df_filtered[pos] == 1].copy()
        
        if pos_data.empty:
            continue
            
        # CRITICAL: Shift defensive stats first
        pos_data[def_cols] = pos_data.groupby('TEAM_ID')[def_cols].shift(1)
        
        # Calculate rolling defensive averages
        for window in windows:
            rolling_stats = pos_data.groupby('TEAM_ID').rolling(
                window=window, 
                on='GAME_DATE'
            )[def_cols].mean().reset_index()
            
            # Rename columns
            rename_dict = {col: f'TEAM_{pos}_{col}_{window}G' for col in def_cols}
            rolling_stats = rolling_stats.rename(columns=rename_dict)
            
            # Merge with main dataframe
            df_enhanced = df_enhanced.merge(rolling_stats, on=['TEAM_ID', 'GAME_DATE'], how='left')
    
    return df_enhanced

def teamRollingDefenseByPosition(df, team_id_col='TEAM_ID', date_col='GAME_DATE', windows=[3,5,7,10]):
    """Calculate rolling team defensive averages by position over fixed windows."""
    data = df.copy()
    data.sort_values([team_id_col, date_col], inplace=True)

    def_cols = [
        'DEF_FG_PCT_ALLOWED',
        'DEF_3PT_PCT_ALLOWED',
        'PTS_ALLOWED_PER_MIN'
    ]
    positions = ['GUARD', 'FORWARD', 'CENTER']

    team_def_list = []

    for pos in positions:
        # Compute team-level defense per game for this position
        team_pos_def = (
            data[data[pos] == 1]
            .groupby([team_id_col, 'GAME_ID'])[def_cols]
            .mean()
            .reset_index()
        )

        # Calculate rolling averages by team for this position
        for window in windows:
            for col in def_cols:
                roll_col = f'TEAM_{pos}_{col}_ROLLING_AVG_{window}'
                team_pos_def[roll_col] = team_pos_def.groupby(team_id_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=window).mean().round(3)
                )

        # Keep only the rolling average columns and the merge keys
        rolling_cols = [col for col in team_pos_def.columns if 'ROLLING_AVG' in col]
        team_pos_def = team_pos_def[[team_id_col, 'GAME_ID'] + rolling_cols]
        team_def_list.append(team_pos_def)

    # Merge position-level results together
    team_def = team_def_list[0]
    for tmp in team_def_list[1:]:
        team_def = team_def.merge(tmp, on=[team_id_col, 'GAME_ID'], how='outer')

    # Merge back to player-level data
    data = data.merge(team_def, on=[team_id_col, 'GAME_ID'], how='left')
    
    # Calculate OPPONENT rolling averages properly
    # First, we need to create opponent-specific rolling averages
    opp_def_list = []
    
    for pos in positions:
        # Compute opponent team-level defense per game for this position
        opp_pos_def = (
            data[data[pos] == 1]
            .groupby(['OPP_TEAM_ID', 'GAME_ID', date_col])[def_cols]
            .mean()
            .reset_index()
        )
        
        # Sort by opponent team and date to ensure proper rolling calculation
        opp_pos_def = opp_pos_def.sort_values(['OPP_TEAM_ID', date_col])
        
        # Calculate rolling averages by opponent team for this position
        for window in windows:
            for col in def_cols:
                roll_col = f'OPP_{pos}_{col}_ROLLING_AVG_{window}'
                opp_pos_def[roll_col] = opp_pos_def.groupby('OPP_TEAM_ID')[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=window).mean().round(3)
                )
        
        # Keep only the rolling average columns and the merge keys
        rolling_cols = [col for col in opp_pos_def.columns if 'ROLLING_AVG' in col]
        opp_pos_def = opp_pos_def[['OPP_TEAM_ID', 'GAME_ID'] + rolling_cols]
        opp_def_list.append(opp_pos_def)
    
    # Merge opponent position-level results together
    opp_def = opp_def_list[0]
    for tmp in opp_def_list[1:]:
        opp_def = opp_def.merge(tmp, on=['OPP_TEAM_ID', 'GAME_ID'], how='outer')
    
    # Merge opponent rolling averages back to player-level data
    data = data.merge(opp_def, on=['OPP_TEAM_ID', 'GAME_ID'], how='left')
    
    return data

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
            team_strength = historical_data.groupby('OPP_ABBREVIATION')['OPP_DEF_RATING_AVG_TO_DATE'].mean()
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
    df['TEAM_FG3A_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FG3A'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_FTA_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FTA'].transform(
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

def add_opponent_team_rolling_stats(df, team_id_col='TEAM_ID', date_col='GAME_DATE', windows=[3,5,7,10]):
    """
    Add rolling averages for opponent team statistics over specified windows.
    Shows how the opposing team has been performing in their recent games.
    """
    df = df.copy()
    df = df.sort_values([team_id_col, date_col]).reset_index(drop=True)
    
    # Define team stats to calculate rolling averages for
    team_stats = [
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_OFF_RATING', 'TEAM_PTS', 
        'TEAM_FGA', 'TEAM_REB', 'TEAM_AST', 'TEAM_TOV', 'TEAM_BLK', 'TEAM_STL'
    ]
    
    # Filter to only available columns
    available_stats = [stat for stat in team_stats if stat in df.columns]
    
    if not available_stats:
        print("Warning: No team stats found in dataframe for rolling averages")
        return df
    
    # Calculate rolling averages for each window
    for window in windows:
        for stat in available_stats:
            # Rolling average (shifted to prevent leakage)
            rolling_col = f'{stat}_ROLLING_AVG_{window}'
            df[rolling_col] = (
                df.groupby(team_id_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
                .round(2)
            )
    
    # Now create opponent versions by mapping team stats to opponent team stats
    # First, create a mapping of team rolling stats per game
    team_rolling_stats = {}
    
    for window in windows:
        for stat in available_stats:
            rolling_col = f'{stat}_ROLLING_AVG_{window}'
            opp_rolling_col = f'OPP_{stat}_ROLLING_AVG_{window}'
            
            # Create mapping: (GAME_ID, TEAM_ID) -> rolling stat value
            team_game_stats = df.groupby(['GAME_ID', team_id_col])[rolling_col].first().to_dict()
            
            # Map opponent team rolling stats
            df[opp_rolling_col] = df.apply(
                lambda row: team_game_stats.get((row['GAME_ID'], row['OPP_TEAM_ID']), np.nan), 
                axis=1
            )
    
    # Fill NaN values with expanding averages as fallback
    opp_rolling_cols = [col for col in df.columns if col.startswith('OPP_') and '_ROLLING_AVG_' in col]
    
    for col in opp_rolling_cols:
        # Extract the base stat name
        base_stat = col.replace('OPP_', '').replace('_ROLLING_AVG_5', '').replace('_ROLLING_AVG_10', '').replace('_ROLLING_AVG_15', '')
        base_stat_col = f'OPP_{base_stat}_AVG_TO_DATE'
        
        # Fill NaN with expanding average if available
        if base_stat_col in df.columns:
            df[col] = df[col].fillna(df[base_stat_col])
        else:
            # Fill with overall mean as last resort
            df[col] = df[col].fillna(df[col].mean())
    
    # Convert to appropriate data types to save memory
    for col in opp_rolling_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df


def add_opponent_team_form_indicators(df, windows=[3,5,7,10]):
    """
    Add indicators showing if opponent team is in good/bad form recently.
    Compares recent performance to season averages.
    """
    df = df.copy()
    
    # Key stats to analyze for team form
    form_stats = ['TEAM_DEF_RATING', 'TEAM_OFF_RATING', 'TEAM_PTS']
    
    for window in windows:
        for stat in form_stats:
            rolling_col = f'OPP_{stat}_ROLLING_AVG_{window}'
            season_col = f'OPP_{stat}_AVG_TO_DATE'
            
            if rolling_col in df.columns and season_col in df.columns:
                # Form indicator: 1 if recent form is better than season average
                form_col = f'OPP_{stat}_GOOD_FORM_{window}'
                
                if 'DEF_RATING' in stat:
                    # For defense, lower is better
                    df[form_col] = (df[rolling_col] < df[season_col] * 0.98).astype(int)
                else:
                    # For offense/points, higher is better
                    df[form_col] = (df[rolling_col] > df[season_col] * 1.02).astype(int)
                
                # Strength of form (percentage difference)
                form_strength_col = f'OPP_{stat}_FORM_STRENGTH_{window}'
                if 'DEF_RATING' in stat:
                    df[form_strength_col] = ((df[season_col] - df[rolling_col]) / df[season_col] * 100).round(2)
                else:
                    df[form_strength_col] = ((df[rolling_col] - df[season_col]) / df[season_col] * 100).round(2)
    
    return df

def expectedPace(df):
    df = df.copy()
    df['EXPECTED_PACE'] = ((df['TEAM_PACE_AVG_TO_DATE'] + df['OPP_PACE_AVG_TO_DATE']) / 2).round(2)
    df['EXPECTED_PACE_DIFF'] = df['TEAM_PACE_AVG_TO_DATE'] - df['OPP_PACE_AVG_TO_DATE']    
    df['EXPECTED_POINTS'] = (df['TEAM_PTS_AVG_TO_DATE'] + df['OPP_PTS_AVG_TO_DATE']) / 2
    return df

def calculate_game_implied_pace(df):
    df = df.copy()
    
    # Use expected pace as base, adjust for total
    expected_pace = (df['TEAM_PACE_AVG_TO_DATE'] + df['OPP_PACE_AVG_TO_DATE']) / 2
    
    # Adjust based on how total compares to team averages
    team_total_avg = (df['TEAM_PTS_AVG_TO_DATE'] + df['OPP_PTS_AVG_TO_DATE'])
    total_ratio = df['total'] / team_total_avg
    
    df['GAME_IMPLIED_PACE'] = (expected_pace * total_ratio).round(1)
    
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
def process_star_players_data(df, min_minutes=10):
    df = df.copy()
    # Create ACTIVE column based on minutes played
    df['ACTIVE'] = (df['MIN'] >= min_minutes).astype(int)

    # Season-long team star by composite score (only among active players)
    active_players = df[df['ACTIVE'] == 1].copy()
    
    # Calculate mean stats per player per team
    player_stats = (
        active_players.groupby(['TEAM_ID', 'PLAYER_NAME'], dropna=False)
        .agg({
            'USG_PCT': 'mean',
            'TS_PCT': 'mean',
            'EFG_PCT': 'mean',
            'PTS': 'mean',
            'PIE': 'mean',  # Player Impact Estimate
            'NET_RATING': 'mean',
        })
        .reset_index()
    )
    
    # Fill NaN values with 0 for missing metrics
    player_stats = player_stats.fillna(0)
    
    # Normalize metrics within each team (0-1 scale per team)
    normalized_stats = player_stats.copy()
    
    for stat in ['USG_PCT', 'TS_PCT', 'EFG_PCT', 'PTS', 'PIE', 'NET_RATING']:
        # Group by team and normalize
        normalized_stats[f'{stat}_NORM'] = (
            player_stats.groupby('TEAM_ID')[stat]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0)
        )
    
    # Calculate composite star score with weighted metrics
    # Weights prioritize usage, efficiency, and scoring
    normalized_stats['STAR_SCORE'] = (
        0.25 * normalized_stats['USG_PCT_NORM'] +      # Usage - how involved they are
        0.20 * normalized_stats['TS_PCT_NORM'] +       # True shooting - efficiency
        0.15 * normalized_stats['EFG_PCT_NORM'] +      # Effective FG% - shooting efficiency
        0.20 * normalized_stats['PTS_NORM'] +          # Points - scoring volume
        0.15 * normalized_stats['PIE_NORM'] +          # Player impact
        0.05 * normalized_stats['NET_RATING_NORM']     # Net rating
    )
    
    # Select highest scoring player per team as star
    star_rows = (
        normalized_stats.sort_values(['TEAM_ID', 'STAR_SCORE'], ascending=[True, False])
        .groupby(['TEAM_ID'], as_index=False)
        .first()
    )
    
    star_by_team = {
        row.TEAM_ID: row.PLAYER_NAME
        for _, row in star_rows.iterrows()
    }

    df['STAR_NAME'] = df['TEAM_ID'].map(star_by_team)
    df['PLAYER_IS_TEAM_STAR'] = (df['PLAYER_NAME'] == df['STAR_NAME']).astype(int)

    star_active_per_game = (
        df[df['PLAYER_NAME'] == df['STAR_NAME']]
        .groupby(['GAME_ID', 'TEAM_ID'], as_index=False)['ACTIVE']
        .max()
        .rename(columns={'ACTIVE': 'STAR_ACTIVE'})
    )
    df = df.merge(star_active_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['STAR_ACTIVE'] = df['STAR_ACTIVE'].fillna(0).astype(int)
    df['STAR_SAT_OUT'] = ((df['PLAYER_IS_TEAM_STAR'] == 0) & (df['STAR_ACTIVE'] == 0)).astype(int)

    df = df.drop(columns=['STAR_NAME', 'STAR_ACTIVE', 'ACTIVE'])

    return df

def add_usual_starters_availability(df, min_minutes=10, lookback_games=20):
    """Calculate the number of usual starters who are available for each game."""
    df = df.copy()
    df = df.sort_values(['TEAM_ID', 'GAME_DATE', 'PLAYER_NAME']).reset_index(drop=True)
    
    # Check required columns exist
    required_cols = ['GAME_ID', 'TEAM_ID', 'PLAYER_NAME', 'MIN', 'GAME_DATE']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns. Need: {required_cols}")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    # Create indicator for players who played (sufficient minutes)
    df['PLAYED'] = (df['MIN'] >= min_minutes).astype(int)
    
    # For each game, determine who started (based on START_POSITION if available, 
    # otherwise use first 5 players by minutes)
    if 'START_POSITION' in df.columns:
        df['STARTED'] = df['START_POSITION'].notna().astype(int)
    else:
        # If no START_POSITION, approximate by top 5 players by minutes per team per game
        df['STARTED'] = df.groupby(['GAME_ID', 'TEAM_ID']).apply(
            lambda x: pd.Series(
                (x['MIN'].rank(ascending=False, method='min') <= 5).astype(int).values,
                index=x.index
            )
        ).reset_index(drop=True)
    
    # Calculate rolling window of games to determine "usual starters"
    # We'll use a lookback window approach
    df['USUAL_STARTERS_AVAILABLE'] = 0
    df['USUAL_STARTERS_OUT'] = 0
    
    # Get unique games per team for efficient processing
    unique_games = df.groupby(['GAME_ID', 'TEAM_ID']).first().reset_index()[['GAME_ID', 'TEAM_ID', 'GAME_DATE']]
    unique_games = unique_games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    
    # For each team, process games chronologically
    for team_id in df['TEAM_ID'].unique():
        team_games = df[df['TEAM_ID'] == team_id].copy()
        team_games = team_games.sort_values(['GAME_DATE', 'PLAYER_NAME']).reset_index(drop=True)
        
        # Get unique game IDs for this team in chronological order
        team_unique_games = unique_games[unique_games['TEAM_ID'] == team_id].copy()
        team_unique_games = team_unique_games.sort_values('GAME_DATE').reset_index(drop=True)
        
        # For each game in this team's schedule
        for idx in range(len(team_unique_games)):
            current_game_id = team_unique_games.iloc[idx]['GAME_ID']
            
            # Look back at previous games to determine usual starters
            lookback_start = max(0, idx - lookback_games)
            historical_game_ids = team_unique_games.iloc[lookback_start:idx]['GAME_ID'].tolist()
            historical_games = team_games[team_games['GAME_ID'].isin(historical_game_ids)]
            
            if len(historical_games) == 0:
                # First game(s) - can't determine usual starters yet
                mask = df['GAME_ID'] == current_game_id
                df.loc[mask, 'USUAL_STARTERS_AVAILABLE'] = 5
                df.loc[mask, 'USUAL_STARTERS_OUT'] = 0
                continue
            
            # Count how many times each player started in historical games
            historical_starters = historical_games.groupby('PLAYER_NAME')['STARTED'].sum().reset_index()
            historical_starters.columns = ['PLAYER_NAME', 'START_COUNT']
            historical_starters = historical_starters.sort_values('START_COUNT', ascending=False)
            
            # Identify usual starters (top 5 by start count)
            if len(historical_starters) >= 5:
                usual_starters = set(historical_starters.head(5)['PLAYER_NAME'].tolist())
            else:
                # If less than 5 historical starters, use all of them
                usual_starters = set(historical_starters['PLAYER_NAME'].tolist())
                # If still no historical starters, use top 5 by average minutes
                if len(usual_starters) == 0:
                    avg_mins = historical_games.groupby('PLAYER_NAME')['MIN'].mean().reset_index()
                    avg_mins = avg_mins.sort_values('MIN', ascending=False)
                    usual_starters = set(avg_mins.head(5)['PLAYER_NAME'].tolist())
            
            # For the current game, count how many usual starters played
            current_game_players = team_games[team_games['GAME_ID'] == current_game_id]
            usual_starters_in_game = current_game_players[current_game_players['PLAYER_NAME'].isin(usual_starters)]
            
            usual_starters_available = usual_starters_in_game['PLAYED'].sum()
            usual_starters_out = len(usual_starters) - usual_starters_available
            
            # Update the dataframe
            mask = df['GAME_ID'] == current_game_id
            df.loc[mask, 'USUAL_STARTERS_AVAILABLE'] = usual_starters_available
            df.loc[mask, 'USUAL_STARTERS_OUT'] = usual_starters_out
    
    # Remove temporary columns
    if 'STARTED' in df.columns:
        df = df.drop(columns=['PLAYED', 'STARTED'])
    else:
        df = df.drop(columns=['PLAYED'])
    
    return df

def add_opponent_usual_starters_availability(df, min_minutes=10, lookback_games=20):
    """Calculate the number of usual opponent starters who are available for each game."""
    df = df.copy()
    df = df.sort_values(['TEAM_ID', 'GAME_DATE', 'PLAYER_NAME']).reset_index(drop=True)
    
    # Check required columns exist
    required_cols = ['GAME_ID', 'TEAM_ID', 'PLAYER_NAME', 'MIN', 'GAME_DATE']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns. Need: {required_cols}")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    # Create indicator for players who played (sufficient minutes)
    df['PLAYED'] = (df['MIN'] >= min_minutes).astype(int)
    
    # For each game, determine who started (based on START_POSITION if available, 
    # otherwise use first 5 players by minutes)
    if 'START_POSITION' in df.columns:
        df['STARTED'] = df['START_POSITION'].notna().astype(int)
    else:
        # If no START_POSITION, approximate by top 5 players by minutes per team per game
        df['STARTED'] = df.groupby(['GAME_ID', 'TEAM_ID']).apply(
            lambda x: pd.Series(
                (x['MIN'].rank(ascending=False, method='min') <= 5).astype(int).values,
                index=x.index
            )
        ).reset_index(drop=True)
    
    # Calculate rolling window of games to determine "usual starters" for opponents
    # We'll use a lookback window approach
    df['OPP_USUAL_STARTERS_AVAILABLE'] = 0
    df['OPP_USUAL_STARTERS_OUT'] = 0
    
    # Get unique games per team for efficient processing
    unique_games = df.groupby(['GAME_ID', 'TEAM_ID']).first().reset_index()[['GAME_ID', 'TEAM_ID', 'GAME_DATE']]
    unique_games = unique_games.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)
    
    # For each team, process games chronologically to determine their usual starters
    for team_id in df['TEAM_ID'].unique():
        team_games = df[df['TEAM_ID'] == team_id].copy()
        team_games = team_games.sort_values(['GAME_DATE', 'PLAYER_NAME']).reset_index(drop=True)
        
        # Get unique game IDs for this team in chronological order
        team_unique_games = unique_games[unique_games['TEAM_ID'] == team_id].copy()
        team_unique_games = team_unique_games.sort_values('GAME_DATE').reset_index(drop=True)
        
        # For each game in this team's schedule
        for idx in range(len(team_unique_games)):
            current_game_id = team_unique_games.iloc[idx]['GAME_ID']
            
            # Get the opponent team ID for this game
            opp_team_mask = df['GAME_ID'] == current_game_id
            if not opp_team_mask.any():
                continue
                
            # Get opponent team ID (the other team in this game)
            game_teams = df[opp_team_mask]['TEAM_ID'].unique()
            if len(game_teams) != 2:
                continue
            
            opp_team_id = game_teams[game_teams != team_id][0]
            
            # Get opponent's games data
            opp_games = df[df['TEAM_ID'] == opp_team_id].copy()
            opp_games = opp_games.sort_values(['GAME_DATE', 'PLAYER_NAME']).reset_index(drop=True)
            
            # Get unique game IDs for opponent in chronological order
            opp_unique_games = unique_games[unique_games['TEAM_ID'] == opp_team_id].copy()
            opp_unique_games = opp_unique_games.sort_values('GAME_DATE').reset_index(drop=True)
            
            # Find which opponent game index this current_game_id represents
            opp_game_indices = opp_unique_games.index[opp_unique_games['GAME_ID'] == current_game_id].tolist()
            if not opp_game_indices:
                continue
            opp_game_idx = opp_game_indices[0]
            
            # Look back at previous games to determine opponent's usual starters
            opp_lookback_start = max(0, opp_game_idx - lookback_games)
            opp_historical_game_ids = opp_unique_games.iloc[opp_lookback_start:opp_game_idx]['GAME_ID'].tolist()
            opp_historical_games = opp_games[opp_games['GAME_ID'].isin(opp_historical_game_ids)]
            
            if len(opp_historical_games) == 0:
                # First game(s) for opponent - can't determine usual starters yet
                mask = (df['GAME_ID'] == current_game_id) & (df['TEAM_ID'] == team_id)
                df.loc[mask, 'OPP_USUAL_STARTERS_AVAILABLE'] = 5
                df.loc[mask, 'OPP_USUAL_STARTERS_OUT'] = 0
                continue
            
            # Count how many times each player started in opponent's historical games
            opp_historical_starters = opp_historical_games.groupby('PLAYER_NAME')['STARTED'].sum().reset_index()
            opp_historical_starters.columns = ['PLAYER_NAME', 'START_COUNT']
            opp_historical_starters = opp_historical_starters.sort_values('START_COUNT', ascending=False)
            
            # Identify opponent's usual starters (top 5 by start count)
            if len(opp_historical_starters) >= 5:
                opp_usual_starters = set(opp_historical_starters.head(5)['PLAYER_NAME'].tolist())
            else:
                # If less than 5 historical starters, use all of them
                opp_usual_starters = set(opp_historical_starters['PLAYER_NAME'].tolist())
                # If still no historical starters, use top 5 by average minutes
                if len(opp_usual_starters) == 0:
                    avg_mins = opp_historical_games.groupby('PLAYER_NAME')['MIN'].mean().reset_index()
                    avg_mins = avg_mins.sort_values('MIN', ascending=False)
                    opp_usual_starters = set(avg_mins.head(5)['PLAYER_NAME'].tolist())
            
            # For the current game, count how many opponent usual starters played
            current_game_players = opp_games[opp_games['GAME_ID'] == current_game_id]
            opp_usual_starters_in_game = current_game_players[current_game_players['PLAYER_NAME'].isin(opp_usual_starters)]
            
            opp_usual_starters_available = opp_usual_starters_in_game['PLAYED'].sum()
            opp_usual_starters_out = len(opp_usual_starters) - opp_usual_starters_available
            
            # Update the dataframe for all players of this team in this game
            mask = (df['GAME_ID'] == current_game_id) & (df['TEAM_ID'] == team_id)
            df.loc[mask, 'OPP_USUAL_STARTERS_AVAILABLE'] = opp_usual_starters_available
            df.loc[mask, 'OPP_USUAL_STARTERS_OUT'] = opp_usual_starters_out
    
    # Remove temporary columns
    if 'STARTED' in df.columns:
        df = df.drop(columns=['PLAYED', 'STARTED'])
    else:
        df = df.drop(columns=['PLAYED'])
    
    return df

    
########################################################################################
# UTILITY AND HELPER FUNCTIONS
########################################################################################

def add_performance_without_stars_columns(df, min_games=2):
    """
    Add columns showing player performance delta when star teammates are out.
    Now calculates delta from baseline performance for MIN, PTS, and USG_PCT.
    """
    df = df.copy()
    df = df.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
    
    metrics = ['MIN', 'PTS', 'USG_PCT']
    
    def calculate_without_star_stats(player_group):
        player_group = player_group.copy()
        
        player_group['STAR_SAT_OUT_SHIFTED'] = player_group['STAR_SAT_OUT'].shift(1)
        
        star_out_mask = player_group['STAR_SAT_OUT_SHIFTED'] == 1
        star_in_mask = player_group['STAR_SAT_OUT_SHIFTED'] == 0
        
        if star_out_mask.sum() >= min_games and star_in_mask.sum() >= min_games:
            star_out_data = player_group[star_out_mask]
            star_in_data = player_group[star_in_mask]
            
            # Calculate baseline performance when star is in
            baseline_performance = {}
            for metric in metrics:
                if metric in player_group.columns:
                    baseline_performance[metric] = star_in_data[metric].shift(1).mean()
            
            # Calculate performance when star is out
            star_out_performance = {}
            for metric in metrics:
                if metric in player_group.columns:
                    star_out_performance[metric] = star_out_data[metric].shift(1).mean()
            
            # Calculate delta (star out performance - baseline performance)
            for metric in metrics:
                if metric in player_group.columns:
                    baseline = baseline_performance[metric]
                    star_out = star_out_performance[metric]
                    delta = star_out - baseline if not pd.isna(baseline) and not pd.isna(star_out) else np.nan
                    player_group[f'{metric}_DELTA_STAR_OUT'] = round(delta, 2)
                else:
                    player_group[f'{metric}_DELTA_STAR_OUT'] = np.nan
            
            player_group['GAMES_WITHOUT_STAR'] = star_out_mask.sum()
            player_group['GAMES_WITH_STAR'] = star_in_mask.sum()
        else:
            # Set to NaN for all metrics if insufficient games
            for metric in metrics:
                player_group[f'{metric}_DELTA_STAR_OUT'] = np.nan
            
            player_group['GAMES_WITHOUT_STAR'] = 0
            player_group['GAMES_WITH_STAR'] = 0
        
        # Drop temporary column
        player_group = player_group.drop('STAR_SAT_OUT_SHIFTED', axis=1)
        
        return player_group
    
    # Apply to each player
    result = df.groupby('PLAYER_NAME', group_keys=False).apply(calculate_without_star_stats)
    
    return result


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


##############################################################################################################
# VOLATILITY FEATURES
##############################################################################################################
def add_volatility_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    # Count/volume stats - use standard deviation
    count_stats = ['PTS', 'AST', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV', 'MIN']
    
    # Percentage/rate stats - use coefficient of variation
    pct_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'TS_PCT', 'USG_PCT', 'EFG_PCT']
    
    windows = [3, 5, 7, 10]
    
    # Standard deviation for count stats
    for window in windows:
        for stat in count_stats:
            if stat in df.columns:
                df[f'{stat}_VOLATILITY_{window}_TO_DATE'] = (
                    df.groupby(player_id_col)[stat]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
                )
    
    # CV for percentage stats (more comparable across players)
    for window in windows:
        for stat in pct_stats:
            if stat in df.columns:
                rolling_std = df.groupby(player_id_col)[stat].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
                )
                rolling_mean = df.groupby(player_id_col)[stat].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=2).mean()
                )
                df[f'{stat}_CV_{window}_TO_DATE'] = rolling_std / rolling_mean
                df[f'{stat}_CV_{window}_TO_DATE'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df


# def add_standard_deviation_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[5, 10, 15, 25, 40]):
#     df = df.copy()
#     df.sort_values([player_id_col, date_col], inplace=True)
    
#     # Define stats to calculate standard deviation for
#     std_stats = [ 'PTS', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT', 'TOV', 'TS_PCT', 'USG_PCT','MIN', 'PACE', 'PIE', 'E_OFF_RATING', 'NET_RATING', 'TCHS', 'POSS', 'EFG_PCT',
#                 'percentagePointsPaint', 'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFreeThrow', 'CFGA', 'UFGA' ]
    
#     # Filter to only available columns
#     available_stats = [stat for stat in std_stats if stat in df.columns]
    
#     if not available_stats:
#         print("Warning: No stats found in dataframe for standard deviation calculation")
#         return df
    
#     # Calculate standard deviation for each window
#     for window in windows:
#         for stat in available_stats:
#             # Rolling standard deviation (shifted to prevent leakage)
#             std_col = f'{stat}_STD_LAST_{window}'
#             df[std_col] = (
#                 df.groupby(player_id_col)[stat]
#                 .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
#                 .round(3)
#             )
    
#     # Fill NaN values with appropriate defaults
#     std_cols = [col for col in df.columns if '_STD_LAST_' in col]
    
#     for col in std_cols:
#         df[col] = df[col].fillna(0)  # No standard deviation for first games
    
#     # Convert to appropriate data types to save memory
#     for col in std_cols:
#         if df[col].dtype == 'float64':
#             df[col] = df[col].astype('float32')
    
#     return df


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
    
    key_stats = [ 'PTS', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTA', 'FTM', 'FT_PCT', 'TOV', 'TS_PCT', 'USG_PCT','MIN', 'PACE', 'PIE', 'E_OFF_RATING', 'NET_RATING', 'TCHS', 'POSS', 'EFG_PCT',
                'percentagePointsPaint', 'percentagePointsMidrange2pt', 'percentagePoints3pt', 'percentagePointsFreeThrow', 'CFGA', 'UFGA' ]
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

def add_interaction_features(df):
    eplison = 1e-8
    df = df.copy()
    
    df['TEAM_OFF_MINUS_OPP_DEF'] = df['TEAM_OFF_RATING_AVG_TO_DATE'] - df['OPP_DEF_RATING_AVG_TO_DATE']
    df['USG_X_PACE'] = df['USG_PCT_AVG_TO_DATE'] * df['EXPECTED_PACE']
    df['MIN_X_PACE'] = df['MIN_AVG_TO_DATE'] * df['EXPECTED_PACE']
    df['TS_X_USG'] = df['TS_PCT_AVG_TO_DATE'] * df['USG_PCT_AVG_TO_DATE']
    df['EFG_X_MIN'] = df['EFG_PCT_AVG_TO_DATE'] * df['MIN_AVG_TO_DATE']
    df['FGA_PER_TCHS_X_TEAM_OFF'] = (
        df['FGA_AVG_TO_DATE'] / (df['TCHS_AVG_TO_DATE'] + eplison)
    ) * df['TEAM_OFF_RATING_AVG_TO_DATE']
    df['EXPECTED_USAGE_MIN'] = df['USG_PCT_ROLLING_AVG_5'] * (df['MIN_ROLLING_AVG_5'] + eplison)
    df['EFG_X_OPP_DEF'] = df['EFG_PCT_AVG_TO_DATE'] * df['OPP_DEF_RATING_AVG_TO_DATE']

    # star player interactions
    df['PTS_X_STAR_PLAYER'] = df['PTS_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['FGA_X_STAR_PLAYER'] = df['FGA_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['USG_PCT_X_STAR_PLAYER'] = df['USG_PCT_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['EFG_PCT_X_STAR_PLAYER'] = df['EFG_PCT_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['E_OFF_RATING_X_STAR_PLAYER'] = df['E_OFF_RATING_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['NET_RATING_X_STAR_PLAYER'] = df['NET_RATING_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['TCHS_X_STAR_PLAYER'] = df['TCHS_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['POSS_X_STAR_PLAYER'] = df['POSS_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['EFG_PCT_X_STAR_PLAYER'] = df['EFG_PCT_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']
    df['E_OFF_RATING_X_STAR_PLAYER'] = df['E_OFF_RATING_AVG_TO_DATE'] * df['PLAYER_IS_TEAM_STAR']

    # Points per minute interactions
    df['PTS_PER_MIN'] = round(df['PTS_AVG_TO_DATE'] / (df['MIN_AVG_TO_DATE'] + eplison), 3)
    df['PTS_PER_40'] = round(df['PTS_AVG_TO_DATE'] * 40 / (df['MIN_AVG_TO_DATE'] + eplison), 3)
    df['PTS_PER_TCHS'] = round(df['PTS_AVG_TO_DATE'] / (df['TCHS_AVG_TO_DATE'] + eplison), 3)
    df['PTS_PER_TCHS_ROLLING_AVG_5'] = round(df['PTS_ROLLING_AVG_5'] / (df['TCHS_ROLLING_AVG_5'] + eplison), 3)
    df['PTS_PER_TCHS_ROLLING_AVG_15'] = round(df['PTS_ROLLING_AVG_15'] / (df['TCHS_ROLLING_AVG_15'] + eplison), 3)
    df['PTS_PER_TCHS_ROLLING_AVG_40'] = round(df['PTS_ROLLING_AVG_40'] / (df['TCHS_ROLLING_AVG_40'] + eplison), 3)
    df['PTS_PER_TCHS_X_USG'] = round(df['PTS_PER_TCHS'] * df['USG_PCT_AVG_TO_DATE'], 3)
    df['PTS_PER_MIN_ROLLING_AVG_5'] = round(df['PTS_ROLLING_AVG_5'] / (df['MIN_ROLLING_AVG_5'] + eplison), 3)
    df['PTS_PER_MIN_ROLLING_AVG_15'] = round(df['PTS_ROLLING_AVG_15'] / (df['MIN_ROLLING_AVG_15'] + eplison), 3)   
    df['PTS_PER_MIN_ROLLING_AVG_40'] = round(df['PTS_ROLLING_AVG_40'] / (df['MIN_ROLLING_AVG_40'] + eplison), 3)
    df['PTS_PER_MIN_X_PACE'] = round(df['PTS_PER_MIN'] * df['EXPECTED_PACE'], 3)
    df['PTS_PER_MIN_X_POSS'] = round(df['PTS_PER_MIN'] * df['POSS_AVG_TO_DATE'], 3)
    df['PTS_PER_MIN_X_USG'] = round(df['PTS_PER_MIN'] * df['USG_PCT_AVG_TO_DATE'], 3)
    df['PTS_PER_MIN_X_TEAM_OFF'] = round(df['PTS_PER_MIN'] * df['TEAM_OFF_RATING_AVG_TO_DATE'], 3)
    df['PTS_PER_MIN_X_OPP_DEF_RATING'] = round(df['PTS_PER_MIN'] * df['OPP_DEF_RATING_AVG_TO_DATE'], 3)
    df['PTS_X_OPP_DEF_RATING'] = round(df['PTS_AVG_TO_DATE'] * df['OPP_DEF_RATING_AVG_TO_DATE'], 3)
    df['PTS_PER_MIN_X_MIN_LAG'] = round(df['PTS_PER_MIN'] * df['MIN_LAG_1'], 3)

    df['3PA_RATE'] = df['FG3A_AVG_TO_DATE'] / (df['FGA_AVG_TO_DATE'] + eplison)
    df['3PA_RATE_ROLLING_AVG_5'] = df['FG3A_ROLLING_AVG_5'] / (df['FGA_ROLLING_AVG_5'] + eplison)
    df['3PA_RATE_ROLLING_AVG_10'] = df['FG3A_ROLLING_AVG_15'] / (df['FGA_ROLLING_AVG_15'] + eplison)
    df['3PA_TEAM_RATE'] = df['FG3A_AVG_TO_DATE'] / (df['TEAM_FG3A_AVG_TO_DATE'] + eplison)
    
    df['FT_RATE'] = df['FTA_AVG_TO_DATE'] / (df['FGA_AVG_TO_DATE'] + eplison)
    df['FT_RATE_ROLLING_AVG_5'] = df['FTA_ROLLING_AVG_5'] / (df['FGA_ROLLING_AVG_5'] + eplison)
    df['FT_RATE_ROLLING_AVG_10'] = df['FTA_ROLLING_AVG_10'] / (df['FGA_ROLLING_AVG_10'] + eplison)

    # team sharing
    df['PLAYER_FG3A_SHARE'] = df['FG3A_AVG_TO_DATE'] / (df['TEAM_FG3A_AVG_TO_DATE'] + eplison)
    df['PLAYER_FGA_SHARE'] = df['FGA_AVG_TO_DATE'] / (df['TEAM_FGA_AVG_TO_DATE'] + eplison)
    df['PLAYER_FT_RATE'] = df['FTA_AVG_TO_DATE'] / (df['TEAM_FTA_AVG_TO_DATE'] + eplison)
    df['PLAYER_PTS_SHARE'] = df['PTS_AVG_TO_DATE'] / (df['TEAM_PTS_AVG_TO_DATE'] + eplison)
    
    # New interaction features
    # Points per possession (true shooting possessions formula)
    df['PTS_PER_POSSESSION'] = df['PTS_AVG_TO_DATE'] / (df['FGA_AVG_TO_DATE'] + 0.44 * df['FTA_AVG_TO_DATE'] + df['TOV_AVG_TO_DATE'] + eplison)
    
    # Usage x Efficiency
    df['USAGE_X_EFFICIENCY'] = df['USG_PCT_AVG_TO_DATE'] * df['TS_PCT_AVG_TO_DATE']
    
    # Pace-adjusted expected points
    df['PACE_ADJUSTED_EXPECTED_PTS'] = df['PTS_ROLLING_AVG_5'] * (df['EXPECTED_PACE'] / 100)
    
    # Star out x usage boost (checking if star_out features exist)
    if 'STAR_SAT_OUT' in df.columns and 'USG_PCT_DELTA_STAR_OUT' in df.columns:
        df['STAR_OUT_X_USAGE_BOOST'] = df['STAR_SAT_OUT'] * df['USG_PCT_DELTA_STAR_OUT']
    elif 'PLAYER_PERFORMANCE_WITHOUT_STARS_PTS_ROLLING_AVG_5' in df.columns:
        # Fallback: calculate star out multiplier
        df['STAR_OUT_X_USAGE_BOOST'] = df['PLAYER_PERFORMANCE_WITHOUT_STARS_PTS_ROLLING_AVG_5'] / (df['PTS_ROLLING_AVG_5'] + eplison) * df['USG_PCT_ROLLING_AVG_5']
    else:
        df['STAR_OUT_X_USAGE_BOOST'] = 0
    
    # Role tier interactions to help model understand typical role expectations
    if 'IS_STARTER_TIER' in df.columns:
        df['PTS_X_STARTER_TIER'] = df['PTS_AVG_TO_DATE'] * df['IS_STARTER_TIER']
        df['USG_X_STARTER_TIER'] = df['USG_PCT_AVG_TO_DATE'] * df['IS_STARTER_TIER']
        df['MIN_X_STARTER_TIER'] = df['MIN_AVG_TO_DATE'] * df['IS_STARTER_TIER']
        
        df['STARTING_X_BENCH_TIER'] = df['STARTING'] * df['IS_BENCH_TIER']
        df['STARTING_X_ROLE_TIER'] = df['STARTING'] * df['IS_ROLE_TIER']
        df['STARTING_X_STARTER_TIER'] = df['STARTING'] * df['IS_STARTER_TIER']
    
    if 'IS_ROLE_TIER' in df.columns:
        df['PTS_X_ROLE_TIER'] = df['PTS_AVG_TO_DATE'] * df['IS_ROLE_TIER']
        df['USG_X_ROLE_TIER'] = df['USG_PCT_AVG_TO_DATE'] * df['IS_ROLE_TIER']
        df['MIN_X_ROLE_TIER'] = df['MIN_AVG_TO_DATE'] * df['IS_ROLE_TIER']

    
    if 'IS_BENCH_TIER' in df.columns:
        df['PTS_X_BENCH_TIER'] = df['PTS_AVG_TO_DATE'] * df['IS_BENCH_TIER']
        df['USG_X_BENCH_TIER'] = df['USG_PCT_AVG_TO_DATE'] * df['IS_BENCH_TIER']
        df['MIN_X_BENCH_TIER'] = df['MIN_AVG_TO_DATE'] * df['IS_BENCH_TIER']
    
    df['PLAYER_3PT_RATE_X_OPP_GUARD_3PT_DEF_ALLOWED'] = df['GUARD'] * df['3PA_RATE'] * (
        df['OPP_GUARD_DEF_3PT_PCT_ALLOWED'] )
    
    df['PLAYER_3PT_RATE_X_OPP_FORWARD_3PT_DEF_ALLOWED'] = df['FORWARD'] * df['3PA_RATE'] * (
    df['OPP_FORWARD_DEF_3PT_PCT_ALLOWED'] )
    
    df['PLAYER_3PT_RATE_X_OPP_CENTER_3PT_DEF_ALLOWED'] = df['CENTER'] * df['3PA_RATE'] * (
    df['OPP_CENTER_DEF_3PT_PCT_ALLOWED'] )

    df['PLAYER_3PT_FG_PCT_X_OPP_GUARD_3PT_DEF_ALLOWED'] = df['GUARD'] * df['FG3_PCT_AVG_TO_DATE'] * (
    df['OPP_GUARD_DEF_3PT_PCT_ALLOWED'] )
    
    df['PLAYER_3PT_FG_PCT_X_OPP_FORWARD_3PT_DEF_ALLOWED'] = df['FORWARD'] * df['FG3_PCT_AVG_TO_DATE'] * (
    df['OPP_FORWARD_DEF_3PT_PCT_ALLOWED'] )
    
    df['PLAYER_3PT_FG_PCT_X_OPP_CENTER_3PT_DEF_ALLOWED'] = df['CENTER'] * df['FG3_PCT_AVG_TO_DATE'] * (
    df['OPP_CENTER_DEF_3PT_PCT_ALLOWED'] )

    df['PLAYER_X_MATCHUP_GUARD_FG_PCT'] = df['GUARD'] * (df['OPP_GUARD_DEF_FG_PCT_ALLOWED'] - df['FG_PCT_AVG_TO_DATE'] )
    
    df['PLAYER_X_MATCHUP_FORWARD_FG_PCT'] = df['FORWARD'] * (df['OPP_FORWARD_DEF_FG_PCT_ALLOWED'] - df['FG_PCT_AVG_TO_DATE']  )
    
    df['PLAYER_X_MATCHUP_CENTER_FG_PCT'] = df['CENTER'] * (df['OPP_CENTER_DEF_FG_PCT_ALLOWED'] - df['FG_PCT_AVG_TO_DATE'] )
    
    return df
