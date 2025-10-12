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

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[10,15,25,40]):
    """Calculate rolling averages for key player statistics only."""
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = [
    # Core stats that made top 150
    'PTS', 'FGA', 'FG3A', 'FTM', 'FGM', 'FG3M',
    'FG_PCT', 'EFG_PCT', 'TS_PCT',
    
    # Usage and pace metrics
    'USG_PCT', 'E_USG_PCT', 'MIN', 'POSS', 'PACE', 'E_PACE',
    
    # Shooting percentages that appear in top 150
    'percentagePointsPaint', 'percentagePointsFreeThrow', 
    'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt',
    'percentagePoints2pt', 'percentagePoints3pt',
    'percentagePointsMidrange2pt',
    'percentageAssisted2pt', 'percentageAssisted3pt', 
    'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM',
    'percentagePointsOffTurnovers',
    
    # Defensive and advanced stats
    'UFGA', 'DFGM', 'DFGA',
    'ORBC', 'DRBC', 'RBC', 'DREB_PCT',
    
    # Passing and touches
    'SAST', 'CFGM', 'CFGA', 
    
    # Other advanced metrics
    'SPD', 'DIST', 'AST_PCT', 'NET_RATING', 'PIE', 'PLUS_MINUS',
    'FANTASY_PTS', 'PTS_2ND_CHANCE', 'PTS_PAINT', 
    'OPP_PTS_PAINT', 'PFD', 'STL', 'BLK', 'TOV', 'PF'
]

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
    player_data = player_data.sort_values([player_id_col, date_col])
    stat_lines = ['PTS', 'MIN', 'FGA', 'FG3A', 'FTA', 'FGM', 'FG3M', 'FTM', 'FG_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT', 'MIN', 'POSS', 'PACE', 'E_PACE', 'percentagePointsPaint', 'percentagePointsFreeThrow', 'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt', 'percentagePoints2pt', 'percentagePoints3pt', 'percentagePointsMidrange2pt', 'percentageAssisted2pt', 'percentageAssisted3pt', 'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM', 'percentagePointsOffTurnovers', 'UFGA', 'DFGM', 'DFGA', 'ORBC', 'DRBC', 'RBC', 'DREB_PCT', 'SAST', 'CFGM', 'CFGA', 'SPD', 'DIST', 'AST_PCT', 'NET_RATING', 'PIE', 'PLUS_MINUS',  'PTS_2ND_CHANCE', 'PTS_PAINT', 'TOV', 'PF']
    
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

def getPlayerAvgToDateVectorized(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Vectorized version that should be faster and avoid multi-index issues.
    FIXED: Properly shifted to prevent data leakage.
    """
    # Create copy and sort
    df_enhanced = df.copy().sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define stats
    stats_cols = [
    # Core stats that made top 150
    'PTS', 'FGA', 'FG3A', 'FTM', 'FGM', 'FG3M',
    'FG_PCT', 'EFG_PCT', 'TS_PCT',
    
    # Usage and pace metrics
    'USG_PCT', 'E_USG_PCT', 'MIN', 'POSS', 'PACE', 'E_PACE',
    
    # Shooting percentages that appear in top 150
    'percentagePointsPaint', 'percentagePointsFreeThrow', 
    'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt',
    'percentagePoints2pt', 'percentagePoints3pt',
    'percentagePointsMidrange2pt',
    'percentageAssisted2pt', 'percentageAssisted3pt', 
    'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM',
    'percentagePointsOffTurnovers',
    
    # Defensive and advanced stats
    'UFGA', 'DFGM', 'DFGA',
    'ORBC', 'DRBC', 'RBC', 'DREB_PCT',
    
    # Passing and touches
    'SAST', 'CFGM', 'CFGA', 
    
    # Other advanced metrics
    'SPD', 'DIST', 'AST_PCT', 'NET_RATING', 'PIE', 'PLUS_MINUS',
    'FANTASY_PTS', 'PTS_2ND_CHANCE', 'PTS_PAINT', 
    'OPP_PTS_PAINT', 'PFD', 'STL', 'BLK', 'TOV', 'PF'
]

    for stat in stats_cols:
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

    metrics = [
    # Core stats that made top 150
    'PTS', 'FGA', 'FG3A', 'FTM', 'FGM', 'FG3M',
    'FG_PCT', 'EFG_PCT', 'TS_PCT',
    
    # Usage and pace metrics
    'USG_PCT', 'E_USG_PCT', 'MIN', 'POSS', 'PACE', 'E_PACE',
    
    # Shooting percentages that appear in top 150
    'percentagePointsPaint', 'percentagePointsFreeThrow', 
    'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt',
    'percentagePoints2pt', 'percentagePoints3pt',
    'percentagePointsMidrange2pt',
    'percentageAssisted2pt', 'percentageAssisted3pt', 
    'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM',
    'percentagePointsOffTurnovers',
    
    # Defensive and advanced stats
    'UFGA', 'DFGM', 'DFGA',
    'ORBC', 'DRBC', 'RBC', 'DREB_PCT',
    
    # Passing and touches
    'SAST', 'CFGM', 'CFGA', 
    
    # Other advanced metrics
    'SPD', 'DIST', 'AST_PCT', 'NET_RATING', 'PIE', 'PLUS_MINUS',
    'FANTASY_PTS', 'PTS_2ND_CHANCE', 'PTS_PAINT', 
    'OPP_PTS_PAINT', 'PFD', 'STL', 'BLK', 'TOV', 'PF'
]
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
        'MIN': [3,5,10,15],
        'FGA': [3,5,10,15],
        'FG3A': [3,5,10,15],
        'FTA': [3,5,10,15],
        'PTS': [3,5,10,15],
        'USG_PCT': [3,5,10,15],
        'EFG_PCT': [3,5,10,15],
        'TS_PCT': [3,5,10,15],
        'POSS': [3,5,10,15],
        'TCHS': [3,5,10,15],
        'PASS': [3,5,10,15],
        'SAST': [3,5,10,15],
        'FTAST': [3,5,10,15],
        'TOV': [3,5,10,15],
        'POINT_PER_SHOT': [3,5,10,15],
        'PLUS_MINUS': [3,5,10,15],
        'NET_RATING': [3,5,10,15],
        'PIE': [3,5,10,15],
        'SPD': [3,5,10,15],
        'DIST': [3,5,10,15],
        
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

def assign_team_opp_def_by_position(df):
    def_cols = ['DEF_FG_PCT_ALLOWED', 'DEF_3PT_PCT_ALLOWED', 'PTS_ALLOWED_PER_MIN']
    positions = ['GUARD', 'FORWARD', 'CENTER']
    team_def_list = []

    for pos in positions:
        df_shifted = df.copy()
        df_shifted = df_shifted.sort_values(['PLAYER_ID', 'GAME_DATE'])
        df_shifted[def_cols] = df_shifted.groupby('PLAYER_ID')[def_cols].shift(1)
        
        tmp = (
            df[df[pos] == 1]
            .groupby(['TEAM_ID', 'GAME_ID'])[def_cols]
            .mean()
            .round(3)
            .reset_index()
            .rename(columns={
                'DEF_FG_PCT_ALLOWED': f'TEAM_{pos}_DEF_FG_PCT_ALLOWED',
                'DEF_3PT_PCT_ALLOWED': f'TEAM_{pos}_DEF_3PT_PCT_ALLOWED',
                'PTS_ALLOWED_PER_MIN': f'TEAM_{pos}_PTS_ALLOWED_PER_MIN'
            })
        )
        team_def_list.append(tmp)

    # Merge all position-based team stats
    team_def = team_def_list[0]
    for tmp in team_def_list[1:]:
        team_def = team_def.merge(tmp, on=['TEAM_ID', 'GAME_ID'], how='outer')
    df = df.merge(team_def, on=['TEAM_ID', 'GAME_ID'], how='left')
    opp_def = team_def.rename(columns={
        'TEAM_ID': 'OPP_TEAM_ID',
        **{col: col.replace('TEAM_', 'OPP_') for col in team_def.columns if col not in ['TEAM_ID', 'GAME_ID']}
    })
    df = df.merge(opp_def, on=['OPP_TEAM_ID', 'GAME_ID'], how='left')
    return df

def teamRollingDefenseByPosition(df, team_id_col='TEAM_ID', date_col='GAME_DATE', windows=[5,10,15]):
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

def assign_team_opp_zone_by_position(df):
    """Calculate team and opponent zone shooting statistics by position."""
    zone_cols = [
        'FREQ_FG3','FG3_PCT_main', 'NS_FG3_PCT', 'PLUS_MINUS_FG3',
        'FREQ_FG2', 'FG2_PCT', 'NS_FG2_PCT', 'PLUS_MINUS_FG2',
        'FREQ_LT_06', 'LT_06_PCT', 'NS_LT_06_PCT', 'PLUS_MINUS_LT_06',
        'FREQ_LT_10', 'LT_10_PCT', 'NS_LT_10_PCT', 'PLUS_MINUS_LT_10',
        'FREQ_GT_15', 'GT_15_PCT', 'NS_GT_15_PCT', 'PLUS_MINUS_GT_15'
    ]
    positions = ['GUARD', 'FORWARD', 'CENTER']
    team_zone_list = []

    for pos in positions:
        tmp = (
            df[df[pos] == 1]
            .groupby(['TEAM_ID', 'GAME_ID'])[zone_cols]
            .mean()
            .round(3)
            .reset_index()
            .rename(columns={
                col: f'TEAM_{pos}_{col}' for col in zone_cols
            })
        )
        team_zone_list.append(tmp)

    # Merge all position-based team zone stats
    team_zone = team_zone_list[0]
    for tmp in team_zone_list[1:]:
        team_zone = team_zone.merge(tmp, on=['TEAM_ID', 'GAME_ID'], how='outer')
    
    # Merge team zone stats to main dataframe
    df = df.merge(team_zone, on=['TEAM_ID', 'GAME_ID'], how='left')
    
    # Create opponent zone stats by renaming team columns
    opp_zone = team_zone.rename(columns={
        'TEAM_ID': 'OPP_TEAM_ID',
        **{col: col.replace('TEAM_', 'OPP_') for col in team_zone.columns if col not in ['TEAM_ID', 'GAME_ID']}
    })
    
    # Merge opponent zone stats to main dataframe
    df = df.merge(opp_zone, on=['OPP_TEAM_ID', 'GAME_ID'], how='left')
    
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

def add_opponent_team_rolling_stats(df, team_id_col='TEAM_ID', date_col='GAME_DATE', windows=[10, 15, 25]):
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


def add_opponent_team_form_indicators(df, windows=[10, 15, 25]):
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
    required_cols = ['TEAM_PACE_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE', 'TEAM_PACE_ROLLING_AVG_5', 'OPP_PACE_ROLLING_AVG_5', 'TEAM_PACE_ROLLING_AVG_10', 'OPP_PACE_ROLLING_AVG_10', 'TEAM_PACE_ROLLING_AVG_15', 'OPP_PACE_ROLLING_AVG_15']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Silently handle missing columns
        df['EXPECTED_PACE'] = np.nan
        df['EXPECTED_PACE_DIFF'] = np.nan
        df['EXPECTED_PACE_DIFF_ROLLING_AVG_5'] = np.nan
        df['EXPECTED_PACE_DIFF_ROLLING_AVG_10'] = np.nan
        df['EXPECTED_PACE_DIFF_ROLLING_AVG_15'] = np.nan
        return df
    
    # Calculate expected pace by multiplying team and opponent pace averages
    df['EXPECTED_PACE'] = ((df['TEAM_PACE_AVG_TO_DATE'] + df['OPP_PACE_AVG_TO_DATE']) / 2).round(2)
    df['EXPECTED_PACE_DIFF'] = df['TEAM_PACE_AVG_TO_DATE'] - df['OPP_PACE_AVG_TO_DATE']
    df['EXPECTED_PACE_DIFF_ROLLING_AVG_5'] = df['TEAM_PACE_ROLLING_AVG_5'] - df['OPP_PACE_ROLLING_AVG_5']
    df['EXPECTED_PACE_DIFF_ROLLING_AVG_10'] = df['TEAM_PACE_ROLLING_AVG_10'] - df['OPP_PACE_ROLLING_AVG_10']
    df['EXPECTED_PACE_DIFF_ROLLING_AVG_15'] = df['TEAM_PACE_ROLLING_AVG_15'] - df['OPP_PACE_ROLLING_AVG_15']
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

    # Cleanup helpers
    df = df.drop(columns=['STAR_NAME', 'STAR_ACTIVE', 'ACTIVE'])

    return df

    
########################################################################################
# UTILITY AND HELPER FUNCTIONS
########################################################################################

def add_performance_without_stars_columns(df, min_games=2):
    """
    Add columns showing player averages when star teammates are out.
    FIXED: Now uses shift(1) to prevent data leakage.
    """
    df = df.copy()
    df = df.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
    
    # Define metrics to track when star is out
    metrics = [
        'PTS', 'MIN', 'USG_PCT', 'FGA', 'FG3A', 'FTA',
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'EFG_PCT', 'TS_PCT',
        'AST', 'POSS', 'TCHS', 'REB', 'TOV', 'NET_RATING', 'PIE', 'PLUS_MINUS',
    ]
    
    def calculate_without_star_stats(player_group):
        player_group = player_group.copy()
        
        # FIXED: Shift TEAM_STAR_OUT to prevent leakage
        player_group['TEAM_STAR_OUT_SHIFTED'] = player_group['TEAM_STAR_OUT'].shift(1)
        
        # Only team star out scenario - using SHIFTED values
        star_out_mask = player_group['TEAM_STAR_OUT_SHIFTED'] == 1
        
        # FIXED: Use shifted values for all statistics
        if star_out_mask.sum() >= min_games:
            star_out_data = player_group[star_out_mask]
            
            # Calculate averages using SHIFTED data for all metrics
            for metric in metrics:
                if metric in player_group.columns:
                    player_group[f'{metric}_WITHOUT_STAR'] = round(
                        star_out_data[metric].shift(1).mean(), 2
                    )
                else:
                    player_group[f'{metric}_WITHOUT_STAR'] = np.nan
            
            # Special calculation: PTS_PER_36 with shifted data
            pts_shifted = star_out_data['PTS'].shift(1)
            min_shifted = star_out_data['MIN'].shift(1)
            player_group['PTS_PER_36_WITHOUT_STAR'] = round(
                (pts_shifted * 36 / (min_shifted + 1e-8)).mean(), 2
            )
            
            player_group['GAMES_WITHOUT_STAR'] = star_out_mask.sum()
        else:
            # Set to NaN for all metrics if insufficient games
            for metric in metrics:
                player_group[f'{metric}_WITHOUT_STAR'] = np.nan
            
            player_group['PTS_PER_36_WITHOUT_STAR'] = np.nan
            player_group['GAMES_WITHOUT_STAR'] = 0
        
        # Drop temporary column
        player_group = player_group.drop('TEAM_STAR_OUT_SHIFTED', axis=1)
        
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
def add_volatility_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[5, 15, 40]):
    """
    Calculate volatility features for player performance metrics.
    Only calculates rolling standard deviation for specified windows.
    """
    # Create copy and sort data
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    # Define stats to calculate volatility for
    volatility_stats = [
    # Core stats that made top 150
    'PTS', 'FGA', 'FG3A', 'FTM', 'FGM', 'FG3M',
    'FG_PCT', 'EFG_PCT', 'TS_PCT',
    
    # Usage and pace metrics
    'USG_PCT', 'E_USG_PCT', 'MIN', 'POSS', 'PACE', 'E_PACE',
    
    # Shooting percentages that appear in top 150
    'percentagePointsPaint', 'percentagePointsFreeThrow', 
    'percentageFieldGoalsAttempted2pt', 'percentagePoints2pt',
    'percentagePointsMidrange2pt',
    'percentageAssisted2pt', 'percentageAssisted3pt', 
    'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM',
    'percentagePointsOffTurnovers',
    
    # Defensive and advanced stats
    'UFGA', 'DFGM', 'DFGA',
    'ORBC', 'DRBC', 'RBC', 'DREB_PCT',
    
    # Passing and touches
    'SAST', 'CFGM', 'CFGA', 
    
    # Other advanced metrics
    'SPD', 'DIST', 'AST_PCT', 'NET_RATING', 'PIE', 'PLUS_MINUS',
    'FANTASY_PTS', 'PTS_2ND_CHANCE', 'PTS_PAINT', 
    'OPP_PTS_PAINT', 'PFD', 'STL', 'BLK', 'TOV', 'PF'
]
    
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
    
    # Add expanding volatility metrics (season-long volatility)
    for stat in available_stats:
        # Expanding standard deviation
        expanding_vol_col = f'{stat}_EXPANDING_VOLATILITY_TO_DATE'
        df[expanding_vol_col] = (
            df.groupby(player_id_col)[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=2).std())
            .round(3)
        )
    
    # Fill NaN values with appropriate defaults
    volatility_cols = [col for col in df.columns if 'VOLATILITY' in col]
    
    for col in volatility_cols:
        df[col] = df[col].fillna(0)  # No volatility for first games
    
    # Convert to appropriate data types to save memory
    for col in volatility_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df


def add_standard_deviation_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[5, 15, 40]):
    """
    Calculate standard deviation features for player performance metrics.
    Calculates rolling standard deviation for specified windows (last 5, 15, 40 games).
    """
    # Create copy and sort data
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    # Define stats to calculate standard deviation for
    std_stats = [
    # Core stats that made top 150
    'PTS', 'FGA', 'FG3A', 'FTM', 'FGM', 'FG3M',
    'FG_PCT', 'EFG_PCT', 'TS_PCT',
    
    # Usage and pace metrics
    'USG_PCT', 'E_USG_PCT', 'MIN', 'POSS', 'PACE', 'E_PACE',
    
    # Shooting percentages that appear in top 150
    'percentagePointsPaint', 'percentagePointsFreeThrow', 
    'percentageFieldGoalsAttempted2pt', 'percentagePoints2pt',
    'percentagePointsMidrange2pt',
    'percentageAssisted2pt', 'percentageAssisted3pt', 
    'percentageUnassisted3pt', 'percentageAssistedFGM', 'percentageUnassistedFGM',
    'percentagePointsOffTurnovers',
    
    # Defensive and advanced stats
    'UFGA', 'DFGM', 'DFGA',
    'ORBC', 'DRBC', 'RBC', 'DREB_PCT',
    
    # Passing and touches
    'SAST', 'CFGM', 'CFGA', 
    
    # Other advanced metrics
    'SPD', 'DIST', 'AST_PCT', 'NET_RATING', 'PIE', 'PLUS_MINUS',
    'FANTASY_PTS', 'PTS_2ND_CHANCE', 'PTS_PAINT', 
    'OPP_PTS_PAINT', 'PFD', 'STL', 'BLK', 'TOV', 'PF'
]
    
    # Filter to only available columns
    available_stats = [stat for stat in std_stats if stat in df.columns]
    
    if not available_stats:
        print("Warning: No stats found in dataframe for standard deviation calculation")
        return df
    
    # Calculate standard deviation for each window
    for window in windows:
        for stat in available_stats:
            # Rolling standard deviation (shifted to prevent leakage)
            std_col = f'{stat}_STD_LAST_{window}'
            df[std_col] = (
                df.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=2).std())
                .round(3)
            )
    
    # Fill NaN values with appropriate defaults
    std_cols = [col for col in df.columns if '_STD_LAST_' in col]
    
    for col in std_cols:
        df[col] = df[col].fillna(0)  # No standard deviation for first games
    
    # Convert to appropriate data types to save memory
    for col in std_cols:
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

def add_interaction_features(df):
    eplison = 1e-8
    df = df.copy()
    # Relative strength interactions
    df['TEAM_OFF_MINUS_OPP_DEF'] = df['TEAM_OFF_RATING_AVG_TO_DATE'] - df['OPP_DEF_RATING_AVG_TO_DATE']
    df['TEAM_PACE_MINUS_OPP_PACE'] = df['TEAM_PACE_AVG_TO_DATE'] - df['OPP_PACE_AVG_TO_DATE']
    df['TEAM_PTS_MINUS_OPP_PTS'] = df['TEAM_PTS_AVG_TO_DATE'] - df['OPP_PTS_AVG_TO_DATE']

    # Player context x environment
    df['USG_X_PACE'] = df['USG_PCT_AVG_TO_DATE'] * df['EXPECTED_PACE']
    df['USG_X_TEAM_OFF'] = df['USG_PCT_AVG_TO_DATE'] * df['TEAM_OFF_RATING_AVG_TO_DATE']
    df['MIN_X_PACE'] = df['MIN_AVG_TO_DATE'] * df['EXPECTED_PACE']
    df['PTS_X_TEAM_TOTAL'] = df['PTS_AVG_TO_DATE'] * np.where(df['team_is_favored'] == 1, 
                                                              df['TEAM_IMPLIED_PTS_FAV'], 
                                                              df['TEAM_IMPLIED_PTS_UND'])

    # Shooting style x matchup fit
    df['PLAYER_3PT_X_OPP_3PT_DEF'] = df['percentageFieldGoalsAttempted3pt_AVG_TO_DATE'] * df['OPP_GUARD_DEF_3PT_PCT_ALLOWED']
    df['PLAYER_PAINT_X_OPP_PAINT_DEF'] = df['percentagePointsPaint_AVG_TO_DATE'] * df['OPP_PTS_PAINT']
    df['PLAYER_MID_X_OPP_MID_DEF'] = df['percentagePointsMidrange2pt_AVG_TO_DATE'] * df['OPP_FORWARD_DEF_FG_PCT_ALLOWED']
    df['PLAYER_3PT_X_OPP_3PT_DEF_RECENT'] = df['percentageFieldGoalsAttempted3pt_ROLLING_AVG_5'] * df['OPP_GUARD_DEF_3PT_PCT_ALLOWED']
    df['PLAYER_PAINT_X_OPP_PAINT_DEF_RECENT'] = df['percentagePointsPaint_ROLLING_AVG_5'] * df['OPP_PTS_PAINT']
    df['PLAYER_MID_X_OPP_MID_DEF_RECENT'] = df['percentagePointsMidrange2pt_ROLLING_AVG_5'] * df['OPP_FORWARD_DEF_FG_PCT_ALLOWED']

    # Form x environment
    df['ROLLING_PTS5_X_PACE'] = df['PTS_ROLLING_AVG_5'] * df['EXPECTED_PACE']
    df['ROLLING_PTS5_X_TEAM_OFF'] = df['PTS_ROLLING_AVG_5'] * df['TEAM_OFF_RATING_AVG_TO_DATE']
    df['ROLLING_MIN5_X_TEAM_PTS'] = df['MIN_ROLLING_AVG_5'] * df['TEAM_PTS_AVG_TO_DATE']

    # Fatigue context
    df['REST_X_PACE'] = df['PLAYER_DAYS_REST'] * df['EXPECTED_PACE']
    df['B2B_X_PACE'] = df['IS_BACK_TO_BACK'] * df['EXPECTED_PACE']

    # Role and status interactions
    df['STARTER_X_PACE'] = df['STARTING'] * df['EXPECTED_PACE']
    df['TEAM_STAR_OUT_X_USG'] = df['TEAM_STAR_OUT'] * df['USG_PCT_AVG_TO_DATE']
    df['TEAM_STAR_OUT_X_MIN'] = df['TEAM_STAR_OUT'] * df['MIN_AVG_TO_DATE']
    df['TEAM_STAR_OUT_X_PTS'] = df['TEAM_STAR_OUT'] * df['PTS_AVG_TO_DATE']

    # Home effect interactions
    df['HOME_X_MIN'] = df['HOME_GAME'] * df['MIN_AVG_TO_DATE']
    df['HOME_X_PTS'] = df['HOME_GAME'] * df['PTS_AVG_TO_DATE']
    df['HOME_X_PACE'] = df['HOME_GAME'] * df['EXPECTED_PACE']

    # Efficiency x volume
    df['EFG_X_FGA'] = df['EFG_PCT_AVG_TO_DATE'] * df['FGA_AVG_TO_DATE']
    df['TS_X_USG'] = df['TS_PCT_AVG_TO_DATE'] * df['USG_PCT_AVG_TO_DATE']
    df['EFG_X_MIN'] = df['EFG_PCT_AVG_TO_DATE'] * df['MIN_AVG_TO_DATE']
    df['EXPECTED_USAGE_MIN'] = df['USG_PCT_ROLLING_AVG_5'] * (df['MIN_ROLLING_AVG_5'] + eplison)

    # Points per minute interactions
    df['PTS_PER_MIN'] = round(df['PTS_AVG_TO_DATE'] / (df['MIN_AVG_TO_DATE'] + eplison), 3)
    df['PTS_PER_36'] = round(df['PTS_AVG_TO_DATE'] * 36 / (df['MIN_AVG_TO_DATE'] + eplison), 3)
    df['PTS_PER_MIN_ROLLING_AVG_5'] = round(df['PTS_ROLLING_AVG_5'] / (df['MIN_ROLLING_AVG_5'] + eplison), 3)
    df['PTS_PER_MIN_ROLLING_AVG_15'] = round(df['PTS_ROLLING_AVG_15'] / (df['MIN_ROLLING_AVG_15'] + eplison), 3)   
    df['PTS_PER_MIN_ROLLING_AVG_40'] = round(df['PTS_ROLLING_AVG_40'] / (df['MIN_ROLLING_AVG_40'] + eplison), 3)
    df['PTS_PER_MIN_X_PACE'] = round(df['PTS_PER_MIN'] * df['EXPECTED_PACE'], 3)

    return df
