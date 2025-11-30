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

# ================================================================================================
# ROLLING AVERAGES AND TIME SERIES FEATURES - FIXED FOR DATA LEAKAGE
# ================================================================================================

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[5,10,40]):
    """Calculate rolling averages for key player statistics only."""
    df = player_data.copy()
    
    # Reset index to ensure unique indices
    df.reset_index(drop=True, inplace=True)
    
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = ['PTS', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
    'USG_PCT', 'MIN', 'POSS', 'PLUS_MINUS', 'TS_PCT', 
    'E_OFF_RATING', 'NET_RATING', 'EFG_PCT', 'TCHS']
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
    stats_lines = ['STARTING', 'PTS', 'AST', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM',
    'TS_PCT', 'USG_PCT', 'MIN', 'E_OFF_RATING', 'NET_RATING', 'EFG_PCT', 'POSS', 'TCHS']
    
    for stat_line in stats_lines:
        if stat_line not in player_data.columns:
            continue
            
        for lag in range(1, 4):
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
    # Create copy and sort
    df_enhanced = df.copy().sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define stats
    stats_cols = ['PTS', 'AST', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
    'FTA', 'FTM', 'MIN','TS_PCT', 'USG_PCT', 'E_OFF_RATING', 'NET_RATING', 'EFG_PCT', 'POSS', 
    'TCHS', 'PLUS_MINUS']
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

    metrics = ['PTS', 'AST', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM',
    'TS_PCT', 'USG_PCT', 'MIN', 'POSS', 'TCHS',
    'UFGA', 'UFGM', 'PLUS_MINUS',
    'E_OFF_RATING', 'NET_RATING', 'EFG_PCT']
    
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
        
        # Calculate location-specific averages (grouped by player and HOME_GAME)
        # This gives us separate expanding means for home games and away games
        loc_avg = shifted_expanding_mean(df[metric], [df[player_id_col], df['HOME_GAME']])
        
        home_col = f'PLAYER_HOME_AVG_{metric}_TO_DATE'
        away_col = f'PLAYER_AWAY_AVG_{metric}_TO_DATE'
        home_delta_col = f'PLAYER_HOME_{metric}_DELTA'
        away_delta_col = f'PLAYER_AWAY_{metric}_DELTA'
        
        # Extract home average from home games, away average from away games
        # loc_avg already has location-specific values due to grouping by HOME_GAME
        df[home_col] = np.where(df['HOME_GAME'] == 1, loc_avg, np.nan)
        df[away_col] = np.where(df['HOME_GAME'] == 0, loc_avg, np.nan)
        
        # Forward fill within each player group so every game has both home and away averages
        # This ensures that even home games have away_avg (from previous away games)
        # and even away games have home_avg (from previous home games)
        df[home_col] = df.groupby(player_id_col)[home_col].transform(lambda x: x.ffill())
        df[away_col] = df.groupby(player_id_col)[away_col].transform(lambda x: x.ffill())
        
        # Fill first games with overall average or global mean
        df.loc[first_game_mask, home_col] = df.loc[first_game_mask, home_col].fillna(overall_avg)
        df.loc[first_game_mask, away_col] = df.loc[first_game_mask, away_col].fillna(overall_avg)
        
        # Final fallback to overall_avg or global_means
        df[home_col] = df[home_col].fillna(overall_avg)
        df[away_col] = df[away_col].fillna(overall_avg)
        df[home_col] = df[home_col].fillna(global_means[metric]).astype('float32').round(2)
        df[away_col] = df[away_col].fillna(global_means[metric]).astype('float32').round(2)
        
        # Calculate deltas (home/away performance - overall performance)
        # Now every game has both home and away averages, so deltas are correctly calculated
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


def statAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION'):
    """
    Calculate historical performance against each opponent.
    Uses expanding average (all previous games vs that team).
    """
    df = player_data.copy()
    df.reset_index(drop=True, inplace=True)
    df.sort_values([player_id_col, 'GAME_DATE'], inplace=True)
    
    # Metrics to track historical performance against teams
    metrics = ['PTS', 'AST', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM',
    'MIN', 'POSS', 'USG_PCT', 'EFG_PCT', 'TS_PCT',
    'E_OFF_RATING', 'NET_RATING', 'PLUS_MINUS', 'POSS', 'TCHS']
    
    # Available metrics only
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Create player-opponent grouper
    player_opp_group = df.groupby([player_id_col, opp_col])
    
    # Track how many games played against this opponent
    df['GAMES_VS_OPP'] = player_opp_group.cumcount()
    
    # Calculate expanding averages for each metric
    for metric in available_metrics:
        # Historical average vs this opponent (shifted to prevent leakage)
        matchup_col = f'MATCHUP_AVG_{metric}_TO_DATE'
        df[matchup_col] = (
            player_opp_group[metric]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            .round(2)
        )
        
        # Calculate delta: matchup average - overall average
        # This shows how player performs vs this opponent compared to their overall average
        overall_avg_col = f'{metric}_AVG_TO_DATE'
        matchup_delta_col = f'MATCHUP_{metric}_DELTA'
        
        if overall_avg_col in df.columns:
            df[matchup_delta_col] = (df[matchup_col] - df[overall_avg_col]).round(2)
        else:
            # If overall average doesn't exist, delta is just the matchup average
            df[matchup_delta_col] = df[matchup_col].round(2)
    
    # Handle NaN values
    matchup_cols = [col for col in df.columns if 'MATCHUP_AVG_' in col]
    matchup_delta_cols = [col for col in df.columns if 'MATCHUP_' in col and '_DELTA' in col]
    
    # For first game vs opponent, fill with 0 (no history yet)
    for col in matchup_cols:
        df[col] = df[col].fillna(0)
    
    # Fill delta NaN values with 0 (no difference from average)
    for col in matchup_delta_cols:
        df[col] = df[col].fillna(0)
    
    # Memory optimization
    for col in matchup_cols + matchup_delta_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    df['GAMES_VS_OPP'] = df['GAMES_VS_OPP'].astype('int8')
    
    return df

def assign_team_opp_def_by_position(df, min_minutes=10):
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

    # Calculate average minutes per player to filter by average (not per-game)
    df_work = df.copy()
    if 'PLAYER_AVG_MIN' not in df_work.columns:
        df_work['PLAYER_AVG_MIN'] = df_work.groupby('PLAYER_ID')['MIN'].transform('mean')
    
    # Filter for players with at least min_minutes on average
    df_filtered = df_work[df_work['PLAYER_AVG_MIN'] >= min_minutes].copy()
    df_filtered = df_filtered.sort_values(['TEAM_ID', 'GAME_DATE'])

    for pos in positions:
        # Get players at this position who average at least min_minutes
        pos_data = df_filtered[df_filtered[pos] == 1].copy()
        
        if pos_data.empty:
            continue
            
        # Calculate team defensive averages by position for each game (RAW/UNSHIFTED)
        # This represents how the team actually defended at that position in that game
        tmp = (
            pos_data
            .groupby(['TEAM_ID', 'GAME_ID', 'GAME_DATE'])[available_def_cols]
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
        
        # Calculate LEAGUE AVERAGE for this position metric (Shifted Expanding Mean)
        tmp = tmp.sort_values('GAME_DATE')
        
        for original_col, new_col in rename_dict.items():
            # Only calculate league average for ratings we care about
            league_avg_col = f'LEAGUE_AVG_{new_col.replace("TEAM_", "")}'
            
            # Calculate expanding mean of all teams' ratings up to previous day
            # Shift 1 to avoid including current game in the average (data leakage prevention)
            tmp[league_avg_col] = tmp[new_col].shift(1).expanding().mean().round(2)
            
            # Ensure same average for all games on the same date
            tmp[league_avg_col] = tmp.groupby('GAME_DATE')[league_avg_col].transform('first')
            
            # Fill initial NaNs with the first available value or global mean
            tmp[league_avg_col] = tmp[league_avg_col].fillna(tmp[new_col].mean())
        
        team_def_list.append(tmp)

    if not team_def_list:
        return df

    # Merge all position-based team stats
    team_def = team_def_list[0]
    for tmp in team_def_list[1:]:
        # Merge on TEAM_ID and GAME_ID (and GAME_DATE for safety/consistency)
        team_def = team_def.merge(tmp, on=['TEAM_ID', 'GAME_ID', 'GAME_DATE'], how='outer')
    
    # Merge with main dataframe
    # Using left join to keep all player rows
    df = df.merge(team_def.drop('GAME_DATE', axis=1), on=['TEAM_ID', 'GAME_ID'], how='left')
    
    # Create opponent versions (OPP_TEAM_ID needs to map to TEAM_ID stats)
    # We take the team_def dataframe and rename columns to OPP_
    # We exclude LEAGUE_AVG columns from renaming as they are global context
    opp_cols_rename = {col: col.replace('TEAM_', 'OPP_') for col in team_def.columns if 'TEAM_' in col}
    opp_cols_rename['TEAM_ID'] = 'OPP_TEAM_ID'
    
    opp_def = team_def.drop('GAME_DATE', axis=1).rename(columns=opp_cols_rename)
    
    # Drop LEAGUE_AVG columns from opp_def to avoid duplication/conflicts
    cols_to_drop = [c for c in opp_def.columns if 'LEAGUE_AVG' in c]
    opp_def = opp_def.drop(cols_to_drop, axis=1)
    
    df = df.merge(opp_def, on=['OPP_TEAM_ID', 'GAME_ID'], how='left')
    
    # Clean up temporary column
    if 'PLAYER_AVG_MIN' in df.columns:
        # Only drop if it wasn't in original df
        pass 
    
    return df

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
# TEAM STATISTICS AND CONTEXT - FIXED FOR DATA LEAKAGE
# ================================================================================================

def getOpponentStats(df, team_abbreviation='LAL'):
    """Get unique team stats per game with season-to-date averages - FIXED"""
    team_df = df[df['TEAM_ABBREVIATION'] == team_abbreviation].copy()
    team_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 
        'TEAM_DEF_RATING', 'TEAM_OFF_RATING', 'TEAM_PACE', 'TEAM_PTS', 'TEAM_FGA', 'TEAM_FGM', 'TEAM_FG3A', 'TEAM_FG3M', 'TEAM_FTA', 'TEAM_FTM', 'TEAM_REB', 'TEAM_AST', 'TEAM_TOV', 'TEAM_BLK', 'TEAM_STL'
    ]
    
    available_team_cols = [col for col in team_cols if col in team_df.columns]
    
    unique_games = team_df[available_team_cols].drop_duplicates(subset=['GAME_ID'])
    
    unique_games = unique_games.sort_values('GAME_DATE')
    
    # FIXED: All averages now use shift(1) to prevent data leakage
    unique_games['DEF_RATING_AVG_TO_DATE'] = unique_games['TEAM_DEF_RATING'].shift(1).expanding().mean().round(2)
    if 'TEAM_OFF_RATING' in unique_games.columns:
        unique_games['OFF_RATING_AVG_TO_DATE'] = unique_games['TEAM_OFF_RATING'].shift(1).expanding().mean().round(2)
    unique_games['PACE_AVG_TO_DATE'] = unique_games['TEAM_PACE'].shift(1).expanding().mean().round(2)
    unique_games['PTS_AVG_TO_DATE'] = unique_games['TEAM_PTS'].shift(1).expanding().mean().round(2)
    unique_games['FGA_AVG_TO_DATE'] = unique_games['TEAM_FGA'].shift(1).expanding().mean().round(2)
    unique_games['FGM_AVG_TO_DATE'] = unique_games['TEAM_FGM'].shift(1).expanding().mean().round(2)
    unique_games['FG3A_AVG_TO_DATE'] = unique_games['TEAM_FG3A'].shift(1).expanding().mean().round(2)
    unique_games['FG3M_AVG_TO_DATE'] = unique_games['TEAM_FG3M'].shift(1).expanding().mean().round(2)
    unique_games['FTA_AVG_TO_DATE'] = unique_games['TEAM_FTA'].shift(1).expanding().mean().round(2)
    unique_games['FTM_AVG_TO_DATE'] = unique_games['TEAM_FTM'].shift(1).expanding().mean().round(2)
    unique_games['REB_AVG_TO_DATE'] = unique_games['TEAM_REB'].shift(1).expanding().mean().round(2)
    unique_games['AST_AVG_TO_DATE'] = unique_games['TEAM_AST'].shift(1).expanding().mean().round(2)
    unique_games['TOV_AVG_TO_DATE'] = unique_games['TEAM_TOV'].shift(1).expanding().mean().round(2)
    unique_games['BLK_AVG_TO_DATE'] = unique_games['TEAM_BLK'].shift(1).expanding().mean().round(2)
    unique_games['STL_AVG_TO_DATE'] = unique_games['TEAM_STL'].shift(1).expanding().mean().round(2)
    unique_games['GAMES_PLAYED'] = range(1, len(unique_games) + 1)
    
    output_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 'GAMES_PLAYED',
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_PTS',
        'DEF_RATING_AVG_TO_DATE', 'OFF_RATING_AVG_TO_DATE', 'PACE_AVG_TO_DATE', 'PTS_AVG_TO_DATE', 
        'FGA_AVG_TO_DATE', 'FGM_AVG_TO_DATE', 'FG3A_AVG_TO_DATE', 'FG3M_AVG_TO_DATE',
        'FTA_AVG_TO_DATE', 'FTM_AVG_TO_DATE',
        'REB_AVG_TO_DATE', 'AST_AVG_TO_DATE', 'TOV_AVG_TO_DATE', 'BLK_AVG_TO_DATE', 'STL_AVG_TO_DATE'
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
                'OPP_OFF_RATING_AVG_TO_DATE': row.get('OFF_RATING_AVG_TO_DATE', None),
                'OPP_PACE_AVG_TO_DATE': row['PACE_AVG_TO_DATE'],
                'OPP_PTS_AVG_TO_DATE': row['PTS_AVG_TO_DATE'],
                'OPP_FGA_AVG_TO_DATE': row.get('FGA_AVG_TO_DATE', None),
                'OPP_FGM_AVG_TO_DATE': row.get('FGM_AVG_TO_DATE', None),
                'OPP_FG3A_AVG_TO_DATE': row.get('FG3A_AVG_TO_DATE', None),
                'OPP_FG3M_AVG_TO_DATE': row.get('FG3M_AVG_TO_DATE', None),
                'OPP_FTA_AVG_TO_DATE': row.get('FTA_AVG_TO_DATE', None),
                'OPP_FTM_AVG_TO_DATE': row.get('FTM_AVG_TO_DATE', None),
                'OPP_REB_AVG_TO_DATE': row['REB_AVG_TO_DATE'],
                'OPP_AST_AVG_TO_DATE': row['AST_AVG_TO_DATE'],
                'OPP_TOV_AVG_TO_DATE': row['TOV_AVG_TO_DATE'],
                'OPP_BLK_AVG_TO_DATE': row['BLK_AVG_TO_DATE'],
                'OPP_STL_AVG_TO_DATE': row['STL_AVG_TO_DATE']
            }
    
    # Assign opponent stats using vectorized lookup
    df_enhanced = df.copy()
    lookup_keys = list(zip(df_enhanced['GAME_ID'], df_enhanced['OPP_ABBREVIATION']))
    
    for col in ['OPP_DEF_RATING_AVG_TO_DATE', 'OPP_OFF_RATING_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE', 
                'OPP_PTS_AVG_TO_DATE', 'OPP_FGA_AVG_TO_DATE', 'OPP_FGM_AVG_TO_DATE', 
                'OPP_FG3A_AVG_TO_DATE', 'OPP_FG3M_AVG_TO_DATE', 'OPP_FTA_AVG_TO_DATE', 'OPP_FTM_AVG_TO_DATE',
                'OPP_REB_AVG_TO_DATE', 'OPP_AST_AVG_TO_DATE', 'OPP_TOV_AVG_TO_DATE', 
                'OPP_BLK_AVG_TO_DATE', 'OPP_STL_AVG_TO_DATE']:
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
    df['TEAM_FGM_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FGM'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_FG3A_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FG3A'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_FG3M_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FG3M'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_FTA_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FTA'].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df['TEAM_FTM_AVG_TO_DATE'] = df.groupby('TEAM_ID')['TEAM_FTM'].transform(
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
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_OFF_RATING', 'TEAM_PTS', 'TEAM_FG3A', 'TEAM_FTA',
        'TEAM_FGA', 'TEAM_REB', 'TEAM_AST', 'TEAM_TOV', 'TEAM_BLK', 'TEAM_STL', 'TEAM_FGM', 'TEAM_FG3M', 'TEAM_FTM'
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


def add_team_rolling_stats(df, team_id_col='TEAM_ID', date_col='GAME_DATE', windows=[3, 5, 7, 10]):
    """
    Add rolling averages for player's team statistics over specified windows.
    Shows how the player's own team has been performing in their recent games.
    
    Args:
        df: DataFrame with player and team data
        team_id_col: Column name for team identifier (default: 'TEAM_ID')
        date_col: Column name for date (default: 'GAME_DATE')
        windows: List of window sizes for rolling averages (default: [5, 10, 15])
        
    Returns:
        DataFrame with team rolling average features added
    """
    df = df.copy()
    df = df.sort_values([team_id_col, date_col]).reset_index(drop=True)
    
    # Define team stats to calculate rolling averages for
    team_stats = [
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_OFF_RATING', 'TEAM_PTS', 
        'TEAM_FGA', 'TEAM_REB', 'TEAM_AST', 'TEAM_TOV', 'TEAM_BLK', 'TEAM_STL',
        'TEAM_FG3A', 'TEAM_FTA', 'TEAM_FGM', 'TEAM_FG3M', 'TEAM_FTM'
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
    
    # Convert to appropriate data types to save memory
    rolling_cols = [col for col in df.columns if col.startswith('TEAM_') and '_ROLLING_AVG_' in col]
    for col in rolling_cols:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df

def expectedPace(df):
    df = df.copy()
    df['EXPECTED_PACE'] = ((df['TEAM_PACE_AVG_TO_DATE'] + df['OPP_PACE_AVG_TO_DATE']) / 2).round(2)
    df['PACE_DIFFERENTIAL'] = df['TEAM_PACE_AVG_TO_DATE'] - df['OPP_PACE_AVG_TO_DATE']    
    df['EXPECTED_POINTS'] = (df['TEAM_PTS_AVG_TO_DATE'] + df['OPP_PTS_AVG_TO_DATE']) / 2
    return df

def calculate_league_avg_team_def_rating(df, team_id_col='TEAM_ID', date_col='GAME_DATE'):
    """
    Calculate league average of TEAM_DEF_RATING using only one player per team per game.
    This avoids double-counting team stats since all players on a team have the same TEAM_DEF_RATING.
    """
    df = df.copy()
    
    if 'TEAM_DEF_RATING' not in df.columns:
        print("Warning: TEAM_DEF_RATING column not found")
        return df
    
    # Get unique team-game combinations (one row per team per game)
    # Use first player from each team-game combination
    unique_team_games = (
        df[[team_id_col, date_col, 'TEAM_DEF_RATING', 'GAME_ID']]
        .drop_duplicates(subset=[team_id_col, 'GAME_ID'])
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    
    # Calculate expanding mean with shift(1) to prevent data leakage
    # This gives league average up to (but not including) each game
    unique_team_games['LEAGUE_AVG_TEAM_DEF_RATING'] = (
        unique_team_games['TEAM_DEF_RATING']
        .shift(1)
        .expanding()
        .mean()
        .round(2)
    )
    
    # Forward fill within each date so all games on same date have same league avg
    unique_team_games['LEAGUE_AVG_TEAM_DEF_RATING'] = (
        unique_team_games.groupby(date_col)['LEAGUE_AVG_TEAM_DEF_RATING']
        .transform('first')
    )
    
    # Fill first game(s) with a default (mean of first date's games)
    if unique_team_games['LEAGUE_AVG_TEAM_DEF_RATING'].isna().any():
        first_date = unique_team_games[date_col].min()
        first_date_games = unique_team_games[unique_team_games[date_col] == first_date]
        default_league_avg = first_date_games['TEAM_DEF_RATING'].mean()
        unique_team_games['LEAGUE_AVG_TEAM_DEF_RATING'] = (
            unique_team_games['LEAGUE_AVG_TEAM_DEF_RATING'].fillna(default_league_avg)
        )
    
    # Merge back to original dataframe using GAME_ID and TEAM_ID
    # All players on same team in same game get the same league average
    league_avg_map = unique_team_games.set_index(['GAME_ID', team_id_col])['LEAGUE_AVG_TEAM_DEF_RATING'].to_dict()
    
    # Create a key for mapping
    df['_league_avg_key'] = list(zip(df['GAME_ID'], df[team_id_col]))
    df['LEAGUE_AVG_TEAM_DEF_RATING'] = df['_league_avg_key'].map(league_avg_map)
    
    # Fallback: if mapping fails, use overall mean
    if df['LEAGUE_AVG_TEAM_DEF_RATING'].isna().any():
        overall_mean = df['TEAM_DEF_RATING'].mean()
        df['LEAGUE_AVG_TEAM_DEF_RATING'] = df['LEAGUE_AVG_TEAM_DEF_RATING'].fillna(overall_mean)
    
    # Clean up temporary column
    df = df.drop('_league_avg_key', axis=1)
    
    return df

def calculate_league_avg_team_pace(df, team_id_col='TEAM_ID', date_col='GAME_DATE'):
    """
    Calculate league average of TEAM_PACE using only one player per team per game.
    This avoids double-counting team stats since all players on a team have the same TEAM_PACE.
    
    Args:
        df: DataFrame with player and team data
        team_id_col: Column name for team identifier (default: 'TEAM_ID')
        date_col: Column name for date (default: 'GAME_DATE')
    
    Returns:
        DataFrame with 'LEAGUE_AVG_TEAM_PACE' column added
    """
    df = df.copy()
    
    if 'TEAM_PACE' not in df.columns:
        print("Warning: TEAM_PACE column not found")
        return df
    
    # Get unique team-game combinations (one row per team per game)
    unique_team_games = (
        df[[team_id_col, date_col, 'TEAM_PACE', 'GAME_ID']]
        .drop_duplicates(subset=[team_id_col, 'GAME_ID'])
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    
    # Calculate expanding mean with shift(1) to prevent data leakage
    unique_team_games['LEAGUE_AVG_TEAM_PACE'] = (
        unique_team_games['TEAM_PACE']
        .shift(1)
        .expanding()
        .mean()
        .round(2)
    )
    
    # Forward fill within each date so all games on same date have same league avg
    unique_team_games['LEAGUE_AVG_TEAM_PACE'] = (
        unique_team_games.groupby(date_col)['LEAGUE_AVG_TEAM_PACE']
        .transform('first')
    )
    
    # Fill first game(s) with a default (mean of first date's games)
    if unique_team_games['LEAGUE_AVG_TEAM_PACE'].isna().any():
        first_date = unique_team_games[date_col].min()
        first_date_games = unique_team_games[unique_team_games[date_col] == first_date]
        default_league_avg = first_date_games['TEAM_PACE'].mean()
        unique_team_games['LEAGUE_AVG_TEAM_PACE'] = (
            unique_team_games['LEAGUE_AVG_TEAM_PACE'].fillna(default_league_avg)
        )
    
    # Merge back to original dataframe
    league_avg_map = unique_team_games.set_index(['GAME_ID', team_id_col])['LEAGUE_AVG_TEAM_PACE'].to_dict()
    
    df['_league_avg_key'] = list(zip(df['GAME_ID'], df[team_id_col]))
    df['LEAGUE_AVG_TEAM_PACE'] = df['_league_avg_key'].map(league_avg_map)
    
    # Fallback: if mapping fails, use overall mean
    if df['LEAGUE_AVG_TEAM_PACE'].isna().any():
        overall_mean = df['TEAM_PACE'].mean()
        df['LEAGUE_AVG_TEAM_PACE'] = df['LEAGUE_AVG_TEAM_PACE'].fillna(overall_mean)
    
    df = df.drop('_league_avg_key', axis=1)
    
    return df

def calculate_league_avg_team_off_rating(df, team_id_col='TEAM_ID', date_col='GAME_DATE'):
    """
    Calculate league average of TEAM_OFF_RATING using only one player per team per game.
    This avoids double-counting team stats since all players on a team have the same TEAM_OFF_RATING.
    
    Args:
        df: DataFrame with player and team data
        team_id_col: Column name for team identifier (default: 'TEAM_ID')
        date_col: Column name for date (default: 'GAME_DATE')
    
    Returns:
        DataFrame with 'LEAGUE_AVG_TEAM_OFF_RATING' column added
    """
    df = df.copy()
    
    if 'TEAM_OFF_RATING' not in df.columns:
        print("Warning: TEAM_OFF_RATING column not found")
        return df
    
    # Get unique team-game combinations (one row per team per game)
    unique_team_games = (
        df[[team_id_col, date_col, 'TEAM_OFF_RATING', 'GAME_ID']]
        .drop_duplicates(subset=[team_id_col, 'GAME_ID'])
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    
    # Calculate expanding mean with shift(1) to prevent data leakage
    unique_team_games['LEAGUE_AVG_TEAM_OFF_RATING'] = (
        unique_team_games['TEAM_OFF_RATING']
        .shift(1)
        .expanding()
        .mean()
        .round(2)
    )
    
    # Forward fill within each date so all games on same date have same league avg
    unique_team_games['LEAGUE_AVG_TEAM_OFF_RATING'] = (
        unique_team_games.groupby(date_col)['LEAGUE_AVG_TEAM_OFF_RATING']
        .transform('first')
    )
    
    # Fill first game(s) with a default (mean of first date's games)
    if unique_team_games['LEAGUE_AVG_TEAM_OFF_RATING'].isna().any():
        first_date = unique_team_games[date_col].min()
        first_date_games = unique_team_games[unique_team_games[date_col] == first_date]
        default_league_avg = first_date_games['TEAM_OFF_RATING'].mean()
        unique_team_games['LEAGUE_AVG_TEAM_OFF_RATING'] = (
            unique_team_games['LEAGUE_AVG_TEAM_OFF_RATING'].fillna(default_league_avg)
        )
    
    # Merge back to original dataframe
    league_avg_map = unique_team_games.set_index(['GAME_ID', team_id_col])['LEAGUE_AVG_TEAM_OFF_RATING'].to_dict()
    
    df['_league_avg_key'] = list(zip(df['GAME_ID'], df[team_id_col]))
    df['LEAGUE_AVG_TEAM_OFF_RATING'] = df['_league_avg_key'].map(league_avg_map)
    
    # Fallback: if mapping fails, use overall mean
    if df['LEAGUE_AVG_TEAM_OFF_RATING'].isna().any():
        overall_mean = df['TEAM_OFF_RATING'].mean()
        df['LEAGUE_AVG_TEAM_OFF_RATING'] = df['LEAGUE_AVG_TEAM_OFF_RATING'].fillna(overall_mean)
    
    df = df.drop('_league_avg_key', axis=1)
    
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
def process_star_players_data(df, min_minutes=10, min_games=5, name_dict=None):
    """
    Process star players data with optional name normalization.
    
    Args:
        df: DataFrame with player game data
        min_minutes: Minimum minutes to be considered active (default: 10)
        min_games: Minimum number of active games to be eligible for star (default: 5)
        name_dict: Optional dictionary to normalize player names (e.g., {'Luka Doncic': 'Luka Dončić'})
                   If None, will try to import from PRODUCTION.teamInfo
    """
    df = df.copy()
    
    # Try to import nameDict if not provided
    if name_dict is None:
        try:
            from PRODUCTION.teamInfo import nameDict
            name_dict = nameDict
        except ImportError:
            name_dict = None
    
    # Normalize player names if name_dict is provided
    if name_dict is not None:
        # Create reverse mapping for normalization (map variations to canonical form)
        # Also create forward mapping for consistency
        normalized_names = {}
        for variant, canonical in name_dict.items():
            normalized_names[variant] = canonical
            # Also map canonical to itself if not already present
            if canonical not in normalized_names:
                normalized_names[canonical] = canonical
        
        # Normalize PLAYER_NAME column
        df['PLAYER_NAME_NORM'] = df['PLAYER_NAME'].map(lambda x: normalized_names.get(x, x))
    else:
        df['PLAYER_NAME_NORM'] = df['PLAYER_NAME']
    
    # Create ACTIVE column based on minutes played
    df['ACTIVE'] = (df['MIN'] >= min_minutes).astype(int)

    # Season-long team star by composite score (only among active players)
    active_players = df[df['ACTIVE'] == 1].copy()
    
    # Count games per player per team to filter by min_games
    player_game_counts = (
        active_players.groupby(['TEAM_ID', 'PLAYER_NAME_NORM'], dropna=False)
        .size()
        .reset_index(name='GAME_COUNT')
    )
    
    # Filter to only players with enough games
    eligible_players = player_game_counts[player_game_counts['GAME_COUNT'] >= min_games]
    
    # Filter active_players to only eligible players using merge
    active_players = active_players.merge(
        eligible_players[['TEAM_ID', 'PLAYER_NAME_NORM']],
        on=['TEAM_ID', 'PLAYER_NAME_NORM'],
        how='inner'
    )
    
    # Calculate mean stats per player per team (using normalized names)
    player_stats = (
        active_players.groupby(['TEAM_ID', 'PLAYER_NAME_NORM'], dropna=False)
        .agg({
            'USG_PCT': 'mean',
            'TS_PCT': 'mean',
            'EFG_PCT': 'mean',
            'PTS': 'mean',
            'PIE': 'mean',  # Player Impact Estimate
            'NET_RATING': 'mean',
        })
        .reset_index()
        .rename(columns={'PLAYER_NAME_NORM': 'PLAYER_NAME'})
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

    # Map normalized star name back to dataframe
    df['STAR_NAME'] = df['TEAM_ID'].map(star_by_team)
    # Compare using normalized names to handle name variations
    df['PLAYER_IS_TEAM_STAR'] = (df['PLAYER_NAME_NORM'] == df['STAR_NAME']).astype(int)

    star_active_per_game = (
        df[df['PLAYER_NAME_NORM'] == df['STAR_NAME']]
        .groupby(['GAME_ID', 'TEAM_ID'], as_index=False)['ACTIVE']
        .max()
        .rename(columns={'ACTIVE': 'STAR_ACTIVE'})
    )
    df = df.merge(star_active_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['STAR_ACTIVE'] = df['STAR_ACTIVE'].fillna(0).astype(int)
    df['STAR_SAT_OUT'] = ((df['PLAYER_IS_TEAM_STAR'] == 0) & (df['STAR_ACTIVE'] == 0)).astype(int)

    df = df.drop(columns=['STAR_NAME', 'STAR_ACTIVE', 'ACTIVE', 'PLAYER_NAME_NORM'])

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
    
    metrics = ['MIN', 'PTS', 'FGA','FTA', 'FG3A', 'FG3M', 'FTM', 'FGM', 'TOV', 'USG_PCT', 'TS_PCT', 'EFG_PCT',
    'UFGA', 'UFGM', 'PF', 'AST_TO_TOV', 'PLUS_MINUS',
    'E_OFF_RATING', 'NET_RATING']
    
    def calculate_without_star_stats(player_group):
        player_group = player_group.copy()
        
        player_group['STAR_SAT_OUT_SHIFTED'] = player_group['STAR_SAT_OUT'].shift(1)
        
        star_out_mask = player_group['STAR_SAT_OUT_SHIFTED'] == 1
        star_in_mask = player_group['STAR_SAT_OUT_SHIFTED'] == 0
        
        if star_out_mask.sum() >= min_games and star_in_mask.sum() >= min_games:
            star_out_data = player_group[star_out_mask]
            star_in_data = player_group[star_in_mask]
            
            baseline_performance = {}
            for metric in metrics:
                if metric in player_group.columns:
                    baseline_performance[metric] = star_in_data[metric].mean()
            star_out_performance = {}
            for metric in metrics:
                if metric in player_group.columns:
                    star_out_performance[metric] = star_out_data[metric].mean()
            
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


##############################################################################################################
# VOLATILITY FEATURES
##############################################################################################################
def add_volatility_features(df, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[5, 7, 10, 20, 25]):
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    # Count/volume stats - use standard deviation
    count_stats = ['PTS', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'MIN', 'AST', 'E_OFF_RATING', 'PLUS_MINUS', 'POSS', 'TCHS']
    
    # Percentage/rate stats - use coefficient of variation
    pct_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'TS_PCT', 'USG_PCT', 'EFG_PCT']
    
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


def get_standard_deviation(df, stats=None, windows=None, player_id_col='PLAYER_ID', date_col='GAME_DATE', 
                           min_periods=2, expanding=False, suffix='_STD'):
    df = df.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    # Default stats if not provided
    if stats is None:
        stats = ['PTS', 'MIN', 'FGA', 'FTA', 'USG_PCT', 'E_OFF_RATING', 'TS_PCT', 'NET_RATING', 'FG3A', 'PLUS_MINUS', 'PF', 'OREB', 'PTS_OFF_TOV', 'PTS_2ND_CHANCE', 'PTS_FB', 'PTS_PAINT']
    
    # Filter to only stats that exist in dataframe
    stats = [stat for stat in stats if stat in df.columns]
    
    if expanding:
        # Expanding standard deviation (to-date)
        for stat in stats:
            col_name = f'{stat}{suffix}_TO_DATE'
            df[col_name] = (
                df.groupby(player_id_col)[stat]
                .transform(lambda x: x.shift(1).expanding(min_periods=min_periods).std())
                .round(4)
            )
    else:
        # Rolling standard deviation
        if windows is None:
            windows = [3, 5, 7, 15]
        
        for window in windows:
            for stat in stats:
                col_name = f'{stat}{suffix}_{window}_TO_DATE'
                df[col_name] = (
                    df.groupby(player_id_col)[stat]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).std())
                    .round(4)
                )
    
    return df

def add_interaction_features(df):
    epsilon = 1e-8
    df = df.copy()
    
    df['OPP_DEF_RATING_OVER_LEAGUE_AVG'] = round(df['OPP_DEF_RATING_AVG_TO_DATE'] / (df['LEAGUE_AVG_TEAM_DEF_RATING'] + epsilon), 2)
    df['OPP_DEF_RATING_L3_OVER_LEAGUE_AVG'] = round(df['OPP_TEAM_DEF_RATING_ROLLING_AVG_3'] / (df['LEAGUE_AVG_TEAM_DEF_RATING'] + epsilon), 2)
    df['OPP_PACE_OVER_LEAGUE_AVG'] = round(df['OPP_PACE_AVG_TO_DATE'] / (df['LEAGUE_AVG_TEAM_PACE'] + epsilon), 2)
    df['OPP_PACE_L3_OVER_LEAGUE_AVG'] = round(df['OPP_TEAM_PACE_ROLLING_AVG_3'] / (df['LEAGUE_AVG_TEAM_PACE'] + epsilon), 2)
    df['OPP_OFF_RATING_OVER_LEAGUE_AVG'] = round(df['OPP_OFF_RATING_AVG_TO_DATE'] / (df['LEAGUE_AVG_TEAM_OFF_RATING'] + epsilon), 2)
    df['OPP_OFF_RATING_L3_OVER_LEAGUE_AVG'] = round(df['OPP_TEAM_OFF_RATING_ROLLING_AVG_3'] / (df['LEAGUE_AVG_TEAM_OFF_RATING'] + epsilon), 2)
    
    df['GUARD_DEF_RATING_OVER_LEAGUE_AVG'] = df['GUARD'] * round(df['OPP_GUARD_DEF_RATING'] / df['LEAGUE_AVG_GUARD_DEF_RATING'], 2)
    df['FORWARD_DEF_RATING_OVER_LEAGUE_AVG'] = df['FORWARD'] * round(df['OPP_FORWARD_DEF_RATING'] / df['LEAGUE_AVG_FORWARD_DEF_RATING'], 2)
    df['CENTER_DEF_RATING_OVER_LEAGUE_AVG'] = df['CENTER'] * round(df['OPP_CENTER_DEF_RATING'] / df['LEAGUE_AVG_CENTER_DEF_RATING'], 2)
    
    df['TEAM_OFF_RATING_OVER_LEAGUE_AVG'] = round(df['TEAM_OFF_RATING_AVG_TO_DATE'] / (df['LEAGUE_AVG_TEAM_OFF_RATING'] + epsilon), 2)
    df['TEAM_OFF_RATING_L3_OVER_LEAGUE_AVG'] = round(df['TEAM_OFF_RATING_ROLLING_AVG_3'] / (df['LEAGUE_AVG_TEAM_OFF_RATING'] + epsilon), 2)
    df['TEAM_OFF_RATING_L5_OVER_LEAGUE_AVG'] = round(df['TEAM_OFF_RATING_ROLLING_AVG_5'] / (df['LEAGUE_AVG_TEAM_OFF_RATING'] + epsilon), 2)
    df['TEAM_PACE_OVER_LEAGUE_AVG'] = round(df['TEAM_PACE_AVG_TO_DATE'] / (df['LEAGUE_AVG_TEAM_PACE'] + epsilon), 2)
    df['TEAM_PACE_L3_OVER_LEAGUE_AVG'] = round(df['TEAM_PACE_ROLLING_AVG_3'] / (df['LEAGUE_AVG_TEAM_PACE'] + epsilon), 2)
    df['TEAM_PACE_L5_OVER_LEAGUE_AVG'] = round(df['TEAM_PACE_ROLLING_AVG_5'] / (df['LEAGUE_AVG_TEAM_PACE'] + epsilon), 2)

    df['PTS_L5_OVER_BASELINE'] = round(df['PTS_ROLLING_AVG_5'] / (df['PTS_AVG_TO_DATE'] + epsilon), 2)
    df['PTS_L10_OVER_BASELINE'] = round(df['PTS_ROLLING_AVG_10'] / (df['PTS_AVG_TO_DATE'] + epsilon), 2)
    df['PTS_L7_OVER_BASELINE'] = round(df['PTS_ROLLING_AVG_7'] / (df['PTS_AVG_TO_DATE'] + epsilon), 2)
    df['MIN_L5_OVER_BASELINE'] = round(df['MIN_ROLLING_AVG_5'] / (df['MIN_AVG_TO_DATE'] + epsilon), 2)
    df['MIN_L10_OVER_BASELINE'] = round(df['MIN_ROLLING_AVG_10'] / (df['MIN_AVG_TO_DATE'] + epsilon), 2)
    df['MIN_L7_OVER_BASELINE'] = round(df['MIN_ROLLING_AVG_7'] / (df['MIN_AVG_TO_DATE'] + epsilon), 2)
    df['FGA_L5_OVER_BASELINE'] = round(df['FGA_ROLLING_AVG_5'] / (df['FGA_AVG_TO_DATE'] + epsilon), 2)
    df['FGA_L10_OVER_BASELINE'] = round(df['FGA_ROLLING_AVG_10'] / (df['FGA_AVG_TO_DATE'] + epsilon), 2)
    df['FGA_L7_OVER_BASELINE'] = round(df['FGA_ROLLING_AVG_7'] / (df['FGA_AVG_TO_DATE'] + epsilon), 2)
    df['FG_PCT_L10_OVER_BASELINE'] = round(df['FG_PCT_ROLLING_AVG_10'] / (df['FG_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FG_PCT_L5_OVER_BASELINE'] = round(df['FG_PCT_ROLLING_AVG_5'] / (df['FG_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FG_PCT_L7_OVER_BASELINE'] = round(df['FG_PCT_ROLLING_AVG_7'] / (df['FG_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FG3_PCT_L10_OVER_BASELINE'] = round(df['FG3_PCT_ROLLING_AVG_10'] / (df['FG3_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FG3_PCT_L5_OVER_BASELINE'] = round(df['FG3_PCT_ROLLING_AVG_5'] / (df['FG3_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FG3_PCT_L7_OVER_BASELINE'] = round(df['FG3_PCT_ROLLING_AVG_7'] / (df['FG3_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FT_PCT_L10_OVER_BASELINE'] = round(df['FT_PCT_ROLLING_AVG_10'] / (df['FT_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FT_PCT_L5_OVER_BASELINE'] = round(df['FT_PCT_ROLLING_AVG_5'] / (df['FT_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FT_PCT_L7_OVER_BASELINE'] = round(df['FT_PCT_ROLLING_AVG_7'] / (df['FT_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['FTA_L10_OVER_BASELINE'] = round(df['FTA_ROLLING_AVG_10'] / (df['FTA_AVG_TO_DATE'] + epsilon), 2)
    df['FTA_L5_OVER_BASELINE'] = round(df['FTA_ROLLING_AVG_5'] / (df['FTA_AVG_TO_DATE'] + epsilon), 2)
    df['FTA_L7_OVER_BASELINE'] = round(df['FTA_ROLLING_AVG_7'] / (df['FTA_AVG_TO_DATE'] + epsilon), 2)
    df['FG3A_L10_OVER_BASELINE'] = round(df['FG3A_ROLLING_AVG_10'] / (df['FG3A_AVG_TO_DATE'] + epsilon), 2)
    df['FG3A_L5_OVER_BASELINE'] = round(df['FG3A_ROLLING_AVG_5'] / (df['FG3A_AVG_TO_DATE'] + epsilon), 2)
    df['FG3A_L7_OVER_BASELINE'] = round(df['FG3A_ROLLING_AVG_7'] / (df['FG3A_AVG_TO_DATE'] + epsilon), 2)
    df['USG_PCT_L10_OVER_BASELINE'] = round(df['USG_PCT_ROLLING_AVG_10'] / (df['USG_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['USG_PCT_L5_OVER_BASELINE'] = round(df['USG_PCT_ROLLING_AVG_5'] / (df['USG_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['USG_PCT_L7_OVER_BASELINE'] = round(df['USG_PCT_ROLLING_AVG_7'] / (df['USG_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['TS_PCT_L10_OVER_BASELINE'] = round(df['TS_PCT_ROLLING_AVG_10'] / (df['TS_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['TS_PCT_L5_OVER_BASELINE'] = round(df['TS_PCT_ROLLING_AVG_5'] / (df['TS_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['TS_PCT_L7_OVER_BASELINE'] = round(df['TS_PCT_ROLLING_AVG_7'] / (df['TS_PCT_AVG_TO_DATE'] + epsilon), 2)
    df['NET_RATING_L10_OVER_BASELINE'] = round(df['NET_RATING_ROLLING_AVG_10'] / (df['NET_RATING_AVG_TO_DATE'] + epsilon), 2)
    df['NET_RATING_L5_OVER_BASELINE'] = round(df['NET_RATING_ROLLING_AVG_5'] / (df['NET_RATING_AVG_TO_DATE'] + epsilon), 2)
    df['NET_RATING_L7_OVER_BASELINE'] = round(df['NET_RATING_ROLLING_AVG_7'] / (df['NET_RATING_AVG_TO_DATE'] + epsilon), 2)
    df['E_OFF_RATING_L10_OVER_BASELINE'] = round(df['E_OFF_RATING_ROLLING_AVG_10'] / (df['E_OFF_RATING_AVG_TO_DATE'] + epsilon), 2)
    df['E_OFF_RATING_L5_OVER_BASELINE'] = round(df['E_OFF_RATING_ROLLING_AVG_5'] / (df['E_OFF_RATING_AVG_TO_DATE'] + epsilon), 2)
    df['E_OFF_RATING_L7_OVER_BASELINE'] = round(df['E_OFF_RATING_ROLLING_AVG_7'] / (df['E_OFF_RATING_AVG_TO_DATE'] + epsilon), 2)
    df['TCHS_L10_OVER_BASELINE'] = round(df['TCHS_ROLLING_AVG_10'] / (df['TCHS_AVG_TO_DATE'] + epsilon), 2)
    df['TCHS_L5_OVER_BASELINE'] = round(df['TCHS_ROLLING_AVG_5'] / (df['TCHS_AVG_TO_DATE'] + epsilon), 2)
    df['TCHS_L7_OVER_BASELINE'] = round(df['TCHS_ROLLING_AVG_7'] / (df['TCHS_AVG_TO_DATE'] + epsilon), 2)
    df['PLUS_MINUS_L10_OVER_BASELINE'] = round(df['PLUS_MINUS_ROLLING_AVG_10'] / (df['PLUS_MINUS_AVG_TO_DATE'] + epsilon), 2)
    df['PLUS_MINUS_L5_OVER_BASELINE'] = round(df['PLUS_MINUS_ROLLING_AVG_5'] / (df['PLUS_MINUS_AVG_TO_DATE'] + epsilon), 2)
    df['PLUS_MINUS_L7_OVER_BASELINE'] = round(df['PLUS_MINUS_ROLLING_AVG_7'] / (df['PLUS_MINUS_AVG_TO_DATE'] + epsilon), 2)
    df['POSS_L10_OVER_BASELINE'] = round(df['POSS_ROLLING_AVG_10'] / (df['POSS_AVG_TO_DATE'] + epsilon), 2)
    df['POSS_L5_OVER_BASELINE'] = round(df['POSS_ROLLING_AVG_5'] / (df['POSS_AVG_TO_DATE'] + epsilon), 2)
    df['POSS_L7_OVER_BASELINE'] = round(df['POSS_ROLLING_AVG_7'] / (df['POSS_AVG_TO_DATE'] + epsilon), 2)
    df['TCHS_L10_OVER_BASELINE'] = round(df['TCHS_ROLLING_AVG_10'] / (df['TCHS_AVG_TO_DATE'] + epsilon), 2)
    df['TCHS_L5_OVER_BASELINE'] = round(df['TCHS_ROLLING_AVG_5'] / (df['TCHS_AVG_TO_DATE'] + epsilon), 2)
    df['TCHS_L7_OVER_BASELINE'] = round(df['TCHS_ROLLING_AVG_7'] / (df['TCHS_AVG_TO_DATE'] + epsilon), 2)

    # ===== SHORT VS LONG TERM DIVERGENCES =====
    df['PTS_10G_VS_SEASON_RATIO'] = df['PTS_ROLLING_AVG_10'] / (df['PTS_AVG_TO_DATE'] + epsilon)
    df['PTS_10G_VS_40G_RATIO'] = df.get('PTS_ROLLING_AVG_10', df['PTS_ROLLING_AVG_10']) / (df.get('PTS_ROLLING_AVG_40', df['PTS_ROLLING_AVG_20']) + epsilon)
    df['FGA_10G_VS_SEASON_RATIO'] = df['FGA_ROLLING_AVG_10'] / (df['FGA_AVG_TO_DATE'] + epsilon)
    df['FGA_10G_VS_40G_RATIO'] = df.get('FGA_ROLLING_AVG_10', df['FGA_ROLLING_AVG_10']) / (df.get('FGA_ROLLING_AVG_40', df['FGA_ROLLING_AVG_20']) + epsilon)
    df['TCHS_5G_VS_SEASON_RATIO'] = df.get('TCHS_ROLLING_AVG_5', 0) / (df.get('TCHS_AVG_TO_DATE', 1) + epsilon)
    df['TCHS_10G_VS_SEASON_RATIO'] = df.get('TCHS_ROLLING_AVG_10', 0) / (df.get('TCHS_AVG_TO_DATE', 1) + epsilon)
    
    # ===== VARIANCE STABILITY (Keep only what's in features) =====
    df['PTS_VARIANCE_STABILITY'] = df.get('PTS_VOLATILITY_10_TO_DATE', 0) / (df.get('PTS_VOLATILITY_40_TO_DATE', 1) + epsilon)
    df['MIN_VARIANCE_STABILITY'] = df.get('MIN_VOLATILITY_15_TO_DATE', 0) / (df.get('MIN_VOLATILITY_40_TO_DATE', 1) + epsilon)
    df['USG_PCT_VARIANCE_STABILITY'] = df.get('USG_PCT_CV_10_TO_DATE', 0) / (df.get('USG_PCT_CV_40_TO_DATE', 1) + epsilon)
    df['TS_PCT_VARIANCE_STABILITY'] = df.get('TS_PCT_CV_10_TO_DATE', 0) / (df.get('TS_PCT_CV_40_TO_DATE', 1) + epsilon)
    df['FTA_VARIANCE_STABILITY'] = df.get('FTA_VOLATILITY_10_TO_DATE', 0) / (df.get('FTA_VOLATILITY_40_TO_DATE', 1) + epsilon)
    df['FGA_VARIANCE_STABILITY'] = df.get('FGA_VOLATILITY_10_TO_DATE', 0) / (df.get('FGA_VOLATILITY_40_TO_DATE', 1) + epsilon)
    df['FG3A_VARIANCE_STABILITY'] = df.get('FG3A_VOLATILITY_10_TO_DATE', 0) / (df.get('FG3A_VOLATILITY_40_TO_DATE', 1) + epsilon)
    df['FG_PCT_VARIANCE_STABILITY'] = df.get('FG_PCT_CV_10_TO_DATE', 0) / (df.get('FG_PCT_CV_40_TO_DATE', 1) + epsilon)
    df['FG3_PCT_VARIANCE_STABILITY'] = df.get('FG3_PCT_CV_10_TO_DATE', 0) / (df.get('FG3_PCT_CV_40_TO_DATE', 1) + epsilon)
    df['FT_PCT_VARIANCE_STABILITY'] = df.get('FT_PCT_CV_10_TO_DATE', 0) / (df.get('FT_PCT_CV_40_TO_DATE', 1) + epsilon)
    df['TCHS_VARIANCE_STABILITY'] = df.get('TCHS_VOLATILITY_15_TO_DATE', 0) / (df.get('TCHS_VOLATILITY_40_TO_DATE', 1) + epsilon)    

    # ===== STAR DYNAMICS (Keep only what's in features) =====
    if 'STAR_SAT_OUT' in df.columns:
        df['PTS_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('PTS_DELTA_STAR_OUT', 0)
        df['MIN_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('MIN_DELTA_STAR_OUT', 0)
        df['FGA_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('FGA_DELTA_STAR_OUT', 0)
        df['FGM_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('FGM_DELTA_STAR_OUT', 0)
        df['FG3A_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('FG3A_DELTA_STAR_OUT', 0)
        df['FG3M_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('FG3M_DELTA_STAR_OUT', 0)
        df['FTA_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('FTA_DELTA_STAR_OUT', 0)
        df['FTM_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('FTM_DELTA_STAR_OUT', 0)
        df['EFG_PCT_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('EFG_PCT_DELTA_STAR_OUT', 0)
        df['TS_PCT_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('TS_PCT_DELTA_STAR_OUT', 0)
        df['USG_PCT_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('USG_PCT_DELTA_STAR_OUT', 0)
        df['E_OFF_RATING_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('E_OFF_RATING_DELTA_STAR_OUT', 0)
        df['NET_RATING_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('NET_RATING_DELTA_STAR_OUT', 0)
        df['PLUS_MINUS_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('PLUS_MINUS_DELTA_STAR_OUT', 0)
        df['PTS_OFF_TOV_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('PTS_OFF_TOV_DELTA_STAR_OUT', 0)
        df['PTS_2ND_CHANCE_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('PTS_2ND_CHANCE_DELTA_STAR_OUT', 0)
        df['PTS_FB_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('PTS_FB_DELTA_STAR_OUT', 0)
        df['PTS_PAINT_BOOST_STAR_OUT'] = df['STAR_SAT_OUT'] * df.get('PTS_PAINT_DELTA_STAR_OUT', 0)
    
    # ===== HOME/AWAY EXPECTATIONS (Keep only what's in features) =====
    if 'HOME_GAME' in df.columns:
        # Use deltas instead of raw averages: base_avg + location_delta
        df['PTS_EXPECTATION_LOCATION'] = (
            df.get('PTS_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_PTS_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_PTS_DELTA', 0)
        )
        df['MIN_EXPECTATION_LOCATION'] = (
            df.get('MIN_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_MIN_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_MIN_DELTA', 0)
        )
        df['AST_EXPECTATION_LOCATION'] = (
            df.get('AST_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_AST_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_AST_DELTA', 0)
        )
        df['USG_PCT_EXPECTATION_LOCATION'] = (
            df.get('USG_PCT_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_USG_PCT_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_USG_PCT_DELTA', 0)
        )
        df['E_OFF_RATING_EXPECTATION_LOCATION'] = (
            df.get('E_OFF_RATING_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_E_OFF_RATING_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_E_OFF_RATING_DELTA', 0)
        )
        df['NET_RATING_EXPECTATION_LOCATION'] = (
            df.get('NET_RATING_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_NET_RATING_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_NET_RATING_DELTA', 0)
        )
        df['EFG_PCT_EXPECTATION_LOCATION'] = (
            df.get('EFG_PCT_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_EFG_PCT_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_EFG_PCT_DELTA', 0)
        )
        df['FGA_EXPECTATION_LOCATION'] = (
            df.get('FGA_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_FGA_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_FGA_DELTA', 0)
        )
        df['FGM_EXPECTATION_LOCATION'] = (
            df.get('FGM_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_FGM_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_FGM_DELTA', 0)
        )
        df['FTA_EXPECTATION_LOCATION'] = (
            df.get('FTA_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_FTA_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_FTA_DELTA', 0)
        )
        df['FG3A_EXPECTATION_LOCATION'] = (
            df.get('FG3A_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_FG3A_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_FG3A_DELTA', 0)
        )
        df['FG3M_EXPECTATION_LOCATION'] = (
            df.get('FG3M_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_FG3M_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_FG3M_DELTA', 0)
        )
        df['TS_PCT_EXPECTATION_LOCATION'] = (
            df.get('TS_PCT_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_TS_PCT_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_TS_PCT_DELTA', 0)
        )
        df['POSS_EXPECTATION_LOCATION'] = (
            df.get('POSS_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_POSS_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_POSS_DELTA', 0)
        )
        df['TCHS_EXPECTATION_LOCATION'] = (
            df.get('TCHS_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_TCHS_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_TCHS_DELTA', 0)
        )
        df['PLUS_MINUS_EXPECTATION_LOCATION'] = (
            df.get('PLUS_MINUS_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_PLUS_MINUS_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_PLUS_MINUS_DELTA', 0)
        )
        df['PTS_OFF_TOV_EXPECTATION_LOCATION'] = (
            df.get('PTS_OFF_TOV_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_PTS_OFF_TOV_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_PTS_OFF_TOV_DELTA', 0)
        )
        df['PTS_2ND_CHANCE_EXPECTATION_LOCATION'] = (
            df.get('PTS_2ND_CHANCE_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_PTS_2ND_CHANCE_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_PTS_2ND_CHANCE_DELTA', 0)
        )
        df['PTS_FB_EXPECTATION_LOCATION'] = (
            df.get('PTS_FB_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_PTS_FB_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_PTS_FB_DELTA', 0)
        )
        df['PTS_PAINT_EXPECTATION_LOCATION'] = (
            df.get('PTS_PAINT_AVG_TO_DATE', 0) +
            df['HOME_GAME'] * df.get('PLAYER_HOME_PTS_PAINT_DELTA', 0) +
            (1 - df['HOME_GAME']) * df.get('PLAYER_AWAY_PTS_PAINT_DELTA', 0)
        )
    return df