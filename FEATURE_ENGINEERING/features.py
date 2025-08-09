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
            return float(min_str)
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
    return (feet * 12) + inches


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

def encode_teams(df):
    """One-hot encode player team and opponent team."""
    df_teams = pd.get_dummies(df['TEAM_ABBREVIATION'], prefix='TEAM_').astype(int)
    df_opps = pd.get_dummies(df['OPP_ABBREVIATION'], prefix='OPP_').astype(int)
    df_encoded = pd.concat([df, df_teams, df_opps], axis=1)
    return df_encoded


# ================================================================================================
# ROLLING AVERAGES AND TIME SERIES FEATURES
# ================================================================================================

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', windows=[3,5,7]):
    """Calculate rolling averages for key player statistics."""
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)

    stats_cols = [
        'MIN', 'PTS', 'FGA', 'FG3A', 'FTA', 'USG_PCT', 'POINT_PER_SHOT', 'TS_PCT', 'OFF_RATING'
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

def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    """Add lag features for specified statistic."""
    player_data = player_data.sort_values([player_id_col, date_col])
    
    for lag in range(1, 5):
        lag_col = f'{stat_line}_LAG_{lag}'
        
        # Create Lag
        player_data[lag_col] = player_data.groupby(player_id_col)[stat_line].shift(lag)
        
        # Compute expanding mean up to current row, aligned correctly using transform
        rolling_mean = player_data.groupby(player_id_col)[stat_line].transform(lambda x: x.shift(1).expanding().mean())
        
        # Fill NaNs in lag with rolling mean
        player_data[lag_col] = player_data[lag_col].fillna(rolling_mean)
        
        # Fill remaining NaNs (e.g., first game) with 0
        player_data[lag_col] = player_data[lag_col].fillna(0)
    
    return player_data


# ================================================================================================
# PLAYER AVERAGE TO DATE FUNCTIONS
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
    stats_to_average = ['PTS', 'MIN', 'FGA', 'FTA', 'USG_PCT', 'TS_PCT']
    
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
        
        # Calculate expanding averages for each stat
        for stat in stats_to_average:
            if stat in df_enhanced.columns:
                # Calculate expanding mean and shift by 1
                expanding_avg = player_data[stat].expanding().mean().shift(1)
                df_enhanced.loc[player_mask, f'{stat}_AVG_TO_DATE'] = expanding_avg.round(2)
    
    return df_enhanced

def getPlayerAvgToDateVectorized(df, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Vectorized version that should be faster and avoid multi-index issues.
    """
    # Create copy and sort
    df_enhanced = df.copy().sort_values([player_id_col, date_col]).reset_index(drop=True)
    
    # Define stats
    stats_to_average = ['PTS', 'MIN', 'FGA', 'FTA', 'USG_PCT', 'TS_PCT']
    
    # Use transform to avoid multi-index issues
    for stat in stats_to_average:
        if stat in df_enhanced.columns:
            df_enhanced[f'{stat}_AVG_TO_DATE'] = (
                df_enhanced.groupby(player_id_col)[stat]
                .transform(lambda x: x.expanding().mean().shift(1))
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
# HOME/AWAY AND MATCHUP SPECIFIC FEATURES
# ================================================================================================

def HomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    """
    Calculate home/away rolling averages (expanding) for key metrics,
    prevent data leakage via shift(1), and fill missing values with the opposite location's current average.
    All results are rounded to 2 decimal places.
    """
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)
    
    metrics = ['PTS', 'USG_PCT', 'POSS', 'PACE', 'OFF_RATING']
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

def statAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION', stat_line='PTS'):
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
        'PTS': [3],
        'USG_PCT': [3],
        'POSS': [3],
        'OFF_RATING': [3]
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


# ================================================================================================
# OPPONENT AND DEFENSIVE FEATURES
# ================================================================================================

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

def add_all_opponent_features(player_data, stat_line='PTS'):
    """Wrapper function to add all opponent-related features"""
    player_data = dynamic_defense_ranking(player_data)
    return player_data


# ================================================================================================
# TEAM STATISTICS AND CONTEXT
# ================================================================================================

def getOpponentStats(df, team_abbreviation='LAL'):
    """Get unique team stats per game with season-to-date averages."""
    team_df = df[df['TEAM_ABBREVIATION'] == team_abbreviation].copy()
    team_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_PTS'
    ]
    
    available_team_cols = [col for col in team_cols if col in team_df.columns]
    
    unique_games = team_df[available_team_cols].drop_duplicates(subset=['GAME_ID'])
    
    unique_games = unique_games.sort_values('GAME_DATE')
    
    unique_games['DEF_RATING_AVG_TO_DATE'] = unique_games['TEAM_DEF_RATING'].expanding().mean().shift(1).round(2)
    unique_games['PACE_AVG_TO_DATE'] = unique_games['TEAM_PACE'].expanding().mean().shift(1).round(2)
    unique_games['PTS_AVG_TO_DATE'] = unique_games['TEAM_PTS'].expanding().mean().shift(1).round(2)
    
    unique_games['GAMES_PLAYED'] = range(1, len(unique_games) + 1)
    
    output_cols = [
        'GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION', 'OPP_ABBREVIATION', 'GAMES_PLAYED',
        'TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_PTS',
        'DEF_RATING_AVG_TO_DATE', 'PACE_AVG_TO_DATE', 'PTS_AVG_TO_DATE'
    ]
    
    available_cols = [col for col in output_cols if col in unique_games.columns]
    
    return unique_games[available_cols].reset_index(drop=True)

def assign_opponent_team_stats_dict(df):
    """Assign opponent team stats using dictionary lookup for efficiency."""
    team_stats_dict = {}
    
    for team in df['TEAM_ABBREVIATION'].unique():
        team_stats = getOpponentStats(df, team)
        for _, row in team_stats.iterrows():
            key = (row['GAME_ID'], team)
            team_stats_dict[key] = {
                'OPP_DEF_RATING_AVG_TO_DATE': row['DEF_RATING_AVG_TO_DATE'],
                'OPP_PACE_AVG_TO_DATE': row['PACE_AVG_TO_DATE'],
                'OPP_PTS_AVG_TO_DATE': row['PTS_AVG_TO_DATE']
            }
    
    # Assign opponent stats using vectorized lookup
    df_enhanced = df.copy()
    lookup_keys = list(zip(df_enhanced['GAME_ID'], df_enhanced['OPP_ABBREVIATION']))
    
    for col in ['OPP_DEF_RATING_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE', 'OPP_PTS_AVG_TO_DATE']:
        df_enhanced[col] = [team_stats_dict.get(key, {}).get(col, None) for key in lookup_keys]
    
    return df_enhanced

def teamContext(df):
    """Add team context features with season-to-date averages."""
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
    """Calculate team starter spacing metric based on 3PT%."""
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
    """Calculate expected pace based on team and opponent starter averages."""
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
    """Process star players data for a single season and return simple star metrics."""
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
    """Wrapper function to add all lineup-related features."""
    df = teamUsualStarters(df)
    df = oppTeamUsualStarters(df)
    df = team_starter_spacing(df)
    df = pace_expectation(df)
    df = process_star_players_data(df, star_players)
    return df


# ================================================================================================
# PACE AND USAGE FEATURES
# ================================================================================================

def add_game_pace_adjustment(df):
    """Calculate game pace adjustments with data leakage prevention using shift(1)."""
    # Create copy and pre-sort for time-series operations
    df = df.copy()
    df.sort_values(['GAME_DATE'], inplace=True)
    
    # Create team grouper objects once
    team_group = df.groupby('TEAM_ID')
    opp_team_group = df.groupby('OPP_TEAM_ID')
    
    # Shift team paces by 1 to prevent data leakage
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
    pace_adjustments.columns = ['OPP_TEAM_ID', 'OPP_GAME_PACE_ADJUSTMENT']
    
    # Merge adjustments and convert to float32 to save memory
    df = df.merge(pace_adjustments, on='OPP_TEAM_ID', how='left')
    df['OPP_GAME_PACE_ADJUSTMENT'] = df['OPP_GAME_PACE_ADJUSTMENT'].astype('float32')
    
    # Add rolling windows for pace adjustments
    windows = [3, 5, 7]
    
    for window in windows:
        col_name = f'OPP_GAME_PACE_ADJ_ROLL_{window}'
        
        # Calculate rolling average on the adjustment
        df[col_name] = (
            opp_team_group['OPP_GAME_PACE_ADJUSTMENT']
            .shift(1)  # Additional shift for rolling calculation
            .rolling(window=window, min_periods=1)
            .mean()
            .round(2)
        )
        
        # Fill first games with overall mean
        first_games_mask = opp_team_group.cumcount() == 0
        df.loc[first_games_mask, col_name] = df['OPP_GAME_PACE_ADJUSTMENT'].mean()
        
        # Convert to float32 to save memory
        df[col_name] = df[col_name].astype('float32')
    
    return df

def playerUsageAndOppurtunity(data):
    """Calculate player usage and opportunity metrics with rolling averages."""
    # Create copy and pre-sort for time-series operations
    df = data.copy()
    df.sort_values(['PLAYER_ID', 'GAME_DATE'], inplace=True)
    
    # Create player grouper object once (reuse for efficiency)
    player_group = df.groupby('PLAYER_ID')
    
    # Define metrics and their combinations
    metric_combinations = {
        'MIN_X_USG': ('MIN', 'USG_PCT'),
        'USG_X_DRTG': ('USG_PCT', 'OPP_DEF_RATING'),
        'TEAM_PACE_X_MIN': ('TEAM_PACE', 'MIN'),
        'USG_X_POSS': ('USG_PCT', 'POSS'),
        'MIN_X_OFF_RATING': ('MIN', 'OFF_RATING'),
        'USG_X_TS_PCT': ('USG_PCT', 'TS_PCT')
    }
    
    # Pre-calculate shifted values for all base metrics
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
        if col_name == 'TEAM_PACE_X_MIN':
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
            
            # Calculate rolling average on the combined metric
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
# ADVANCED FEATURE ENGINEERING
# ================================================================================================

def feature_engineering(df):
    """Comprehensive feature engineering with rolling averages and derived metrics."""
    df = df.sort_values(['GAME_DATE']).copy()

    # --- Rolling Averages ---
    rolling_features = [
        ('PLAYER_ID', 'MIN', 'AVG_MIN_PRE_GAME'),
        ('PLAYER_ID', 'TCHS', 'AVG_TCHS_PRE_GAME'),
        ('PLAYER_ID', 'USG_PCT', 'AVG_USG_PCT_PRE_GAME'),
        ('PLAYER_ID', 'AST_PCT', 'AVG_AST_PCT_PRE_GAME'),
        ('PLAYER_ID', 'POSS', 'AVG_POSS_PRE_GAME'),
        ('PLAYER_ID', 'DIST', 'AVG_DIST'),
        ('PLAYER_ID', 'SPD', 'AVG_SPD'),
        ('PLAYER_ID', 'ORBC', 'AVG_ORBC'),
        ('PLAYER_ID', 'DRBC', 'AVG_DRBC'),
        ('PLAYER_ID', 'STL', 'AVG_STL'),
        ('PLAYER_ID', 'BLK', 'AVG_BLK'),
        ('PLAYER_ID', 'PFD', 'AVG_PFD'),
        ('PLAYER_ID', 'AST', 'AVG_AST'),
        ('PLAYER_ID', 'TOV', 'AVG_TOV'),
        ('TEAM_ID', 'PACE', 'AVG_PACE_PRE_GAME'),
        ('TEAM_ID', 'TEAM_DEF_RATING', 'AVG_DEF_RATING_PRE_GAME'),
        ('TEAM_ID', 'TEAM_OFF_RATING', 'AVG_TEAM_OFF_RATING'),
        ('TEAM_ID', 'TEAM_PACE', 'AVG_TEAM_PACE'),
        ('TEAM_ID', 'TEAM_TOV', 'AVG_TEAM_TOV'),
        ('TEAM_ID', 'TEAM_FGA', 'AVG_TEAM_FGA'),
        ('TEAM_ID', 'TEAM_REB', 'AVG_TEAM_REB'),
        ('TEAM_ID', 'TEAM_AST', 'AVG_TEAM_AST'),
        ('OPP_TEAM_ID', 'OPP_DEF_RATING', 'AVG_OPP_DEF_RATING'),
        ('OPP_TEAM_ID', 'OPP_PACE', 'AVG_OPP_PACE'),
        ('OPP_TEAM_ID', 'OPP_STL', 'AVG_OPP_STL'),
        ('OPP_TEAM_ID', 'OPP_BLK', 'AVG_OPP_BLK'),
        ('OPP_TEAM_ID', 'OPP_REB', 'AVG_OPP_REB'),
        ('GAME_ID', 'GAME_PACE', 'AVG_GAME_PACE_PRE_GAME')
    ]

    for group_col, target_col, new_col in rolling_features:
        df[new_col] = df.groupby(group_col)[target_col].transform(lambda x: x.shift(1).expanding().mean())

    # --- Feature Engineering ---
    df['PACE_IMPACT'] = (
        df['AVG_PACE_PRE_GAME'] +
        df['AVG_TEAM_PACE'] +
        df['AVG_GAME_PACE_PRE_GAME'] +
        df['AVG_OPP_PACE']
    ) / 4

    df['PACE_USAGE'] = df['AVG_PACE_PRE_GAME'] * df['AVG_USG_PCT_PRE_GAME']
    df['MIN_PACE'] = df['AVG_MIN_PRE_GAME'] * df['AVG_PACE_PRE_GAME']
    df['DEF_AST_INTERACTION'] = df['AVG_DEF_RATING_PRE_GAME'] * df['AVG_AST_PCT_PRE_GAME']
    df['TOUCH_USAGE'] = df['AVG_TCHS_PRE_GAME'] * df['AVG_USG_PCT_PRE_GAME']
    df['POSS_PER_MIN'] = df['AVG_POSS_PRE_GAME'] / (df['AVG_MIN_PRE_GAME'] + 1e-6)

    df['DEF_RATING_DIFF'] = df['AVG_TEAM_OFF_RATING'] - df['AVG_OPP_DEF_RATING']
    df['PACE_DIFF'] = df['AVG_TEAM_PACE'] - df['AVG_OPP_PACE']

    df['OPP_STL_PRESSURE'] = df['AVG_OPP_STL'] / (df['AVG_TEAM_TOV'] + 1)
    df['OPP_BLOCK_PRESSURE'] = df['AVG_OPP_BLK'] / (df['AVG_TEAM_FGA'] + 1)
    df['OPP_DEF_REBOUND_CONTROL'] = df['AVG_OPP_REB'] / (df['AVG_TEAM_REB'] + df['AVG_OPP_REB'] + 1)

    df['DIST_PER_MIN'] = df['AVG_DIST'] / (df['AVG_MIN_PRE_GAME'] + 1e-6)
    df['SPD_INTENSITY'] = df['AVG_SPD'] * df['DIST_PER_MIN']
    df['EFFORT_METRIC'] = (
        (df['AVG_ORBC'] + df['AVG_DRBC'] + df['AVG_STL'] + df['AVG_BLK'] + df['AVG_PFD']) /
        (df['AVG_MIN_PRE_GAME'] + 1)
    )

    df['TEAM_AST_SHARE'] = df['AVG_AST'] / (df['AVG_TEAM_AST'] + 1)
    df['TEAM_TOV_SHARE'] = df['AVG_TOV'] / (df['AVG_TEAM_TOV'] + 1)

    df['EXPECTED_POSS_ADJUSTED'] = df['AVG_POSS_PRE_GAME'] * (df['PACE_IMPACT'] / 100)

    # --- NaN Handling ---
    # 1. Fill Rolling Averages NaNs with Global Averages
    for col in [colname for _, _, colname in rolling_features]:
        df[col] = df[col].fillna(df[col].mean(skipna=True))

    # 2. Fill Engineered Features selectively
    df['POSS_PER_MIN'] = df['POSS_PER_MIN'].fillna(df['POSS_PER_MIN'].mean(skipna=True))
    df['TEAM_AST_SHARE'] = df['TEAM_AST_SHARE'].fillna(0)
    df['TEAM_TOV_SHARE'] = df['TEAM_TOV_SHARE'].fillna(0)
    df['OPP_STL_PRESSURE'] = df['OPP_STL_PRESSURE'].fillna(0)
    df['OPP_BLOCK_PRESSURE'] = df['OPP_BLOCK_PRESSURE'].fillna(0)
    df['OPP_DEF_REBOUND_CONTROL'] = df['OPP_DEF_REBOUND_CONTROL'].fillna(0)
    df['EFFORT_METRIC'] = df['EFFORT_METRIC'].fillna(0)
    df['DIST_PER_MIN'] = df['DIST_PER_MIN'].fillna(0)
    df['SPD_INTENSITY'] = df['SPD_INTENSITY'].fillna(0)

    # 3. Final Sweep
    df = df.fillna(0)

    # --- Round Engineered Features ---
    engineered_cols = [
        'PACE_IMPACT', 'PACE_USAGE', 'MIN_PACE', 'DEF_AST_INTERACTION', 'TOUCH_USAGE',
        'POSS_PER_MIN', 'DEF_RATING_DIFF', 'PACE_DIFF', 'OPP_STL_PRESSURE',
        'OPP_BLOCK_PRESSURE', 'OPP_DEF_REBOUND_CONTROL', 'DIST_PER_MIN', 'SPD_INTENSITY',
        'EFFORT_METRIC', 'TEAM_AST_SHARE', 'TEAM_TOV_SHARE', 'EXPECTED_POSS_ADJUSTED'
    ]

    df[engineered_cols] = df[engineered_cols].round(2)

    return df

def starterUsageRank(df):
    """Calculate usage ranking among starters."""
    starters_df = df[df['START_POSITION'].notnull()].copy()

    # Use AVG_USG_PCT_PRE_GAME (pre-game rolling average)
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

def analyzeTeammateSplitsPTS(game_data, min_minutes=10):
    """
    Analyze player's performance splits based on presence of star teammates
    with optimized performance.
    """
    # Filter minutes once at the start
    game_data = game_data[game_data['MIN'] >= min_minutes].copy()
    
    # Precompute starters once for all players
    game_starters = (
        game_data[game_data['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg(PLAYER_NAMES=('PLAYER_NAME', list))
        .reset_index()
    )
    
    # Precompute star players lookup dictionary
    star_players_dict = (
        game_data[game_data['IS_STAR'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])['PLAYER_NAME']
        .apply(set)
        .to_dict()
    )
    
    def get_star_info_vectorized(group):
        game_ids = group['GAME_ID']
        team_ids = group['TEAM_ID']
        player_name = group['PLAYER_NAME'].iloc[0]
        
        # Create a mask for star teammates
        star_counts = []
        for game_id, team_id in zip(game_ids, team_ids):
            star_set = star_players_dict.get((game_id, team_id), set())
            # Remove the current player if they're a star
            star_count = len(star_set - {player_name})
            star_counts.append(star_count)
            
        group = group.copy()
        group['NUM_STAR_TEAMMATES'] = star_counts
        group['HAS_STAR_TEAMMATE'] = (group['NUM_STAR_TEAMMATES'] > 0).astype(int)
        
        # Vectorized calculations for per-36 stats
        group['PTS_PER_36'] = round(group['PTS'] * (36 / group['MIN']), 2)
        group['FGA_PER_36'] = round(group['FGA'] * (36 / group['MIN']), 2)
        
        # Vectorized calculations for with/without star stats
        with_star_mask = group['HAS_STAR_TEAMMATE'] == 1
        without_star_mask = ~with_star_mask
        
        # Calculate averages once
        with_star_pts36 = group.loc[with_star_mask, 'PTS_PER_36'].mean() if with_star_mask.any() else 0
        without_star_pts36 = group.loc[without_star_mask, 'PTS_PER_36'].mean() if without_star_mask.any() else 0
        with_star_usg = group.loc[with_star_mask, 'USG_PCT'].mean() if with_star_mask.any() else 0
        without_star_usg = group.loc[without_star_mask, 'USG_PCT'].mean() if without_star_mask.any() else 0
        
        # Assign values using vectorized operations
        group['AVG_PTS_PER_36_WITH_STAR'] = round(with_star_pts36, 2)
        group['AVG_PTS_PER_36_WITHOUT_STAR'] = round(without_star_pts36, 2)
        group['AVG_USG_PCT_WITH_STAR'] = round(with_star_usg, 2)
        group['AVG_USG_PCT_WITHOUT_STAR'] = round(without_star_usg, 2)
        
        return group
    
    # Process all players at once using groupby
    result = (
        game_data
        .groupby('PLAYER_NAME', group_keys=False)
        .apply(get_star_info_vectorized)
    )
    
    return result

def analyzeTeammateSplitsWhenAnyStarSitsPTS(game_data, min_minutes=10):
    """
    Optimized version to calculate historical averages when any star teammate sits.
    """
    # Initial filtering
    game_data = game_data[game_data['MIN'] >= min_minutes].copy()
    
    # Precompute starters once
    game_starters = (
        game_data[game_data['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg(PLAYER_NAMES=('PLAYER_NAME', set))  # Using set for faster lookup
        .reset_index()
    )
    
    # Precompute star players by team
    team_stars = (
        game_data[game_data['IS_STAR'] == 1]
        .groupby('TEAM_ID')['PLAYER_NAME']
        .agg(set)
        .to_dict()
    )
    
    # Precompute starter sets for quick lookup
    starter_dict = (
        game_starters
        .set_index(['GAME_ID', 'TEAM_ID'])['PLAYER_NAMES']
        .to_dict()
    )
    
    result_frames = []
    
    for (player_name, team_id), group in game_data.groupby(['PLAYER_NAME', 'TEAM_ID']):
        # Get star teammates for this player's team
        team_star_teammates = team_stars.get(team_id, set()) - {player_name}
        
        if not team_star_teammates:  # No star teammates on team
            group['STAR_OUT'] = 0
            group['AVG_PTS_WHEN_STAR_OUT'] = 0
            group['AVG_USG_PCT_WHEN_STAR_OUT'] = 0
            group['AVG_MIN_WHEN_STAR_OUT'] = 0
            result_frames.append(group)
            continue
            
        # Vectorized operation to check for missing stars
        def check_missing_stars(row):
            game_starters_set = starter_dict.get((row['GAME_ID'], row['TEAM_ID']), set())
            return int(any(star not in game_starters_set for star in team_star_teammates))
        
        # Apply the check across all games
        group['STAR_OUT'] = group.apply(check_missing_stars, axis=1)
        
        # Calculate averages when star is out
        star_out_mask = group['STAR_OUT'] == 1
        
        if star_out_mask.any():
            star_out_games = group[star_out_mask]
            avg_pts = round(star_out_games['PTS'].mean(), 2)
            avg_usg = round(star_out_games['USG_PCT'].mean(), 2)
            avg_min = round(star_out_games['MIN'].mean(), 2)
        else:
            avg_pts = avg_usg = avg_min = 0
            
        # Assign values
        group['AVG_PTS_WHEN_STAR_OUT'] = avg_pts
        group['AVG_USG_PCT_WHEN_STAR_OUT'] = avg_usg
        group['AVG_MIN_WHEN_STAR_OUT'] = avg_min
        
        # Calculate HAS_STAR_TEAMMATE
        def check_active_stars(row):
            game_starters_set = starter_dict.get((row['GAME_ID'], row['TEAM_ID']), set())
            return int(any(star in game_starters_set for star in team_star_teammates))
        
        group['HAS_STAR_TEAMMATE'] = group.apply(check_active_stars, axis=1)
        
        result_frames.append(group)
    
    # Concatenate all groups back together
    result = pd.concat(result_frames, ignore_index=True)
    
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