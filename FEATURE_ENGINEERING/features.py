import pandas as pd
import numpy as np


#grabs players rest days between games
def add_rest_day_features(df):
    '''
    Add rest day features for both teams and individual players.
    Optimized for space and time efficiency.
    '''
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
    df['PLAYER_B2B'] = (df['PLAYER_DAYS_REST'] <= 1).astype('int8')
    
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

def convert_height_to_inches(height_str):
    if pd.isna(height_str):
        return np.nan
    # Split the string into feet and inches
    feet, inches = map(int, height_str.split('-'))
    # Convert to total inches
    return (feet * 12) + inches

#rolling averages for points against each team
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
        'PTS': [3,5,7],
        'USG_PCT': [3,5,7],
        'POSS': [3,5,7],
        'OFF_RATING': [3,5,7]
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

########################################################################################
#rolling averages
########################################################################################

def rollingAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    """
    Calculate rolling averages with data leakage prevention (via shift(1)).
    Includes rolling means for 3, 5, and 7 games, and rolling std for the main stat_line.
    Results rounded to 2 decimal places (hundredths).
    """
    df = player_data.copy()
    df.sort_values([player_id_col, date_col], inplace=True)

    rolling_features = {
        'PTS': [
            'MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'USG_PCT', 
            'TS_PCT', 'EFG_PCT', 'PACE', 'POSS', 'OFF_RATING', 'PointsPerShot', 'TOV',
            'TEAM_FGA', 'TEAM_FG3A', 'TEAM_FG_PCT', 'TEAM_FG3_PCT', 'TEAM_AST', 
            'TEAM_REB', 'TEAM_PACE', 'TEAM_PTS', 'OPP_DEF_RATING', 'OPP_PACE'
        ]
    }

    if stat_line not in rolling_features:
        raise ValueError(f"Invalid stat_line: {stat_line}. Must be one of {list(rolling_features.keys())}")

    player_group = df.groupby(player_id_col)
    windows = [3, 5, 7]
    features_to_process = rolling_features[stat_line]
    global_means = df[features_to_process].mean()
    global_stds = df[features_to_process].std()

    for feature in features_to_process:
        if feature not in df.columns:
            continue

        shifted = player_group[feature].shift(1)
        first_games_mask = player_group.cumcount() == 0

        for window in windows:
            col_mean = f'{feature}_ROLL_AVG_{window}'
            df[col_mean] = shifted.rolling(window=window, min_periods=1).mean()
            df.loc[first_games_mask, col_mean] = global_means[feature]
            df[col_mean] = df[col_mean].round(2)

            if feature == stat_line:
                col_std = f'{feature}_STD_AVG_{window}'
                df[col_std] = shifted.rolling(window=window, min_periods=1).std()
                df.loc[first_games_mask, col_std] = global_stds[feature]
                df[col_std] = df[col_std].round(2)

    return df

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
import numpy as np

def calculate_absolute_vs_defense(player_data, player_id_col='PLAYER_ID', stat_line='PTS'):
    df = player_data.copy()
    df.sort_values([player_id_col, 'GAME_DATE'], inplace=True)
    
    metrics = {
        'PTS': ['PTS', 'FGA', 'FTA', 'FG3A', 'USG_PCT', 'TS_PCT', 'POSS', 'OFF_RATING']
    }

    if stat_line not in metrics:
        raise ValueError(f"stat_line must be one of: {list(metrics.keys())}")

    player_group = df.groupby(player_id_col)
    df['SHIFTED_DEF_CATEGORY'] = player_group['DEF_CATEGORY'].shift(1)

    shifted_metrics = {}
    for metric in metrics[stat_line]:
        if metric in df.columns:
            shifted_metrics[metric] = player_group[metric].shift(1)

    # Calculate expanding averages for strong and weak defense
    for metric in metrics[stat_line]:
        if metric not in df.columns:
            continue

        for def_type, def_val in [('STRONG', 1), ('WEAK', 0)]:
            mask = df['SHIFTED_DEF_CATEGORY'] == def_val
            col_name = f'{metric}_VS_DEF_{def_type}'
            df[col_name] = np.nan

            for player_id in df[player_id_col].unique():
                player_mask = df[player_id_col] == player_id
                combined_mask = player_mask & mask
                values = shifted_metrics[metric][combined_mask].expanding().mean()
                df.loc[combined_mask, col_name] = values

            # Round now and cast
            df[col_name] = df[col_name].round(2).astype('float32')

        # Fill NaNs with the opposite-defense value
        strong_col = f'{metric}_VS_DEF_STRONG'
        weak_col   = f'{metric}_VS_DEF_WEAK'

        df[strong_col] = df[strong_col].fillna(df[weak_col])
        df[weak_col]   = df[weak_col].fillna(df[strong_col])

        # After fill, re-round to enforce formatting
        df[strong_col] = np.round(df[strong_col].astype('float64'), 2).astype('float32')
        df[weak_col]   = np.round(df[weak_col].astype('float64'), 2).astype('float32')

        # Calculate difference
        diff_col = f'{metric}_VS_DEF_DIFF'
        df[diff_col] = (df[weak_col] - df[strong_col]).apply(lambda x: round(x, 2)).astype('float32')

    # Rolling averages
    windows = [3, 5, 7]
    for metric in metrics[stat_line]:
        if metric not in df.columns:
            continue

        for def_type in ['STRONG', 'WEAK']:
            base_col = f'{metric}_VS_DEF_{def_type}'

            for window in windows:
                roll_col = f'{base_col}_ROLL_{window}'

                df[roll_col] = (
                    player_group[base_col]
                    .shift(1)  # prevent leakage
                    .rolling(window=window, min_periods=1)
                    .mean()
                )

                # Fill first games with mean of column (across all players)
                first_games_mask = player_group.cumcount() == 0
                df.loc[first_games_mask, roll_col] = df[base_col].mean()

                # Round and convert
                df[roll_col] = np.round(df[roll_col].astype('float64'), 2).astype('float32')
    # Clean up
    df.drop('SHIFTED_DEF_CATEGORY', axis=1, inplace=True)
    return df


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

#-----------------------------------------------------------------------------------------------------------
def encode_teams(df):
    # One-hot encode player team and opponent team
    df_teams = pd.get_dummies(df['TEAM_ABBREVIATION'], prefix='TEAM_').astype(int)
    df_opps = pd.get_dummies(df['OPP_ABBREVIATION'], prefix='OPP_').astype(int)
    df_encoded = pd.concat([df, df_teams, df_opps], axis=1)
    return df_encoded

def add_game_pace_adjustment(df):
    """
    Calculate game pace adjustments with data leakage prevention using shift(1).
    """
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
    """
    Calculate player usage and opportunity metrics with optimized performance 
    and data leakage prevention using shift(1).
    """
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
