import numpy as np
import pandas as pd
from collections import defaultdict
from collections import deque

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
        'MIN', 'PF'
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
    df = player_data.copy()
    df = df.sort_values([player_id_col, date_col])

    metric = 'MIN'
    if metric not in df.columns:
        return df

    global_mean = df[metric].mean()

    # Overall per player expanding avg, shifted 1, used as a neutral fallback
    overall_shifted = df.groupby(player_id_col)[metric].shift(1)
    overall_cumcount = overall_shifted.groupby(df[player_id_col]).expanding().count().reset_index(level=[0], drop=True)
    overall_cumsum = overall_shifted.groupby(df[player_id_col]).expanding().sum().reset_index(level=[0], drop=True)
    overall_avg = overall_cumsum / overall_cumcount

    # Per location expanding avg, shifted 1
    g_loc = df.groupby([player_id_col, 'HOME_GAME'])[metric]
    loc_shifted = g_loc.shift(1)

    # Compute expanding mean per player, per location
    loc_cumcount = loc_shifted.groupby([df[player_id_col], df['HOME_GAME']]).expanding().count().reset_index(level=[0,1], drop=True)
    loc_cumsum = loc_shifted.groupby([df[player_id_col], df['HOME_GAME']]).expanding().sum().reset_index(level=[0,1], drop=True)
    loc_avg = loc_cumsum / loc_cumcount

    # Write into separate columns by location
    df['PLAYER_HOME_AVG_MIN'] = np.where(df['HOME_GAME'] == 1, loc_avg, np.nan)
    df['PLAYER_AWAY_AVG_MIN'] = np.where(df['HOME_GAME'] == 0, loc_avg, np.nan)

    # First game for a player, set both to global mean
    first_game_mask = df.groupby(player_id_col).cumcount() == 0
    df.loc[first_game_mask, 'PLAYER_HOME_AVG_MIN'] = global_mean
    df.loc[first_game_mask, 'PLAYER_AWAY_AVG_MIN'] = global_mean

    # Fill remaining NaNs with the player's overall prior average, then global mean
    for col in ['PLAYER_HOME_AVG_MIN', 'PLAYER_AWAY_AVG_MIN']:
        df[col] = df[col].fillna(overall_avg)
        df[col] = df[col].fillna(global_mean)
        df[col] = df[col].astype('float32').round(2)

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

def _rolling_slope(y, window):
    if len(y) < window:
        return np.nan
    x = np.arange(window, dtype=float)
    yw = y[-window:].astype(float)
    x_mean = x.mean()
    y_mean = yw.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x_mean) * (yw - y_mean)).sum() / denom
    return slope

def add_minutes_trend_features(df, date_col='GAME_DATE'):
    d = df.sort_values([ 'PLAYER_ID', date_col ])
    # shift to avoid leakage
    d['MIN_PRIOR'] = d.groupby('PLAYER_ID')['MIN'].shift(1)

    # slope of minutes over last 5 prior games
    d['MIN_SLOPE_5'] = (
        d.groupby('PLAYER_ID')['MIN_PRIOR']
         .apply(lambda s: s.rolling(5).apply(lambda w: _rolling_slope(w.values, 5), raw=False))
         .reset_index(level=0, drop=True)
    )

    # rolling std of minutes over last 3, 5, 7 prior games
    for k in [3, 5, 7]:
        d[f'STD_MIN_{k}'] = (
            d.groupby('PLAYER_ID')['MIN_PRIOR']
             .rolling(k).std()
             .reset_index(level=0, drop=True)
        )

    # clean up
    d.drop(columns=['MIN_PRIOR'], inplace=True)
    return d

def add_usage_shift(df, window=5, date_col='GAME_DATE'):
    d = df.sort_values(['PLAYER_ID', date_col, 'GAME_ID']).copy()

    # rolling avg USG over last N prior games
    d[f'USG_LAST_{window}'] = (
        d.groupby('PLAYER_ID')['USG_PCT']
         .shift(1)
         .rolling(window)
         .mean()
    )

    # season prior mean without MultiIndex
    g = d.groupby(['PLAYER_ID', 'TEAM_SEASON_ID'])
    prior_sum = g['USG_PCT'].cumsum().shift(1)
    prior_cnt = g.cumcount()  # 0 for first row in season
    d['SEASON_USG_PRIOR'] = prior_sum / prior_cnt.replace(0, np.nan)

    d[f'USG_SHIFT_{window}'] = d[f'USG_LAST_{window}'] - d['SEASON_USG_PRIOR']
    d.drop(columns=[f'USG_LAST_{window}', 'SEASON_USG_PRIOR'], inplace=True)
    return d

def add_rotation_stability(df, window=10, date_col='GAME_DATE'):
    d = df.sort_values(['TEAM_ID', date_col, 'GAME_ID', 'PLAYER_ID']).copy()

    # build starters per team game as lists
    starters = (
        d[d['STARTING'] == 1]
        .groupby(['TEAM_ID', date_col, 'GAME_ID'])['PLAYER_ID']
        .apply(list)
        .reset_index()
        .sort_values(['TEAM_ID', date_col, 'GAME_ID'])
    )

    records = {}
    for team_id, grp in starters.groupby('TEAM_ID', sort=False):
        prev = deque(maxlen=window)
        for _, row in grp.iterrows():
            gid = row['GAME_ID']
            cur = set(row['PLAYER_ID'])
            if len(prev) == 0:
                stability = np.nan
            else:
                overlaps = [len(cur.intersection(p)) / 5.0 for p in prev]
                stability = float(np.mean(overlaps))
            records[(team_id, gid)] = stability
            prev.append(cur)

    d[f'ROTATION_STABILITY_{window}'] = d.set_index(['TEAM_ID', 'GAME_ID']).index.map(records)
    d.reset_index(drop=True, inplace=True)
    return d

def add_lineup_cohesion(df, date_col='GAME_DATE'):
    d = df.sort_values(['TEAM_ID', date_col, 'GAME_ID', 'PLAYER_ID']).copy()

    starters = (
        d[d['STARTING'] == 1]
        .groupby(['TEAM_ID', date_col, 'GAME_ID'])['PLAYER_ID']
        .apply(list)
        .reset_index()
        .sort_values(['TEAM_ID', date_col, 'GAME_ID'])
    )

    records = {}
    for team_id, grp in starters.groupby('TEAM_ID', sort=False):
        pair_counts = defaultdict(int)  # prior co-start counts for pairs
        for _, row in grp.iterrows():
            gid = row['GAME_ID']
            lineup = row['PLAYER_ID']
            if len(lineup) < 2:
                records[(team_id, gid)] = np.nan
            else:
                pairs = []
                for i in range(len(lineup)):
                    for j in range(i + 1, len(lineup)):
                        a, b = lineup[i], lineup[j]
                        key = (a, b) if a < b else (b, a)
                        pairs.append(pair_counts[key])
                cohesion = float(np.mean(pairs)) if len(pairs) else np.nan
                records[(team_id, gid)] = cohesion
                # update counts after computing prior value
                for i in range(len(lineup)):
                    for j in range(i + 1, len(lineup)):
                        a, b = lineup[i], lineup[j]
                        key = (a, b) if a < b else (b, a)
                        pair_counts[key] += 1

    d['LINEUP_COHESION_IDX'] = d.set_index(['TEAM_ID', 'GAME_ID']).index.map(records)
    d.reset_index(drop=True, inplace=True)
    return d

def process_star_players(season_df, star_players, usg_col='USG_PCT'):
    """Process star and usage data for a single season."""
    df = season_df.copy()

    # IS_STAR
    df['IS_STAR'] = df['PLAYER_NAME'].isin(star_players).astype(int)

    # Count starters and stars among starters per team-game
    starters = (
        df[df['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])['PLAYER_NAME']
        .agg(list)
        .reset_index()
        .rename(columns={'PLAYER_NAME': 'STARTERS_LIST'})
    )
    starters['NUM_STARS_ON_TEAM'] = starters['STARTERS_LIST'].apply(
        lambda players: sum(p in star_players for p in players)
    )

    # Merge star starter count
    df = df.merge(
        starters[['GAME_ID', 'TEAM_ID', 'NUM_STARS_ON_TEAM']],
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    df['NUM_STARS_ON_TEAM'] = df['NUM_STARS_ON_TEAM'].fillna(0).astype(int)

    # HIGHEST_USG_RATE per team-game
    if usg_col not in df.columns:
        raise KeyError(f"Column '{usg_col}' not in DataFrame")

    df['HIGHEST_USG_RATE'] = 0
    valid = df[usg_col].notna()
    max_usg = df.loc[valid].groupby(['GAME_ID', 'TEAM_ID'])[usg_col].transform('max')
    df.loc[valid & (df[usg_col] == max_usg), 'HIGHEST_USG_RATE'] = 1
    df['HIGHEST_USG_RATE'] = df['HIGHEST_USG_RATE'].astype(int)

    return df

def add_fatigue_features(df):
    d = df.copy()
    d['GAME_DATE'] = pd.to_datetime(d['GAME_DATE'])
    d.sort_values(['PLAYER_ID','GAME_DATE','GAME_ID'], inplace=True)

    g = d.groupby('PLAYER_ID', sort=False)
    prev_dist = g['DIST'].shift(1)
    prev_spd  = g['SPD'].shift(1)
    prev_min  = g['MIN'].shift(1)

    d['WORKLOAD_RAW_L1']     = (prev_dist.fillna(0) * prev_spd.fillna(0)).astype('float32')
    d['WORKLOAD_PER_MIN_L1'] = (d['WORKLOAD_RAW_L1'] / (prev_min.fillna(0) + 1e-6)).astype('float32')

    s = d['WORKLOAD_PER_MIN_L1']
    d['ACUTE_WORKLOAD_3'] = s.groupby(d['PLAYER_ID']).rolling(3,  min_periods=1).mean().reset_index(level=0, drop=True).astype('float32')
    d['CHRONIC_WORKLOAD_12'] = s.groupby(d['PLAYER_ID']).rolling(12, min_periods=3).mean().reset_index(level=0, drop=True).astype('float32')
    d['ACWR'] = (d['ACUTE_WORKLOAD_3'] / d['CHRONIC_WORKLOAD_12']).replace([np.inf,-np.inf], np.nan).fillna(1.0).astype('float32')

    sp = g['SPD'].shift(1)
    d['SPD_MEAN_5']  = sp.groupby(d['PLAYER_ID']).rolling(5,  min_periods=1).mean().reset_index(level=0, drop=True).astype('float32')
    d['SPD_MEAN_15'] = sp.groupby(d['PLAYER_ID']).rolling(15, min_periods=3).mean().reset_index(level=0, drop=True).astype('float32')
    d['SPD_TREND_5v15'] = (d['SPD_MEAN_5'] - d['SPD_MEAN_15']).astype('float32')

    di = g['DIST'].shift(1)
    d['DIST_MEAN_5']  = di.groupby(d['PLAYER_ID']).rolling(5,  min_periods=1).mean().reset_index(level=0, drop=True).astype('float32')
    d['DIST_MEAN_15'] = di.groupby(d['PLAYER_ID']).rolling(15, min_periods=3).mean().reset_index(level=0, drop=True).astype('float32')
    d['DIST_TREND_5v15'] = (d['DIST_MEAN_5'] - d['DIST_MEAN_15']).astype('float32')

    if 'PLAYER_DAYS_REST' not in d.columns: d['PLAYER_DAYS_REST'] = 3
    if 'IS_BACK_TO_BACK' not in d.columns:  d['IS_BACK_TO_BACK'] = 0
    rest_term = 1.0 / (1.0 + d['PLAYER_DAYS_REST'].clip(lower=0))
    d['FATIGUE_PROXY'] = (0.6*d['ACWR'].clip(0,3) + 0.2*d['IS_BACK_TO_BACK'].astype('float32') + 0.2*rest_term.astype('float32')).astype('float32')
    return d