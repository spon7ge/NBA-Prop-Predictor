import pandas as pd

def min_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    df = _rolling_player(df)
    df = _lag_features(df)
    df = _starter_features(df)
    df = _season_averages(df)
    df = _team_context(df)
    df = _schedule_features(df)
    df = _volatility_features(df)
    df = _opponent_stats(df)
    df = _expectedPace(df)
    df = _detect_star_players(df)
    df = _positional_vacancy_features(df)

    return df


# ── 1. Rolling player performance ─────────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'REB', 'AST_PCT', 'OREB_PCT', 'DREB_PCT', 'TM_TOV_PCT',
    'PLUS_MINUS', 'PF', 'POSS', 'PIE', 'USG_PCT', 'TS_PCT', 'NET_RATING', 'OFF_RATING', 'DEF_RATING']
    for window in [3,5,10]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window).mean().round(2))
            )

    return df

# ── 2. Lag features ───────────────────────────────────────────────────────────

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PIE', 'STARTING', 'POSS', "USG_PCT", "AST", "REB", "STL", "BLK"]

    for lag in [1, 2, 3]:
        for col in cols:
            df[f'{col}_lag{lag}'] = df.groupby('PLAYER_ID')[col].shift(lag)

    return df


# ── 3. Starter / role ─────────────────────────────────────────────────────────

def _starter_features(df: pd.DataFrame) -> pd.DataFrame:
    df['STARTER_ROLL10_PCT'] = (
        df.groupby('PLAYER_ID')['STARTING']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    return df


# ── 4. Season averages (expanding) ───────────────────────────────────────────

def _season_averages(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT','OREB', 'DREB','REB',
    'PLUS_MINUS', 'PF', 'POSS', 'PIE',
    'USG_PCT', 'TS_PCT', 'NET_RATING',]

    for col in cols:
        df[f'{col}_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON_YEAR'])[col]
            .transform(lambda x: x.shift(1).expanding().mean().round(2))
        )

    return df


# ── 6. Team context ───────────────────────────────────────────────────────────

def _team_context(df: pd.DataFrame) -> pd.DataFrame:
    # ── Team pace: one row per team per game, then map back ──────────────────
    team_pace = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_PACE']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_pts = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_PTS']
        .transform(lambda x: x.shift(1).rolling(5).mean().round(2))
    )
    team_net_rating = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_NET_RATING']
        .transform(lambda x: x.shift(1).rolling(5).mean().round(2))
    )

    team_pace_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PACE_roll10=team_pace.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PACE_roll10']]
    )
    team_pts_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PTS_roll5=team_pts.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PTS_roll5']]
    )
    team_net_rating_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_NET_RATING_roll5=team_net_rating.values)[['TEAM_ID', 'GAME_ID', 'TEAM_NET_RATING_roll5']]
    )    

    df = df.merge(team_pace_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_pts_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_net_rating_map, on=['TEAM_ID', 'GAME_ID'], how='left')

    # ── MIN share and POSS share are player-level so these are fine as-is ─────
    df["MIN_share_proxy"] = round(df["MIN_roll10"] / (48 * 10), 2)
    #percentage of team's points scored by the player
    df["PTS_share_proxy"] = round(df["PTS_roll5"] / (df["TEAM_PTS_roll5"]), 2)

    df["TEAM_POSS_roll5"] = df.groupby(["TEAM_ID", "SEASON_YEAR"])["POSS"].transform(lambda x: x.shift(1).rolling(5).mean().round(2))
    df["TEAM_POSS_share"] = df["POSS_roll5"] / (df["TEAM_POSS_roll5"] + 1e-6)

    return df

# ── 8. Schedule / rest ────────────────────────────────────────────────────────

def _schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    df['DAYS_REST'] = (
        df.groupby('PLAYER_ID')['GAME_DATE']
        .diff().dt.days.fillna(3)
    )

    df['IS_B2B']  = (df['DAYS_REST'] == 1).astype(int)
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs', na=False).astype(int)

    df['GAME_NUMBER'] = (
        df.groupby(['PLAYER_ID', 'SEASON_YEAR'])
        .cumcount()
    )

    return df


# ── 9. Volatility ─────────────────────────────────────────────────────────────

def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df['MIN_std10'] = (
        df.groupby('PLAYER_ID')['MIN']
        .transform(lambda x: x.shift(1).rolling(10).std().round(2))
    )
    df['MIN_std5'] = (
        df.groupby('PLAYER_ID')['MIN']
        .transform(lambda x: x.shift(1).rolling(5).std().round(2))
    )
    df['MIN_std20'] = (
        df.groupby('PLAYER_ID')['MIN']
        .transform(lambda x: x.shift(1).rolling(20).std().round(2))
    )

    return df
# ---- Opponent stats ---------------------------------------------------------------
def _opponent_stats(df):
    stat_cols = ['TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_POSS','TEAM_NET_RATING']
    available_stat_cols = [c for c in stat_cols if c in df.columns]

    team_game = (
        df[['GAME_ID', 'GAME_DATE', 'TEAM_ABBREVIATION'] + available_stat_cols]
        .drop_duplicates(subset=['GAME_ID', 'TEAM_ABBREVIATION'])
        .sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
    )

    opp_cols = []
    for col in available_stat_cols:
        base = col.replace('TEAM_', '')
        for window in [5, 10]:
            avg_col = f'{base}_roll{window}'
            team_game[avg_col] = (
                team_game.groupby('TEAM_ABBREVIATION')[col]
                .transform(lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean().round(2))
            )
            opp_cols.append(avg_col)

    opp_rename = {col: f'OPP_{col}' for col in opp_cols}

    team_game_opp = (
        team_game[['GAME_ID', 'TEAM_ABBREVIATION'] + opp_cols]
        .rename(columns={**{'TEAM_ABBREVIATION': 'OPP_OPP_ABBREVIATION_base'}, **opp_rename})
    )

    return df.merge(team_game_opp, on=['GAME_ID', 'OPP_OPP_ABBREVIATION_base'], how='left')

# ---- Expected Pace and Points -----------------------
def _expectedPace(df):
    df = df.copy()
    df['EXPECTED_PACE'] = ((df['TEAM_PACE_roll10'] + df['OPP_PACE_roll10']) / 2).round(2)
    df['PACE_DIFFERENTIAL'] = df['TEAM_PACE_roll10'] - df['OPP_PACE_roll10']    
    return df

# ---- Finding Star Players -----------------------
def _detect_star_players(df, min_minutes=10, min_games=10, name_dict=None):
    df = df.copy()
    
    # Try to import nameDict if not provided
    if name_dict is None:
        try:
            from src.utils.team_info import nameDict
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

def _positional_vacancy_features(df):
    # 1. Map granular positions to 3 main rotation buckets
    pos_map = {
        'G': 'G', 'PG': 'G', 'SG': 'G',
        'F': 'F', 'SF': 'F', 'PF': 'F',
        'C': 'C', 'F-C': 'C', 'C-F': 'C', 'G-F': 'F'
    }
    df['rot_group'] = df['pos'].map(pos_map)

    # 2. Get the "Expected Minutes" for every player (using your best lag feature)
    # This represents what the player USUALLY plays.
    df['exp_min'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.rolling(10).mean().shift(1))

    # 3. Calculate total "Expected Minutes" per position group per team per game
    # This tells us how many minutes the ACTIVE players usually take up.
    df['active_pos_min_sum'] = df.groupby(['GAME_ID', 'TEAM_ID', 'rot_group'])['exp_min'].transform('sum')

    # 4. Define the "Regulation Cap" for each group
    # (2 Guards * 48m = 96, 2 Forwards * 48m = 96, 1 Center * 48m = 48)
    cap_map = {'G': 96, 'F': 96, 'C': 48}
    df['pos_cap'] = df['rot_group'].map(cap_map)

    # 5. The Feature: POS_VACANCY
    # Positive value = Minutes are "up for grabs" in this rotation.
    # Negative value = Rotation is "crowded" (likely someone will see a dip).
    df['POS_VACANCY'] = df['pos_cap'] - df['active_pos_min_sum']
    
    # 6. Interaction: Only relevant if the player is actually in that group
    # (Self-explanatory for the model, but helps with scaling)
    df['PLAYER_POS_OPPORTUNITY'] = df['exp_min'] + df['POS_VACANCY']

    return df

