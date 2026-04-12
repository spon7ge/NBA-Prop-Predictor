import pandas as pd

def ppm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    df = _rolling_player(df)
    df = _lag_features(df)
    df = _starter_features(df)
    df = _season_averages(df)
    df = _efficiency_trends(df)
    df = _team_context(df)
    df = _schedule_features(df)
    df = _volatility_features(df)
    df = _opponent_stats(df)
    df = _expectedPace(df)
    df = _detect_star_players(df)

    return df


# ── 1. Rolling player performance ─────────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'USG_PCT', 'PLUS_MINUS', 'POSS', 'PTS_PER_MIN', 'PTS_PER_POSS',
    'TS_PCT', 'AST_TO', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'AST', 'TOV'
    ]

    for window in [3,5,10]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window).mean().round(2))
            )

    return df

# ── 2. Lag features ───────────────────────────────────────────────────────────

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PIE', 'STARTING', 'POSS', "USG_PCT"]

    for lag in [1, 2]:
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
    cols = ['MIN', 'PTS', 'USG_PCT', 'POSS', 'PF', 'OFF_RATING', 'DEF_RATING', 'PTS_PER_MIN']

    for col in cols:
        df[f'{col}_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON_YEAR'])[col]
            .transform(lambda x: x.shift(1).expanding().mean().round(2))
        )

    return df


# ── 5. Efficiency trends ──────────────────────────────────────────────────────

def _efficiency_trends(df: pd.DataFrame) -> pd.DataFrame:
    df['PTS_PER_MIN_roll5'] = df['PTS_roll5'] / df['MIN_roll5'].replace(0, pd.NA)

    df['NET_RATING_roll5'] = (
        df.groupby('PLAYER_ID')['NET_RATING']
        .transform(lambda x: x.shift(1).rolling(5).mean().round(2))
    )

    df['TS_PCT_roll5'] = (
        df.groupby('PLAYER_ID')['TS_PCT']
        .transform(lambda x: x.shift(1).rolling(5).mean().round(2))
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
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_poss = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_POSS']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_ts_pct = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_TS_PCT']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )

    team_pace_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PACE_roll10=team_pace.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PACE_roll10']]
    )
    team_pts_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PTS_roll10=team_pts.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PTS_roll10']]
    )
    team_poss_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_POSS_roll10=team_poss.values)[['TEAM_ID', 'GAME_ID', 'TEAM_POSS_roll10']]
    )
    team_ts_pct_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_TS_PCT_roll10=team_ts_pct.values)[['TEAM_ID', 'GAME_ID', 'TEAM_TS_PCT_roll10']]
    )

    df = df.merge(team_pace_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_pts_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_poss_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_ts_pct_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    # ── MIN share and POSS share are player-level so these are fine as-is ─────
    df["MIN_share_proxy"] = round(df["MIN_roll10"] / (48 * 10), 2)
    #percentage of team's points scored by the player
    df["PTS_share_proxy"] = round(df["PTS_roll10"] / (df["TEAM_PTS_roll10"]), 2)

    # df["TEAM_POSS_roll5"] = df.groupby(["TEAM_ID", "SEASON_YEAR"])["POSS"].transform(lambda x: x.shift(1).rolling(5).mean().round(2))
    # df["TEAM_POSS_share"] = df["POSS_roll5"] / (df["TEAM_POSS_roll5"] + 1e-6)

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

    return df
# ---- Opponent stats ---------------------------------------------------------------
def _opponent_stats(df):
    stat_cols = ['TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_POSS']
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
    
    # 1. Normalize Names
    if name_dict:
        df['PLAYER_NAME_NORM'] = df['PLAYER_NAME'].map(lambda x: name_dict.get(x, x))
    else:
        df['PLAYER_NAME_NORM'] = df['PLAYER_NAME']
    
    # 2. Filter for qualified players
    active_mask = df['MIN'] >= min_minutes
    stats_df = df[active_mask].groupby(['TEAM_ID', 'PLAYER_NAME_NORM']).agg({
        'USG_PCT': 'mean',
        'TS_PCT': 'mean',
        'EFG_PCT': 'mean',
        'PTS': 'mean',
        'PIE': 'mean',
        'NET_RATING': 'mean'
    }).reset_index()
    
    # Filter by games played
    game_counts = df[active_mask].groupby(['TEAM_ID', 'PLAYER_NAME_NORM']).size()
    eligible = game_counts[game_counts >= min_games].index
    stats_df = stats_df.set_index(['TEAM_ID', 'PLAYER_NAME_NORM']).loc[eligible].reset_index()

    # 3. Normalize metrics (Min-Max per Team)
    cols = ['USG_PCT', 'TS_PCT', 'EFG_PCT', 'PTS', 'PIE', 'NET_RATING']
    for col in cols:
        stats_df[f'{col}_NORM'] = stats_df.groupby('TEAM_ID')[col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        )
    
    # 4. Refined Weighted Star Score
    # Prioritizing PIE (30%) as the best catch-all, Usage (25%), PTS (20%)
    stats_df['STAR_SCORE'] = (
        0.25 * stats_df['USG_PCT_NORM'] +
        0.15 * stats_df['TS_PCT_NORM'] +
        0.10 * stats_df['EFG_PCT_NORM'] +
        0.20 * stats_df['PTS_NORM'] +
        0.30 * stats_df['PIE_NORM'] 
    )
    
    # 5. Extract Top 3 per team
    top_3 = (stats_df.sort_values(['TEAM_ID', 'STAR_SCORE'], ascending=[True, False])
             .groupby('TEAM_ID').head(3))
    
    # 6. Flag players in the main dataframe
    # Create a set of "Star" identifiers (TeamID + PlayerName)
    top_stars = set(zip(top_3['TEAM_ID'], top_3['PLAYER_NAME_NORM']))
    
    df['IS_TOP_STAR'] = df.apply(
        lambda x: 1 if (x['TEAM_ID'], x['PLAYER_NAME_NORM']) in top_stars else 0, axis=1
    )
    
    # 7. Calculate total stars active per game
    # This identifies how many of the top 3 are available for a given game
    active_stars_per_game = (
        df[df['IS_TOP_STAR'] == 1 & (df['MIN'] >= min_minutes)]
        .groupby(['GAME_ID', 'TEAM_ID'])['IS_TOP_STAR']
        .sum()
        .reset_index(name='ACTIVE_STARS_COUNT')
    )
    
    df = df.merge(active_stars_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['ACTIVE_STARS_COUNT'] = df['ACTIVE_STARS_COUNT'].fillna(0).astype(int)
    
    return df.drop(columns=['PLAYER_NAME_NORM'])