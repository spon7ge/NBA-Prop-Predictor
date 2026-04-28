import pandas as pd

def apm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    df = _rolling_player(df)
    df = _ewm_player(df)
    df = _lag_features(df)
    # df = _starter_features(df)
    df = _season_averages(df)
    df = _efficiency_trends(df)
    df = _team_context(df)
    df = _schedule_features(df)
    df = _volatility_features(df)
    df = _opponent_stats(df)
    df = _opponent_season_stats_and_rank(df)
    df = _expectedPace(df)
    df = _detect_star_players(df)
    df = _team_allowed_context(df)
    
    return df


# ── 1. Rolling player performance ─────────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'USG_PCT', 'PLUS_MINUS', 'POSS', 'AST_TO','AST', 'AST_PCT','AST_RATIO', 'TOV', 'AST_PER_MIN', 'TOV_PER_MIN', 'POSS_PER_MIN', 'SAST_PER_MIN', 'FTAST_PER_MIN', 'PASS_PER_MIN', 'TCHS_PER_MIN']

    for window in [3,5,10]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window).mean().round(2)))

    return df

def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'AST_PER_MIN', 'AST', 'TOV_PER_MIN', 'TOV', 'POSS_PER_MIN', 'AST_PCT', 'AST_TO', 'MIN', 
        'USG_PCT', 'SAST_PER_MIN', 'FTAST_PER_MIN', 'PASS_PER_MIN', 'TCHS_PER_MIN'
    ]

    for span in [3, 5, 10]:
        for col in cols:
            df[f'{col}_{span}_ewm'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(
                    lambda x: x.shift(1).ewm(span=span, adjust=False).mean().round(2)
                )
            )

    return df


# ── 2. Lag features ───────────────────────────────────────────────────────────

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['AST_PER_MIN', 'AST', 'TOV_PER_MIN', 'TOV', 'POSS_PER_MIN', 'SAST_PER_MIN', 'FTAST_PER_MIN', 'PASS_PER_MIN', 'TCHS_PER_MIN']

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
    cols = ['MIN', 'AST', 'USG_PCT', 'POSS', 'PF', 'OFF_RATING', 'AST_PER_MIN', 'SAST_PER_MIN', 'FTAST_PER_MIN', 'PASS_PER_MIN', 'TCHS_PER_MIN']

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
    team_ast = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_AST']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )
    team_net_rating = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .groupby('TEAM_ID')['TEAM_NET_RATING']
        .transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    )

    team_pace_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_PACE_roll10=team_pace.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PACE_roll10']]
    )
    team_ast_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_AST_roll10=team_ast.values)[['TEAM_ID', 'GAME_ID', 'TEAM_AST_roll10']]
    )
    team_net_rating_map = (
        df.drop_duplicates(subset=['TEAM_ID', 'GAME_ID'])
        .sort_values(['TEAM_ID', 'GAME_DATE'])
        .assign(TEAM_NET_RATING_roll10=team_net_rating.values)[['TEAM_ID', 'GAME_ID', 'TEAM_NET_RATING_roll10']]
    )    

    df = df.merge(team_pace_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_ast_map, on=['TEAM_ID', 'GAME_ID'], how='left')
    df = df.merge(team_net_rating_map, on=['TEAM_ID', 'GAME_ID'], how='left')

    # ── MIN share and POSS share are player-level so these are fine as-is ─────
    df["MIN_share_proxy"] = round(df["MIN_roll10"] / (48 * 10), 2)
    #percentage of team's points scored by the player
    df["AST_share_proxy_roll10"] = round(df["AST_roll10"] / (df["TEAM_AST_roll10"]), 2)

    df["TEAM_POSS_roll10"] = df.groupby(["TEAM_ID", "SEASON_YEAR"])["POSS"].transform(lambda x: x.shift(1).rolling(10).mean().round(2))
    df["TEAM_POSS_share_roll10"] = df["POSS_roll10"] / (df["TEAM_POSS_roll10"] + 1e-6)

    return df
# ── 7. Schedule / rest ────────────────────────────────────────────────────────

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
    df['AST_std10'] = (
        df.groupby('PLAYER_ID')['MIN']
        .transform(lambda x: x.shift(1).rolling(10).std().round(2))
    )
    df['TOV_std10'] = (
        df.groupby('PLAYER_ID')['TOV']
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
        0.35 * stats_df['USG_PCT_NORM'] +
        0.20 * stats_df['TS_PCT_NORM'] +
        0.30 * stats_df['PTS_NORM'] +
        0.15 * stats_df['PIE_NORM'])
    
    # 5. Extract Top 3 per team + single top player per team
    sorted_stars = stats_df.sort_values(['TEAM_ID', 'STAR_SCORE'], ascending=[True, False])
    top_3 = sorted_stars.groupby('TEAM_ID').head(3)
    top_1 = sorted_stars.groupby('TEAM_ID').head(1)[['TEAM_ID', 'PLAYER_NAME_NORM']]
    top_1_set = set(zip(top_1['TEAM_ID'], top_1['PLAYER_NAME_NORM']))

    # 6. Flag players in the main dataframe
    top_stars = set(zip(top_3['TEAM_ID'], top_3['PLAYER_NAME_NORM']))

    df['IS_TOP_STAR'] = df.apply(
        lambda x: 1 if (x['TEAM_ID'], x['PLAYER_NAME_NORM']) in top_stars else 0, axis=1
    )
    df['IS_TOP_1_STAR'] = df.apply(
        lambda x: 1 if (x['TEAM_ID'], x['PLAYER_NAME_NORM']) in top_1_set else 0, axis=1
    )

    # 7. Calculate total top-3 stars active per game
    active_stars_per_game = (
        df[(df['IS_TOP_STAR'] == 1) & (df['MIN'] >= min_minutes)]
        .groupby(['GAME_ID', 'TEAM_ID'])['IS_TOP_STAR']
        .sum()
        .reset_index(name='ACTIVE_STARS_COUNT')
    )

    df = df.merge(active_stars_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['ACTIVE_STARS_COUNT'] = df['ACTIVE_STARS_COUNT'].fillna(0).astype(int)

    # 8. Flag whether the #1 star was active per game (played >= min_minutes)
    top1_active_per_game = (
        df[(df['IS_TOP_1_STAR'] == 1) & (df['MIN'] >= min_minutes)]
        .groupby(['GAME_ID', 'TEAM_ID'])['IS_TOP_1_STAR']
        .max()
        .reset_index(name='TOP_STAR_ACTIVE')
    )

    df = df.merge(top1_active_per_game, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['TOP_STAR_ACTIVE'] = df['TOP_STAR_ACTIVE'].fillna(0).astype(int)

    return df.drop(columns=['PLAYER_NAME_NORM'])

def _team_allowed_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds team-level "allowed" features using only games before the current one
    (shift(1) within TEAM_ID + SEASON_YEAR), then merges back to player rows.
    """
    required = [
        "TEAM_ID",
        "SEASON_YEAR",
        "GAME_ID",
        "GAME_DATE",
        "OPP_POSS",
        "OPP_AST",
        "OPP_TOV",
        "OPP_DEF_RATING",
        "OPP_PTS",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return df

    team_game = (
        df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
        .sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"])
        .copy()
    )

    g = team_game.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False)

    # Per-game allowed/forced averages to date (prior games only)
    def prior_expanding_mean(x):
        return x.shift(1).expanding().mean().round(3)

    for raw_col, new_col in [
        ("OPP_POSS",       "TEAM_POSS_ALLOWED"),
        ("OPP_AST",        "TEAM_AST_ALLOWED"),
        ("OPP_TOV",        "TEAM_TOV_FORCED"),
        ("OPP_DEF_RATING", "TEAM_DEF_RATING_ALLOWED"),
        ("OPP_PTS",        "TEAM_PTS_ALLOWED"),
    ]:
        team_game[new_col] = g[raw_col].transform(prior_expanding_mean)

    allowed_cols = [
        "TEAM_ID",
        "GAME_ID",
        "TEAM_POSS_ALLOWED",
        "TEAM_AST_ALLOWED",
        "TEAM_TOV_FORCED",
        "TEAM_DEF_RATING_ALLOWED",
        "TEAM_PTS_ALLOWED",
    ]

    allowed_map = team_game[allowed_cols]
    out = df.merge(allowed_map, on=["TEAM_ID", "GAME_ID"], how="left")

    if "OPP_TEAM_ID" not in out.columns:
        return out

    opp_rename = {
        "TEAM_ID":                  "OPP_TEAM_ID",
        "TEAM_POSS_ALLOWED":        "OPP_TEAM_POSS_ALLOWED",
        "TEAM_AST_ALLOWED":         "OPP_TEAM_AST_ALLOWED",
        "TEAM_TOV_FORCED":          "OPP_TEAM_TOV_FORCED",
        "TEAM_DEF_RATING_ALLOWED":  "OPP_TEAM_DEF_RATING_ALLOWED",
        "TEAM_PTS_ALLOWED":         "OPP_TEAM_PTS_ALLOWED",
    }

    opp_allowed_map = allowed_map.rename(columns=opp_rename)
    return out.merge(opp_allowed_map, on=["OPP_TEAM_ID", "GAME_ID"], how="left")

def _opponent_season_stats_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opponent (team) features: YTD means through prior games only, plus weekly league
    rank among teams (1 = best) using each team's S2D stat at the last game in the
    ISO week, broadcast to all games in that week.
    """
    if 'OPP_OPP_ABBREVIATION_base' not in df.columns or 'SEASON_YEAR' not in df.columns:
        return df

    stat_cols = ['TEAM_DEF_RATING', 'TEAM_PACE', 'TEAM_POSS']
    available = [c for c in stat_cols if c in df.columns]
    if not available:
        return df

    need = ['GAME_ID', 'GAME_DATE', 'SEASON_YEAR', 'TEAM_ABBREVIATION'] + available
    team_game = (
        df[need]
        .drop_duplicates(subset=['GAME_ID', 'TEAM_ABBREVIATION'])
        .copy()
    )
    team_game['GAME_DATE'] = pd.to_datetime(team_game['GAME_DATE'], utc=False)
    team_game = team_game.sort_values(['SEASON_YEAR', 'TEAM_ABBREVIATION', 'GAME_DATE'])

    g = team_game.groupby(['SEASON_YEAR', 'TEAM_ABBREVIATION'], sort=False)

    season_renames = {}
    for col in available:
        base = col.replace('TEAM_', '')
        sname = f'{base}_season_to_date'
        team_game[sname] = g[col].transform(
            lambda x: x.shift(1).expanding().mean().round(2)
        )
        season_renames[col] = sname

    team_game['RANK_WEEK'] = team_game['GAME_DATE'].dt.to_period('W-SAT')
    last_idx = team_game.groupby(
        ['SEASON_YEAR', 'RANK_WEEK', 'TEAM_ABBREVIATION'], sort=False
    )['GAME_DATE'].idxmax()
    weekly = team_game.loc[last_idx].copy()

    # Def rating: lower is better. Pace / poss: higher = rank 1.
    rank_asc = {
        'TEAM_DEF_RATING': True,
        'TEAM_PACE': False,
        'TEAM_POSS': False,
    }
    rank_cols = []
    for col, sname in season_renames.items():
        asc = rank_asc.get(col, True)
        rk = sname.replace('_season_to_date', '_szn_league_rank')
        rank_cols.append(rk)
        weekly[rk] = (
            weekly.groupby(['SEASON_YEAR', 'RANK_WEEK'])[sname]
            .rank(ascending=asc, method='min')
        )

    w = weekly[['SEASON_YEAR', 'RANK_WEEK', 'TEAM_ABBREVIATION', *rank_cols]]
    team_game = team_game.merge(
        w, on=['SEASON_YEAR', 'RANK_WEEK', 'TEAM_ABBREVIATION'], how='left'
    )

    out_cols = list(season_renames.values()) + rank_cols
    opp_rename = {c: f'OPP_{c}' for c in out_cols}
    team_game_opp = (
        team_game[['GAME_ID', 'TEAM_ABBREVIATION', *out_cols]]
        .rename(columns={**{'TEAM_ABBREVIATION': 'OPP_OPP_ABBREVIATION_base'}, **opp_rename})
    )

    return df.merge(team_game_opp, on=['GAME_ID', 'OPP_OPP_ABBREVIATION_base'], how='left')