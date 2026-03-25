import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


# ── 1. Rolling player performance ─────────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'USG_PCT', 'PF', 'PLUS_MINUS', 'POSS', 'PTS_PER_MIN', 'PF_PER_MIN', 'TS_PCT', 'AST_TO', 'NET_RATING']

    for window in [3,5,10]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window).mean().round(2))
            )

    return df

# ── 2. Lag features ───────────────────────────────────────────────────────────

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PIE', 'STARTING', 'POSS']

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
    cols = ['MIN', 'PTS', 'USG_PCT', 'POSS', 'PF', 'OFF_RATING', 'DEF_RATING']

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
        .transform(lambda x: x.shift(1).rolling(5).mean().round(2))
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
        .assign(TEAM_PACE_roll5=team_pace.values)[['TEAM_ID', 'GAME_ID', 'TEAM_PACE_roll5']]
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
    df["MIN_share_proxy"] = round(df["MIN_roll5"] / (48 * 5), 2)
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
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs').astype(int)

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



