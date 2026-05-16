import numpy as np
import pandas as pd

def min_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    df = _rolling_player(df)
    df = _ewm_player(df)
    df = _lag_features(df)
    df = _starter_features(df)
    df = _season_averages(df)
    df = _star_out_flag(df)
    df = _star_position_match(df)
    df = _min_by_star_status(df)
    df = _team_context(df)
    df = _schedule_features(df)
    df = _volatility_features(df)
    df = _opponent_stats(df)
    df = _expectedPace(df)
    df = _detect_star_players(df)
    df = _HomeAwayAverages(df)
    df = _pbp_stint_features(df)

    return df


# ── 1. Rolling player performance ─────────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PF', 'POSS', 'PIE', 'USG_PCT', 'TS_PCT', 'NET_RATING']
    for window in [5,10,20]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean().round(2))
            )

    return df

def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PF_PER_MIN', 'POSS', 'PIE', 'USG_PCT', 'TS_PCT', 'NET_RATING', 'DIST', 'SPD', 'TCHS']

    for span in [3, 5, 10, 20]:
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
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PIE', 'STARTING', 'POSS', "USG_PCT"]

    for lag in [1, 2]:
        for col in cols:
            df[f'{col}_lag{lag}'] = df.groupby('PLAYER_ID')[col].shift(lag)

    return df


# ── 3. Starter / role ─────────────────────────────────────────────────────────

def _starter_features(df: pd.DataFrame) -> pd.DataFrame:
    df['STARTER_ROLL10_PCT'] = (
        df.groupby('PLAYER_ID')['STARTING']
        .transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean().round(2)
        )
    )
    return df


# ── 4. Season averages (expanding) ───────────────────────────────────────────

def _season_averages(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['MIN', 'PTS', 'PLUS_MINUS', 'PF', 'POSS', 'PIE', 'USG_PCT', 'TS_PCT', 'NET_RATING']

    for col in cols:
        # Convert to numeric once to save processing time
        series = pd.to_numeric(df[col], errors='coerce')
        group = df.groupby(['PLAYER_ID', 'SEASON_YEAR'], sort=False)

        # 1. Calculate the EWM (The primary "Recent Form" feature)
        ewm_series = group[col].transform(
            lambda x: x.shift(1).ewm(alpha=0.3, min_periods=3, adjust=False).mean()
        )

        # 2. Calculate the Expanding Mean (The "Season Baseline" fallback)
        # min_periods=1 ensures we get a value as early as Game 2
        expanding_mean = group[col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

        # 3. Fill NaNs in EWM with the Expanding Mean, then round
        df[f'{col}_ewm_season_avg'] = ewm_series.fillna(expanding_mean).round(2)

    return df


# ── 5. Star-out flag ──────────────────────────────────────────────────────────

def _star_out_flag(
    df: pd.DataFrame,
    usg_scale: float = 80.0,
    min_games: int = 5,
) -> pd.DataFrame:
    """
    Identify each team's star for every team-game using leakage-free season
    averages (PTS_ewm_season_avg + USG_PCT_ewm_season_avg * usg_scale) and
    flag games where that star did not play (STAR_OUT_FLAG).

    Both season averages are already shift(1) EWM, so the composite score
    for a player at a given game reflects ONLY data from prior games. To
    find the star for a team's game on date D we look at every player who
    has ever played for that team in the same SEASON_YEAR and use their
    most recent row strictly before D. This avoids leakage from the game
    being predicted.

    Parameters
    ----------
    usg_scale : float
        Scale applied to USG_PCT (0-1 fraction) so it is comparable to PTS.
        Default 80 ≈ a 25% usage player contributes 20 "PTS-equivalent" to
        their star score.
    min_games : int
        Minimum prior games required for a player to be considered the
        team's star. Falls back to any player with history if no one
        qualifies (useful for very early-season games).

    Output columns
    --------------
    STAR_PLAYER_ID : Int64 (nullable)
        The player flagged as the team's star going into the game.
    STAR_OUT_FLAG : int {0, 1}
        1 if that star has no row for (TEAM_ID, GAME_ID), else 0.
    """
    df = df.copy()

    pts_col = "PTS_ewm_season_avg"
    usg_col = "USG_PCT_ewm_season_avg"

    if pts_col not in df.columns or usg_col not in df.columns:
        df["STAR_PLAYER_ID"] = pd.NA
        df["STAR_OUT_FLAG"] = 0
        return df

    df["_STAR_SCORE"] = (
        df[pts_col].fillna(0).astype(float)
        + df[usg_col].fillna(0).astype(float) * usg_scale
    )

    team_games = (
        df[["TEAM_ID", "SEASON_YEAR", "GAME_ID", "GAME_DATE"]]
        .drop_duplicates()
        .sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"])
    )

    rosters = (
        df.groupby(["TEAM_ID", "GAME_ID"])["PLAYER_ID"]
        .apply(set)
        .to_dict()
    )

    hist_cols = ["PLAYER_ID", "TEAM_ID", "SEASON_YEAR", "GAME_DATE", "_STAR_SCORE"]
    player_hist = df[hist_cols].sort_values("GAME_DATE")

    star_rows = []
    for (team_id, season), tg in team_games.groupby(["TEAM_ID", "SEASON_YEAR"]):
        team_hist = player_hist[
            (player_hist["TEAM_ID"] == team_id)
            & (player_hist["SEASON_YEAR"] == season)
        ]

        for _, g in tg.iterrows():
            game_id = g["GAME_ID"]
            game_date = g["GAME_DATE"]

            prior = team_hist[team_hist["GAME_DATE"] < game_date]
            if prior.empty:
                star_rows.append({
                    "TEAM_ID": team_id,
                    "GAME_ID": game_id,
                    "STAR_PLAYER_ID": pd.NA,
                    "STAR_OUT_FLAG": 0,
                })
                continue

            latest = prior.groupby("PLAYER_ID", as_index=False).tail(1)
            counts = prior.groupby("PLAYER_ID").size()
            qualified_ids = counts[counts >= min_games].index
            pool = latest[latest["PLAYER_ID"].isin(qualified_ids)]
            if pool.empty:
                pool = latest

            star_id = pool.loc[pool["_STAR_SCORE"].idxmax(), "PLAYER_ID"]
            roster = rosters.get((team_id, game_id), set())
            star_out = int(star_id not in roster)

            star_rows.append({
                "TEAM_ID": team_id,
                "GAME_ID": game_id,
                "STAR_PLAYER_ID": star_id,
                "STAR_OUT_FLAG": star_out,
            })

    stars = pd.DataFrame(star_rows)
    df = df.merge(stars, on=["TEAM_ID", "GAME_ID"], how="left")
    df["STAR_OUT_FLAG"] = df["STAR_OUT_FLAG"].fillna(0).astype(int)
    df = df.drop(columns=["_STAR_SCORE"])
    return df


# ── 5a. Star ↔ player position match ──────────────────────────────────────────

def _star_position_match(df: pd.DataFrame, pos_col: str = "pos") -> pd.DataFrame:
    """
    Attach the star's position and a match flag against the current player's
    position. Requires STAR_PLAYER_ID (from `_star_out_flag`) and a `pos`
    column with values like 'PG', 'SG', 'SF', 'PF', 'C'.

    Output columns
    --------------
    STAR_POS : object
        The star player's position.
    STAR_SAME_POS : Int8 (0/1, NaN if either position unknown)
        1 if STAR_POS == player's position (exact, e.g. SG == SG).
    STAR_SAME_POS_GROUP : Int8 (0/1, NaN if either position unknown)
        1 if both positions fall in the same broad bucket (G / F / C),
        based on the first character.
    """
    df = df.copy()

    if "STAR_PLAYER_ID" not in df.columns or pos_col not in df.columns:
        df["STAR_POS"] = pd.NA
        df["STAR_SAME_POS"] = pd.NA
        df["STAR_SAME_POS_GROUP"] = pd.NA
        return df

    # Player → most common position seen for that player (positions are
    # essentially static, but mode handles any one-off mislabels).
    pos_lookup = (
        df.dropna(subset=[pos_col])
          .groupby("PLAYER_ID")[pos_col]
          .agg(lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA)
    )

    df["STAR_POS"] = df["STAR_PLAYER_ID"].map(pos_lookup)

    both_known = df[pos_col].notna() & df["STAR_POS"].notna()

    same_exact = (df[pos_col].astype("string") == df["STAR_POS"].astype("string"))
    df["STAR_SAME_POS"] = same_exact.where(both_known).astype("Int8")

    def _bucket(s: pd.Series) -> pd.Series:
        return s.astype("string").str[0]  # 'G', 'F', or 'C'

    same_group = (_bucket(df[pos_col]) == _bucket(df["STAR_POS"]))
    df["STAR_SAME_POS_GROUP"] = same_group.where(both_known).astype("Int8")

    return df


# ── 5b. Minutes conditioned on star status ────────────────────────────────────

def _min_by_star_status(df: pd.DataFrame, min_sample: int = 2) -> pd.DataFrame:
    """
    Per player, leakage-free historical average minutes split by the team's
    star status going into each game:

        MIN_WHEN_STAR_IN   = avg MIN in prior same-season games where STAR_OUT_FLAG == 0
        MIN_WHEN_STAR_OUT  = avg MIN in prior same-season games where STAR_OUT_FLAG == 1
        MIN_STAR_OUT_DELTA = MIN_WHEN_STAR_OUT - MIN_WHEN_STAR_IN

    Uses a shift(1) expanding mean within (PLAYER_ID, SEASON_YEAR) restricted
    to games matching each flag value, so the value at row t only reflects
    games strictly before t. Cells with fewer than `min_sample` prior matching
    games are left as NaN to avoid noisy small-sample estimates.

    Requires STAR_OUT_FLAG (from `_star_out_flag`) and MIN in df.
    """
    df = df.copy()

    if "STAR_OUT_FLAG" not in df.columns or "MIN" not in df.columns:
        df["MIN_WHEN_STAR_IN"] = np.nan
        df["MIN_WHEN_STAR_OUT"] = np.nan
        df["MIN_STAR_OUT_DELTA"] = np.nan
        return df

    df = df.sort_values(["PLAYER_ID", "SEASON_YEAR", "GAME_DATE"])

    def _conditional_expanding_mean(group: pd.DataFrame, flag_val: int) -> pd.Series:
        mask = (group["STAR_OUT_FLAG"] == flag_val)
        masked_min = group["MIN"].where(mask)
        # shift(1) so the value at row t uses only games strictly before t
        shifted = masked_min.shift(1)
        running_sum = shifted.expanding().sum()
        running_cnt = shifted.expanding().count()
        avg = running_sum / running_cnt.replace(0, np.nan)
        avg = avg.where(running_cnt >= min_sample)
        return avg.round(2)

    df["MIN_WHEN_STAR_IN"] = (
        df.groupby(["PLAYER_ID", "SEASON_YEAR"], group_keys=False)
          .apply(lambda g: _conditional_expanding_mean(g, 0))
    )
    df["MIN_WHEN_STAR_OUT"] = (
        df.groupby(["PLAYER_ID", "SEASON_YEAR"], group_keys=False)
          .apply(lambda g: _conditional_expanding_mean(g, 1))
    )
    df["MIN_STAR_OUT_DELTA"] = (
        df["MIN_WHEN_STAR_OUT"] - df["MIN_WHEN_STAR_IN"]
    ).round(2)

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

def _detect_star_players(
    df,
    min_minutes: int = 10,
    min_games: int = 10,
    name_dict: dict | None = None,
    current_season: str | None = None,   # e.g. "2024-25" — fixes trade bug
    recent_n_games: int | None = None,   # alternative: use last N games per player-team
):
    df = df.copy()

    # ── 1. normalize names ───────────────────────────────────────────────────
    df["PLAYER_NAME_NORM"] = (
        df["PLAYER_NAME"].map(lambda x: name_dict.get(x, x))
        if name_dict else df["PLAYER_NAME"]
    )

    # ── 2. resolve current team per player (trade fix) ───────────────────────
    # Use only the player's most recent TEAM_ID so traded players
    # are evaluated under their current team, not their old one.
    current_team = (
        df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME_NORM")["TEAM_ID"]
        .last()
    )
    df["CURRENT_TEAM_ID"] = df["PLAYER_NAME_NORM"].map(current_team)

    # ── 3. build the scoring window ──────────────────────────────────────────
    # Prefer season filter, fall back to recent_n_games, fall back to all data.
    scoring_df = df.copy()

    if current_season is not None and "SEASON" in df.columns:
        scoring_df = scoring_df[scoring_df["SEASON"] == current_season]
    elif recent_n_games is not None:
        scoring_df = (
            scoring_df.sort_values("GAME_DATE")
            .groupby(["PLAYER_NAME_NORM", "CURRENT_TEAM_ID"])
            .tail(recent_n_games)
        )

    # Only score within current-team games — eliminates old-team contamination
    scoring_df = scoring_df[scoring_df["TEAM_ID"] == scoring_df["CURRENT_TEAM_ID"]]

    # ── 4. filter for qualified minutes ─────────────────────────────────────
    active = scoring_df[scoring_df["MIN"] >= min_minutes]

    stats_df = active.groupby(["CURRENT_TEAM_ID", "PLAYER_NAME_NORM"]).agg(
        USG_PCT    = ("USG_PCT",    "mean"),
        TS_PCT     = ("TS_PCT",     "mean"),
        EFG_PCT    = ("EFG_PCT",    "mean"),
        PTS        = ("PTS",        "mean"),
        PIE        = ("PIE",        "mean"),
        NET_RATING = ("NET_RATING", "mean"),
        GAMES      = ("MIN",        "count"),
    ).reset_index()

    stats_df = stats_df[stats_df["GAMES"] >= min_games]

    if stats_df.empty:
        df["IS_TOP_STAR"] = 0
        df["IS_TOP_1_STAR"] = 0
        df["ACTIVE_STARS_COUNT"] = 0
        df["TOP_STAR_ACTIVE"] = 0
        df["TOP_PLAYER"] = None
        df["SECOND_TOP_PLAYER"] = None
        df["THIRD_TOP_PLAYER"] = None
        return df.drop(columns=["PLAYER_NAME_NORM", "CURRENT_TEAM_ID"])

    # ── 5. rank-based normalization (robust to outliers) ─────────────────────
    # Percentile rank within team rather than min-max — a single outlier
    # can't compress the rest of the team's scores with this approach.
    score_cols = ["USG_PCT", "TS_PCT", "EFG_PCT", "PTS", "PIE", "NET_RATING"]
    for col in score_cols:
        stats_df[f"{col}_RANK"] = stats_df.groupby("CURRENT_TEAM_ID")[col].rank(
            pct=True, method="average", na_option="bottom"
        )

    # ── 6. star score — weights match comment, all 6 metrics used ────────────
    stats_df["STAR_SCORE"] = (
        0.25 * stats_df["USG_PCT_RANK"]    +   # usage: how much offense runs through them
        0.20 * stats_df["PIE_RANK"]        +   # overall impact catch-all
        0.20 * stats_df["PTS_RANK"]        +   # raw scoring load
        0.15 * stats_df["TS_PCT_RANK"]     +   # scoring efficiency
        0.10 * stats_df["EFG_PCT_RANK"]    +   # field goal quality
        0.10 * stats_df["NET_RATING_RANK"]     # on/off impact
    )

    # ── 7. top 3 per team ────────────────────────────────────────────────────
    sorted_stars = stats_df.sort_values(
        ["CURRENT_TEAM_ID", "STAR_SCORE"], ascending=[True, False]
    )
    top_3 = sorted_stars.groupby("CURRENT_TEAM_ID").head(3).copy()
    top_3["STAR_RANK"] = top_3.groupby("CURRENT_TEAM_ID").cumcount() + 1

    top_star_map    = top_3[top_3["STAR_RANK"] == 1].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]
    second_star_map = top_3[top_3["STAR_RANK"] == 2].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]
    third_star_map  = top_3[top_3["STAR_RANK"] == 3].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]

    df["TOP_PLAYER"]        = df["CURRENT_TEAM_ID"].map(top_star_map)
    df["SECOND_TOP_PLAYER"] = df["CURRENT_TEAM_ID"].map(second_star_map)
    df["THIRD_TOP_PLAYER"]  = df["CURRENT_TEAM_ID"].map(third_star_map)

    top_stars_set = set(zip(top_3["CURRENT_TEAM_ID"], top_3["PLAYER_NAME_NORM"]))
    top1_set      = set(zip(
        top_3[top_3["STAR_RANK"] == 1]["CURRENT_TEAM_ID"],
        top_3[top_3["STAR_RANK"] == 1]["PLAYER_NAME_NORM"],
    ))

    # ── 8. flag players in main df ───────────────────────────────────────────
    # Use CURRENT_TEAM_ID for lookup so traded players resolve to new team
    df["IS_TOP_STAR"]   = df.apply(
        lambda r: 1 if (r["CURRENT_TEAM_ID"], r["PLAYER_NAME_NORM"]) in top_stars_set else 0,
        axis=1,
    )
    df["IS_TOP_1_STAR"] = df.apply(
        lambda r: 1 if (r["CURRENT_TEAM_ID"], r["PLAYER_NAME_NORM"]) in top1_set else 0,
        axis=1,
    )

    # ── 9. active stars per game ─────────────────────────────────────────────
    active_stars_per_game = (
        df[(df["IS_TOP_STAR"] == 1) & (df["MIN"] >= min_minutes)]
        .groupby(["GAME_ID", "CURRENT_TEAM_ID"])["PLAYER_NAME_NORM"]
        .nunique()                          # distinct players, not sum of flag
        .reset_index(name="ACTIVE_STARS_COUNT")
        .assign(ACTIVE_STARS_COUNT=lambda x: x["ACTIVE_STARS_COUNT"].clip(upper=3))
    )
    df = df.merge(active_stars_per_game, on=["GAME_ID", "CURRENT_TEAM_ID"], how="left")
    df["ACTIVE_STARS_COUNT"] = df["ACTIVE_STARS_COUNT"].fillna(0).astype(int)

    # ── 10. top-1 star active flag ───────────────────────────────────────────
    top1_active_per_game = (
        df[(df["IS_TOP_1_STAR"] == 1) & (df["MIN"] >= min_minutes)]
        .groupby(["GAME_ID", "CURRENT_TEAM_ID"])["IS_TOP_1_STAR"]
        .max()
        .reset_index(name="TOP_STAR_ACTIVE")
    )
    df = df.merge(top1_active_per_game, on=["GAME_ID", "CURRENT_TEAM_ID"], how="left")
    df["TOP_STAR_ACTIVE"] = df["TOP_STAR_ACTIVE"].fillna(0).astype(int)

    # ── 11. individual star active flags per game ────────────────────────────
    for rank, col_name in [(1, "TOP_PLAYER_ACTIVE"), (2, "SECOND_PLAYER_ACTIVE"), (3, "THIRD_PLAYER_ACTIVE")]:
        star_name_map = top_3[top_3["STAR_RANK"] == rank].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]

        # resolve each row's star name for that rank
        df[f"_STAR_{rank}_NAME"] = df["CURRENT_TEAM_ID"].map(star_name_map)

        # find games where that specific player hit min_minutes
        star_active = (
            df[
                (df["PLAYER_NAME_NORM"] == df[f"_STAR_{rank}_NAME"]) &
                (df["MIN"] >= min_minutes)
            ]
            .groupby(["GAME_ID", "CURRENT_TEAM_ID"])
            .size()
            .gt(0)
            .astype(int)
            .reset_index(name=col_name)
        )

        df = df.merge(star_active, on=["GAME_ID", "CURRENT_TEAM_ID"], how="left")
        df[col_name] = df[col_name].fillna(0).astype(int)
        df.drop(columns=[f"_STAR_{rank}_NAME"], inplace=True)

    return df.drop(columns=["PLAYER_NAME_NORM", "CURRENT_TEAM_ID"])

# ------- HOME/AWAY AVERAGES -------------------------------

def _HomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE'):
    df = player_data.copy()
    df.reset_index(drop=True, inplace=True)
    df.sort_values([player_id_col, date_col], inplace=True)

    if 'IS_HOME' not in df.columns:
        return df

    metrics = ['PTS_PER_MIN', 'TCHS_PER_MIN', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'USG_PCT', 'POSS', 'MIN',]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return df

    global_means = df[metrics].mean()

    for metric in metrics:
        overall_avg = df.groupby(player_id_col)[metric].transform(lambda x: x.shift(1).expanding().mean())
        loc_avg    = df.groupby([player_id_col, 'IS_HOME'])[metric].transform(lambda x: x.shift(1).expanding().mean())

        home_col = f'HOME_AVG_{metric}_TO_DATE'
        away_col = f'AWAY_AVG_{metric}_TO_DATE'

        df[home_col] = np.where(df['IS_HOME'] == 1, loc_avg, np.nan)
        df[away_col] = np.where(df['IS_HOME'] == 0, loc_avg, np.nan)

        df[home_col] = df.groupby(player_id_col)[home_col].transform('ffill')
        df[away_col] = df.groupby(player_id_col)[away_col].transform('ffill')

        df[home_col] = df[home_col].fillna(overall_avg).fillna(global_means[metric]).astype('float32').round(2)
        df[away_col] = df[away_col].fillna(overall_avg).fillna(global_means[metric]).astype('float32').round(2)

        df[f'PLAYER_HOME_{metric}_DELTA'] = (df[home_col] - overall_avg).fillna(0).round(2)
        df[f'PLAYER_AWAY_{metric}_DELTA'] = (df[away_col] - overall_avg).fillna(0).round(2)

    return df


# ── PBP stint rolling features ────────────────────────────────────────────────

def _pbp_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['avg_stint_min', 'max_stint_min', 'total_stints', 'avg_entry_sec']
    present = [c for c in cols if c in df.columns]
    if not present:
        return df

    for col in present:
        # season avg: expanding mean within player-season, no leakage
        df[f'{col}_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON_YEAR'])[col]
            .transform(lambda x: x.shift(1).expanding().mean().round(2))
        )
        # EWM-10: exponentially weighted recent form
        df[f'{col}_10_ewm'] = (
            df.groupby('PLAYER_ID')[col]
            .transform(lambda x: x.shift(1).ewm(span=10, adjust=False).mean().round(2))
        )

    return df