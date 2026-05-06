import pandas as pd
import numpy as np
from src.features.feature_engineer.min_features import _detect_star_players

def ppm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    df = _per_min_features(df)
    df = _rolling_player(df)
    df = _ewm_player(df)
    # df = _lag_features(df)
    df = _starter_features(df)
    df = _season_averages(df)
    # df = _lineup_usg_shift(df)
    df = _team_context(df)
    df = _team_allowed_context(df)
    df = _team_allowed_rolling(df)
    df = _opp_pos_ppm_allowed(df)
    df = _own_pos_allowed_features(df)
    df = _schedule_features(df)
    df = _volatility_features(df)
    df = _opponent_stats(df)
    df = _expectedPace(df)
    df = _detect_star_players(df)
    df = _HomeAwayAverages(df)
    # df = _role_tier_features(df)
    # df = _role_tier_interactions(df)


    return df

def _per_min_features(df: pd.DataFrame) -> pd.DataFrame:
    df['PTS_PER_MIN'] = df['PTS'] / df['MIN'].replace(0, np.nan)
    df['POSS_PER_MIN'] = df['POSS'] / df['MIN'].replace(0, np.nan)
    df['FGA_PER_MIN'] = df['FGA'] / df['MIN'].replace(0, np.nan)
    df['FGM_PER_MIN'] = df['FGM'] / df['MIN'].replace(0, np.nan)
    df['3PA_PER_MIN'] = df['FG3A'] / df['MIN'].replace(0, np.nan)
    df['3PM_PER_MIN'] = df['FG3M'] / df['MIN'].replace(0, np.nan)
    df['FTA_PER_MIN'] = df['FTA'] / df['MIN'].replace(0, np.nan)
    df['FTM_PER_MIN'] = df['FTM'] / df['MIN'].replace(0, np.nan)
    df['TCHS_PER_MIN']  = df['TCHS'] / df['MIN'].replace(0, np.nan)
    df['PASS_PER_MIN']  = df['PASS'] / df['MIN'].replace(0, np.nan)
    df['UFGA_PER_MIN']  = df['UFGA'] / df['MIN'].replace(0, np.nan)
    df['CFGA_PER_MIN']  = df['CFGA'] / df['MIN'].replace(0, np.nan)
    df['SAST_PER_MIN']  = df['SAST'] / df['MIN'].replace(0, np.nan)
    df['DFGA_PER_MIN']  = df['DFGA'] / df['MIN'].replace(0, np.nan)
    return df

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'MIN', 'PTS', 'USG_PCT', 'POSS', 'PTS_PER_MIN', 'POSS_PER_MIN',
        'TS_PCT', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'FGM_PER_MIN', '3PM_PER_MIN', 'FTM_PER_MIN',
        'TCHS_PER_MIN', 'PASS_PER_MIN', 'UFGA_PER_MIN', 'CFGA_PER_MIN', 'SAST_PER_MIN', 'UFG_PCT', 'CFG_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT'
    ]

    for window in [5,10,20]:
        for col in cols:
            df[f'{col}_roll{window}'] = (
                df.groupby('PLAYER_ID')[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean().round(2))
            )

    return df

def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'MIN', 'PTS', 'USG_PCT', 'POSS', 'PTS_PER_MIN', 'POSS_PER_MIN',
        'TS_PCT', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'FGM_PER_MIN', '3PM_PER_MIN', 'FTM_PER_MIN',
        'TCHS_PER_MIN', 'PASS_PER_MIN', 'UFGA_PER_MIN', 'CFGA_PER_MIN', 'SAST_PER_MIN', 'UFG_PCT', 'CFG_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'DFGA_PER_MIN'
    ]

    for span in [5, 10, 15, 20]:
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
    cols = [
        'MIN', 'PTS', 'USG_PCT', 'POSS', 'PTS_PER_MIN', 'POSS_PER_MIN',
        'TS_PCT', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'FGM_PER_MIN', '3PM_PER_MIN', 'FTM_PER_MIN',
        'TCHS_PER_MIN', 'PASS_PER_MIN', 'UFGA_PER_MIN', 'CFGA_PER_MIN', 'SAST_PER_MIN', 'UFG_PCT', 'CFG_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'DFGA_PER_MIN'
    ]
    for lag in [1, 2]:
        for col in cols:
            df[f'{col}_lag{lag}'] = df.groupby('PLAYER_ID')[col].shift(lag)

    return df


# ── 3. Starter / role ─────────────────────────────────────────────────────────

def _starter_features(df: pd.DataFrame) -> pd.DataFrame:
    df['STARTER_ROLL10_PCT'] = (
        df.groupby('PLAYER_ID')['STARTING']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean().round(2))
    )
    return df


# ── 4. Season averages (expanding) ───────────────────────────────────────────

def _season_averages(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'MIN', 'PTS', 'USG_PCT', 'POSS', 'PTS_PER_MIN', 'POSS_PER_MIN',
        'TS_PCT', 'FGA_PER_MIN', '3PA_PER_MIN', 'FTA_PER_MIN', 'FGM_PER_MIN', '3PM_PER_MIN', 'FTM_PER_MIN',
        'TCHS_PER_MIN', 'PASS_PER_MIN', 'UFGA_PER_MIN', 'CFGA_PER_MIN', 'SAST_PER_MIN', 'UFG_PCT', 'CFG_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'DFGA_PER_MIN'
    ]
    for col in cols:
        df[f'{col}_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON_YEAR'])[col]
            .transform(lambda x: x.shift(1).expanding().mean().round(2))
        )

    return df

# ── 5b. Lineup USG shift ──────────────────────────────────────────────────────

def _lineup_usg_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per team-game, identify the team's top USG player going into that game using
    USG_PCT_season_avg (already a shift(1)-expanding mean, so leakage-safe) and
    emit features capturing whether the top player is active/starting and how
    much USG must be absorbed by the rest of the lineup when they are out.

    Outputs:
        TOP_USG_PLAYER_VALUE     - prior season-avg USG of the identified top player
        TOP_USG_PLAYER_ACTIVE    - 1 if that player has a row in the team's game
        TOP_USG_PLAYER_STARTING  - 1 if that player started the team's game
        LINEUP_USG_SHIFT         - TOP_USG_PLAYER_VALUE when top player is out, else 0
    """
    required = {"TEAM_ID", "SEASON_YEAR", "GAME_ID", "GAME_DATE",
                "PLAYER_ID", "USG_PCT_season_avg", "STARTING"}
    if not required.issubset(df.columns):
        return df

    df = df.copy()

    # Build per-(TEAM_ID, SEASON_YEAR) timelines of each player's most recently
    # known USG_PCT_season_avg, forward-filled across the team's game dates so a
    # player who sits today still has their prior season-to-date USG available.
    tops = []
    for (tid, season), grp in df.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False):
        dates = np.sort(grp["GAME_DATE"].unique())
        pivot = (
            grp.pivot_table(
                index="GAME_DATE",
                columns="PLAYER_ID",
                values="USG_PCT_season_avg",
                aggfunc="last",
            )
            .reindex(dates)
            .ffill()
        )
        if pivot.shape[1] == 0:
            continue

        all_nan = pivot.isna().all(axis=1)
        top_pid = pivot.idxmax(axis=1)
        top_val = pivot.max(axis=1)
        top_pid[all_nan] = np.nan
        top_val[all_nan] = np.nan

        tops.append(pd.DataFrame({
            "TEAM_ID": tid,
            "SEASON_YEAR": season,
            "GAME_DATE": dates,
            "TOP_USG_PLAYER_ID": top_pid.values,
            "TOP_USG_PLAYER_VALUE": top_val.values,
        }))

    if not tops:
        return df

    top_map = pd.concat(tops, ignore_index=True)
    df = df.merge(top_map, on=["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], how="left")

    # Flag the top player's row (if present), then aggregate per team-game.
    is_top = (df["PLAYER_ID"] == df["TOP_USG_PLAYER_ID"]).astype(int)
    is_top_starter = (is_top.astype(bool) & (df["STARTING"] == 1)).astype(int)
    df["_is_top_row"] = is_top
    df["_is_top_starter_row"] = is_top_starter

    active = (
        df.groupby(["TEAM_ID", "GAME_ID"])["_is_top_row"].max()
        .rename("TOP_USG_PLAYER_ACTIVE").reset_index()
    )
    starter = (
        df.groupby(["TEAM_ID", "GAME_ID"])["_is_top_starter_row"].max()
        .rename("TOP_USG_PLAYER_STARTING").reset_index()
    )

    df = df.merge(active, on=["TEAM_ID", "GAME_ID"], how="left")
    df = df.merge(starter, on=["TEAM_ID", "GAME_ID"], how="left")

    # When we have no history to identify a top player (e.g., season opener),
    # leave the flags as NaN instead of implying the player is out.
    unknown = df["TOP_USG_PLAYER_ID"].isna()
    df.loc[unknown, ["TOP_USG_PLAYER_ACTIVE", "TOP_USG_PLAYER_STARTING"]] = np.nan

    df["LINEUP_USG_SHIFT"] = np.where(
        df["TOP_USG_PLAYER_ACTIVE"] == 0,
        df["TOP_USG_PLAYER_VALUE"].fillna(0.0),
        0.0,
    ).round(3)

    df = df.drop(columns=["_is_top_row", "_is_top_starter_row", "TOP_USG_PLAYER_ID"])
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

# ── Team defense allowed (prior games only) ───────────────────────────────────
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
        "OPP_FGM",
        "OPP_FGA",
        "OPP_FG3M",
        "OPP_FG3A",
        "OPP_FTA",
        "OPP_PTS",
        "OPP_POSS",
        "OPP_PFD",
        "OPP_DEF_RATING",
        "OPP_EFG_PCT",
        "OPP_TS_PCT",
        "OPP_DREB_PCT",
        "OPP_TM_TOV_PCT",
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

    # Cumulative % allowed to date (prior games only)
    cum_opp_fgm_prior  = g["OPP_FGM"].transform(lambda x: x.cumsum().shift(1))
    cum_opp_fga_prior  = g["OPP_FGA"].transform(lambda x: x.cumsum().shift(1))
    cum_opp_fg3m_prior = g["OPP_FG3M"].transform(lambda x: x.cumsum().shift(1))
    cum_opp_fg3a_prior = g["OPP_FG3A"].transform(lambda x: x.cumsum().shift(1))

    team_game["TEAM_FG%_ALLOWED"] = (
        (cum_opp_fgm_prior / cum_opp_fga_prior.replace(0, np.nan)).round(3)
    )
    team_game["TEAM_FG3%_ALLOWED"] = (
        (cum_opp_fg3m_prior / cum_opp_fg3a_prior.replace(0, np.nan)).round(3)
    )

    # Per-game allowed averages to date (prior games only)
    def prior_expanding_mean(x):
        return x.shift(1).expanding().mean().round(3)

    for raw_col, new_col in [
        ("OPP_FTA",        "TEAM_FTA_ALLOWED"),
        ("OPP_FG3A",       "TEAM_FG3A_ALLOWED"),
        ("OPP_FG3M",       "TEAM_FG3M_ALLOWED"),
        ("OPP_PTS",        "TEAM_PTS_ALLOWED"),
        ("OPP_POSS",       "TEAM_POSS_ALLOWED"),
        ("OPP_PFD",        "TEAM_PFD_ALLOWED"),
        # High-signal additions
        ("OPP_DEF_RATING", "TEAM_DEF_RATING_ALLOWED"),
        ("OPP_EFG_PCT",    "TEAM_EFG_PCT_ALLOWED"),
        ("OPP_TS_PCT",     "TEAM_TS_PCT_ALLOWED"),
        ("OPP_DREB_PCT",   "TEAM_DREB_PCT_ALLOWED"),
        ("OPP_TM_TOV_PCT", "TEAM_TM_TOV_PCT_ALLOWED"),
    ]:
        team_game[new_col] = g[raw_col].transform(prior_expanding_mean)

    allowed_cols = [
        "TEAM_ID",
        "GAME_ID",
        "TEAM_FG%_ALLOWED",
        "TEAM_FG3%_ALLOWED",
        "TEAM_FG3A_ALLOWED",
        "TEAM_FG3M_ALLOWED",
        "TEAM_FTA_ALLOWED",
        "TEAM_PTS_ALLOWED",
        "TEAM_POSS_ALLOWED",
        "TEAM_PFD_ALLOWED",
        # High-signal additions
        "TEAM_DEF_RATING_ALLOWED",
        "TEAM_EFG_PCT_ALLOWED",
        "TEAM_TS_PCT_ALLOWED",
        "TEAM_DREB_PCT_ALLOWED",
        "TEAM_TM_TOV_PCT_ALLOWED",
    ]

    allowed_map = team_game[allowed_cols]
    out = df.merge(allowed_map, on=["TEAM_ID", "GAME_ID"], how="left")

    if "OPP_TEAM_ID" not in out.columns:
        return out

    opp_rename = {
        "TEAM_ID":                  "OPP_TEAM_ID",
        "TEAM_FG%_ALLOWED":         "OPP_TEAM_FG%_ALLOWED",
        "TEAM_FG3%_ALLOWED":        "OPP_TEAM_FG3%_ALLOWED",
        "TEAM_FG3A_ALLOWED":        "OPP_TEAM_FG3A_ALLOWED",
        "TEAM_FG3M_ALLOWED":        "OPP_TEAM_FG3M_ALLOWED",
        "TEAM_FTA_ALLOWED":         "OPP_TEAM_FTA_ALLOWED",
        "TEAM_PTS_ALLOWED":         "OPP_TEAM_PTS_ALLOWED",
        "TEAM_POSS_ALLOWED":        "OPP_TEAM_POSS_ALLOWED",
        "TEAM_PFD_ALLOWED":         "OPP_TEAM_PFD_ALLOWED",
        # High-signal additions
        "TEAM_DEF_RATING_ALLOWED":  "OPP_TEAM_DEF_RATING_ALLOWED",
        "TEAM_EFG_PCT_ALLOWED":     "OPP_TEAM_EFG_PCT_ALLOWED",
        "TEAM_TS_PCT_ALLOWED":      "OPP_TEAM_TS_PCT_ALLOWED",
        "TEAM_DREB_PCT_ALLOWED":    "OPP_TEAM_DREB_PCT_ALLOWED",
        "TEAM_TM_TOV_PCT_ALLOWED":  "OPP_TEAM_TM_TOV_PCT_ALLOWED",
    }

    opp_allowed_map = allowed_map.rename(columns=opp_rename)
    return out.merge(opp_allowed_map, on=["OPP_TEAM_ID", "GAME_ID"], how="left")

def _team_allowed_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling means of per-game opponent stats (prior games only) for the player's
    team and the opponent, aligned with _team_allowed_context input columns.
    """
    required = [
        "TEAM_ID",
        "SEASON_YEAR",
        "GAME_ID",
        "GAME_DATE",
        "OPP_FGM",
        "OPP_FGA",
        "OPP_FG3M",
        "OPP_FG3A",
        "OPP_FTA",
        "OPP_PTS",
        "OPP_POSS",
        "OPP_PFD",
        "OPP_DEF_RATING",
        "OPP_EFG_PCT",
        "OPP_TS_PCT",
        "OPP_DREB_PCT",
        "OPP_TM_TOV_PCT",
    ]
    if any(c not in df.columns for c in required):
        return df
    if "OPP_TEAM_ID" not in df.columns:
        return df

    team_game = (
        df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
        .sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"])
        .copy()
    )
    g = team_game.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False)

    # Derived per-game ratio columns
    team_game["OPP_FGPCT_GAME"]  = team_game["OPP_FGM"] / team_game["OPP_FGA"].replace(0, np.nan)
    team_game["OPP_FG3PCT_GAME"] = team_game["OPP_FG3M"] / team_game["OPP_FG3A"].replace(0, np.nan)

    # (raw_col, output_prefix) — raw_col is what we roll, output_prefix is the TEAM_*_ALLOWED_roll{w} name
    roll_stat_cols = [
        ("OPP_FGPCT_GAME",   "TEAM_FG%_ALLOWED"),
        ("OPP_FG3PCT_GAME",  "TEAM_FG3%_ALLOWED"),
        ("OPP_FTA",          "TEAM_FTA_ALLOWED"),
        ("OPP_FG3M",         "TEAM_FG3M_ALLOWED"),
        ("OPP_FG3A",         "TEAM_FG3A_ALLOWED"),
        ("OPP_PTS",          "TEAM_PTS_ALLOWED"),
        ("OPP_POSS",         "TEAM_POSS_ALLOWED"),
        ("OPP_PFD",          "TEAM_PFD_ALLOWED"),
        # High-signal additions
        ("OPP_DEF_RATING",   "TEAM_DEF_RATING_ALLOWED"),
        ("OPP_EFG_PCT",      "TEAM_EFG_PCT_ALLOWED"),
        ("OPP_TS_PCT",       "TEAM_TS_PCT_ALLOWED"),
        ("OPP_DREB_PCT",     "TEAM_DREB_PCT_ALLOWED"),
        ("OPP_TM_TOV_PCT",   "TEAM_TM_TOV_PCT_ALLOWED"),
    ]

    roll_windows = [3, 5, 10]
    for window in roll_windows:
        for raw_col, prefix in roll_stat_cols:
            team_game[f"{prefix}_roll{window}"] = g[raw_col].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=3).mean().round(3)
            )

    team_roll_cols = [c for c in team_game.columns if c.startswith("TEAM_") and "_ALLOWED_roll" in c]
    allowed_roll_map = team_game[["TEAM_ID", "GAME_ID"] + team_roll_cols]

    out = df.merge(allowed_roll_map, on=["TEAM_ID", "GAME_ID"], how="left")

    opp_rename = {c: c.replace("TEAM_", "OPP_TEAM_", 1) for c in team_roll_cols}
    opp_allowed_roll_map = (
        allowed_roll_map
        .rename(columns=opp_rename)
        .rename(columns={"TEAM_ID": "OPP_TEAM_ID"})
    )

    return out.merge(opp_allowed_roll_map, on=["OPP_TEAM_ID", "GAME_ID"], how="left")


# ── Position-level PPM / PTS allowed (season-to-date, leakage-safe) ──────────

def _opp_pos_ppm_allowed(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (defending team, season, position), compute season-to-date defender
    "allowed" features, with a shift(1) so the current game is excluded
    (prior games only -> no leakage):

        PPM_ALLOWED_TO_{POS} = cumulative PTS allowed / cumulative MIN allowed
        PTS_ALLOWED_TO_{POS} = expanding mean of PTS allowed per game

    Pivoted wide so every player row gets one feature per position, merged via
    the player's OPP_TEAM_ID (the defender they are facing).
    """
    required = {"TEAM_ID", "OPP_TEAM_ID", "SEASON_YEAR", "GAME_ID", "GAME_DATE",
                "POS", "PTS", "MIN"}
    if not required.issubset(df.columns):
        return df

    df = df.copy()
    pos_norm = df["POS"].fillna("UNK").astype(str).str.upper().str.strip()

    work = pd.DataFrame({
        "DEF_TEAM_ID": df["OPP_TEAM_ID"],
        "SEASON_YEAR": df["SEASON_YEAR"],
        "GAME_ID":     df["GAME_ID"],
        "GAME_DATE":   df["GAME_DATE"],
        "POS":         pos_norm,
        "PTS":         df["PTS"],
        "MIN":         df["MIN"],
    })

    pos_game = (
        work.groupby(
            ["DEF_TEAM_ID", "SEASON_YEAR", "GAME_ID", "GAME_DATE", "POS"],
            as_index=False,
        )
        .agg(PTS=("PTS", "sum"), MIN=("MIN", "sum"))
        .sort_values(["DEF_TEAM_ID", "SEASON_YEAR", "POS", "GAME_DATE"])
    )

    g = pos_game.groupby(["DEF_TEAM_ID", "SEASON_YEAR", "POS"], sort=False)
    cum_pts_prior = g["PTS"].transform(lambda x: x.cumsum().shift(1))
    cum_min_prior = g["MIN"].transform(lambda x: x.cumsum().shift(1))
    pos_game["PPM_ALLOWED"] = (
        cum_pts_prior / cum_min_prior.replace(0, np.nan)
    ).round(4)
    pos_game["PTS_ALLOWED"] = (
        g["PTS"].transform(lambda x: x.shift(1).expanding().mean())
    ).round(2)

    def _pivot(value_col: str, prefix: str) -> pd.DataFrame:
        wide = (
            pos_game.pivot_table(
                index=["DEF_TEAM_ID", "GAME_ID"],
                columns="POS",
                values=value_col,
                aggfunc="last",
            )
            .reset_index()
        )
        wide.columns.name = None
        rename = {
            c: f"{prefix}_{c}"
            for c in wide.columns
            if c not in {"DEF_TEAM_ID", "GAME_ID"}
        }
        return wide.rename(columns=rename).rename(
            columns={"DEF_TEAM_ID": "OPP_TEAM_ID"}
        )

    ppm_wide = _pivot("PPM_ALLOWED", "PPM_ALLOWED_TO")
    pts_wide = _pivot("PTS_ALLOWED", "PTS_ALLOWED_TO")

    df = df.merge(ppm_wide, on=["OPP_TEAM_ID", "GAME_ID"], how="left")
    df = df.merge(pts_wide, on=["OPP_TEAM_ID", "GAME_ID"], how="left")
    return df


# ── Self-position selectors + matchup expectation ────────────────────────────

def _own_pos_allowed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the wide PPM_ALLOWED_TO_{POS} / PTS_ALLOWED_TO_{POS} columns into
    single per-row features that pick the value matching the player's own POS.

    Adds:
        OPP_PPM_ALLOWED_TO_OWN_POS
        OPP_PTS_ALLOWED_TO_OWN_POS
        EXP_PTS_FROM_OPP_POS  (= MIN_roll10 * OPP_PPM_ALLOWED_TO_OWN_POS)
    """
    if "POS" not in df.columns:
        return df

    df = df.copy()
    pos = df["POS"].fillna("UNK").astype(str).str.upper().str.strip()

    ppm_out = pd.Series(np.nan, index=df.index)
    pts_out = pd.Series(np.nan, index=df.index)
    for p in pos.unique():
        mask = (pos == p)
        ppm_col = f"PPM_ALLOWED_TO_{p}"
        pts_col = f"PTS_ALLOWED_TO_{p}"
        if ppm_col in df.columns:
            ppm_out.loc[mask] = df.loc[mask, ppm_col]
        if pts_col in df.columns:
            pts_out.loc[mask] = df.loc[mask, pts_col]

    df["OPP_PPM_ALLOWED_TO_OWN_POS"] = ppm_out.round(4)
    df["OPP_PTS_ALLOWED_TO_OWN_POS"] = pts_out.round(2)

    if "MIN_roll10" in df.columns:
        df["EXP_PTS_FROM_OPP_POS"] = (
            df["MIN_roll10"] * df["OPP_PPM_ALLOWED_TO_OWN_POS"]
        ).round(2)

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

    df['PTS_PER_MIN_std5'] = (
        df.groupby('PLAYER_ID')['PTS_PER_MIN']
        .transform(lambda x: x.shift(1).rolling(5).std().round(2))
    )
    df['PTS_PER_MIN_std10'] = (
        df.groupby('PLAYER_ID')['PTS_PER_MIN']
        .transform(lambda x: x.shift(1).rolling(10).std().round(2))
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