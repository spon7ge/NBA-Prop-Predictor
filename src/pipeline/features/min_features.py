import numpy as np
import pandas as pd

MIN_FEATURES = [
    "MIN_10_ewm",
    "MIN_SEASON_MEAN",
    "STARTER_ROLL10_PCT",
    "CONSEC_STARTS",
    "MIN_RATE_OF_CHANGE",
    "TEAM_MIN_RANK_L10",
    "TEAM_USG_RANK_L10",
    "MIN_P10_L10",
    "MIN_P90_L10",
    "MIN_STD_L10",
    "MIN_SEASON_STD",
    "SPD_10_ewm",
    "USG_PCT_lag1",
    "USG_PCT_lag2",
    "AST_PCT_lag1",
    "AST_PCT_lag2",
    "PIE_lag1",
    "PIE_lag2",
    "GAMES_PLAYED_LAST_7_DAYS",
    "GAMES_PLAYED_LAST_14_DAYS",
    "MIN_SUM_LAST_7_DAYS",
    "TOP_PLAYER_ACTIVE",
    "ACTIVE_STARS_COUNT",
]

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25", 4: "S26"}


def min_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the MIN quantile model."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns:
        df["STARTING"] = df["START_POSITION"].notna().astype(int)
    if "PF_PER_MIN" not in df.columns:
        df["PF_PER_MIN"] = df["PF"] / df["MIN"].replace(0, np.nan)

    df = _rolling_player(df)
    df = _ewm_player(df)
    df = _lag_features(df)
    df = _fatigue_features(df)
    df = _starter_features(df)
    df = _detect_star_players(df)
    df = _quantile_model_features(df)
    return df


def prepare_season_df(df: pd.DataFrame, season_label: str | None = None) -> pd.DataFrame:
    """Feature-engineer one season and optionally tag it for holdout splits."""
    season_df = min_features(df)
    if season_label is not None:
        season_df["SEASON"] = season_label
    return season_df[(season_df["MIN"] >= 5) & (season_df["MIN"] <= 48)]


def build_min_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build the concatenated MIN-model training frame across seasons."""
    season_map = season_map or DEFAULT_SEASON_MAP
    res = [
        prepare_season_df(season_df, season_label=season_map[i])
        for i, season_df in enumerate(season_dfs)
    ]
    df = pd.concat(res, ignore_index=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


# ── Rolling / EWM (model inputs only) ────────────────────────────────────────

def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("MIN", "USG_PCT"):
        df[f"{col}_roll10"] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean().round(2))
        )
    return df


def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    for col, span in (("MIN", 10), ("SPD", 10)):
        if col not in df.columns:
            continue
        df[f"{col}_{span}_ewm"] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.shift(1).ewm(span=span, adjust=False).mean().round(2))
        )
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ("USG_PCT", "AST_PCT", "PIE")
    for lag in (1, 2):
        for col in cols:
            if col not in df.columns:
                continue
            df[f"{col}_lag{lag}"] = df.groupby("PLAYER_ID")[col].shift(lag)
    return df


def _fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-window fatigue proxies from prior games only (excludes current game)."""
    df = df.copy()
    games_7 = np.zeros(len(df), dtype=int)
    games_14 = np.zeros(len(df), dtype=int)
    min_sum_7 = np.zeros(len(df), dtype=float)

    for idx in df.groupby("PLAYER_ID").groups.values():
        pos = df.index.get_indexer(idx)
        order = np.argsort(df.loc[idx, "GAME_DATE"].to_numpy())
        pos = pos[order]
        dates = df.loc[idx, "GAME_DATE"].to_numpy(dtype="datetime64[ns]")[order]
        mins = df.loc[idx, "MIN"].to_numpy(dtype=float)[order]

        for i in range(1, len(pos)):
            d = dates[i]
            prior_dates = dates[:i]
            prior_mins = mins[:i]
            mask_7 = prior_dates >= (d - np.timedelta64(7, "D"))
            mask_14 = prior_dates >= (d - np.timedelta64(14, "D"))
            games_7[pos[i]] = mask_7.sum()
            games_14[pos[i]] = mask_14.sum()
            min_sum_7[pos[i]] = prior_mins[mask_7].sum()

    df["GAMES_PLAYED_LAST_7_DAYS"] = games_7
    df["GAMES_PLAYED_LAST_14_DAYS"] = games_14
    df["MIN_SUM_LAST_7_DAYS"] = min_sum_7.round(1)
    return df


# ── Starter / role ─────────────────────────────────────────────────────────────

def _starter_features(df: pd.DataFrame) -> pd.DataFrame:
    df["STARTER_ROLL10_PCT"] = (
        df.groupby("PLAYER_ID")["STARTING"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean().round(2))
    )
    return df


# ── MIN quantile model features (formerly inline in min_quantile_model.ipynb) ──

def _quantile_model_features(df: pd.DataFrame) -> pd.DataFrame:
    player_min = df.groupby("PLAYER_ID")["MIN"]

    df["MIN_SEASON_MEAN"] = player_min.transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["MIN_SEASON_STD"] = player_min.transform(
        lambda x: x.expanding().std().shift(1)
    )
    df["MIN_RATE_OF_CHANGE"] = (
        df.groupby("PLAYER_ID")["MIN_roll10"].transform(lambda x: x.diff(4))
    )

    df["TEAM_MIN_RANK_L10"] = (
        df.groupby(["TEAM_ID", "GAME_DATE"])["MIN_roll10"]
        .rank(ascending=False, method="dense")
    )
    df["TEAM_USG_RANK_L10"] = (
        df.groupby(["TEAM_ID", "GAME_DATE"])["USG_PCT_roll10"]
        .rank(ascending=False, method="dense")
    )

    df["MIN_STD_L10"] = player_min.transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).std()
    )
    df["MIN_P10_L10"] = player_min.transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).quantile(0.1)
    )
    df["MIN_P90_L10"] = player_min.transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).quantile(0.9)
    )

    # Includes current-game start status; safe if STARTING is known pregame.
    df["CONSEC_STARTS"] = (
        df.groupby("PLAYER_ID")["STARTING"]
        .transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        )
    )
    return df


# ── Star-player context (shared with PPM / live pipeline) ──────────────────────

def _detect_star_players(
    df,
    min_minutes: int = 10,
    min_games: int = 10,
    name_dict: dict | None = None,
    current_season: str | None = None,
    recent_n_games: int | None = None,
):
    df = df.copy()

    df["PLAYER_NAME_NORM"] = (
        df["PLAYER_NAME"].map(lambda x: name_dict.get(x, x))
        if name_dict
        else df["PLAYER_NAME"]
    )

    current_team = (
        df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME_NORM")["TEAM_ID"]
        .last()
    )
    df["CURRENT_TEAM_ID"] = df["PLAYER_NAME_NORM"].map(current_team)

    scoring_df = df.copy()
    if current_season is not None and "SEASON" in df.columns:
        scoring_df = scoring_df[scoring_df["SEASON"] == current_season]
    elif recent_n_games is not None:
        scoring_df = (
            scoring_df.sort_values("GAME_DATE")
            .groupby(["PLAYER_NAME_NORM", "CURRENT_TEAM_ID"])
            .tail(recent_n_games)
        )

    scoring_df = scoring_df[scoring_df["TEAM_ID"] == scoring_df["CURRENT_TEAM_ID"]]
    active = scoring_df[scoring_df["MIN"] >= min_minutes]

    stats_df = active.groupby(["CURRENT_TEAM_ID", "PLAYER_NAME_NORM"]).agg(
        USG_PCT=("USG_PCT", "mean"),
        TS_PCT=("TS_PCT", "mean"),
        EFG_PCT=("EFG_PCT", "mean"),
        PTS=("PTS", "mean"),
        PIE=("PIE", "mean"),
        NET_RATING=("NET_RATING", "mean"),
        GAMES=("MIN", "count"),
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

    score_cols = ["USG_PCT", "TS_PCT", "EFG_PCT", "PTS", "PIE", "NET_RATING"]
    for col in score_cols:
        stats_df[f"{col}_RANK"] = stats_df.groupby("CURRENT_TEAM_ID")[col].rank(
            pct=True, method="average", na_option="bottom"
        )

    stats_df["STAR_SCORE"] = (
        0.25 * stats_df["USG_PCT_RANK"]
        + 0.20 * stats_df["PIE_RANK"]
        + 0.20 * stats_df["PTS_RANK"]
        + 0.15 * stats_df["TS_PCT_RANK"]
        + 0.10 * stats_df["EFG_PCT_RANK"]
        + 0.10 * stats_df["NET_RATING_RANK"]
    )

    sorted_stars = stats_df.sort_values(
        ["CURRENT_TEAM_ID", "STAR_SCORE"], ascending=[True, False]
    )
    top_3 = sorted_stars.groupby("CURRENT_TEAM_ID").head(3).copy()
    top_3["STAR_RANK"] = top_3.groupby("CURRENT_TEAM_ID").cumcount() + 1

    top_star_map = top_3[top_3["STAR_RANK"] == 1].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]
    second_star_map = top_3[top_3["STAR_RANK"] == 2].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]
    third_star_map = top_3[top_3["STAR_RANK"] == 3].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]

    df["TOP_PLAYER"] = df["CURRENT_TEAM_ID"].map(top_star_map)
    df["SECOND_TOP_PLAYER"] = df["CURRENT_TEAM_ID"].map(second_star_map)
    df["THIRD_TOP_PLAYER"] = df["CURRENT_TEAM_ID"].map(third_star_map)

    top_stars_set = set(zip(top_3["CURRENT_TEAM_ID"], top_3["PLAYER_NAME_NORM"]))
    top1_set = set(zip(
        top_3[top_3["STAR_RANK"] == 1]["CURRENT_TEAM_ID"],
        top_3[top_3["STAR_RANK"] == 1]["PLAYER_NAME_NORM"],
    ))

    df["IS_TOP_STAR"] = df.apply(
        lambda r: 1 if (r["CURRENT_TEAM_ID"], r["PLAYER_NAME_NORM"]) in top_stars_set else 0,
        axis=1,
    )
    df["IS_TOP_1_STAR"] = df.apply(
        lambda r: 1 if (r["CURRENT_TEAM_ID"], r["PLAYER_NAME_NORM"]) in top1_set else 0,
        axis=1,
    )

    active_stars_per_game = (
        df[(df["IS_TOP_STAR"] == 1) & (df["MIN"] >= min_minutes)]
        .groupby(["GAME_ID", "CURRENT_TEAM_ID"])["PLAYER_NAME_NORM"]
        .nunique()
        .reset_index(name="ACTIVE_STARS_COUNT")
        .assign(ACTIVE_STARS_COUNT=lambda x: x["ACTIVE_STARS_COUNT"].clip(upper=3))
    )
    df = df.merge(active_stars_per_game, on=["GAME_ID", "CURRENT_TEAM_ID"], how="left")
    df["ACTIVE_STARS_COUNT"] = df["ACTIVE_STARS_COUNT"].fillna(0).astype(int)

    top1_active_per_game = (
        df[(df["IS_TOP_1_STAR"] == 1) & (df["MIN"] >= min_minutes)]
        .groupby(["GAME_ID", "CURRENT_TEAM_ID"])["IS_TOP_1_STAR"]
        .max()
        .reset_index(name="TOP_STAR_ACTIVE")
    )
    df = df.merge(top1_active_per_game, on=["GAME_ID", "CURRENT_TEAM_ID"], how="left")
    df["TOP_STAR_ACTIVE"] = df["TOP_STAR_ACTIVE"].fillna(0).astype(int)

    for rank, col_name in [
        (1, "TOP_PLAYER_ACTIVE"),
        (2, "SECOND_PLAYER_ACTIVE"),
        (3, "THIRD_PLAYER_ACTIVE"),
    ]:
        star_name_map = top_3[top_3["STAR_RANK"] == rank].set_index("CURRENT_TEAM_ID")["PLAYER_NAME_NORM"]
        df[f"_STAR_{rank}_NAME"] = df["CURRENT_TEAM_ID"].map(star_name_map)
        star_active = (
            df[
                (df["PLAYER_NAME_NORM"] == df[f"_STAR_{rank}_NAME"])
                & (df["MIN"] >= min_minutes)
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
