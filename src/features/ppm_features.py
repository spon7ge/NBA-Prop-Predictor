"""PPM quantile model feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Order must match training / saved quantile bundle `feature_names`.
PPM_FEATURES = [
    "PTS_PER_MIN_season_avg",
    "PTS_PER_MIN_X_OPP_PTS_ALLOWED",
    "CFGA_PER_MIN_X_OPP_FG_PCT_ALLOWED",
    "FGA_PER_MIN_X_OPP_DEF_RATING",
    "3PA_PER_MIN_X_OPP_TEAM_FG3A_ALLOWED",
    "FTA_PER_MIN_X_OPP_FTA_ALLOWED",
    "PPM_SEASON_STD",
    "TEAM_PTS_PER_MIN_RANK_L10",
    "TEAM_USG_RANK_L10",
    "TEAM_MIN_RANK_L10",
    "PPM_P10_L10",
    "PPM_P90_L10",
]

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25", 4: "S26"}

_PER_MIN_COLS = ("PTS_PER_MIN", "FGA_PER_MIN", "FTA_PER_MIN", "3PA_PER_MIN", "CFGA_PER_MIN")
_ROLL_COLS = ("MIN", "USG_PCT", "PTS_PER_MIN")
_EWM_COLS = ("PTS_PER_MIN", "FGA_PER_MIN", "CFGA_PER_MIN", "3PA_PER_MIN", "FTA_PER_MIN")


def ppm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the PPM quantile model."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns:
        df["STARTING"] = df["START_POSITION"].notna().astype(int)

    df = _per_min_features(df)
    df = _rolling_player(df)
    df = _ewm_player(df)
    df = _season_averages(df)
    df = _team_allowed_context(df)
    df = _opponent_def_rating_roll10(df)
    df = _quantile_model_features(df)
    return df


def prepare_season_df(df: pd.DataFrame, season_label: str | None = None) -> pd.DataFrame:
    """Feature-engineer one season and optionally tag it for holdout splits."""
    season_df = ppm_features(df)
    if season_label is not None:
        season_df["SEASON"] = season_label
    min_reg = season_df["MIN"].clip(upper=48)
    return season_df[(min_reg >= 10) | (season_df["STARTING"] == 1)]


def build_ppm_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build the concatenated PPM-model training frame across seasons."""
    season_map = season_map or DEFAULT_SEASON_MAP
    res = [
        prepare_season_df(season_df, season_label=season_map[i])
        for i, season_df in enumerate(season_dfs)
    ]
    df = pd.concat(res, ignore_index=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


# ── Base rates / rolling / EWM (model inputs only) ─────────────────────────────

def _per_min_features(df: pd.DataFrame) -> pd.DataFrame:
    per_min_map = {
        "PTS_PER_MIN": ("PTS", "MIN"),
        "FGA_PER_MIN": ("FGA", "MIN"),
        "FTA_PER_MIN": ("FTA", "MIN"),
        "3PA_PER_MIN": ("FG3A", "MIN"),
        "CFGA_PER_MIN": ("CFGA", "MIN"),
    }
    for out_col, (num_col, den_col) in per_min_map.items():
        df[out_col] = df[num_col] / df[den_col].replace(0, np.nan)
    return df


def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    for col in _ROLL_COLS:
        df[f"{col}_roll10"] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean().round(2))
        )
    return df


def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    for col in _EWM_COLS:
        df[f"{col}_10_ewm"] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.shift(1).ewm(span=10, adjust=False).mean().round(2))
        )
    return df


def _season_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["PTS_PER_MIN_season_avg"] = (
        df.groupby(["PLAYER_ID", "SEASON_YEAR"])["PTS_PER_MIN"]
        .transform(lambda x: x.shift(1).expanding().mean().round(2))
    )
    return df


# ── Opponent context ───────────────────────────────────────────────────────────

def _team_allowed_context(df: pd.DataFrame) -> pd.DataFrame:
    """Team-level allowed stats (prior games only), mapped to player rows."""
    required = [
        "TEAM_ID",
        "SEASON_YEAR",
        "GAME_ID",
        "GAME_DATE",
        "OPP_FGM",
        "OPP_FGA",
        "OPP_FG3A",
        "OPP_FTA",
        "OPP_PTS",
    ]
    if any(c not in df.columns for c in required):
        return df

    team_game = (
        df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
        .sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"])
        .copy()
    )
    g = team_game.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False)

    cum_opp_fgm_prior = g["OPP_FGM"].transform(lambda x: x.cumsum().shift(1))
    cum_opp_fga_prior = g["OPP_FGA"].transform(lambda x: x.cumsum().shift(1))
    team_game["TEAM_FG%_ALLOWED"] = (
        cum_opp_fgm_prior / cum_opp_fga_prior.replace(0, np.nan)
    ).round(3)

    def prior_expanding_mean(x):
        return x.shift(1).expanding().mean().round(3)

    for raw_col, new_col in [
        ("OPP_FTA", "TEAM_FTA_ALLOWED"),
        ("OPP_FG3A", "TEAM_FG3A_ALLOWED"),
        ("OPP_PTS", "TEAM_PTS_ALLOWED"),
    ]:
        team_game[new_col] = g[raw_col].transform(prior_expanding_mean)

    allowed_cols = [
        "TEAM_ID",
        "GAME_ID",
        "TEAM_FG%_ALLOWED",
        "TEAM_FG3A_ALLOWED",
        "TEAM_FTA_ALLOWED",
        "TEAM_PTS_ALLOWED",
    ]
    allowed_map = team_game[allowed_cols]
    out = df.merge(allowed_map, on=["TEAM_ID", "GAME_ID"], how="left")

    if "OPP_TEAM_ID" not in out.columns:
        return out

    opp_rename = {
        "TEAM_ID": "OPP_TEAM_ID",
        "TEAM_FG%_ALLOWED": "OPP_TEAM_FG%_ALLOWED",
        "TEAM_FG3A_ALLOWED": "OPP_TEAM_FG3A_ALLOWED",
        "TEAM_FTA_ALLOWED": "OPP_TEAM_FTA_ALLOWED",
        "TEAM_PTS_ALLOWED": "OPP_TEAM_PTS_ALLOWED",
    }
    opp_allowed_map = allowed_map.rename(columns=opp_rename)
    return out.merge(opp_allowed_map, on=["OPP_TEAM_ID", "GAME_ID"], how="left")


def _opponent_def_rating_roll10(df: pd.DataFrame) -> pd.DataFrame:
    """Map opponent L10 defensive rating onto player rows (leakage-safe shift+roll)."""
    if "TEAM_DEF_RATING" not in df.columns:
        return df

    team_game = (
        df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
        .sort_values(["TEAM_ID", "GAME_DATE"])
        .loc[:, ["TEAM_ID", "GAME_ID", "GAME_DATE", "TEAM_DEF_RATING"]]
        .copy()
    )
    team_game["OPP_DEF_RATING_roll10"] = (
        team_game.groupby("TEAM_ID")["TEAM_DEF_RATING"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean().round(2))
    )
    opp_map = team_game[["TEAM_ID", "GAME_ID", "OPP_DEF_RATING_roll10"]].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID"}
    )

    if "OPP_TEAM_ID" in df.columns:
        return df.merge(opp_map, on=["OPP_TEAM_ID", "GAME_ID"], how="left")

    opp_abbr_col = next(
        (c for c in ("OPP_OPP_ABBREVIATION_base", "OPP_TEAM_ABBREVIATION") if c in df.columns),
        None,
    )
    if opp_abbr_col is None or "TEAM_ABBREVIATION" not in df.columns:
        return df

    abbr_map = (
        df[["GAME_ID", "TEAM_ABBREVIATION", "TEAM_DEF_RATING", "GAME_DATE"]]
        .drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION"])
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    )
    abbr_map["OPP_DEF_RATING_roll10"] = (
        abbr_map.groupby("TEAM_ABBREVIATION")["TEAM_DEF_RATING"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean().round(2))
    )
    abbr_map = abbr_map[["GAME_ID", "TEAM_ABBREVIATION", "OPP_DEF_RATING_roll10"]].rename(
        columns={"TEAM_ABBREVIATION": opp_abbr_col}
    )
    if "OPP_DEF_RATING_roll10" in df.columns:
        df = df.drop(columns=["OPP_DEF_RATING_roll10"])
    return df.merge(abbr_map, on=["GAME_ID", opp_abbr_col], how="left")


# ── PPM quantile model features (formerly inline in ppm_quantile_model.ipynb) ──

def _quantile_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TEAM_PTS_PER_MIN_RANK_L10"] = (
        df.groupby(["TEAM_ID", "GAME_DATE"])["PTS_PER_MIN_roll10"]
        .rank(ascending=False, method="dense")
    )
    df["TEAM_USG_RANK_L10"] = (
        df.groupby(["TEAM_ID", "GAME_DATE"])["USG_PCT_roll10"]
        .rank(ascending=False, method="dense")
    )
    df["TEAM_MIN_RANK_L10"] = (
        df.groupby(["TEAM_ID", "GAME_DATE"])["MIN_roll10"]
        .rank(ascending=False, method="dense")
    )

    df["PTS_PER_MIN_X_OPP_PTS_ALLOWED"] = (
        df["PTS_PER_MIN_10_ewm"] * df["OPP_TEAM_PTS_ALLOWED"]
    )
    df["CFGA_PER_MIN_X_OPP_FG_PCT_ALLOWED"] = (
        df["CFGA_PER_MIN_10_ewm"] * df["OPP_TEAM_FG%_ALLOWED"]
    )
    df["FGA_PER_MIN_X_OPP_DEF_RATING"] = (
        df["FGA_PER_MIN_10_ewm"] * df["OPP_DEF_RATING_roll10"]
    )
    df["3PA_PER_MIN_X_OPP_TEAM_FG3A_ALLOWED"] = (
        df["3PA_PER_MIN_10_ewm"] * df["OPP_TEAM_FG3A_ALLOWED"]
    )
    df["FTA_PER_MIN_X_OPP_FTA_ALLOWED"] = (
        df["FTA_PER_MIN_10_ewm"] * df["OPP_TEAM_FTA_ALLOWED"]
    )

    player_ppm = df.groupby("PLAYER_ID")["PTS_PER_MIN"]
    df["PPM_SEASON_STD"] = player_ppm.transform(
        lambda x: x.expanding().std().shift(1)
    )
    df["PPM_P10_L10"] = player_ppm.transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).quantile(0.1)
    )
    df["PPM_P90_L10"] = player_ppm.transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).quantile(0.9)
    )
    return df
