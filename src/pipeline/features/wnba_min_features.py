"""WNBA MIN quantile model feature engineering.

Separate from NBA ``min_features`` — WNBA box/advanced coverage differs
(no reliance on tracking SPD / NBA-tuned star detection by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features._common import days_rest, fatigue_features

TARGET = "MIN"

# Order must match training / saved WNBA quantile bundle `feature_names`.
WNBA_MIN_FEATURES = [
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
    "USG_PCT_lag1",
    "USG_PCT_lag2",
    "AST_PCT_lag1",
    "PIE_lag1",
    "GAMES_PLAYED_LAST_7_DAYS",
    "GAMES_PLAYED_LAST_14_DAYS",
    "MIN_SUM_LAST_7_DAYS",
    "DAYS_REST",
]

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25"}


def wnba_min_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the WNBA MIN quantile model."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns:
        if "START_POSITION" in df.columns:
            df["STARTING"] = df["START_POSITION"].notna().astype(int)
        else:
            df["STARTING"] = 0

    df = _rolling_player(df)
    df = _ewm_player(df)
    df = _lag_features(df)
    df = fatigue_features(df)
    df = days_rest(df)
    df = _starter_features(df)
    df = _quantile_model_features(df)
    return df


def prepare_season_df(df: pd.DataFrame, season_label: str | None = None) -> pd.DataFrame:
    """Feature-engineer one WNBA season and optionally tag it for holdout splits."""
    season_df = wnba_min_features(df)
    if season_label is not None:
        season_df["SEASON"] = season_label
    # WNBA regulation is 40 minutes; allow OT headroom.
    return season_df[(season_df["MIN"] >= 5) & (season_df["MIN"] <= 45)]


def build_wnba_min_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build the concatenated WNBA MIN-model training frame across seasons."""
    season_map = season_map or DEFAULT_SEASON_MAP
    res = [
        prepare_season_df(season_df, season_label=season_map[i])
        for i, season_df in enumerate(season_dfs)
    ]
    df = pd.concat(res, ignore_index=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def validate_wnba_min_dataset(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    key_cols: tuple[str, ...] = ("GAME_ID", "PLAYER_ID"),
    require_season: bool = True,
) -> dict[str, object]:
    """Assert an engineered WNBA MIN frame is training-ready."""
    features = list(feature_cols) if feature_cols is not None else list(WNBA_MIN_FEATURES)
    if df.empty:
        raise ValueError("WNBA MIN dataset is empty")

    required = list(key_cols) + [TARGET, "GAME_DATE", *features]
    if require_season:
        required.append("SEASON")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"WNBA MIN dataset missing {len(missing)} column(s): {missing[:20]}"
            + ("..." if len(missing) > 20 else "")
        )

    dup_mask = df.duplicated(subset=list(key_cols), keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        n_keys = int(df.loc[dup_mask, list(key_cols)].drop_duplicates().shape[0])
        sample = (
            df.loc[dup_mask, list(key_cols)]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"WNBA MIN dataset has {n_dup_rows:,} duplicate rows across {n_keys:,} "
            f"{key_cols} keys. Sample: {sample}"
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    n_bad_target = int((y.isna() | ~np.isfinite(y)).sum())
    if n_bad_target:
        raise ValueError(
            f"WNBA MIN dataset has {n_bad_target:,} non-finite {TARGET} values"
        )

    summary: dict[str, object] = {
        "rows": len(df),
        "cols": df.shape[1],
        "features": len(features),
        "duplicate_keys": 0,
        "bad_target": 0,
    }
    if "SEASON" in df.columns:
        summary["seasons"] = df["SEASON"].value_counts().sort_index().to_dict()
    return summary


def _rolling_player(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("MIN", "USG_PCT"):
        if col not in df.columns:
            continue
        df[f"{col}_roll10"] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean().round(2))
        )
    return df


def _ewm_player(df: pd.DataFrame) -> pd.DataFrame:
    if "MIN" not in df.columns:
        return df
    df["MIN_10_ewm"] = (
        df.groupby("PLAYER_ID")["MIN"]
        .transform(lambda x: x.shift(1).ewm(span=10, adjust=False).mean().round(2))
    )
    return df


def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Lighter lag set than NBA — AST_PCT/PIE lag2 omitted until ablation justifies them.
    specs = (
        ("USG_PCT", (1, 2)),
        ("AST_PCT", (1,)),
        ("PIE", (1,)),
    )
    for col, lags in specs:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("PLAYER_ID")[col].shift(lag)
    return df


def _starter_features(df: pd.DataFrame) -> pd.DataFrame:
    df["STARTER_ROLL10_PCT"] = (
        df.groupby("PLAYER_ID")["STARTING"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean().round(2))
    )
    return df


def _quantile_model_features(df: pd.DataFrame) -> pd.DataFrame:
    player_min = df.groupby("PLAYER_ID")["MIN"]

    df["MIN_SEASON_MEAN"] = player_min.transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["MIN_SEASON_STD"] = player_min.transform(
        lambda x: x.expanding().std().shift(1)
    )
    if "MIN_roll10" in df.columns:
        df["MIN_RATE_OF_CHANGE"] = (
            df.groupby("PLAYER_ID")["MIN_roll10"].transform(lambda x: x.diff(4))
        )
        df["TEAM_MIN_RANK_L10"] = (
            df.groupby(["TEAM_ID", "GAME_DATE"])["MIN_roll10"]
            .rank(ascending=False, method="dense")
        )
    if "USG_PCT_roll10" in df.columns:
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

    df["CONSEC_STARTS"] = (
        df.groupby("PLAYER_ID")["STARTING"]
        .transform(
            lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        )
    )
    return df
