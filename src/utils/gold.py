"""Gold layer: model-ready feature frames derived from silver game logs."""

from __future__ import annotations

import pandas as pd

from src.features.min_features import MIN_FEATURES, build_min_dataset, prepare_season_df
from src.features.ppm_features import (
    PPM_FEATURES,
    build_ppm_dataset,
    prepare_season_df as prepare_ppm_season_df,
)

# Final MIN-model columns (features + target/metadata for training).
GOLD_MIN_MODEL_COLUMNS = MIN_FEATURES + ["MIN", "GAME_DATE", "PLAYER_NAME", "STARTING"]

# Final PPM-model columns (features + target/metadata for training).
GOLD_PPM_MODEL_COLUMNS = PPM_FEATURES + [
    "PTS_PER_MIN",
    "MIN",
    "GAME_DATE",
    "PLAYER_NAME",
    "STARTING",
]

_GOLD_KEYS = ("GAME_ID", "PLAYER_ID")
_GOLD_OPTIONAL = ("SEASON",)


def prepare_gold_min_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize the MIN-model gold upload frame."""
    required = list(_GOLD_KEYS) + GOLD_MIN_MODEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gold MIN frame missing columns: {missing}")

    keep = list(_GOLD_KEYS) + [
        c for c in _GOLD_OPTIONAL if c in df.columns
    ] + GOLD_MIN_MODEL_COLUMNS
    out = df[keep].copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date
    return out


def prepare_gold_ppm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize the PPM-model gold upload frame."""
    required = list(_GOLD_KEYS) + GOLD_PPM_MODEL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gold PPM frame missing columns: {missing}")

    keep = list(_GOLD_KEYS) + [
        c for c in _GOLD_OPTIONAL if c in df.columns
    ] + GOLD_PPM_MODEL_COLUMNS
    out = df[keep].copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date
    return out


def build_min_gold_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Feature-engineer seasons and return the gold MIN-model upload frame."""
    return prepare_gold_min_df(build_min_dataset(season_dfs, season_map=season_map))


def build_ppm_gold_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Feature-engineer seasons and return the gold PPM-model upload frame."""
    return prepare_gold_ppm_df(build_ppm_dataset(season_dfs, season_map=season_map))


def build_min_gold_from_silver(
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
) -> pd.DataFrame:
    """Build gold MIN features for one silver slice (season processed in isolation)."""
    from src.utils.db import read_df

    df = read_df(
        "player_gamelogs",
        schema="silver",
        where="season_year = %(season)s AND season_type = %(season_type)s",
        params={"season": season_year, "season_type": season_type},
    )
    if df.empty:
        return pd.DataFrame(
            columns=list(_GOLD_KEYS) + list(_GOLD_OPTIONAL) + GOLD_MIN_MODEL_COLUMNS
        )

    df.columns = [c.upper() for c in df.columns]
    label = season_label or season_year
    return prepare_gold_min_df(prepare_season_df(df, season_label=label))


def build_ppm_gold_from_silver(
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
) -> pd.DataFrame:
    """Build gold PPM features for one silver slice (season processed in isolation)."""
    from src.utils.db import read_df

    df = read_df(
        "player_gamelogs",
        schema="silver",
        where="season_year = %(season)s AND season_type = %(season_type)s",
        params={"season": season_year, "season_type": season_type},
    )
    if df.empty:
        return pd.DataFrame(
            columns=list(_GOLD_KEYS) + list(_GOLD_OPTIONAL) + GOLD_PPM_MODEL_COLUMNS
        )

    df.columns = [c.upper() for c in df.columns]
    label = season_label or season_year
    return prepare_gold_ppm_df(prepare_ppm_season_df(df, season_label=label))
