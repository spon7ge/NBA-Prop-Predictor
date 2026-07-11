"""Gold layer: model-ready feature frames derived from silver game logs.

One gold table per prop × league (mirrors silver naming)::

    gold.nba_player_min_model / gold.wnba_player_min_model
    gold.nba_player_ppm_model / gold.wnba_player_ppm_model
    ...
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.pipeline.features.apm_features import (
    APM_FEATURES,
    build_apm_dataset,
    prepare_season_df as prepare_apm_season_df,
)
from src.pipeline.features.min_features import (
    MIN_FEATURES,
    build_min_dataset,
    prepare_season_df as prepare_nba_min_season_df,
)
from src.pipeline.features.ppm_features import (
    PPM_FEATURES,
    build_ppm_dataset,
    prepare_season_df as prepare_ppm_season_df,
)
from src.pipeline.features.rpm_features import (
    RPM_FEATURES,
    build_rpm_dataset,
    prepare_season_df as prepare_rpm_season_df,
)
from src.pipeline.features.wnba_min_features import (
    WNBA_MIN_FEATURES,
    build_wnba_min_dataset,
    prepare_season_df as prepare_wnba_min_season_df,
)

LeagueKey = Literal["nba", "wnba"]

_GOLD_KEYS = ("GAME_ID", "PLAYER_ID")
_GOLD_OPTIONAL = ("SEASON",)

_SILVER_TABLE: dict[LeagueKey, str] = {
    "nba": "nba_player_gamelogs",
    "wnba": "wnba_player_gamelogs",
}

_META = ["MIN", "GAME_DATE", "PLAYER_NAME", "STARTING"]

# Features + target + shared metadata for each prop's gold upload frame.
GOLD_NBA_MIN_MODEL_COLUMNS = MIN_FEATURES + _META
GOLD_WNBA_MIN_MODEL_COLUMNS = WNBA_MIN_FEATURES + _META
GOLD_PPM_MODEL_COLUMNS = PPM_FEATURES + ["PTS_PER_MIN", *_META]
GOLD_APM_MODEL_COLUMNS = APM_FEATURES + ["AST_PER_MIN", *_META]
GOLD_RPM_MODEL_COLUMNS = RPM_FEATURES + ["REB_PER_MIN", *_META]

# Back-compat alias (NBA).
GOLD_MIN_MODEL_COLUMNS = GOLD_NBA_MIN_MODEL_COLUMNS

GOLD_PROP_COLUMNS: dict[LeagueKey, dict[str, list[str]]] = {
    "nba": {
        "min": GOLD_NBA_MIN_MODEL_COLUMNS,
        "ppm": GOLD_PPM_MODEL_COLUMNS,
        "apm": GOLD_APM_MODEL_COLUMNS,
        "rpm": GOLD_RPM_MODEL_COLUMNS,
    },
    "wnba": {
        "min": GOLD_WNBA_MIN_MODEL_COLUMNS,
        # PPM/APM/RPM still share NBA feature modules until wnba_*_features exist.
        "ppm": GOLD_PPM_MODEL_COLUMNS,
        "apm": GOLD_APM_MODEL_COLUMNS,
        "rpm": GOLD_RPM_MODEL_COLUMNS,
    },
}

_GOLD_PROP_SUFFIX = {
    "min": "player_min_model",
    "ppm": "player_ppm_model",
    "apm": "player_apm_model",
    "rpm": "player_rpm_model",
}

GOLD_PROP_TABLES: dict[LeagueKey, dict[str, str]] = {
    league: {prop: f"{league}_{suffix}" for prop, suffix in _GOLD_PROP_SUFFIX.items()}
    for league in ("nba", "wnba")
}


def gold_table(prop: str, league: LeagueKey = "nba") -> str:
    """Return ``gold`` table name for a prop × league."""
    prop = prop.lower()
    if league not in GOLD_PROP_TABLES:
        raise ValueError(f"Unknown league {league!r}; expected one of {sorted(GOLD_PROP_TABLES)}")
    if prop not in GOLD_PROP_TABLES[league]:
        raise ValueError(f"Unknown prop {prop!r}; expected one of {sorted(_GOLD_PROP_SUFFIX)}")
    return GOLD_PROP_TABLES[league][prop]


def gold_columns(prop: str, league: LeagueKey = "nba") -> list[str]:
    """Return gold upload columns for a prop × league."""
    prop = prop.lower()
    if league not in GOLD_PROP_COLUMNS:
        raise ValueError(f"Unknown league {league!r}; expected one of {sorted(GOLD_PROP_COLUMNS)}")
    if prop not in GOLD_PROP_COLUMNS[league]:
        raise ValueError(f"Unknown prop {prop!r}; expected one of {sorted(GOLD_PROP_COLUMNS[league])}")
    return GOLD_PROP_COLUMNS[league][prop]


def prepare_gold_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select and normalize a prop gold upload frame."""
    required = list(_GOLD_KEYS) + columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gold frame missing columns: {missing}")

    keep = list(_GOLD_KEYS) + [c for c in _GOLD_OPTIONAL if c in df.columns] + columns
    out = df[keep].copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date
    return out


def prepare_gold_min_df(df: pd.DataFrame, *, league: LeagueKey = "nba") -> pd.DataFrame:
    return prepare_gold_df(df, gold_columns("min", league=league))


def prepare_gold_ppm_df(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_gold_df(df, GOLD_PPM_MODEL_COLUMNS)


def prepare_gold_apm_df(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_gold_df(df, GOLD_APM_MODEL_COLUMNS)


def prepare_gold_rpm_df(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_gold_df(df, GOLD_RPM_MODEL_COLUMNS)


def build_min_gold_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
    *,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    if league == "wnba":
        return prepare_gold_min_df(
            build_wnba_min_dataset(season_dfs, season_map=season_map),
            league="wnba",
        )
    return prepare_gold_min_df(
        build_min_dataset(season_dfs, season_map=season_map),
        league="nba",
    )


def build_ppm_gold_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    return prepare_gold_ppm_df(build_ppm_dataset(season_dfs, season_map=season_map))


def build_apm_gold_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    return prepare_gold_apm_df(build_apm_dataset(season_dfs, season_map=season_map))


def build_rpm_gold_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    return prepare_gold_rpm_df(build_rpm_dataset(season_dfs, season_map=season_map))


def _empty_gold(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(_GOLD_KEYS) + list(_GOLD_OPTIONAL) + columns)


def _read_silver_slice(
    season_year: str,
    season_type: str,
    *,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    from src.utils.db import read_df

    if league not in _SILVER_TABLE:
        raise ValueError(f"Unknown league {league!r}; expected one of {sorted(_SILVER_TABLE)}")

    df = read_df(
        _SILVER_TABLE[league],
        schema="silver",
        where="season_year = %(season)s AND season_type = %(season_type)s",
        params={"season": season_year, "season_type": season_type},
    )
    if df.empty:
        return df
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    return df


def build_min_gold_from_silver(
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    cols = gold_columns("min", league=league)
    df = _read_silver_slice(season_year, season_type, league=league)
    if df.empty:
        return _empty_gold(cols)
    label = season_label or season_year
    if league == "wnba":
        return prepare_gold_min_df(
            prepare_wnba_min_season_df(df, season_label=label),
            league="wnba",
        )
    return prepare_gold_min_df(
        prepare_nba_min_season_df(df, season_label=label),
        league="nba",
    )


def build_ppm_gold_from_silver(
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    df = _read_silver_slice(season_year, season_type, league=league)
    if df.empty:
        return _empty_gold(GOLD_PPM_MODEL_COLUMNS)
    label = season_label or season_year
    return prepare_gold_ppm_df(prepare_ppm_season_df(df, season_label=label))


def build_apm_gold_from_silver(
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    df = _read_silver_slice(season_year, season_type, league=league)
    if df.empty:
        return _empty_gold(GOLD_APM_MODEL_COLUMNS)
    label = season_label or season_year
    return prepare_gold_apm_df(prepare_apm_season_df(df, season_label=label))


def build_rpm_gold_from_silver(
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    df = _read_silver_slice(season_year, season_type, league=league)
    if df.empty:
        return _empty_gold(GOLD_RPM_MODEL_COLUMNS)
    label = season_label or season_year
    return prepare_gold_rpm_df(prepare_rpm_season_df(df, season_label=label))


def build_gold_from_silver(
    prop: str,
    season_year: str,
    season_type: str,
    *,
    season_label: str | None = None,
    league: LeagueKey = "nba",
) -> pd.DataFrame:
    """Dispatch silver → gold for one prop category and league."""
    builders = {
        "min": build_min_gold_from_silver,
        "ppm": build_ppm_gold_from_silver,
        "apm": build_apm_gold_from_silver,
        "rpm": build_rpm_gold_from_silver,
    }
    prop = prop.lower()
    if prop not in builders:
        raise ValueError(f"Unknown prop {prop!r}; expected one of {sorted(builders)}")
    return builders[prop](
        season_year,
        season_type,
        season_label=season_label,
        league=league,
    )
