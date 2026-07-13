"""RPM quantile model feature engineering — TOP-14 feature subset.

Combines:
* New rebound-specific features (season avg / EWM / quantiles / slope / ratio)
* Survivors from SHAP + permutation + ablation (``keep=True`` only)

Dropped ``keep=False`` from the prior 18-feature set:
``player_REB_PER_MIN_*`` rate windows, ``player_REB_PCT_*``,
``player_OREB/DREB_PCT_*``, ``RBC/DRBC_TREND_5v20``, ``DAYS_REST``,
``STARTING_rate_last10``.

Leakage-safe: same-game box-score columns are only used as lagged rolling /
expanding averages. Exposes the gold / ml API (`RPM_FEATURES`, `rpm_features`,
`prepare_season_df`, `build_rpm_dataset`) plus `build_features` for notebook tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET = "REB_PER_MIN"

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25", 4: "S26"}

ROLLING_WINDOWS = [5, 10]
TEAM_ROLLING_WINDOWS = [10]

# Rolling inputs for keep=True survivors + helpers used by new features.
PLAYER_ROLL_STATS = [
    "REB_PER_MIN",
    "ORBC",
    "DRBC",
    "MIN",
]
PLAYER_SEASON_STATS = [
    "MIN",
]

TEAM_STATS_TO_ROLL: list[str] = []
OPP_REBOUNDING_STATS_TO_ROLL = ["TEAM_REB_PCT"]

RPM_FEATURES = [
    # New rebound features
    "REB_PER_MIN_season_avg",
    "REB_PER_MIN_10_ewm",
    "OREB_DREB_RATIO",
    "POSITION_ENC",
    "RBC_PER_MIN_10_ewm",
    "RPM_SEASON_STD",
    "REB_ROLL10_SLOPE",
    "REB_PER_MIN_P10_L10",
    "REB_PER_MIN_P90_L10",
    # keep=True survivors from SHAP / perm / ablation
    "player_ORBC_roll10_mean",
    "player_DRBC_roll10_mean",
    "player_MIN_roll5_mean",
    "player_MIN_season_mean",
    "opp_team_TEAM_REB_PCT_roll10_mean",
]

assert len(RPM_FEATURES) == 14, f"expected 14 features, got {len(RPM_FEATURES)}"


def rpm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the RPM model."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns and "START_POSITION" in df.columns:
        df["STARTING"] = df["START_POSITION"].notna().astype(int)

    if TARGET not in df.columns and {"REB", "MIN"}.issubset(df.columns):
        df[TARGET] = df["REB"] / df["MIN"].replace(0, np.nan)

    if "RBC" in df.columns and "MIN" in df.columns:
        df["RBC_PER_MIN"] = df["RBC"] / df["MIN"].replace(0, np.nan)

    df = _encode_categoricals(df)
    df = _add_rolling_and_expanding(
        df, group_col="PLAYER_ID", stat_cols=PLAYER_ROLL_STATS,
        windows=ROLLING_WINDOWS, prefix="player",
    )
    if "SEASON_YEAR" in df.columns:
        df = _add_season_to_date(
            df, group_cols=["PLAYER_ID", "SEASON_YEAR"],
            stat_cols=PLAYER_SEASON_STATS, prefix="player",
        )
    else:
        # Fallback when SEASON_YEAR is absent: career expanding mean as season proxy.
        df = _add_season_to_date(
            df, group_cols=["PLAYER_ID"],
            stat_cols=PLAYER_SEASON_STATS, prefix="player",
        )

    df = _add_rpm_quantile_features(df)
    df = _build_team_context(df)

    for col in RPM_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


def prepare_season_df(df: pd.DataFrame, season_label: str | None = None) -> pd.DataFrame:
    """Feature-engineer one season and optionally tag it for holdout splits."""
    season_df = rpm_features(df)
    if season_label is not None:
        season_df["SEASON"] = season_label
    if "STARTING" not in season_df.columns:
        season_df["STARTING"] = 0
    min_reg = season_df["MIN"].clip(upper=48)
    return season_df[(min_reg >= 10) | (season_df["STARTING"] == 1)]


def build_rpm_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build the concatenated RPM-model training frame across seasons."""
    season_map = season_map or DEFAULT_SEASON_MAP
    res = [
        prepare_season_df(season_df, season_label=season_map[i])
        for i, season_df in enumerate(season_dfs)
    ]
    df = pd.concat(res, ignore_index=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def validate_rpm_dataset(
    df: pd.DataFrame,
    *,
    key_cols: tuple[str, ...] = ("GAME_ID", "PLAYER_ID"),
    require_season: bool = True,
) -> dict[str, object]:
    """Assert an engineered RPM frame is training-ready.

    Checks required columns, no duplicate ``(GAME_ID, PLAYER_ID)`` rows, and
    finite target values.
    """
    if df.empty:
        raise ValueError("RPM dataset is empty")

    required = list(key_cols) + [TARGET, "MIN", "GAME_DATE", *RPM_FEATURES]
    if require_season:
        required.append("SEASON")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"RPM dataset missing {len(missing)} column(s): {missing[:20]}"
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
            f"RPM dataset has {n_dup_rows:,} duplicate rows across {n_keys:,} "
            f"{key_cols} keys. Sample: {sample}"
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    n_bad_target = int((y.isna() | ~np.isfinite(y)).sum())
    if n_bad_target:
        raise ValueError(
            f"RPM dataset has {n_bad_target:,} non-finite {TARGET} values"
        )

    summary: dict[str, object] = {
        "rows": len(df),
        "cols": df.shape[1],
        "features": len(RPM_FEATURES),
        "duplicate_keys": 0,
        "bad_target": 0,
    }
    if "SEASON" in df.columns:
        summary["seasons"] = df["SEASON"].value_counts().sort_index().to_dict()
    return summary


def feature_cols_from_df(df: pd.DataFrame) -> list[str]:
    """The fixed top-14 feature columns present on an engineered frame."""
    return [c for c in RPM_FEATURES if c in df.columns]


def build_features(df: pd.DataFrame, drop_first_n_games: int = 0):
    """Notebook helper: return ``(X, y, feature_cols)`` ready for XGBoost."""
    df = rpm_features(df)

    if drop_first_n_games > 0:
        if "SEASON_YEAR" in df.columns:
            df = df.sort_values(["PLAYER_ID", "SEASON_YEAR", "GAME_DATE"])
            games_played = df.groupby(["PLAYER_ID", "SEASON_YEAR"]).cumcount()
        else:
            df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
            games_played = df.groupby("PLAYER_ID").cumcount()
        df = df[games_played >= drop_first_n_games]

    feature_cols = feature_cols_from_df(df)
    valid = df[TARGET].notna() & np.isfinite(df[TARGET])
    df = df[valid]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y, feature_cols


# ── Internal helpers ───────────────────────────────────────────────────────────

def _add_rolling_and_expanding(df, group_col, stat_cols, windows, prefix):
    df = df.sort_values([group_col, "GAME_DATE"])
    g = df.groupby(group_col, sort=False)

    for col in stat_cols:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        shifted_grouped = shifted.groupby(df[group_col])

        for w in windows:
            df[f"{prefix}_{col}_roll{w}_mean"] = shifted_grouped.transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            df[f"{prefix}_{col}_roll{w}_std"] = shifted_grouped.transform(
                lambda s, w=w: s.rolling(w, min_periods=2).std()
            )

        df[f"{prefix}_{col}_ewma"] = shifted_grouped.transform(
            lambda s: s.ewm(span=max(windows), min_periods=1).mean()
        )
        df[f"{prefix}_{col}_expanding_mean"] = shifted_grouped.transform(
            lambda s: s.expanding(min_periods=1).mean()
        )

    return df


def _add_season_to_date(df, group_cols, stat_cols, prefix):
    df = df.sort_values(group_cols + ["GAME_DATE"])
    g = df.groupby(group_cols, sort=False)
    for col in stat_cols:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        shifted_grouped = shifted.groupby([df[c] for c in group_cols])
        df[f"{prefix}_{col}_season_mean"] = shifted_grouped.transform(
            lambda s: s.expanding(min_periods=1).mean()
        )
    return df


def _rolling_slope(series: pd.Series, window: int = 10, min_periods: int = 5) -> pd.Series:
    """Leakage-safe linear slope over the prior ``window`` observations."""

    def _slope(y: np.ndarray) -> float:
        mask = np.isfinite(y)
        n = int(mask.sum())
        if n < min_periods:
            return np.nan
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x[mask], y[mask], 1)[0])

    return series.shift(1).rolling(window, min_periods=min_periods).apply(_slope, raw=True)


def _add_rpm_quantile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the new named RPM features (leakage-safe)."""
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    g_rpm = df.groupby("PLAYER_ID", sort=False)[TARGET]
    shifted = g_rpm.shift(1)
    shifted_g = shifted.groupby(df["PLAYER_ID"])

    # Season-to-date mean / std of REB_PER_MIN
    if "SEASON_YEAR" in df.columns:
        g_season = df.groupby(["PLAYER_ID", "SEASON_YEAR"], sort=False)[TARGET]
        season_shifted = g_season.shift(1)
        season_shifted_g = season_shifted.groupby([df["PLAYER_ID"], df["SEASON_YEAR"]])
        df["REB_PER_MIN_season_avg"] = season_shifted_g.transform(
            lambda s: s.expanding(min_periods=1).mean()
        )
        df["RPM_SEASON_STD"] = season_shifted_g.transform(
            lambda s: s.expanding(min_periods=2).std()
        )
    else:
        df["REB_PER_MIN_season_avg"] = shifted_g.transform(
            lambda s: s.expanding(min_periods=1).mean()
        )
        df["RPM_SEASON_STD"] = shifted_g.transform(
            lambda s: s.expanding(min_periods=2).std()
        )

    df["REB_PER_MIN_10_ewm"] = shifted_g.transform(
        lambda s: s.ewm(span=10, min_periods=1).mean()
    )
    df["REB_PER_MIN_P10_L10"] = shifted_g.transform(
        lambda s: s.rolling(10, min_periods=5).quantile(0.1)
    )
    df["REB_PER_MIN_P90_L10"] = shifted_g.transform(
        lambda s: s.rolling(10, min_periods=5).quantile(0.9)
    )
    df["REB_ROLL10_SLOPE"] = df.groupby("PLAYER_ID", sort=False)[TARGET].transform(
        _rolling_slope
    )

    # OREB / DREB ratio — lagged season average (pregame-safe)
    if {"OREB", "DREB"}.issubset(df.columns):
        ratio = df["OREB"] / df["DREB"].replace(0, np.nan)
        if "SEASON_YEAR" in df.columns:
            ratio_shifted = ratio.groupby([df["PLAYER_ID"], df["SEASON_YEAR"]], sort=False).shift(1)
            df["OREB_DREB_RATIO"] = ratio_shifted.groupby(
                [df["PLAYER_ID"], df["SEASON_YEAR"]], sort=False
            ).transform(lambda s: s.expanding(min_periods=1).mean())
        else:
            ratio_shifted = ratio.groupby(df["PLAYER_ID"], sort=False).shift(1)
            df["OREB_DREB_RATIO"] = ratio_shifted.groupby(
                df["PLAYER_ID"], sort=False
            ).transform(lambda s: s.expanding(min_periods=1).mean())
    else:
        df["OREB_DREB_RATIO"] = np.nan

    # RBC per minute EWM
    if "RBC_PER_MIN" in df.columns:
        rbc_shifted = df.groupby("PLAYER_ID", sort=False)["RBC_PER_MIN"].shift(1)
        df["RBC_PER_MIN_10_ewm"] = rbc_shifted.groupby(df["PLAYER_ID"], sort=False).transform(
            lambda s: s.ewm(span=10, min_periods=1).mean()
        )
    else:
        df["RBC_PER_MIN_10_ewm"] = np.nan

    return df


def _build_team_context(df):
    if not {"TEAM_ID", "GAME_ID", "GAME_DATE", "OPP_TEAM_ID"}.issubset(df.columns):
        return df

    stat_pool = list(dict.fromkeys(TEAM_STATS_TO_ROLL + OPP_REBOUNDING_STATS_TO_ROLL))
    team_cols = ["TEAM_ID", "GAME_ID", "GAME_DATE"] + [c for c in stat_pool if c in df.columns]
    if len(team_cols) <= 3:
        return df
    team_df = df[team_cols].drop_duplicates(subset=["TEAM_ID", "GAME_ID"]).copy()

    stat_union = sorted(set(stat_pool) & set(team_df.columns))
    if not stat_union:
        return df

    team_df = _add_rolling_and_expanding(
        team_df, group_col="TEAM_ID", stat_cols=stat_union,
        windows=TEAM_ROLLING_WINDOWS, prefix="team",
    )

    opp_feat_cols = [
        c for c in team_df.columns
        if c.startswith("team_") and any(s in c for s in OPP_REBOUNDING_STATS_TO_ROLL)
    ]
    opp_merge = team_df[["TEAM_ID", "GAME_ID"] + opp_feat_cols].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", **{c: f"opp_{c}" for c in opp_feat_cols}}
    )
    df = df.merge(opp_merge, on=["OPP_TEAM_ID", "GAME_ID"], how="left")

    return df


def _encode_categoricals(df):
    pos_col = "POS" if "POS" in df.columns else "pos" if "pos" in df.columns else None
    if pos_col is not None and "POSITION_ENC" not in df.columns:
        le = LabelEncoder()
        df["POSITION_ENC"] = le.fit_transform(df[pos_col].astype(str))
    # Back-compat alias if older frames still expose POSITION_ENCODED only.
    if "POSITION_ENC" not in df.columns and "POSITION_ENCODED" in df.columns:
        df["POSITION_ENC"] = df["POSITION_ENCODED"]
    return df


if __name__ == "__main__":
    df = pd.read_csv("your_data.csv")
    X, y, feature_cols = build_features(df)
    print(f"Built {len(feature_cols)} features on {len(X)} rows.")
    print(X.head())
