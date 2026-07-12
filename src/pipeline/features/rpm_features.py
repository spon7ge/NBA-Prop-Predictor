"""RPM quantile model feature engineering — TOP-22 feature subset (SHAP/perm/ablation-validated).

Trimmed from the domain-reasoned top-25 down to 22 features that survived
empirical validation: SHAP importance, permutation importance (with
p-value), and leave-one-out ablation delta all had to agree a feature earns
its place (see `keep` column of the validation run this was built from).

Notably `player_REB_PER_MIN_ewma` ranked #1 on SHAP/permutation AND had a
strongly positive ablation delta — unlike the analogous PTS_PER_MIN_ewma
feature in the PPM model, which was dropped for being redundant. Rebounding
rate benefits from the recency emphasis of an EWMA in a way scoring rate
didn't; conclusions don't transfer across targets.

Dropped this round:
- `own_team_TEAM_PACE_roll10_mean` — negative ablation delta. Once opponent
  rebounding strength and the player's own rate/chances are in the model,
  pace added nothing.
- `MIN_TREND_5v20` — negative ablation delta, even though the rebound-chance
  trends (`RBC_TREND_5v20`, `DRBC_TREND_5v20`) both survived.
- `opp_team_TEAM_FG_PCT_roll10_mean` — p=0.34, statistically indistinguishable
  from noise.

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

# Windows: 5/10 for feature means, 20 kept only as the "slow" side of the
# MIN / RBC / DRBC 5-vs-20 role-trend deltas.
ROLLING_WINDOWS = [5, 10, 20]
TEAM_ROLLING_WINDOWS = [10]

# Stats needing rolling/ewma/expanding windows (roll5/roll10/roll20/expanding).
PLAYER_ROLL_STATS = [
    "REB_PER_MIN",
    "REB_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "RBC",
    "ORBC",
    "DRBC",
    "MIN",
]

# Stats needing only a season-to-date expanding mean.
PLAYER_SEASON_STATS = [
    "REB_PER_MIN",
    "REB_PCT",
    "DREB_PCT",
    "MIN",
]

# Own-team context: TEAM_PACE failed validation (negative ablation delta),
# so no own-team feature survives. Opponent context: only their rebound
# percentage survived — TEAM_FG_PCT (opponent miss rate) was noise (p=0.34).
TEAM_STATS_TO_ROLL: list[str] = []
OPP_REBOUNDING_STATS_TO_ROLL = ["TEAM_REB_PCT"]

# Static features: schedule, role, and physical/position proxy, plus the
# rebound-chance trends that survived. MIN_TREND_5v20 dropped (negative
# ablation delta) even though the RBC/DRBC trends both held up.
_STATIC_FEATURES = [
    "DAYS_REST",
    "STARTING_rate_last10",
    "POSITION_ENCODED",
    "RBC_TREND_5v20",
    "DRBC_TREND_5v20",
]

# The 22 features that survived SHAP + permutation importance + ablation
# validation (all keep=True), hardcoded so trimming the pipeline above can
# never silently change what gets exposed.
RPM_FEATURES = [
    # Recent rebounding rate (5)
    "player_REB_PER_MIN_ewma",
    "player_REB_PER_MIN_expanding_mean",
    "player_REB_PER_MIN_season_mean",
    "player_REB_PER_MIN_roll5_mean",
    "player_REB_PER_MIN_roll10_mean",
    # Rebound share / split (5)
    "player_REB_PCT_season_mean",
    "player_REB_PCT_roll10_mean",
    "player_OREB_PCT_roll10_mean",
    "player_DREB_PCT_season_mean",
    "player_DREB_PCT_roll10_mean",
    # Rebound chances (3)
    "player_ORBC_roll10_mean",
    "player_RBC_roll10_mean",
    "player_DRBC_roll10_mean",
    # Minutes (3) — MIN_TREND_5v20 dropped, levels survived
    "player_MIN_roll10_mean",
    "player_MIN_roll5_mean",
    "player_MIN_season_mean",
    # Rebound-chance trend (2)
    "RBC_TREND_5v20",
    "DRBC_TREND_5v20",
    # Physical / role proxy (2)
    "POSITION_ENCODED",
    "STARTING_rate_last10",
    # Matchup / opportunity (1) — TEAM_FG_PCT and TEAM_PACE dropped
    "opp_team_TEAM_REB_PCT_roll10_mean",
    # Schedule (1)
    "DAYS_REST",
]

assert len(RPM_FEATURES) == 22, f"expected 22 features, got {len(RPM_FEATURES)}"


def _rolling_feature_names(stat_cols: list[str], windows: list[int], prefix: str) -> list[str]:
    names: list[str] = []
    for col in stat_cols:
        for w in windows:
            names.append(f"{prefix}_{col}_roll{w}_mean")
            names.append(f"{prefix}_{col}_roll{w}_std")
        names.append(f"{prefix}_{col}_ewma")
        names.append(f"{prefix}_{col}_expanding_mean")
    return names


def rpm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the RPM model (top-22 subset)."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns and "START_POSITION" in df.columns:
        df["STARTING"] = df["START_POSITION"].notna().astype(int)

    if TARGET not in df.columns and {"REB", "MIN"}.issubset(df.columns):
        df[TARGET] = df["REB"] / df["MIN"].replace(0, np.nan)

    df = _encode_categoricals(df)
    df = _add_rest_and_schedule(df)
    if "STARTING" in df.columns:
        df = _add_starting_rate(df)

    df = _add_rolling_and_expanding(
        df, group_col="PLAYER_ID", stat_cols=PLAYER_ROLL_STATS,
        windows=ROLLING_WINDOWS, prefix="player",
    )
    if "SEASON_YEAR" in df.columns:
        df = _add_season_to_date(
            df, group_cols=["PLAYER_ID", "SEASON_YEAR"],
            stat_cols=PLAYER_SEASON_STATS, prefix="player",
        )
    df = _add_minutes_and_role_trend(df)
    df = _build_team_context(df)

    # Guarantee declared model columns exist for gold / ml alignment.
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


def feature_cols_from_df(df: pd.DataFrame) -> list[str]:
    """The fixed top-22 feature columns present on an engineered frame."""
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


def _add_rest_and_schedule(df):
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["DAYS_REST"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
    median_rest = df["DAYS_REST"].median()
    if pd.isna(median_rest):
        median_rest = 3
    df["DAYS_REST"] = df["DAYS_REST"].fillna(median_rest)
    df["IS_BACK_TO_BACK"] = (df["DAYS_REST"] <= 1).astype(int)
    return df


def _add_starting_rate(df):
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    g = df.groupby("PLAYER_ID", sort=False)
    shifted = g["STARTING"].shift(1)
    shifted_grouped = shifted.groupby(df["PLAYER_ID"])
    df["STARTING_rate_last10"] = shifted_grouped.transform(
        lambda s: s.rolling(10, min_periods=1).mean()
    )
    return df


def _add_minutes_and_role_trend(df):
    pairs = [
        ("player_RBC_roll5_mean", "player_RBC_roll20_mean", "RBC_TREND_5v20"),
        ("player_DRBC_roll5_mean", "player_DRBC_roll20_mean", "DRBC_TREND_5v20"),
    ]
    for c5, c20, out in pairs:
        if c5 in df.columns and c20 in df.columns:
            df[out] = df[c5] - df[c20]
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

    own_feat_cols = [
        c for c in team_df.columns
        if c.startswith("team_") and any(s in c for s in TEAM_STATS_TO_ROLL)
    ]
    own_merge = team_df[["TEAM_ID", "GAME_ID"] + own_feat_cols].rename(
        columns={c: f"own_{c}" for c in own_feat_cols}
    )
    df = df.merge(own_merge, on=["TEAM_ID", "GAME_ID"], how="left")

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
    if pos_col is not None and "POSITION_ENCODED" not in df.columns:
        le = LabelEncoder()
        df["POSITION_ENCODED"] = le.fit_transform(df[pos_col].astype(str))
    return df


if __name__ == "__main__":
    df = pd.read_csv("your_data.csv")
    X, y, feature_cols = build_features(df)
    print(f"Built {len(feature_cols)} features on {len(X)} rows.")
    print(X.head())