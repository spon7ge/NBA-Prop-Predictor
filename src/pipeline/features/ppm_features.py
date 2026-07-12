"""PPM quantile model feature engineering — TOP-16 feature subset (SHAP/perm/ablation-validated, round 2).

Trimmed further after a second empirical validation pass: SHAP importance,
permutation importance (with p-value), and leave-one-out ablation delta all
had to agree a feature earns its place (see `keep` column of the validation
run this was built from).

Round 1 (30 -> 21) dropped `player_PTS_PER_MIN_ewma`, IS_HOME, IS_BACK_TO_BACK,
TS_PCT level features, and USG_TREND/FGA_TREND.

Round 2 (21 -> 16) dropped:
- `player_USG_PCT_roll10_mean` — ranked #1 on raw SHAP, but negative ablation
  delta (removing it improved held-out error). The season-level usage mean
  was carrying the real signal; the short-window version was overfitting.
- `player_USG_PCT_roll5_mean` — same pattern, negative ablation delta.
- `player_FTA_roll10_mean`, `player_FGA_roll10_mean` — negative ablation deltas.
- `player_PTS_PER_MIN_season_mean` — negative ablation delta; the roll5/roll10/
  expanding versions of the same stat already cover it.

Leakage-safe: same-game box-score columns are only used as lagged rolling /
expanding averages. Exposes the gold / ml API (`PPM_FEATURES`, `ppm_features`,
`prepare_season_df`, `build_ppm_dataset`) plus `build_features` for notebook tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "PTS_PER_MIN"

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25", 4: "S26"}

# Windows: 5/10 for feature means, 20 kept only as the "slow" side of the
# MIN and TS_PCT 5-vs-20 role-trend deltas (the only two trends that survived).
ROLLING_WINDOWS = [5, 10, 20]
TEAM_ROLLING_WINDOWS = [10]

# Stats needing rolling/ewma/expanding windows (roll5/roll10/roll20/expanding).
# USG_PCT and FT_PCT dropped out of this list in round 2 — only their
# season-to-date means survived, not any short-window version.
PLAYER_ROLL_STATS = [
    "PTS_PER_MIN",
    "FGA",
    "FG3_PCT",
    "EFG_PCT",
    "MIN",
    "TS_PCT",   # level dropped, but needed for TS_TREND_5v20
]

# Stats needing only a season-to-date expanding mean (no short-window roll).
PLAYER_SEASON_STATS = [
    "USG_PCT",
    "FGA",
    "FT_PCT",
    "MIN",
]

# No "own team" features survived validation — only opponent defensive rating.
TEAM_STATS_TO_ROLL: list[str] = []
OPP_DEFENSE_STATS_TO_ROLL = ["TEAM_DEF_RATING"]

# IS_HOME and IS_BACK_TO_BACK both failed validation (near-zero / negative
# ablation impact) and were dropped in round 1.
_STATIC_FEATURES = [
    "DAYS_REST",
    "STARTING_rate_last10",
]

# The 16 features that survived a second SHAP + permutation + ablation
# validation pass (all keep=True), hardcoded so trimming the pipeline above
# can never silently change what gets exposed.
PPM_FEATURES = [
    # Recent scoring rate (3) — season_mean dropped in round 2
    "player_PTS_PER_MIN_expanding_mean",
    "player_PTS_PER_MIN_roll5_mean",
    "player_PTS_PER_MIN_roll10_mean",
    # Usage (1) — both short-window versions dropped in round 2
    "player_USG_PCT_season_mean",
    # Shot volume (2) — roll10 dropped in round 2
    "player_FGA_season_mean",
    "player_FGA_roll5_mean",
    # Free throws (1) — roll10 dropped in round 2, only season_mean survives
    "player_FT_PCT_season_mean",
    # Efficiency (2)
    "player_FG3_PCT_roll10_mean",
    "player_EFG_PCT_roll10_mean",
    # Minutes (2)
    "player_MIN_season_mean",
    "player_MIN_roll10_mean",
    # Role trend (2)
    "MIN_TREND_5v20",
    "TS_TREND_5v20",
    # Context (3)
    "opp_team_TEAM_DEF_RATING_roll10_mean",
    "DAYS_REST",
    "STARTING_rate_last10",
]

assert len(PPM_FEATURES) == 16, f"expected 16 features, got {len(PPM_FEATURES)}"


def _rolling_feature_names(stat_cols: list[str], windows: list[int], prefix: str) -> list[str]:
    names: list[str] = []
    for col in stat_cols:
        for w in windows:
            names.append(f"{prefix}_{col}_roll{w}_mean")
            names.append(f"{prefix}_{col}_roll{w}_std")
        names.append(f"{prefix}_{col}_ewma")
        names.append(f"{prefix}_{col}_expanding_mean")
    return names


def ppm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the PPM model (top-16 subset)."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns and "START_POSITION" in df.columns:
        df["STARTING"] = df["START_POSITION"].notna().astype(int)

    if TARGET not in df.columns and {"PTS", "MIN"}.issubset(df.columns):
        df[TARGET] = df["PTS"] / df["MIN"].replace(0, np.nan)

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
    df = _add_role_trend(df)
    df = _build_team_context(df)

    # Guarantee declared model columns exist for gold / ml alignment.
    for col in PPM_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


def prepare_season_df(df: pd.DataFrame, season_label: str | None = None) -> pd.DataFrame:
    """Feature-engineer one season and optionally tag it for holdout splits."""
    season_df = ppm_features(df)
    if season_label is not None:
        season_df["SEASON"] = season_label
    if "STARTING" not in season_df.columns:
        season_df["STARTING"] = 0
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


def feature_cols_from_df(df: pd.DataFrame) -> list[str]:
    """The fixed top-16 feature columns present on an engineered frame."""
    return [c for c in PPM_FEATURES if c in df.columns]


def build_features(df: pd.DataFrame, drop_first_n_games: int = 0):
    """Notebook helper: return ``(X, y, feature_cols)`` ready for XGBoost."""
    df = ppm_features(df)

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


def _add_role_trend(df):
    pairs = [
        ("player_MIN_roll5_mean", "player_MIN_roll20_mean", "MIN_TREND_5v20"),
        ("player_TS_PCT_roll5_mean", "player_TS_PCT_roll20_mean", "TS_TREND_5v20"),
    ]
    for c5, c20, out in pairs:
        if c5 in df.columns and c20 in df.columns:
            df[out] = df[c5] - df[c20]
    return df


def _build_team_context(df):
    if not {"TEAM_ID", "GAME_ID", "GAME_DATE", "OPP_TEAM_ID"}.issubset(df.columns):
        return df

    stat_pool = list(dict.fromkeys(TEAM_STATS_TO_ROLL + OPP_DEFENSE_STATS_TO_ROLL))
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
        if c.startswith("team_") and any(s in c for s in OPP_DEFENSE_STATS_TO_ROLL)
    ]
    opp_merge = team_df[["TEAM_ID", "GAME_ID"] + opp_feat_cols].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", **{c: f"opp_{c}" for c in opp_feat_cols}}
    )
    df = df.merge(opp_merge, on=["OPP_TEAM_ID", "GAME_ID"], how="left")

    return df


if __name__ == "__main__":
    df = pd.read_csv("your_data.csv")
    X, y, feature_cols = build_features(df)
    print(f"Built {len(feature_cols)} features on {len(X)} rows.")
    print(X.head())