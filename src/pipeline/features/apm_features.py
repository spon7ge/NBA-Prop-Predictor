from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET = "AST_PER_MIN"

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25", 4: "S26"}

# Windows: 5/10 for feature means, 20 kept only as the "slow" side of the
# MIN 5-vs-20 role-trend delta.
ROLLING_WINDOWS = [5, 10, 20]
TEAM_ROLLING_WINDOWS = [10]

# Stats needing rolling/ewma/expanding windows (roll5/roll10/roll20/expanding).
# Only what feeds a surviving feature or the MIN_TREND_5v20 intermediate.
PLAYER_ROLL_STATS = [
    "AST_PER_MIN",  # only its ewma survived, but ewma still needs this listed
    "PASS",
    "SAST",
    "AST_RATIO",
    "MIN",
]

# Only PASS_season_mean survived — everything else in the AST_PER_MIN/PIE/MIN
# season-mean family was subsumed by ewma or MIN_roll5_mean (see docstring).
PLAYER_SEASON_STATS = [
    "PASS",
]

# Own-team context: only TEAM_AST_PCT survived. Opponent context: both
# defensive rating and pace survived.
TEAM_STATS_TO_ROLL = ["TEAM_AST_PCT"]
OPP_DEFENSE_STATS_TO_ROLL = ["TEAM_DEF_RATING", "TEAM_PACE"]

# STARTING_rate_last10 dropped this round (p=0.23, not significant).
_STATIC_FEATURES = [
    "POSITION_ENCODED",
    "MIN_TREND_5v20",
]

# The 11 features that survived a second SHAP + permutation + ablation
# validation pass (all keep=True), hardcoded so trimming the pipeline above
# can never silently change what gets exposed.
APM_FEATURES = [
    # Recent assist rate (1) — sole survivor of the AST_PER_MIN/AST_PCT
    # family; the rest were >0.95 correlated with this and got pruned.
    "player_AST_PER_MIN_ewma",
    # Ball-handling volume (2)
    "player_PASS_roll10_mean",
    "player_PASS_season_mean",
    # Secondary playmaking / role (2)
    "player_SAST_roll10_mean",
    "player_AST_RATIO_roll10_mean",
    # Minutes / role (2)
    "player_MIN_roll5_mean",
    "MIN_TREND_5v20",
    # Role / physical proxy (1)
    "POSITION_ENCODED",
    # Team / matchup context (3)
    "opp_team_TEAM_DEF_RATING_roll10_mean",
    "opp_team_TEAM_PACE_roll10_mean",
    "own_team_TEAM_AST_PCT_roll10_mean",
]

assert len(APM_FEATURES) == 11, f"expected 11 features, got {len(APM_FEATURES)}"


def _rolling_feature_names(stat_cols: list[str], windows: list[int], prefix: str) -> list[str]:
    names: list[str] = []
    for col in stat_cols:
        for w in windows:
            names.append(f"{prefix}_{col}_roll{w}_mean")
            names.append(f"{prefix}_{col}_roll{w}_std")
        names.append(f"{prefix}_{col}_ewma")
        names.append(f"{prefix}_{col}_expanding_mean")
    return names


def apm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature pipeline for the APM model (top-11 subset)."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    if "STARTING" not in df.columns and "START_POSITION" in df.columns:
        df["STARTING"] = df["START_POSITION"].notna().astype(int)

    if TARGET not in df.columns and {"AST", "MIN"}.issubset(df.columns):
        df[TARGET] = df["AST"] / df["MIN"].replace(0, np.nan)

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
    df = _add_minutes_trend(df)
    df = _build_team_context(df)

    # Guarantee declared model columns exist for gold / ml alignment.
    for col in APM_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


def prepare_season_df(df: pd.DataFrame, season_label: str | None = None) -> pd.DataFrame:
    """Feature-engineer one season and optionally tag it for holdout splits."""
    season_df = apm_features(df)
    if season_label is not None:
        season_df["SEASON"] = season_label
    if "STARTING" not in season_df.columns:
        season_df["STARTING"] = 0
    min_reg = season_df["MIN"].clip(upper=48)
    return season_df[(min_reg >= 15) | (season_df["STARTING"] == 1)]


def build_apm_dataset(
    season_dfs: list[pd.DataFrame],
    season_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build the concatenated APM-model training frame across seasons."""
    season_map = season_map or DEFAULT_SEASON_MAP
    res = [
        prepare_season_df(season_df, season_label=season_map[i])
        for i, season_df in enumerate(season_dfs)
    ]
    df = pd.concat(res, ignore_index=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def feature_cols_from_df(df: pd.DataFrame) -> list[str]:
    """The fixed top-11 feature columns present on an engineered frame."""
    return [c for c in APM_FEATURES if c in df.columns]


def build_features(df: pd.DataFrame, drop_first_n_games: int = 0):
    """Notebook helper: return ``(X, y, feature_cols)`` ready for XGBoost."""
    df = apm_features(df)

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


def _add_minutes_trend(df):
    c5 = "player_MIN_roll5_mean"
    c20 = "player_MIN_roll20_mean"
    if c5 in df.columns and c20 in df.columns:
        df["MIN_TREND_5v20"] = df[c5] - df[c20]
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
        if c.startswith("team_") and any(s in c for s in OPP_DEFENSE_STATS_TO_ROLL)
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