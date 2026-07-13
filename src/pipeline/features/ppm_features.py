from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "PTS_PER_MIN"

DEFAULT_SEASON_MAP = {0: "S22", 1: "S23", 2: "S24", 3: "S25", 4: "S26"}

# Spans: 5/10 for feature means, 20 kept only as the "slow" side of the
# MIN and TS_PCT 5-vs-20 role-trend deltas (the only two trends that survived).
EWM_SPANS = [5, 10, 20]
TEAM_EWM_SPANS = [10]

# Stats needing ewm/expanding windows (5/10/20 ewm + expanding).
# USG_PCT and FT_PCT dropped out of this list in round 2 — only their
# season-to-date means survived, not any short-window version.
PLAYER_EWM_STATS = [
    "PTS_PER_MIN",
    "FGA",
    "FG3_PCT",
    "EFG_PCT",
    "MIN",
    "TS_PCT",   # level dropped, but needed for TS_TREND_5v20
]

# Stats needing only a season-to-date expanding mean (no short-window ewm).
PLAYER_SEASON_STATS = [
    "USG_PCT",
    "FGA",
    "FT_PCT",
    "MIN",
]

# No "own team" features survived validation — only opponent defensive rating.
TEAM_STATS_TO_EWM: list[str] = []
OPP_DEFENSE_STATS_TO_EWM = ["TEAM_DEF_RATING"]

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
    "player_PTS_PER_MIN_5_ewm",
    "player_PTS_PER_MIN_10_ewm",
    # Usage (1) — both short-window versions dropped in round 2
    "player_USG_PCT_season_mean",
    # Shot volume (2) — 10_ewm dropped in round 2
    "player_FGA_season_mean",
    "player_FGA_5_ewm",
    # Free throws (1) — 10_ewm dropped in round 2, only season_mean survives
    "player_FT_PCT_season_mean",
    # Efficiency (2)
    "player_FG3_PCT_10_ewm",
    "player_EFG_PCT_10_ewm",
    # Minutes (2)
    "player_MIN_season_mean",
    "player_MIN_10_ewm",
    # Role trend (2) — ewm5 − ewm20
    "MIN_TREND_5v20",
    "TS_TREND_5v20",
    # Context (3)
    "opp_team_TEAM_DEF_RATING_10_ewm",
    "DAYS_REST",
    "STARTING_rate_last10",
]

assert len(PPM_FEATURES) == 16, f"expected 16 features, got {len(PPM_FEATURES)}"


def _ewm_feature_names(stat_cols: list[str], spans: list[int], prefix: str) -> list[str]:
    names: list[str] = []
    for col in stat_cols:
        for span in spans:
            names.append(f"{prefix}_{col}_{span}_ewm")
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

    df = _add_ewm_and_expanding(
        df, group_col="PLAYER_ID", stat_cols=PLAYER_EWM_STATS,
        spans=EWM_SPANS, prefix="player",
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


def validate_ppm_dataset(
    df: pd.DataFrame,
    *,
    key_cols: tuple[str, ...] = ("GAME_ID", "PLAYER_ID"),
    require_season: bool = True,
) -> dict[str, object]:
    """Assert an engineered PPM frame is training-ready.

    Checks:
    * required id / target / feature columns exist
    * no duplicate ``(GAME_ID, PLAYER_ID)`` rows
    * target is finite where present
    * optional ``SEASON`` column present

    Returns a small summary dict. Raises ``ValueError`` / ``KeyError`` on failure.
    """
    if df.empty:
        raise ValueError("PPM dataset is empty")

    required = list(key_cols) + [TARGET, "MIN", "GAME_DATE", *PPM_FEATURES]
    if require_season:
        required.append("SEASON")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"PPM dataset missing {len(missing)} column(s): {missing[:20]}"
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
            f"PPM dataset has {n_dup_rows:,} duplicate rows across {n_keys:,} "
            f"{key_cols} keys. Sample: {sample}"
        )

    y = pd.to_numeric(df[TARGET], errors="coerce")
    n_bad_target = int((y.isna() | ~np.isfinite(y)).sum())
    if n_bad_target:
        raise ValueError(
            f"PPM dataset has {n_bad_target:,} non-finite {TARGET} values"
        )

    summary: dict[str, object] = {
        "rows": len(df),
        "cols": df.shape[1],
        "features": len(PPM_FEATURES),
        "duplicate_keys": 0,
        "bad_target": 0,
    }
    if "SEASON" in df.columns:
        summary["seasons"] = (
            df["SEASON"].value_counts().sort_index().to_dict()
        )
    return summary


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

def _add_ewm_and_expanding(df, group_col, stat_cols, spans, prefix):
    df = df.sort_values([group_col, "GAME_DATE"])
    g = df.groupby(group_col, sort=False)

    for col in stat_cols:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        shifted_grouped = shifted.groupby(df[group_col])

        for span in spans:
            df[f"{prefix}_{col}_{span}_ewm"] = shifted_grouped.transform(
                lambda s, span=span: s.ewm(span=span, min_periods=1).mean()
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
        ("player_MIN_5_ewm", "player_MIN_20_ewm", "MIN_TREND_5v20"),
        ("player_TS_PCT_5_ewm", "player_TS_PCT_20_ewm", "TS_TREND_5v20"),
    ]
    for c5, c20, out in pairs:
        if c5 in df.columns and c20 in df.columns:
            df[out] = df[c5] - df[c20]
    return df


def _build_team_context(df):
    if not {"TEAM_ID", "GAME_ID", "GAME_DATE", "OPP_TEAM_ID"}.issubset(df.columns):
        return df

    stat_pool = list(dict.fromkeys(TEAM_STATS_TO_EWM + OPP_DEFENSE_STATS_TO_EWM))
    team_cols = ["TEAM_ID", "GAME_ID", "GAME_DATE"] + [c for c in stat_pool if c in df.columns]
    if len(team_cols) <= 3:
        return df
    team_df = df[team_cols].drop_duplicates(subset=["TEAM_ID", "GAME_ID"]).copy()

    stat_union = sorted(set(stat_pool) & set(team_df.columns))
    if not stat_union:
        return df

    team_df = _add_ewm_and_expanding(
        team_df, group_col="TEAM_ID", stat_cols=stat_union,
        spans=TEAM_EWM_SPANS, prefix="team",
    )

    opp_feat_cols = [
        c for c in team_df.columns
        if c.startswith("team_") and any(s in c for s in OPP_DEFENSE_STATS_TO_EWM)
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