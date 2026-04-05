import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.team_info import nameDict, projectedStartingFive

# Match min_quantile_model.ipynb `apply_bayesian_minutes(..., confidence_k=20)`
_BAYES_MIN_CONFIDENCE_K = 20


def _bayes_min_proj_for_row(
    df: pd.DataFrame,
    last: pd.Series,
    confidence_k: int,
    starting_override: int | None = None,
) -> float:
    """Role (pos, STARTING) prior blended with player career mean minutes."""
    priors = df.groupby(["pos", "STARTING"], dropna=False)["MIN"].mean()
    stats = df.groupby("PLAYER_ID", as_index=False).agg(
        player_mean_min=("MIN", "mean"),
        games_played=("MIN", "count"),
    )
    pid = last["PLAYER_ID"]
    s = stats.loc[stats["PLAYER_ID"] == pid]
    if s.empty:
        return float("nan")
    n = float(s["games_played"].iloc[0])
    player_mean = float(s["player_mean_min"].iloc[0])
    pos = last["pos"] if "pos" in last.index else None
    if starting_override is not None:
        starting = int(starting_override)
    else:
        starting = last["STARTING"] if "STARTING" in last.index else np.nan
    if pd.isna(starting):
        prior_min = float(df["MIN"].mean())
    else:
        key = (pos, int(starting))
        if key in priors.index:
            prior_min = float(priors.loc[key])
        else:
            prior_min = float(df["MIN"].mean())
    return round((n * player_mean + confidence_k * prior_min) / (n + confidence_k), 2)


MIN_FEATURES = [
    # Tier 1 — Core role
    "BAYES_MIN_PROJ",
    "STARTER_X_MIN_AVG",
    "MIN_EWM_L12",
    "MIN_EWM_L3",
    "STARTING_lag1",
    "STARTER_ROLL10_PCT",
    "ROLE_LOCK",
    "MEDIAN_MIN_L10",
    "MIN_MIN_L5",  # new
    "MIN_MAX_L5",  # new
    "HIGH_MIN_TIER",  # new
    # Tier 2 — Team context
    "TEAM_MIN_RANK_L5",
    "TEAM_POSS_RANK_L5",
    "TEAM_PTS_RANK_L5",
    "TEAM_USG_RANK_L5",
    "POSS_roll5",
    # Tier 3 — Situational
    "DAYS_REST",
    "AGE_X_B2B",
    "USG_EWM_L5",
    "PM_PER_MIN_R10",
    "POSITION_ENC",
    "HOME_X_MIN_AVG",
    "MIN_share_proxy",
    "TEAM_POSS_RANK_BY_POS_L5",
]


def min_pipeline(df, name):
    df = df.sort_values(["GAME_DATE", "PLAYER_ID"]).copy()
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    if "STARTING_lag1" not in df.columns:
        df["STARTING_lag1"] = df.groupby("PLAYER_ID")["STARTING"].shift(1)
    if "STARTER_ROLL10_PCT" not in df.columns:
        df["STARTER_ROLL10_PCT"] = df.groupby("PLAYER_ID")["STARTING"].transform(
            lambda x: x.shift(1).rolling(10).mean().round(2)
        )
    if "POSS_roll5" not in df.columns and "POSS" in df.columns:
        df["POSS_roll5"] = df.groupby("PLAYER_ID")["POSS"].transform(
            lambda x: x.shift(1).rolling(5).mean().round(2)
        )
    if "MIN_roll5" not in df.columns and "MIN" in df.columns:
        df["MIN_roll5"] = df.groupby("PLAYER_ID")["MIN"].transform(
            lambda x: x.shift(1).rolling(5).mean().round(2)
        )
    if "PTS_roll5" not in df.columns and "PTS" in df.columns:
        df["PTS_roll5"] = df.groupby("PLAYER_ID")["PTS"].transform(
            lambda x: x.shift(1).rolling(5).mean().round(2)
        )
    if "USG_PCT_roll5" not in df.columns and "USG_PCT" in df.columns:
        df["USG_PCT_roll5"] = df.groupby("PLAYER_ID")["USG_PCT"].transform(
            lambda x: x.shift(1).rolling(5).mean().round(2)
        )

    if "MIN_roll5" in df.columns:
        df["TEAM_MIN_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "MIN_roll5"
        ].rank(ascending=False, method="dense")
    if "POSS_roll5" in df.columns:
        df["TEAM_POSS_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "POSS_roll5"
        ].rank(ascending=False, method="dense")
        if "pos" in df.columns:
            df["TEAM_POSS_RANK_BY_POS_L5"] = df.groupby(
                ["TEAM_ID", "GAME_DATE", "pos"]
            )["POSS_roll5"].rank(ascending=False, method="dense")
    if "PTS_roll5" in df.columns:
        df["TEAM_PTS_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "PTS_roll5"
        ].rank(ascending=False, method="dense")
    if "USG_PCT_roll5" in df.columns:
        df["TEAM_USG_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "USG_PCT_roll5"
        ].rank(ascending=False, method="dense")

    df["MIN_EWM_L12"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).ewm(span=12, adjust=False).mean()
    )
    df["MIN_EWM_L5"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )
    df["MIN_EWM_L3"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
    )
    df["MIN_MAX_L5"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).rolling(5).max()
    )
    df["MIN_MIN_L5"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).rolling(5).min()
    )
    if "USG_PCT" in df.columns:
        df["USG_EWM_L5"] = df.groupby("PLAYER_ID")["USG_PCT"].transform(
            lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
        )

    df["HIGH_MIN_TIER"] = np.nan
    mask = df["MIN_EWM_L12"].notna()
    df.loc[mask, "HIGH_MIN_TIER"] = pd.cut(
        df.loc[mask, "MIN_EWM_L12"],
        bins=[0, 20, 25, 30, 34, 100],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)

    df["MIN_season_avg"] = df.groupby(["PLAYER_ID", "SEASON_YEAR"])["MIN"].transform(
        lambda x: x.shift(1).expanding().mean().round(2)
    )
    df["MEDIAN_MIN_L10"] = df.groupby("PLAYER_ID")["MIN"].transform(
        lambda x: x.shift(1).rolling(10).median()
    )

    if "MIN_share_proxy" not in df.columns and "MIN_roll5" in df.columns:
        df["MIN_share_proxy"] = round(df["MIN_roll5"] / (48 * 5), 2)

    if "DAYS_REST" not in df.columns:
        df["DAYS_REST"] = (
            df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(3)
        )
    if "IS_B2B" not in df.columns:
        df["IS_B2B"] = (df["DAYS_REST"] == 1).astype(int)
    if "IS_HOME" not in df.columns and "MATCHUP" in df.columns:
        df["IS_HOME"] = df["MATCHUP"].str.contains("vs", na=False).astype(int)

    df["STARTER_X_MIN_AVG"] = df["STARTING"] * df["MIN_season_avg"]
    if "age" in df.columns:
        df["AGE_X_B2B"] = df["age"] * df["IS_B2B"]
    if "IS_HOME" in df.columns:
        df["HOME_X_MIN_AVG"] = df["IS_HOME"] * df["MIN_season_avg"]

    if "MIN_std10" not in df.columns:
        df["MIN_std10"] = df.groupby("PLAYER_ID")["MIN"].transform(
            lambda x: x.shift(1).rolling(10).std().round(2)
        )
    if "STARTER_ROLL10_PCT" in df.columns:
        df["ROLE_LOCK"] = df["STARTER_ROLL10_PCT"] * (
            1 / df["MIN_std10"].replace(0, np.nan)
        )

    if "MIN_lag1" not in df.columns:
        df["MIN_lag1"] = df.groupby("PLAYER_ID")["MIN"].shift(1)
    if "PLUS_MINUS_lag1" not in df.columns and "PLUS_MINUS" in df.columns:
        df["PLUS_MINUS_lag1"] = df.groupby("PLAYER_ID")["PLUS_MINUS"].shift(1)
    if "PLUS_MINUS_lag1" in df.columns and "MIN_lag1" in df.columns:
        df["_PM_PER_MIN"] = df["PLUS_MINUS_lag1"] / df["MIN_lag1"].replace(0, np.nan)
        df["PM_PER_MIN_R10"] = df.groupby("PLAYER_ID")["_PM_PER_MIN"].transform(
            lambda x: x.shift(1).rolling(10).mean()
        )
        df = df.drop(columns=["_PM_PER_MIN"], errors="ignore")

    if "POSITION_ENC" not in df.columns and "pos" in df.columns:
        le = LabelEncoder()
        df["POSITION_ENC"] = le.fit_transform(df["pos"].astype(str))

    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE")
    if len(pdf) < 10:
        return None

    last = pdf.iloc[-1]
    team = last["TEAM_ABBREVIATION"]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_proj = int(canon_name in projected or name in projected)

    bayes_min_proj = _bayes_min_proj_for_row(
        df,
        last,
        confidence_k=_BAYES_MIN_CONFIDENCE_K,
        starting_override=starting_proj,
    )

    def g(col, default=np.nan):
        if col == "BAYES_MIN_PROJ":
            return bayes_min_proj if pd.notna(bayes_min_proj) else default
        if col == "STARTER_X_MIN_AVG":
            if "MIN_season_avg" not in last.index:
                return default
            msa = last["MIN_season_avg"]
            if pd.isna(msa):
                return default
            return float(starting_proj * msa)
        if col not in pdf.columns:
            return default
        v = last[col]
        return float(v) if pd.notna(v) else default

    return [g(col) for col in MIN_FEATURES]
