import numpy as np
import pandas as pd

from src.features.feature_engineer.apm_features import apm_features
from src.utils.helper_functions import findOpp
from src.utils.team_info import *

# Match apm_quantile_model.ipynb `apply_bayesian_apm(..., confidence_k=20)`
_BAYES_APM_CONFIDENCE_K = 20

# Aligned with apm_quantile_model.ipynb `APM_FEATURES`
APM_FEATURES = [
    "BAYES_APM_PROJ",
    "AST_PCT_roll5",
    "AST_RATIO_roll5",
    "AST_TO_roll5",
    "USG_PCT_roll5",
    "MIN_ewm10",
    "AST_PER_MIN_ewm10",
    "AST_PER_MIN_roll5",
    "TEAM_AST_PER_MIN_RANK_L10",
    "TEAM_AST_PER_MIN_RANK_L5",
    "TEAM_USG_RANK_L5",
    "AST_share_proxy_roll5",
    "MIN_share_proxy",
    "TEAM_POSS_share_roll5",
    "OPP_PACE_roll5",
    "OPP_POSS_roll5",
    "OPP_DEF_RATING_roll5",
    "DAYS_REST",
    "IS_B2B",
    "STARTING",
    "GP",
]


def _bayes_apm_proj_for_player(
    df: pd.DataFrame, pid: int, confidence_k: int = 20, min_minutes: float = 1.0
) -> float:
    """Shrink player AST/MIN toward a (pos, STARTING) role prior; same as training."""
    d = df.copy()
    d["_apm"] = np.where(d["MIN"] >= min_minutes, d["AST"] / d["MIN"], np.nan)
    priors = d.groupby(["pos", "STARTING"], dropna=False)["_apm"].mean().to_dict()
    stats = d.groupby("PLAYER_ID", as_index=False).agg(
        player_mean_apm=("_apm", "mean"),
        games_played=("_apm", "count"),
    )
    last = d[d["PLAYER_ID"] == pid].sort_values("GAME_DATE").iloc[-1]
    s = stats.loc[stats["PLAYER_ID"] == pid]
    if s.empty:
        return float("nan")
    player_mean = float(s["player_mean_apm"].iloc[0])
    games_played = float(s["games_played"].iloc[0])
    key = (last["pos"], last["STARTING"])
    pv = priors.get(key)
    if pv is None or (isinstance(pv, float) and np.isnan(pv)):
        prior_apm = float(d["_apm"].mean())
    else:
        prior_apm = float(pv)
    return round(
        (games_played * player_mean + confidence_k * prior_apm) / (games_played + confidence_k),
        4,
    )


def _ewm10_next_row(series: pd.Series) -> float:
    """shift(1).ewm(span=10) value for the row after the last observation (training-aligned)."""
    s = series.reset_index(drop=True).astype(float)
    extended = pd.concat([s, pd.Series([np.nan])], ignore_index=True)
    v = extended.shift(1).ewm(span=10, adjust=False).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _ensure_apm_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror notebook: apm_features + team assist / usage ranks."""
    if "AST_PER_MIN" not in df.columns:
        df["AST_PER_MIN"] = df["AST"] / df["MIN"].replace(0, np.nan)
    if "AST_PCT_roll5" not in df.columns:
        df = apm_features(df)
    if "TEAM_USG_RANK_L5" not in df.columns:
        df["TEAM_USG_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])["USG_PCT_roll5"].rank(
            ascending=False, method="dense"
        )
    if "TEAM_AST_PER_MIN_RANK_L5" not in df.columns:
        df["TEAM_AST_PER_MIN_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "AST_PER_MIN_roll5"
        ].rank(ascending=False, method="dense")
    if "TEAM_AST_PER_MIN_RANK_L10" not in df.columns:
        df["TEAM_AST_PER_MIN_RANK_L10"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "AST_PER_MIN_roll10"
        ].rank(ascending=False, method="dense")
    return df


def apm_pipeline(df, name, date):
    df = df.sort_values(["GAME_DATE", "PLAYER_ID"]).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    slate = pd.Timestamp(date)
    date_str = date if isinstance(date, str) else pd.Timestamp(date).strftime("%Y-%m-%d")

    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE")
    if len(pdf) < 1:
        raise ValueError("No rows for player name in dataframe")
    pid = int(pdf["PLAYER_ID"].iloc[-1])

    df = _ensure_apm_training_columns(df)
    pdf = df[df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
    last = pdf.iloc[-1]

    res = []
    res.append(_bayes_apm_proj_for_player(df, pid, confidence_k=_BAYES_APM_CONFIDENCE_K))

    res.append(float(pdf["AST_PCT"].tail(5).mean()))
    res.append(float(pdf["AST_RATIO"].tail(5).mean()))
    res.append(float(pdf["AST_TO"].tail(5).mean()))
    res.append(float(pdf["USG_PCT"].tail(5).mean()))

    res.append(_ewm10_next_row(pdf["MIN"]))
    res.append(_ewm10_next_row(pdf["AST_PER_MIN"]))
    res.append(float(pdf["AST_PER_MIN"].tail(5).mean()))

    res.append(float(last["TEAM_AST_PER_MIN_RANK_L10"]))
    res.append(float(last["TEAM_AST_PER_MIN_RANK_L5"]))
    res.append(float(last["TEAM_USG_RANK_L5"]))

    res.append(float(last["AST_share_proxy_roll5"]))
    res.append(float(last["MIN_share_proxy"]))
    res.append(float(last["TEAM_POSS_share_roll5"]))

    opp, _ = findOpp(name, pdf, date_str, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_team_poss_roll5 = (
        opp_team.groupby("TEAM_ID")["TEAM_POSS"]
        .rolling(5, min_periods=1)
        .mean()
        .round(2)
        .reset_index(level=0, drop=True)
    )
    opp_team_pace_roll5 = (
        opp_team.groupby("TEAM_ID")["TEAM_PACE"]
        .rolling(5, min_periods=1)
        .mean()
        .round(2)
        .reset_index(level=0, drop=True)
    )
    opp_team_def_rating_roll5 = (
        opp_team.groupby("TEAM_ID")["TEAM_DEF_RATING"]
        .rolling(5, min_periods=1)
        .mean()
        .round(2)
        .reset_index(level=0, drop=True)
    )
    res.append(float(opp_team_pace_roll5.iloc[-1]))
    res.append(float(opp_team_poss_roll5.iloc[-1]))
    res.append(float(opp_team_def_rating_roll5.iloc[-1]))

    last_date = pdf["GAME_DATE"].iloc[-1]
    days_rest = int((slate.normalize() - last_date.normalize()).days)
    res.append(float(days_rest))
    res.append(float(1 if days_rest == 1 else 0))

    team = last["TEAM_ABBREVIATION"]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(
        1 if (canon_name in projected or name in projected) else 0
    )
    res.append(starting_flag)

    res.append(float(len(pdf) - 1))

    return res
