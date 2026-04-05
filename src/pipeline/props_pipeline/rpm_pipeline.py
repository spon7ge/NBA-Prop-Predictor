import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.feature_engineer.rpm_features import rpm_features
from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive

# Match rpm_quantile_model.ipynb `apply_bayesian_rpm(..., confidence_k=20)`
_BAYES_RPM_CONFIDENCE_K = 20

# Aligned with rpm_quantile_model.ipynb `RPM_FEATURES`
RPM_FEATURES = [
    "BAYES_RPM_PROJ",
    "REB_PCT_roll5",
    "OREB_PCT_roll5",
    "DREB_PCT_roll5",
    "REB_PER_MIN_roll5",
    "OREB_PER_MIN_roll5",
    "DREB_PER_MIN_roll5",
    "MIN_ewm10",
    "MIN_ewm5",
    "TEAM_REB_PER_MIN_RANK_L5",
    "TEAM_REB_RANK_L5",
    "TEAM_FG_PCT_roll5",
    "TEAM_PACE_roll5",
    "OPP_PACE_roll5",
    "OPP_DEF_RATING_roll5",
    "OPP_FG_PCT_roll5",
    "OPP_FG3A_roll5",
    "DAYS_REST",
    "IS_B2B",
    "STARTING",
    "POSITION_ENC",
    "GP",
]


def _bayes_rpm_proj_for_player(
    df: pd.DataFrame, pid: int, confidence_k: int = 20, min_minutes: float = 1.0
) -> float:
    """Shrink player REB/MIN toward a (pos, STARTING) role prior; same construction as training."""
    d = df.copy()
    d["_rpm"] = np.where(d["MIN"] >= min_minutes, d["REB"] / d["MIN"], np.nan)
    priors = d.groupby(["pos", "STARTING"], dropna=False)["_rpm"].mean().to_dict()
    stats = d.groupby("PLAYER_ID", as_index=False).agg(
        player_mean_rpm=("_rpm", "mean"),
        games_played=("_rpm", "count"),
    )
    last = d[d["PLAYER_ID"] == pid].sort_values("GAME_DATE").iloc[-1]
    s = stats.loc[stats["PLAYER_ID"] == pid]
    if s.empty:
        return float("nan")
    player_mean = float(s["player_mean_rpm"].iloc[0])
    games_played = float(s["games_played"].iloc[0])
    key = (last["pos"], last["STARTING"])
    pv = priors.get(key)
    if pv is None or (isinstance(pv, float) and np.isnan(pv)):
        prior_rpm = float(d["_rpm"].mean())
    else:
        prior_rpm = float(pv)
    return round(
        (games_played * player_mean + confidence_k * prior_rpm) / (games_played + confidence_k),
        4,
    )


def _ewm_next_row(series: pd.Series, span: int) -> float:
    """shift(1).ewm(span) value for the row after the last observation (training-aligned)."""
    s = series.reset_index(drop=True).astype(float)
    extended = pd.concat([s, pd.Series([np.nan])], ignore_index=True)
    v = extended.shift(1).ewm(span=span, adjust=False).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _ensure_rpm_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror notebook: rpm_features + team rebound ranks + position encoding."""
    if "REB_PER_MIN" not in df.columns:
        df["REB_PER_MIN"] = df["REB"] / df["MIN"].replace(0, np.nan)
    if "OREB_PER_MIN" not in df.columns:
        df["OREB_PER_MIN"] = df["OREB"] / df["MIN"].replace(0, np.nan)
    if "DREB_PER_MIN" not in df.columns:
        df["DREB_PER_MIN"] = df["DREB"] / df["MIN"].replace(0, np.nan)
    if "REB_PCT_roll5" not in df.columns:
        df = rpm_features(df)
    if "TEAM_REB_RANK_L5" not in df.columns:
        df["TEAM_REB_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])["REB_roll5"].rank(
            ascending=False, method="dense"
        )
    if "TEAM_REB_PER_MIN_RANK_L5" not in df.columns:
        df["TEAM_REB_PER_MIN_RANK_L5"] = df.groupby(["TEAM_ID", "GAME_DATE"])[
            "REB_PER_MIN_roll5"
        ].rank(ascending=False, method="dense")
    if "POSITION_ENC" not in df.columns and "pos" in df.columns:
        le = LabelEncoder()
        df["POSITION_ENC"] = le.fit_transform(df["pos"].astype(str))
    return df


def rpm_pipeline(df, name, date):
    df = df.sort_values(["GAME_DATE", "PLAYER_ID"]).copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    slate = pd.Timestamp(date)
    date_str = date if isinstance(date, str) else pd.Timestamp(date).strftime("%Y-%m-%d")

    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE")
    if len(pdf) < 1:
        raise ValueError("No rows for player name in dataframe")
    pid = int(pdf["PLAYER_ID"].iloc[-1])

    df = _ensure_rpm_training_columns(df)
    pdf = df[df["PLAYER_ID"] == pid].sort_values("GAME_DATE")
    last = pdf.iloc[-1]

    res = []
    res.append(_bayes_rpm_proj_for_player(df, pid, confidence_k=_BAYES_RPM_CONFIDENCE_K))

    res.append(float(last["REB_PCT_roll5"]))
    res.append(float(last["OREB_PCT_roll5"]))
    res.append(float(last["DREB_PCT_roll5"]))
    res.append(float(last["REB_PER_MIN_roll5"]))
    res.append(float(last["OREB_PER_MIN_roll5"]))
    res.append(float(last["DREB_PER_MIN_roll5"]))

    res.append(_ewm_next_row(pdf["MIN"], 10))
    res.append(_ewm_next_row(pdf["MIN"], 5))

    res.append(float(last["TEAM_REB_PER_MIN_RANK_L5"]))
    res.append(float(last["TEAM_REB_RANK_L5"]))
    res.append(float(last["TEAM_FG_PCT_roll5"]))
    res.append(float(last["TEAM_PACE_roll5"]))

    opp, _ = findOpp(name, pdf, date_str, max_days_ahead=3)
    if opp is None:
        res.extend([float("nan")] * 4)
    else:
        opp_team = df[df["TEAM_ABBREVIATION"] == opp].sort_values("GAME_DATE")
        opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
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
        opp_team_fg_pct_roll5 = (
            opp_team.groupby("TEAM_ID")["TEAM_FG_PCT"]
            .rolling(5, min_periods=1)
            .mean()
            .round(2)
            .reset_index(level=0, drop=True)
        )
        opp_team_fg3a_roll5 = (
            opp_team.groupby("TEAM_ID")["TEAM_FG3A"]
            .rolling(5, min_periods=1)
            .mean()
            .round(2)
            .reset_index(level=0, drop=True)
        )
        res.append(float(opp_team_pace_roll5.iloc[-1]))
        res.append(float(opp_team_def_rating_roll5.iloc[-1]))
        res.append(float(opp_team_fg_pct_roll5.iloc[-1]))
        res.append(float(opp_team_fg3a_roll5.iloc[-1]))

    last_date = pdf["GAME_DATE"].iloc[-1]
    days_rest = int((slate.normalize() - last_date.normalize()).days)
    res.append(float(days_rest))
    res.append(float(1 if days_rest == 1 else 0))

    team = last["TEAM_ABBREVIATION"]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)

    if "POSITION_ENC" in last.index and pd.notna(last["POSITION_ENC"]):
        res.append(float(last["POSITION_ENC"]))
    else:
        res.append(float("nan"))

    res.append(float(len(pdf) - 1))

    return res
