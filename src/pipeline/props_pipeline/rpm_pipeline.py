import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.feature_engineer.rpm_features import rpm_features
from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive

# Match rpm_quantile_model.ipynb `apply_bayesian_rpm(..., confidence_k=20)`
_BAYES_RPM_CONFIDENCE_K = 20

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


def rpm_pipeline(df, name, date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    pid = int(pdf["PLAYER_ID"].iloc[-1])
    res = []
    res.append(_bayes_rpm_proj_for_player(pdf, pid, confidence_k=_BAYES_RPM_CONFIDENCE_K))
    res.append(float(pdf["REB_PCT"].tail(5).mean()))
    res.append(float(pdf["OREB_PCT"].tail(5).mean()))
    res.append(float(pdf["DREB_PCT"].tail(5).mean()))
    pdf['REB_PER_MIN'] = pdf['REB'] / pdf['MIN'].replace(0, np.nan)
    pdf['OREB_PER_MIN'] = pdf['OREB'] / pdf['MIN'].replace(0, np.nan)
    pdf['DREB_PER_MIN'] = pdf['DREB'] / pdf['MIN'].replace(0, np.nan)
    res.append(float(pdf["REB_PER_MIN"].tail(5).mean()))
    res.append(float(pdf["OREB_PER_MIN"].tail(5).mean()))
    res.append(float(pdf["DREB_PER_MIN"].tail(5).mean()))
    res.append(float(pdf["REB_PER_MIN"].tail(5).median()))
    min_10_ewm = pdf["MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(min_10_ewm) if pd.notna(min_10_ewm) else float("nan"))
    min_5_ewm = pdf["MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(min_5_ewm) if pd.notna(min_5_ewm) else float("nan"))

    last = pdf.iloc[-1]
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    r = gameday["REB_PER_MIN_roll5"].rank(ascending=False, method="dense")
    team_reb_per_min_rank_l5 = float(r[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_reb_per_min_rank_l5 if pd.notna(team_reb_per_min_rank_l5) else float("nan"))
    reb_rank_l10 = gameday["REB_roll10"].rank(ascending=False, method="dense")
    team_reb_rank_l10 = float(reb_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_reb_rank_l10 if pd.notna(team_reb_rank_l10) else float("nan"))

    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    team_fg_pct_roll5 = float(player_team_df["TEAM_FG_PCT_roll5"].tail(5).mean())
    res.append(team_fg_pct_roll5 if pd.notna(team_fg_pct_roll5) else float("nan"))
    team_pace_roll5 = float(player_team_df["TEAM_PACE_roll5"].tail(5).mean())
    res.append(team_pace_roll5 if pd.notna(team_pace_roll5) else float("nan"))

    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_pace_roll5 = float(opp_team["TEAM_PACE"].tail(5).mean())
    res.append(opp_pace_roll5 if pd.notna(opp_pace_roll5) else float("nan"))
    opp_def_rating_roll5 = float(opp_team["TEAM_DEF_RATING"].tail(5).mean())
    res.append(opp_def_rating_roll5 if pd.notna(opp_def_rating_roll5) else float("nan"))
    opp_fg_pct_roll5 = float(opp_team["TEAM_FG_PCT"].tail(5).mean())
    res.append(opp_fg_pct_roll5 if pd.notna(opp_fg_pct_roll5) else float("nan"))
    opp_fg3a_roll5 = float(opp_team["TEAM_FG3A"].tail(5).mean())
    res.append(opp_fg3a_roll5 if pd.notna(opp_fg3a_roll5) else float("nan"))

    last_date = pdf["GAME_DATE"].iloc[-1]
    slate = pd.Timestamp(date).normalize()
    last_norm = pd.Timestamp(last_date).normalize()
    days_rest = int((slate - last_norm).days)
    res.append(float(days_rest))
    res.append(float(1 if days_rest == 1 else 0))

    team = last["TEAM_ABBREVIATION"]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)
    # POSITION_ENC
    player_pos = pdf["pos"].iloc[-1]
    if player_pos == "PG":
        res.append(0)
    elif player_pos == "SG":
        res.append(1)
    elif player_pos == "SF":
        res.append(2)
    elif player_pos == "PF":
        res.append(3)
    elif player_pos == "C":
        res.append(4)
    else:
        res.append(float("nan"))
    res.append(float(len(pdf) - 1))
    return res