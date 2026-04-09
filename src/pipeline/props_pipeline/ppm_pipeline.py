import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive

_BAYES_PPM_CONFIDENCE_K = 20

def _bayes_ppm_proj_for_player(df, pid: int, confidence_k: int = 20, min_minutes: float = 1.0) -> float:
    """Shrink player PTS/MIN toward a (pos, STARTING) role prior; same construction as training."""
    d = df.copy()
    d["_ppm"] = np.where(d["MIN"] >= min_minutes, d["PTS"] / d["MIN"], np.nan)
    priors = d.groupby(["pos", "STARTING"], dropna=False)["_ppm"].mean().to_dict()
    stats = d.groupby("PLAYER_ID", as_index=False).agg(
        player_mean_ppm=("_ppm", "mean"),
        games_played=("_ppm", "count"),
    )
    last = d[d["PLAYER_ID"] == pid].sort_values("GAME_DATE").iloc[-1]
    s = stats.loc[stats["PLAYER_ID"] == pid]
    if s.empty:
        return float("nan")
    player_mean = float(s["player_mean_ppm"].iloc[0])
    games_played = float(s["games_played"].iloc[0])
    key = (last["pos"], last["STARTING"])
    pv = priors.get(key)
    if pv is None or (isinstance(pv, float) and np.isnan(pv)):
        prior_ppm = float(d["_ppm"].mean())
    else:
        prior_ppm = float(pv)
    return round(
        (games_played * player_mean + confidence_k * prior_ppm) / (games_played + confidence_k),
        4,
    )

def ppm_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    pid = int(pdf["PLAYER_ID"].iloc[-1])
    res = []

    # Bayes PPM Proj
    res.append(_bayes_ppm_proj_for_player(pdf, pid, confidence_k=_BAYES_PPM_CONFIDENCE_K))

    # STARTING
    team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)

    # PPM Momentum
    ppm_ewm10 = pdf["PTS_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(ppm_ewm10) if pd.notna(ppm_ewm10) else float("nan"))
    ppm_ewm5 = pdf["PTS_PER_MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(ppm_ewm5 - ppm_ewm10)
    points = pdf['PTS'].tail(10).mean().round(2)
    poss = pdf['POSS'].tail(10).mean().round(2)
    res.append(points / poss if pd.notna(points / poss) else float("nan"))

    # USG_PCT_roll10 and 3PA_PER_MIN_10_ewm
    usg_pct_roll10 = pdf["USG_PCT"].astype(float).tail(10).mean().round(2)
    res.append(float(usg_pct_roll10) if pd.notna(usg_pct_roll10) else float("nan"))
    three_pa_per_min_10_ewm = pdf["3PA_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(three_pa_per_min_10_ewm) if pd.notna(three_pa_per_min_10_ewm) else float("nan"))

    # TEAM_USG_RANK_L10
    team_usg_rank_l10 = pdf["USG_PCT_roll10"].rank(ascending=False, method="dense")
    team_usg_rank_l10 = float(team_usg_rank_l10[pdf["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l10 if pd.notna(team_usg_rank_l10) else float("nan"))

    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    poss_diff_l10 = float(player_team_df["POSS_DIFF_L10"].tail(10).mean().round(2)) - float(player_team_df["TEAM_POSS"].tail(10).mean().round(2))
    res.append(poss_diff_l10 if pd.notna(poss_diff_l10) else float("nan"))
    ts_pct_delta = float(player_team_df["TS_PCT_roll10"].tail(10).mean().round(2)) - float(player_team_df["TEAM_TS_PCT_roll10"].tail(10).mean().round(2))
    res.append(ts_pct_delta if pd.notna(ts_pct_delta) else float("nan"))
    true_usg_l10 = float(pdf["POSS_roll10"].tail(10).mean().round(2)) / float(player_team_df["TEAM_POSS"].tail(10).mean().round(2))
    res.append(true_usg_l10 if pd.notna(true_usg_l10) else float("nan"))
    res.append(float(pdf['TS_PCT'].tail(10).mean().round(2)))

    # Opponent Stats
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_def_rating_roll10 = float(opp_team["TEAM_DEF_RATING"].tail(10).mean().round(2))
    res.append(opp_def_rating_roll10 if pd.notna(opp_def_rating_roll10) else float("nan"))

    # PACE_DIFFERENTIAL
    opp_pace_roll10 = float(opp_team["TEAM_PACE"].tail(10).mean().round(2))
    player_team_pace_roll10 = float(player_team_df["TEAM_PACE"].tail(10).mean().round(2))
    res.append(player_team_pace_roll10 - opp_pace_roll10)

    # DAYS_REST
    last_date = pdf["GAME_DATE"].iloc[-1]
    slate = pd.Timestamp(current_date).normalize()
    last_norm = pd.Timestamp(last_date).normalize()
    days_rest = int((slate - last_norm).days)
    res.append(float(days_rest))

    # POSITION_ENC
    res.append(pdf['POSITION_ENC'].iloc[-1])
    return res