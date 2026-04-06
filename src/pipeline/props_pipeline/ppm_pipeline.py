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

    pts_ewm10 = pdf["PTS"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(pts_ewm10) if pd.notna(pts_ewm10) else float("nan"))
    pts_ewm5 = pdf["PTS"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(pts_ewm5) if pd.notna(pts_ewm5) else float("nan"))
    min_ewm10 = pdf["MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(min_ewm10) if pd.notna(min_ewm10) else float("nan"))
    min_ewm5 = pdf["MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(min_ewm5) if pd.notna(min_ewm5) else float("nan"))
    # Bayes PPM Proj
    res.append(_bayes_ppm_proj_for_player(pdf, pid, confidence_k=_BAYES_PPM_CONFIDENCE_K))

    pts_per_min_10_ewm = pdf["PTS_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(pts_per_min_10_ewm) if pd.notna(pts_per_min_10_ewm) else float("nan"))
    pts_per_min_5_ewm = pdf["PTS_PER_MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(pts_per_min_5_ewm) if pd.notna(pts_per_min_5_ewm) else float("nan"))
    median_pts_per_min_rolling_10 = pdf["PTS_PER_MIN"].astype(float).tail(10).median()
    res.append(
        round(float(median_pts_per_min_rolling_10), 2)
        if pd.notna(median_pts_per_min_rolling_10)
        else float("nan")
    )
    fga_per_min_10_ewm = pdf["FGA_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(fga_per_min_10_ewm) if pd.notna(fga_per_min_10_ewm) else float("nan"))
    fga_per_min_5_ewm = pdf["FGA_PER_MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(fga_per_min_5_ewm) if pd.notna(fga_per_min_5_ewm) else float("nan"))
    three_pa_per_min_10_ewm = pdf["3PA_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(three_pa_per_min_10_ewm) if pd.notna(three_pa_per_min_10_ewm) else float("nan"))
    three_pa_per_min_5_ewm = pdf["3PA_PER_MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(three_pa_per_min_5_ewm) if pd.notna(three_pa_per_min_5_ewm) else float("nan"))

    usg_pct_roll10 = pdf["USG_PCT"].astype(float).tail(10).mean().round(2)
    res.append(float(usg_pct_roll10) if pd.notna(usg_pct_roll10) else float("nan"))
    usg_pct_roll5 = pdf["USG_PCT"].astype(float).tail(5).mean().round(2)
    res.append(float(usg_pct_roll5) if pd.notna(usg_pct_roll5) else float("nan"))
    ts_pct_roll10 = pdf["TS_PCT"].astype(float).tail(10).mean().round(2)
    res.append(float(ts_pct_roll10) if pd.notna(ts_pct_roll10) else float("nan"))
    poss_roll10 = pdf["POSS"].astype(float).tail(10).mean().round(2)
    res.append(float(poss_roll10) if pd.notna(poss_roll10) else float("nan"))

    # Opponent Stats
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_pace_roll5 = float(opp_team["TEAM_PACE"].tail(5).mean().round(2))
    res.append(opp_pace_roll5 if pd.notna(opp_pace_roll5) else float("nan"))
    opp_poss_roll5 = float(opp_team["TEAM_POSS"].tail(5).mean().round(2))
    res.append(opp_poss_roll5 if pd.notna(opp_poss_roll5) else float("nan"))
    opp_def_rating_roll5 = float(opp_team["TEAM_DEF_RATING"].tail(5).mean().round(2))
    res.append(opp_def_rating_roll5 if pd.notna(opp_def_rating_roll5) else float("nan"))

    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    player_team_poss_roll5 = float(player_team_df["TEAM_POSS"].tail(5).mean().round(2))
    res.append(player_team_poss_roll5 if pd.notna(player_team_poss_roll5) else float("nan"))
    player_team_pts = float(player_team_df["TEAM_PTS"].tail(5).mean().round(2))
    pts_roll_5 = float(pdf["PTS"].astype(float).rolling(5).mean().round(2).iloc[-1])
    if pd.isna(pts_roll_5) or player_team_pts == 0 or pd.isna(player_team_pts):
        res.append(float("nan"))
    else:
        res.append(round(pts_roll_5 / player_team_pts, 2))

    min_roll_5 = float(pdf["MIN"].astype(float).tail(5).mean().round(2))
    res.append(
        round(min_roll_5 / 240.0, 2) if pd.notna(min_roll_5) else float("nan")
    )  # MIN_share_proxy = MIN_roll5 / (48 * 5)

    # Team Rankings L10
    last = pdf.iloc[-1]
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    r = gameday["PTS_PER_MIN_roll10"].rank(ascending=False, method="dense")
    player_team_pts_per_min_rank_l10 = float(r[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(player_team_pts_per_min_rank_l10 if pd.notna(player_team_pts_per_min_rank_l10) else float("nan"))
    r = gameday["PTS_PER_MIN_roll5"].rank(ascending=False, method="dense")
    player_team_pts_per_min_rank_l5 = float(r[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(player_team_pts_per_min_rank_l5 if pd.notna(player_team_pts_per_min_rank_l5) else float("nan"))
    r = gameday["USG_PCT_roll10"].rank(ascending=False, method="dense")
    player_team_usg_rank_l10 = float(r[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(player_team_usg_rank_l10 if pd.notna(player_team_usg_rank_l10) else float("nan"))
    r = gameday["TS_PCT_roll10"].rank(ascending=False, method="dense")
    player_team_ts_rank_l10 = float(r[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(player_team_ts_rank_l10 if pd.notna(player_team_ts_rank_l10) else float("nan"))

    team = last["TEAM_ABBREVIATION"]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)

    last_date = pdf["GAME_DATE"].iloc[-1]
    slate = pd.Timestamp(current_date).normalize()
    last_norm = pd.Timestamp(last_date).normalize()
    days_rest = int((slate - last_norm).days)
    res.append(float(1 if days_rest == 1 else 0))
    res.append(float(days_rest))

    res.append(float(len(pdf) - 1))
    return res