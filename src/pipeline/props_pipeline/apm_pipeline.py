import numpy as np
import pandas as pd

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
    "MEDIAN_AST_PER_MIN_L10",
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

def apm_pipeline(df, name, date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    pid = int(pdf["PLAYER_ID"].iloc[-1])
    res = []

    # BAYES_APM_PROJ
    res.append(_bayes_apm_proj_for_player(pdf, pid, confidence_k=_BAYES_APM_CONFIDENCE_K))

    # AST_PCT_roll5, AST_RATIO_roll5, AST_TO_roll5
    res.append(float(pdf["AST_PCT"].tail(5).mean()))
    res.append(float(pdf["AST_RATIO"].tail(5).mean()))
    res.append(float(pdf["AST_TO"].tail(5).mean()))

    # MEDIAN_AST_PER_MIN_L10
    pdf['AST_PER_MIN'] = pdf['AST'] / pdf['MIN'].replace(0, np.nan)
    res.append(float(pdf["AST_PER_MIN"].tail(10).median()))

    # USG_PCT_roll5
    res.append(float(pdf["USG_PCT"].tail(5).mean()))

    # MIN_ewm10
    min_10_ewm = pdf["MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(min_10_ewm) if pd.notna(min_10_ewm) else float("nan"))

    # AST_PER_MIN_ewm10
    ast_per_min_ewm10 = pdf["AST_PER_MIN"].ewm(span=10).mean().iloc[-1]
    res.append(float(ast_per_min_ewm10) if pd.notna(ast_per_min_ewm10) else float("nan"))

    # AST_PER_MIN_roll5
    res.append(float(pdf["AST_PER_MIN"].tail(5).mean()))

    # TEAM_AST_PER_MIN_RANK_L10, TEAM_AST_PER_MIN_RANK_L5, TEAM_USG_RANK_L5
    last = pdf.iloc[-1]
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]

    ast_rank_l10 = gameday["AST_PER_MIN_roll10"].rank(ascending=False, method="dense")
    team_ast_per_min_rank_l10 = float(ast_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_ast_per_min_rank_l10 if pd.notna(team_ast_per_min_rank_l10) else float("nan"))

    ast_rank_l5 = gameday["AST_PER_MIN_roll5"].rank(ascending=False, method="dense")
    team_ast_per_min_rank_l5 = float(ast_rank_l5[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_ast_per_min_rank_l5 if pd.notna(team_ast_per_min_rank_l5) else float("nan"))

    usg_rank_l5 = gameday["USG_PCT_roll5"].rank(ascending=False, method="dense")
    team_usg_rank_l5 = float(usg_rank_l5[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l5 if pd.notna(team_usg_rank_l5) else float("nan"))

    # AST_share_proxy_roll5: player AST_roll5 / team AST_roll5 sum
    team_ast_roll5_sum = gameday["AST_roll5"].sum()
    player_ast_roll5 = float(pdf["AST"].tail(5).mean())
    ast_share_proxy_roll5 = (
        player_ast_roll5 / team_ast_roll5_sum if team_ast_roll5_sum > 0 else float("nan")
    )
    res.append(ast_share_proxy_roll5)

    # MIN_share_proxy: player MIN_ewm10 / team MIN_ewm10 sum
    player_team = last["TEAM_ABBREVIATION"]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    team_min_sum = gameday["MIN"].sum()
    min_share_proxy = (
        float(min_10_ewm) / team_min_sum if (pd.notna(min_10_ewm) and team_min_sum > 0) else float("nan")
    )
    res.append(min_share_proxy)

    # TEAM_POSS_share_roll5: player team possessions roll5 relative to league avg
    team_poss_roll5 = float(player_team_df["TEAM_POSS_roll5"].tail(5).mean())
    res.append(team_poss_roll5 if pd.notna(team_poss_roll5) else float("nan"))

    # OPP_PACE_roll5, OPP_POSS_roll5, OPP_DEF_RATING_roll5
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])

    opp_pace_roll5 = float(opp_team["TEAM_PACE"].tail(5).mean())
    res.append(opp_pace_roll5 if pd.notna(opp_pace_roll5) else float("nan"))

    opp_poss_roll5 = float(opp_team["TEAM_POSS"].tail(5).mean())
    res.append(opp_poss_roll5 if pd.notna(opp_poss_roll5) else float("nan"))

    opp_def_rating_roll5 = float(opp_team["TEAM_DEF_RATING"].tail(5).mean())
    res.append(opp_def_rating_roll5 if pd.notna(opp_def_rating_roll5) else float("nan"))

    # DAYS_REST, IS_B2B
    last_date = pdf["GAME_DATE"].iloc[-1]
    slate = pd.Timestamp(date).normalize()
    last_norm = pd.Timestamp(last_date).normalize()
    days_rest = int((slate - last_norm).days)
    res.append(float(days_rest))
    res.append(float(1 if days_rest == 1 else 0))

    # STARTING
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(player_team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)

    # GP
    res.append(float(len(pdf) - 1))

    return res
