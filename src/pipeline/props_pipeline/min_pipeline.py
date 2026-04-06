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
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    pid = int(pdf["PLAYER_ID"].iloc[-1])
    player_pos = pdf["pos"].iloc[-1]
    res = []

    last = pdf.iloc[-1]

    # --- Tier 1: Core role ---

    # BAYES_MIN_PROJ
    canon_name = nameDict.get(name, name)
    player_team = last["TEAM_ABBREVIATION"]
    projected = projectedStartingFive.get(player_team, [])
    starting_override = 1 if (canon_name in projected or name in projected) else 0
    res.append(_bayes_min_proj_for_row(pdf, last, _BAYES_MIN_CONFIDENCE_K, starting_override=starting_override))

    # STARTER_X_MIN_AVG: starting flag × career avg minutes
    min_avg = float(pdf["MIN"].mean())
    res.append(float(starting_override) * min_avg)

    # MIN_EWM_L12, MIN_EWM_L3
    min_ewm12 = pdf["MIN"].astype(float).ewm(span=12).mean().iloc[-1]
    res.append(float(min_ewm12) if pd.notna(min_ewm12) else float("nan"))
    min_ewm3 = pdf["MIN"].astype(float).ewm(span=3).mean().iloc[-1]
    res.append(float(min_ewm3) if pd.notna(min_ewm3) else float("nan"))

    # STARTING_lag1: previous game's starting status
    starting_lag1 = float(pdf["STARTING"].iloc[-2]) if len(pdf) >= 2 else float("nan")
    res.append(starting_lag1)

    # STARTER_ROLL10_PCT: fraction of last 10 games as starter
    starter_roll10_pct = float(pdf["STARTING"].tail(10).mean())
    res.append(starter_roll10_pct if pd.notna(starter_roll10_pct) else float("nan"))

    # ROLE_LOCK: 1 if player is consistently a starter or bench player (>=90% either way)
    role_lock = float(1 if starter_roll10_pct >= 0.9 or starter_roll10_pct <= 0.1 else 0)
    res.append(role_lock)

    # MEDIAN_MIN_L10
    res.append(float(pdf["MIN"].tail(10).median()))

    # MIN_MIN_L5, MIN_MAX_L5
    res.append(float(pdf["MIN"].tail(5).min()))
    res.append(float(pdf["MIN"].tail(5).max()))

    # HIGH_MIN_TIER: 1 if player averages >= 28 MIN over last 10
    high_min_tier = float(1 if pdf["MIN"].tail(10).mean() >= 28.0 else 0)
    res.append(high_min_tier)

    # --- Tier 2: Team context ---

    # Gameday snapshot for within-team ranks
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]

    # TEAM_MIN_RANK_L5
    min_rank_l5 = gameday["MIN_roll5"].rank(ascending=False, method="dense")
    team_min_rank_l5 = float(min_rank_l5[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l5 if pd.notna(team_min_rank_l5) else float("nan"))

    # TEAM_POSS_RANK_L5
    poss_rank_l5 = gameday["POSS_roll5"].rank(ascending=False, method="dense")
    team_poss_rank_l5 = float(poss_rank_l5[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_poss_rank_l5 if pd.notna(team_poss_rank_l5) else float("nan"))

    # TEAM_PTS_RANK_L5
    pts_rank_l5 = gameday["PTS_roll5"].rank(ascending=False, method="dense")
    team_pts_rank_l5 = float(pts_rank_l5[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_pts_rank_l5 if pd.notna(team_pts_rank_l5) else float("nan"))

    # TEAM_USG_RANK_L5
    usg_rank_l5 = gameday["USG_PCT_roll5"].rank(ascending=False, method="dense")
    team_usg_rank_l5 = float(usg_rank_l5[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l5 if pd.notna(team_usg_rank_l5) else float("nan"))

    # POSS_roll5: player's own possession roll5
    res.append(float(pdf["POSS_roll5"].iloc[-1]) if pd.notna(pdf["POSS_roll5"].iloc[-1]) else float("nan"))

    # --- Tier 3: Situational ---

    # DAYS_REST: derived from last two game dates in player history
    if len(pdf) >= 2:
        last_date = pd.Timestamp(pdf["GAME_DATE"].iloc[-1]).normalize()
        prev_date = pd.Timestamp(pdf["GAME_DATE"].iloc[-2]).normalize()
        days_rest = int((last_date - prev_date).days)
    else:
        days_rest = 1
    res.append(float(days_rest))

    # AGE_X_B2B: age × back-to-back flag
    is_b2b = float(1 if days_rest == 1 else 0)
    age = float(last["AGE"]) if "AGE" in last.index and pd.notna(last["AGE"]) else float("nan")
    res.append(age * is_b2b)

    # USG_EWM_L5
    usg_ewm5 = pdf["USG_PCT"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(float(usg_ewm5) if pd.notna(usg_ewm5) else float("nan"))

    # PM_PER_MIN_R10: plus-minus per minute over last 10 games
    tail10 = pdf.tail(10)
    total_min = tail10["MIN"].sum()
    pm_per_min_r10 = (
        float(tail10["PLUS_MINUS"].sum() / total_min) if total_min > 0 else float("nan")
    )
    res.append(pm_per_min_r10)

    # POSITION_ENC
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

    # HOME_X_MIN_AVG: last game's home flag × career avg minutes
    home_flag = float(last["HOME"]) if "HOME" in last.index and pd.notna(last["HOME"]) else 0.0
    res.append(home_flag * min_avg)

    # MIN_share_proxy: player MIN_ewm12 / sum of team minutes on gameday
    team_min_sum = gameday["MIN"].sum()
    min_share_proxy = (
        float(min_ewm12) / team_min_sum if (pd.notna(min_ewm12) and team_min_sum > 0) else float("nan")
    )
    res.append(min_share_proxy)

    # TEAM_POSS_RANK_BY_POS_L5: rank within same-position teammates by POSS_roll5
    player_pos = last["pos"] if "pos" in last.index else last.get("POSITION", None)
    pos_group = gameday[gameday["pos"] == player_pos] if player_pos is not None else gameday
    poss_rank_by_pos = pos_group["POSS_roll5"].rank(ascending=False, method="dense")
    poss_rank_by_pos_val = poss_rank_by_pos[pos_group["PLAYER_NAME"] == name]
    res.append(float(poss_rank_by_pos_val.iloc[0]) if not poss_rank_by_pos_val.empty else float("nan"))

    return res
