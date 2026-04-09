import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.team_info import nameDict, projectedStartingFive
from src.utils.helper_functions import findOpp

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


def min_pipeline(df, name, current_date):
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

    # MIN_EWM_L10, MIN_EWM_L3
    min_ewm10 = pdf["MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(min_ewm10) if pd.notna(min_ewm10) else float("nan"))
    min_ewm3 = pdf["MIN"].astype(float).ewm(span=3).mean().iloc[-1]

    # MIN_TREND
    res.append(min_ewm3 - min_ewm10)

    # STARTER_ROLL10_PCT: fraction of last 10 games as starter
    starter_roll10_pct = float(pdf["STARTING"].tail(10).mean())
    res.append(starter_roll10_pct if pd.notna(starter_roll10_pct) else float("nan"))

    # MIN_ROLE_Z_SCORE (expanding stats are Series; take value through last row)
    season_mean = float(pdf["MIN"].expanding().mean().iloc[-1])
    season_std = float(pdf["MIN"].expanding().std().iloc[-1])
    recent_mean = float(pdf["MIN"].tail(10).mean())
    if pd.isna(season_std) or season_std == 0:
        res.append(float("nan"))
    else:
        res.append((recent_mean - season_mean) / season_std)

    # ROLE_LOCK
    roll10_std = pdf["MIN"].tail(10).std()
    role_lock = (
        float(starter_roll10_pct) / float(roll10_std)
        if pd.notna(roll10_std) and roll10_std != 0
        else float("nan")
    )
    res.append(float(role_lock) if pd.notna(role_lock) else float("nan"))

    # HIGH_MIN_TIER: 1 if player averages >= 28 MIN over last 10
    # high_min_tier = float(1 if pdf["MIN"].tail(10).mean() >= 28.0 else 0)
    # res.append(high_min_tier)

    # Gameday snapshot for within-team ranks
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]

    # TEAM_MIN_RANK_L10
    min_rank_l10 = gameday["MIN_roll10"].rank(ascending=False, method="dense")
    team_min_rank_l10 = float(min_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    # TEAM_USG_RANK_L10
    usg_rank_l10 = gameday["USG_PCT_roll10"].rank(ascending=False, method="dense")
    team_usg_rank_l10 = float(usg_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l10 if pd.notna(team_usg_rank_l10) else float("nan"))

    # MIN_share_proxy: player MIN_ewm10 / sum of team minutes on gameday
    team_min_sum = gameday["MIN"].sum()
    min_share_proxy = (
        float(pdf["MIN"].tail(10).mean()) / team_min_sum if (pd.notna(pdf["MIN"].tail(10).mean()) and team_min_sum > 0) else float("nan")
    )
    res.append(min_share_proxy)

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

    # MIN VOLATILITY_L10
    min_volatility_l10 = pdf["MIN"].tail(10).std()
    res.append(min_volatility_l10)

    # ROTATION_GAP_L5
    min_max_l10 = pdf["MIN"].tail(10).max()
    min_min_l10 = pdf["MIN"].tail(10).min()
    res.append(min_max_l10 - min_min_l10)

    # POSITION_ENC
    res.append(pdf['POSITION_ENC'].iloc[-1])

    # PACE_DIFFERENTIAL
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_pace_roll10 = float(opp_team["TEAM_PACE"].tail(10).mean().round(2))
    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    player_team_pace_roll10 = float(player_team_df["TEAM_PACE"].tail(10).mean().round(2))
    res.append(player_team_pace_roll10 - opp_pace_roll10)

    # Games Played
    res.append(len(pdf))

    return res
