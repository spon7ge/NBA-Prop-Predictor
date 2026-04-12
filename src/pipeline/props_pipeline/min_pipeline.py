import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import joblib
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam
from src.utils.helper_functions import findOpp

min_bundle = joblib.load("src/models/saved_models/min_quantile_xgb.joblib")
min_quantile_models = min_bundle["quantile_models"]
min_feature_names = min_bundle["feature_names"]
scaler = min_bundle.get("scaler")


CONTINUOUS_COLS = [
    'MIN_season_avg', 'MIN_EWM_L10', 'MIN_TREND', 'STARTER_ROLL10_PCT', 
    'MIN_ROLE_Z_SCORE', 'ROLE_LOCK', 'MIN_share_proxy', 'ROTATION_GAP_L10', 
    'PACE_DIFFERENTIAL'
]

def min_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    pid = int(pdf["PLAYER_ID"].iloc[-1])
    player_pos = pdf["pos"].iloc[-1]
    res = []
    last = pdf.iloc[-1]

    # STARTER_X_MIN_AVG: starting flag × career avg minutes
    min_avg = float(pdf["MIN"].mean())
    if name in projectedStartingFive[last["TEAM_ABBREVIATION"]]:
        starting_override = 1
    else:
        starting_override = 0
    res.append(float(starting_override))
    res.append(min_avg)

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
        min_role_z_score = float("nan")
    else:
        min_role_z_score = (recent_mean - season_mean) / season_std
    res.append(min_role_z_score)

    # ROLE_LOCK
    roll10_std = pdf["MIN"].tail(10).std()
    role_lock = (
        float(starter_roll10_pct) / float(roll10_std)
        if pd.notna(roll10_std) and roll10_std != 0
        else float("nan")
    )
    res.append(float(role_lock) if pd.notna(role_lock) else float("nan"))

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

    # IS_TOP_STAR
    res.append(pdf['IS_TOP_STAR'].iloc[-1])
    
    # ACTIVE_STARS_COUNT
    count = 0
    for star in team3StarsPerTeam[player_team]:
        if star in projectedStartingFive[player_team]:
            count += 1
    res.append(count)


    res_df = pd.DataFrame([res], columns=min_feature_names)
    
    # 3. Scale continuous columns (bundle must include scaler from training notebook)
    if scaler is not None:
        res_df[CONTINUOUS_COLS] = scaler.transform(res_df[CONTINUOUS_COLS])
    res_df = res_df.fillna(0.0)
    
    return res_df.values
