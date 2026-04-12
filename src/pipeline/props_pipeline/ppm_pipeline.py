import numpy as np
import pandas as pd
import joblib

ppm_bundle = joblib.load("src/models/saved_models/ppm_quantile_xgb.joblib")
ppm_quantile_models = ppm_bundle["quantile_models"]
ppm_feature_names = ppm_bundle["feature_names"]
scaler = ppm_bundle["scaler"]

CONTINUOUS_COLS = [
    'PTS_PER_MIN_season_avg', 'PTS_PER_MIN_10_ewm', 'PPM_MOMENTUM', 'PTS_PER_POSS_roll10', 
    'USG_PCT_roll10', '3PA_PER_MIN_10_ewm', 'FTA_PER_MIN_10_ewm', 
    'POSS_DIFF_L10', 'TS_PCT_DELTA', 'TRUE_USG_L10', 'TS_PCT_roll10', 'OPP_DEF_RATING_roll10',
    'PACE_DIFFERENTIAL'
]

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam


def ppm_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    res = []

    # STARTING
    team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)
    
    # PTS_PER_MIN_season_avg
    pts_per_min_season_avg = pdf["PTS_PER_MIN"].mean()
    res.append(float(pts_per_min_season_avg) if pd.notna(pts_per_min_season_avg) else float("nan"))

    # PPM Momentum
    ppm_ewm10 = pdf["PTS_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(ppm_ewm10) if pd.notna(ppm_ewm10) else float("nan"))
    ppm_ewm5 = pdf["PTS_PER_MIN"].astype(float).ewm(span=5).mean().iloc[-1]
    res.append(ppm_ewm5 - ppm_ewm10)
    points = pdf['PTS'].tail(10).mean().round(2)
    poss = pdf['POSS'].tail(10).mean().round(2)
    res.append(points / poss if pd.notna(points / poss) else float("nan"))
    team_pts_per_min_rank_l10 = pdf["TEAM_PTS_PER_MIN_RANK_L10"].iloc[-1]
    res.append(float(team_pts_per_min_rank_l10) if pd.notna(team_pts_per_min_rank_l10) else float("nan"))

    # USG_PCT_roll10 and 3PA_PER_MIN_10_ewm
    usg_pct_roll10 = pdf["USG_PCT"].astype(float).tail(10).mean().round(2)
    res.append(float(usg_pct_roll10) if pd.notna(usg_pct_roll10) else float("nan"))
    three_pa_per_min_10_ewm = pdf["3PA_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(three_pa_per_min_10_ewm) if pd.notna(three_pa_per_min_10_ewm) else float("nan"))
    fta_per_min_10_ewm = pdf["FTA_PER_MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(fta_per_min_10_ewm) if pd.notna(fta_per_min_10_ewm) else float("nan"))

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

    # POSITION_ENC
    res.append(pdf['POSITION_ENC'].iloc[-1])

    # IS_TOP_STAR
    res.append(pdf['IS_TOP_STAR'].iloc[-1])
    
    # ACTIVE_STARS_COUNT
    count = 0
    for star in team3StarsPerTeam[player_team]:
        if star in projectedStartingFive[player_team]:
            count += 1
    res.append(count)
    
    res_df = pd.DataFrame([res], columns=ppm_feature_names)
    
    # 3. Scale continuous columns
    res_df[CONTINUOUS_COLS] = scaler.transform(res_df[CONTINUOUS_COLS])
    res_df = res_df.fillna(0.0)
    
    return res_df.values