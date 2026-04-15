import numpy as np
import pandas as pd
import joblib

papm_bundle = joblib.load("src/models/saved_models/papm_quantile_xgb.joblib")
papm_quantile_models = papm_bundle["quantile_models"]
papm_feature_names = papm_bundle["feature_names"]
scaler = papm_bundle["scaler"]

CONTINUOUS_COLS = [    
    'PTS_PER_MIN_roll10',
    'AST_PER_MIN_roll10',
    'COMBO_TARGET_roll10',
    'USG_PCT_roll10',
    'TS_PCT_roll10',

    # Momentum (Rate of change - captures "streakiness")
    'PPM_MOMENTUM',
    'APM_MOMENTUM',
    'FGAPM_MOMENTUM',
    'USG_PCT_MOMENTUM',
    'TS_PCT_MOMENTUM',
    'COMBO_TARGET_MOMENTUM',

    # Min/Max
    'PPM_MIN_L10',
    'APM_MIN_L10',
    'PPM_MAX_L10',
    'APM_MAX_L10',
    'COMBO_TARGET_MIN_L10',
    'COMBO_TARGET_MAX_L10',
    'PTS_MAX_L10',
    'USG_PCT_MAX_L10',
    'TS_PCT_MAX_L10',
    'AST_MAX_L10',

    #interaction features
    'USG_PCT_X_TS_PCT',

    # Matchup Environment
    'OPP_PACE_roll10',       # Keep, drop OPP_POSS_roll10 (highly correlated)
    'OPP_DEF_RATING_roll10',
    'PACE_DIFFERENTIAL'
]

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam


def papm_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    res = []

    # STARTING
    team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)
    res.append(pdf['POSITION_ENC'].iloc[-1])
    res.append(pdf['IS_TOP_STAR'].iloc[-1])
    
    # ACTIVE_STARS_COUNT
    count = 0
    for star in team3StarsPerTeam[team]:
        if star in projectedStartingFive[team]:
            count += 1
    res.append(count)

    # TEAM RANKINGS
    res.append(pdf['TEAM_COMBO_TARGET_RANK_L10'].iloc[-1])
    res.append(pdf['TEAM_PTS_PER_MIN_RANK_L10'].iloc[-1])
    res.append(pdf['TEAM_AST_PER_MIN_RANK_L10'].iloc[-1])
    res.append(pdf['TEAM_MIN_RANK_L10'].iloc[-1])
    res.append(pdf['TEAM_TS_PCT_RANK_L10'].iloc[-1])

    # Rolling Averages
    res.append(pdf['PTS_PER_MIN'].tail(10).mean().round(2))
    res.append(pdf['AST_PER_MIN'].tail(10).mean().round(2))
    res.append(pdf['COMBO_TARGET'].tail(10).mean().round(2))
    res.append(pdf['USG_PCT'].tail(10).mean().round(2))
    res.append(pdf['TS_PCT'].tail(10).mean().round(2))
    
    # Momentum
    res.append(pdf['PTS_PER_MIN'].tail(5).mean().round(2) - pdf['PTS_PER_MIN'].tail(10).mean().round(2))
    res.append(pdf['AST_PER_MIN'].tail(5).mean().round(2) - pdf['AST_PER_MIN'].tail(10).mean().round(2))
    res.append(pdf['FGA_PER_MIN'].tail(5).mean().round(2) - pdf['FGA_PER_MIN'].tail(10).mean().round(2))
    res.append(pdf['USG_PCT'].tail(5).mean().round(2) - pdf['USG_PCT'].tail(10).mean().round(2))
    res.append(pdf['TS_PCT'].tail(5).mean().round(2) - pdf['TS_PCT'].tail(10).mean().round(2))
    res.append(pdf['COMBO_TARGET'].tail(5).mean().round(2) - pdf['COMBO_TARGET'].tail(10).mean().round(2))

    # Min/Max
    res.append(pdf['PTS_PER_MIN'].tail(10).min().round(2))
    res.append(pdf['AST_PER_MIN'].tail(10).min().round(2))
    res.append(pdf['PTS_PER_MIN'].tail(10).max().round(2))
    res.append(pdf['AST_PER_MIN'].tail(10).max().round(2))
    res.append(pdf['COMBO_TARGET'].tail(10).min().round(2))
    res.append(pdf['COMBO_TARGET'].tail(10).max().round(2))
    res.append(pdf['PTS'].tail(10).max().round(2))
    res.append(pdf['USG_PCT'].tail(10).max().round(2))
    res.append(pdf['TS_PCT'].tail(10).max().round(2))
    res.append(pdf['AST'].tail(10).max().round(2))

    # interaction features
    res.append(pdf['USG_PCT'].tail(10).mean().round(2) * pdf['TS_PCT'].tail(10).mean().round(2))

    # Matchup Environment
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    res.append(opp_team["TEAM_PACE"].tail(10).mean().round(2))
    res.append(opp_team["TEAM_DEF_RATING"].tail(10).mean().round(2))

    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    res.append(player_team_df["TEAM_PACE"].tail(10).mean().round(2) - opp_team["TEAM_PACE"].tail(10).mean().round(2))

    
    res_df = pd.DataFrame([res], columns=papm_feature_names)

    res_df[CONTINUOUS_COLS] = scaler.transform(res_df[CONTINUOUS_COLS])
    res_df = res_df.fillna(0.0)
    
    return res_df.values