import numpy as np
import pandas as pd
import joblib

apm_bundle = joblib.load("src/models/saved_models/apm_quantile_xgb.joblib")
apm_quantile_models = apm_bundle["quantile_models"]
apm_feature_names = apm_bundle["feature_names"]
scaler = apm_bundle.get("scaler")

CONTINUOUS_COLS = [
    "MIN_ewm10",
    "AST_PER_MIN_season_avg",
    "AST_PER_MIN_ewm10",
    "AST_share_proxy_roll10",
    "AST_PCT_roll10",
    "TOV_PER_MIN_ewm10",
    "AST_TO_roll10",
    "OPP_DEF_RATING_roll10",
    "OPP_PACE_roll10",
    "PACE_DIFFERENTIAL",
    "TEAM_AST_roll10",
]

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam


def apm_pipeline(df, name, date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    res = []

    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])

    # MIN_ewm10
    min_10_ewm = pdf["MIN"].astype(float).ewm(span=10).mean().iloc[-1]
    res.append(float(min_10_ewm) if pd.notna(min_10_ewm) else float("nan"))

    # AST_PER_MIN_season_avg
    ast_per_min_season_avg = pdf["AST_PER_MIN"].mean()
    res.append(float(ast_per_min_season_avg) if pd.notna(ast_per_min_season_avg) else float("nan"))

    # AST_PER_MIN_ewm10
    ast_per_min_ewm10 = pdf["AST_PER_MIN"].ewm(span=10).mean().iloc[-1]
    res.append(float(ast_per_min_ewm10) if pd.notna(ast_per_min_ewm10) else float("nan"))

    # AST_share_proxy_roll10
    team_ast_roll10_sum = player_team_df['TEAM_AST'].tail(10).mean()
    player_ast_roll10 = float(pdf["AST"].tail(10).mean())
    ast_share_proxy_roll10 = player_ast_roll10 / team_ast_roll10_sum
    res.append(ast_share_proxy_roll10 if pd.notna(ast_share_proxy_roll10) else float("nan"))

    #AST_PER_MIN RANK_L10
    res.append(pdf['TEAM_AST_PER_MIN_RANK_L10'].iloc[-1])

    # AST_PCT_roll10
    ast_pct_roll10 = float(pdf["AST_PCT"].tail(10).mean())
    res.append(ast_pct_roll10 if pd.notna(ast_pct_roll10) else float("nan"))

    # TOV_PER_MIN_ewm10
    tov_per_min_ewm10 = pdf["TOV_PER_MIN"].ewm(span=10).mean().iloc[-1]
    res.append(float(tov_per_min_ewm10) if pd.notna(tov_per_min_ewm10) else float("nan"))

    # AST_TO_roll10
    ast_to_roll10 = float(pdf["AST_TO"].tail(10).mean())
    res.append(ast_to_roll10 if pd.notna(ast_to_roll10) else float("nan"))

    # OPP_PACE_roll10, OPP_DEF_RATING_roll10
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])

    opp_pace_roll10 = float(opp_team["TEAM_PACE"].tail(10).mean())
    res.append(opp_pace_roll10 if pd.notna(opp_pace_roll10) else float("nan"))

    opp_def_rating_roll10 = float(opp_team["TEAM_DEF_RATING"].tail(10).mean())
    res.append(opp_def_rating_roll10 if pd.notna(opp_def_rating_roll10) else float("nan"))

    player_team = pdf["TEAM_ABBREVIATION"].iloc[-1]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    player_team_pace_roll10 = float(player_team_df["TEAM_PACE"].tail(10).mean())
    # PACE_DIFFERENTIAL
    res.append(player_team_pace_roll10 - opp_pace_roll10)

    # TEAM_AST_roll10
    player_team_ast_roll10 = float(player_team_df["TEAM_AST"].tail(10).mean())
    res.append(player_team_ast_roll10 if pd.notna(player_team_ast_roll10) else float("nan"))

    # team usg rank l10
    res.append(pdf['TEAM_USG_RANK_L10'].iloc[-1])

    # Active stars count
    count = 0
    for star in team3StarsPerTeam[player_team]:
        if star in projectedStartingFive[player_team]:
            count += 1
    res.append(count)

    # POSITION_ENC
    res.append(pdf['POSITION_ENC'].iloc[-1])

    # STARTING
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(player_team, [])
    starting_flag = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_flag)

    return res
