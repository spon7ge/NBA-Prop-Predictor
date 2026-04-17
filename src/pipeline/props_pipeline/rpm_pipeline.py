import numpy as np
import pandas as pd
import joblib

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam


def rpm_pipeline(df, name, date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    res = []

    # REB_PER_MIN_season_avg
    reb_per_min_season_avg = pdf["REB_PER_MIN"].mean()
    res.append(float(reb_per_min_season_avg) if pd.notna(reb_per_min_season_avg) else float("nan"))

    # MEDIAN_REB_PER_MIN_L10 (rolling series → scalar for latest game)
    median_reb_per_min_l10 = (
        pdf["REB_PER_MIN"].tail(10).median().round(2)
    )
    res.append(float(median_reb_per_min_l10) if pd.notna(median_reb_per_min_l10) else float("nan"))

    # POSITION_ENC
    res.append(pdf['POSITION_ENC'].iloc[-1])

    # REB_PCT_roll10
    reb_pct_roll10 = pdf["REB_PCT"].astype(float).tail(10).mean().round(2)
    res.append(float(reb_pct_roll10) if pd.notna(reb_pct_roll10) else float("nan"))

    # REB_PER_MIN_roll10
    reb_per_min_roll10 = pdf["REB_PER_MIN"].astype(float).tail(10).mean().round(2)
    res.append(float(reb_per_min_roll10) if pd.notna(reb_per_min_roll10) else float("nan"))

    # OREB_PER_MIN_season_avg
    oreb_per_min_season_avg = pdf["OREB_PER_MIN"].mean()
    res.append(float(oreb_per_min_season_avg) if pd.notna(oreb_per_min_season_avg) else float("nan"))

    # DREB_PER_MIN_season_avg
    dreb_per_min_season_avg = pdf["DREB_PER_MIN"].mean()
    res.append(float(dreb_per_min_season_avg) if pd.notna(dreb_per_min_season_avg) else float("nan"))

    # TEAM_REB_PER_MIN_RANK_L10
    team_reb_per_min_rank_l10 = pdf["TEAM_REB_PER_MIN_RANK_L10"].iloc[-1]
    res.append(float(team_reb_per_min_rank_l10) if pd.notna(team_reb_per_min_rank_l10) else float("nan"))

    # OPP_DEF_RATING_roll10
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_def_rating_roll10 = float(opp_team["TEAM_DEF_RATING"].tail(10).mean().round(2))
    res.append(opp_def_rating_roll10)

    # DREB_PER_MIN_roll10
    dreb_per_min_roll10 = pdf["DREB_PER_MIN"].astype(float).tail(10).mean().round(2)
    res.append(float(dreb_per_min_roll10) if pd.notna(dreb_per_min_roll10) else float("nan"))
    
    return res