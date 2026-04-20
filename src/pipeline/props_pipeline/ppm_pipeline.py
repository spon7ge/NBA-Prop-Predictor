import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam


def ppm_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]
    res = []

    # ── PTS_PER_MIN_10_ewm ────────────────────────────────────────────────────
    ppm_10_ewm = float(pdf["PTS_PER_MIN"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(ppm_10_ewm if pd.notna(ppm_10_ewm) else float("nan"))

    # ── FGA_PER_MIN_10_ewm ────────────────────────────────────────────────────
    fga_per_min_10_ewm = float(pdf["FGA_PER_MIN"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(fga_per_min_10_ewm if pd.notna(fga_per_min_10_ewm) else float("nan"))

    # ── FTA_PER_MIN_10_ewm ────────────────────────────────────────────────────
    fta_per_min_10_ewm = float(pdf["FTA_PER_MIN"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(fta_per_min_10_ewm if pd.notna(fta_per_min_10_ewm) else float("nan"))

    # ── USG_PCT_10_ewm ────────────────────────────────────────────────────────
    usg_pct_10_ewm = float(pdf["USG_PCT"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(usg_pct_10_ewm if pd.notna(usg_pct_10_ewm) else float("nan"))

    # ── ACTIVE_STARS_COUNT ────────────────────────────────────────────────────
    count = 0
    for star in team3StarsPerTeam[player_team]:
        if star in projectedStartingFive[player_team]:
            count += 1
    res.append(count)

    # ── TEAM_MIN_RANK_L10 ─────────────────────────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    min_rank_l10 = gameday["MIN_roll10"].rank(ascending=False, method="dense")
    team_min_rank_l10 = float(min_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    # ── OPP_POSS_roll10 ───────────────────────────────────────────────────────
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_poss_roll10 = float(opp_team["TEAM_POSS"].tail(10).mean().round(2))
    res.append(opp_poss_roll10 if pd.notna(opp_poss_roll10) else float("nan"))

    # ── Season averages (for interactions) ────────────────────────────────────
    ppm_season_avg     = float(pdf["PTS_PER_MIN"].expanding().mean().iloc[-1])
    fga_season_avg     = float(pdf["FGA_PER_MIN"].expanding().mean().iloc[-1])
    fta_season_avg     = float(pdf["FTA_PER_MIN"].expanding().mean().iloc[-1])
    tpa_season_avg     = float(pdf["3PA_PER_MIN"].expanding().mean().iloc[-1])
    ts_pct_season_avg  = float(pdf["TS_PCT"].expanding().mean().iloc[-1])
    usg_pct_season_avg = float(pdf["USG_PCT"].expanding().mean().iloc[-1])
    min_10_ewm         = float(pdf["MIN"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])

    opp_def_rating_roll10 = float(opp_team["TEAM_DEF_RATING"].tail(10).mean().round(2))
    # Historically IS_TOP_STAR meant any of the top-3; use IS_STAR_TRIO to match trained models.
    is_star_trio = float(last.get("IS_STAR_TRIO", last.get("IS_TOP_STAR", 0)))

    # ── Interactions ──────────────────────────────────────────────────────────
    res.append(is_star_trio * opp_poss_roll10)                  # STAR_TRIO_X_OPP_POSS_L10
    res.append(ppm_season_avg * opp_def_rating_roll10)           # PPM_SEASON_AVG_X_OPP_DRTG_L10
    res.append(fga_season_avg * opp_def_rating_roll10)           # FGA_PER_MIN_X_OPP_DRTG_L10
    res.append(fta_season_avg * opp_def_rating_roll10)           # FTA_PER_MIN_X_OPP_DRTG_L10
    res.append(tpa_season_avg * opp_def_rating_roll10)           # 3PA_PER_MIN_X_OPP_DRTG_L10
    res.append(count * usg_pct_10_ewm)                           # ACTIVE_STARS_X_USG_PCT_L10
    res.append(ts_pct_season_avg * usg_pct_season_avg)           # TS_PCT_X_USG_PCT
    res.append(ppm_season_avg * min_10_ewm)                      # PPM_SEASON_AVG_X_MIN_L10

    return res
