import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp
from src.utils.team_info import projectedStartingFive


def rpm_pipeline(df, name, date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]

    # ── Opponent setup ────────────────────────────────────────────────────────
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])

    # ── REB_PER_MIN_season_avg ────────────────────────────────────────────────
    reb_per_min_season_avg = float(pdf["REB_PER_MIN"].mean())
    res.append(reb_per_min_season_avg if pd.notna(reb_per_min_season_avg) else float("nan"))

    # ── RBC_PER_MIN_10_ewm ────────────────────────────────────────────────────
    rbc_per_min_10_ewm = (
        float(pdf["RBC_PER_MIN"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
        if "RBC_PER_MIN" in pdf.columns else float("nan")
    )
    res.append(rbc_per_min_10_ewm if pd.notna(rbc_per_min_10_ewm) else float("nan"))

    # ── POSITION_ENC ──────────────────────────────────────────────────────────
    res.append(float(pdf["POSITION_ENCODED"].iloc[-1]))

    # ── REB_PER_MIN_roll5 ─────────────────────────────────────────────────────
    reb_per_min_roll5 = float(pdf["REB_PER_MIN"].astype(float).tail(5).mean())
    res.append(reb_per_min_roll5 if pd.notna(reb_per_min_roll5) else float("nan"))

    # ── REB_season_avg ────────────────────────────────────────────────────────
    reb_season_avg = float(pdf["REB"].astype(float).mean()) if "REB" in pdf.columns else float("nan")
    res.append(reb_season_avg if pd.notna(reb_season_avg) else float("nan"))

    # ── OPP_TEAM_REB_ALLOWED ──────────────────────────────────────────────────
    opp_team_reb_allowed = (
        float(opp_team["OPP_REB"].astype(float).mean())
        if "OPP_REB" in opp_team.columns else float("nan")
    )
    res.append(opp_team_reb_allowed if pd.notna(opp_team_reb_allowed) else float("nan"))

    # ── EXPECTED_PACE ─────────────────────────────────────────────────────────
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    team_pace = float(player_team_df["TEAM_PACE"].tail(10).mean()) if "TEAM_PACE" in player_team_df.columns else float("nan")
    opp_pace = float(opp_team["TEAM_PACE"].tail(10).mean()) if "TEAM_PACE" in opp_team.columns else float("nan")
    expected_pace = (team_pace + opp_pace) / 2 if pd.notna(team_pace) and pd.notna(opp_pace) else float("nan")
    res.append(expected_pace)

    # ── RPM_SEASON_STD ────────────────────────────────────────────────────────
    rpm_season_std = float(pdf["REB_PER_MIN"].astype(float).std())
    res.append(rpm_season_std if pd.notna(rpm_season_std) else float("nan"))

    # ── STARTING ──────────────────────────────────────────────────────────────
    projected = projectedStartingFive.get(player_team, [])
    starting = float(1 if (name in projected) else 0)
    res.append(starting)

    # ── IS_PLAYOFF ────────────────────────────────────────────────────────────
    is_playoff = float(last["IS_PLAYOFF"]) if "IS_PLAYOFF" in last.index else float("nan")
    res.append(is_playoff)
    
    return res
