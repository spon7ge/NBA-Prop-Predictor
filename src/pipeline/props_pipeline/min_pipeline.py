import warnings

import numpy as np
import pandas as pd
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam
from src.utils.helper_functions import findOpp


def min_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    pid = int(pdf["PLAYER_ID"].iloc[-1])
    player_pos = pdf["pos"].iloc[-1]
    res = []
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]

    # ── GP ───────────────────────────────────────────────────────────────────
    res.append(len(pdf))

    # ── DAYS_REST ─────────────────────────────────────────────────────────────
    if len(pdf) >= 2:
        last_date = pd.Timestamp(pdf["GAME_DATE"].iloc[-1]).normalize()
        current = pd.Timestamp(current_date).normalize()
        days_rest = int((current - last_date).days)
    elif days_rest > 3:
        days_rest = 3
    res.append(float(days_rest))

    # ── POSITION_ENC ──────────────────────────────────────────────────────────
    res.append(pdf['POSITION_ENC'].iloc[-1])

    # ── STARTING ──────────────────────────────────────────────────────────────
    if name in projectedStartingFive[player_team]:
        starting_override = 1
    else:
        starting_override = 0
    res.append(float(starting_override))

    # ── STARTER_X_MIN_AVG ─────────────────────────────────────────────────────
    min_avg = float(pdf["MIN"].mean())
    res.append(float(starting_override) * min_avg)

    # ── STARTING_lag1 ─────────────────────────────────────────────────────────
    starting_lag1 = float(pdf["STARTING"].iloc[-1]) if len(pdf) >= 1 else float("nan")
    res.append(starting_lag1)

    # ── STARTER_ROLL10_PCT ────────────────────────────────────────────────────
    starter_roll10_pct = float(pdf["STARTING"].tail(10).mean())
    res.append(starter_roll10_pct if pd.notna(starter_roll10_pct) else float("nan"))

    # ── TEAM_MIN_RANK_L10 ─────────────────────────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    min_rank_l10 = gameday["MIN_roll10"].rank(ascending=False, method="dense")
    team_min_rank_l10 = float(min_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    # ── MIN_EWM_L10 ───────────────────────────────────────────────────────────
    min_ewm10 = float(pdf["MIN"].astype(float).ewm(span=10).mean().iloc[-1])
    res.append(min_ewm10 if pd.notna(min_ewm10) else float("nan"))

    # ── MIN_EWM_L3 ────────────────────────────────────────────────────────────
    min_ewm3 = float(pdf["MIN"].astype(float).ewm(span=3).mean().iloc[-1])
    res.append(min_ewm3 if pd.notna(min_ewm3) else float("nan"))

    # ── MIN_lag1 ──────────────────────────────────────────────────────────────
    min_lag1 = float(pdf["MIN"].iloc[-1]) if len(pdf) >= 1 else float("nan")
    res.append(min_lag1)

    # ── ROLE_LOCK ─────────────────────────────────────────────────────────────
    roll10_std = pdf["MIN"].tail(10).std()
    role_lock = (
        float(starter_roll10_pct) / float(roll10_std)
        if pd.notna(roll10_std) and roll10_std != 0
        else float("nan")
    )
    res.append(float(role_lock) if pd.notna(role_lock) else float("nan"))

    # ── ACTIVE_STARS_COUNT ────────────────────────────────────────────────────
    count = 0
    for star in team3StarsPerTeam[player_team]:
        if star in projectedStartingFive[player_team]:
            count += 1
    res.append(count)

    # ── TEAM_USG_RANK_L10 ─────────────────────────────────────────────────────
    usg_rank_l10 = gameday["USG_PCT_roll10"].rank(ascending=False, method="dense")
    team_usg_rank_l10 = float(usg_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l10 if pd.notna(team_usg_rank_l10) else float("nan"))

    # ── MIN_ROLE_Z_SCORE ──────────────────────────────────────────────────────
    season_mean = float(pdf["MIN"].expanding().mean().iloc[-1])
    season_std  = float(pdf["MIN"].expanding().std().iloc[-1])
    recent_mean = float(pdf["MIN"].tail(10).mean())
    if pd.isna(season_std) or season_std == 0:
        min_role_z_score = float("nan")
    else:
        min_role_z_score = (recent_mean - season_mean) / season_std
    res.append(min_role_z_score)

    # ── POSS_season_avg ───────────────────────────────────────────────────────
    poss_season_avg = float(pdf["POSS"].expanding().mean().iloc[-1]) if "POSS" in pdf.columns else float("nan")
    res.append(poss_season_avg)

    # ── MIN_MAX_L10 ───────────────────────────────────────────────────────────
    res.append(float(pdf["MIN"].tail(10).max()))

    # ── MIN_MIN_L10 ───────────────────────────────────────────────────────────
    res.append(float(pdf["MIN"].tail(10).min()))

    # ── SEASON_STD ────────────────────────────────────────────────────────────
    res.append(float(season_std) if pd.notna(season_std) else float("nan"))

    return res