import warnings

import numpy as np
import pandas as pd
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam
from src.utils.helper_functions import findOpp


def min_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    player_team = last["TEAM_ABBREVIATION"]

    # ── STARTING ──────────────────────────────────────────────────────────────
    canon_name = nameDict.get(name, name)
    projected = projectedStartingFive.get(player_team, [])
    starting_override = float(1 if (canon_name in projected or name in projected) else 0)
    res.append(starting_override)

    # ── STARTING_lag1 ─────────────────────────────────────────────────────────
    starting_lag1 = float(pdf["STARTING"].iloc[-1])
    res.append(starting_lag1)

    # ── STAR_OUT_FLAG ─────────────────────────────────────────────────────────
    stars = team3StarsPerTeam.get(player_team, [])
    top_star = stars[0] if len(stars) > 0 else None
    if top_star is None:
        star_out = 0
    else:
        top_star_canon = nameDict.get(top_star, top_star)
        in_lineup = (top_star in projected) or (top_star_canon in projected)
        star_out = int(not in_lineup)
    res.append(float(star_out))

    # ── TEAM_MIN_RANK_L10 ─────────────────────────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    min_rank_l10 = gameday["MIN_roll10"].rank(ascending=False, method="dense")
    team_min_rank_l10 = float(min_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    # ── MIN_10_ewm ────────────────────────────────────────────────────────────
    min_series = pdf["MIN"].astype(float)
    min_10_ewm = float(min_series.ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(min_10_ewm if pd.notna(min_10_ewm) else float("nan"))

    # ── MIN_lag1 ──────────────────────────────────────────────────────────────
    min_lag1 = float(pdf["MIN"].iloc[-1])
    res.append(min_lag1)

    # ── MIN_TREND ─────────────────────────────────────────────────────────────
    min_5_ewm = float(min_series.ewm(span=5, adjust=False).mean().iloc[-1])
    min_trend = min_5_ewm - min_10_ewm
    res.append(min_trend if pd.notna(min_trend) else float("nan"))

    # ── ROLE_LOCK ─────────────────────────────────────────────────────────────
    starter_roll10_pct = float(pdf["STARTING"].tail(10).mean())
    min_std10 = pdf["MIN"].tail(10).std()
    if pd.notna(min_std10) and pd.notna(min_10_ewm) and min_10_ewm != 0:
        role_lock = float(starter_roll10_pct) * float(np.exp(-float(min_std10) / float(min_10_ewm)))
    else:
        role_lock = float("nan")
    res.append(role_lock)

    # ── TEAM_USG_RANK_L10 ─────────────────────────────────────────────────────
    usg_rank_l10 = gameday["USG_PCT_roll10"].rank(ascending=False, method="dense")
    team_usg_rank_l10 = float(usg_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l10 if pd.notna(team_usg_rank_l10) else float("nan"))

    # ── MIN_ROLE_Z_SCORE ──────────────────────────────────────────────────────
    season_mean = float(pdf["MIN"].mean())
    season_std = float(pdf["MIN"].std())
    min_roll10 = float(pdf["MIN"].tail(10).mean())
    if pd.isna(season_std) or season_std == 0:
        min_role_z_score = float("nan")
    else:
        min_role_z_score = (min_roll10 - season_mean) / season_std
    res.append(min_role_z_score)

    # ── POSS_10_ewm ───────────────────────────────────────────────────────────
    poss_10_ewm = (
        float(pdf["POSS"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
        if "POSS" in pdf.columns
        else float("nan")
    )
    res.append(poss_10_ewm if pd.notna(poss_10_ewm) else float("nan"))

    # ── MIN_MAX_L10 ───────────────────────────────────────────────────────────
    res.append(float(pdf["MIN"].tail(10).max()))

    # ── MIN_MIN_L10 ───────────────────────────────────────────────────────────
    res.append(float(pdf["MIN"].tail(10).min()))

    # ── MIN_SEASON_STD ────────────────────────────────────────────────────────
    res.append(float(season_std) if pd.notna(season_std) else float("nan"))

    return res