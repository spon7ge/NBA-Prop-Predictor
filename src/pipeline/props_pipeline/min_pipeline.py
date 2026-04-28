import numpy as np
import pandas as pd
from src.utils.team_info import nameDict, projectedStartingFive


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

    # ── MIN_10_ewm ────────────────────────────────────────────────────────────
    min_series = pdf["MIN"].astype(float)
    min_10_ewm = float(min_series.ewm(span=10, adjust=False).mean().iloc[-1])
    res.append(min_10_ewm if pd.notna(min_10_ewm) else float("nan"))

    # ── TEAM_MIN_RANK_L10 ─────────────────────────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    min_rank_l10 = gameday["MIN_roll10"].rank(ascending=False, method="dense")
    team_min_rank_l10 = float(min_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    # ── STARTER_ROLL10_PCT ────────────────────────────────────────────────────
    starter_roll10_pct = float(pdf["STARTING"].tail(10).mean())
    res.append(starter_roll10_pct if pd.notna(starter_roll10_pct) else float("nan"))

    # ── MIN_SEASON_MEAN ───────────────────────────────────────────────────────
    season_mean = float(pdf["MIN"].mean())
    res.append(season_mean if pd.notna(season_mean) else float("nan"))

    # ── MIN_lag1 ──────────────────────────────────────────────────────────────
    min_lag1 = float(pdf["MIN"].iloc[-1])
    res.append(min_lag1)

    # ── MIN_SEASON_STD ────────────────────────────────────────────────────────
    season_std = float(pdf["MIN"].std())
    res.append(season_std if pd.notna(season_std) else float("nan"))

    # ── MIN_MAX_L10 ───────────────────────────────────────────────────────────
    res.append(float(pdf["MIN"].tail(10).max()))

    # ── IS_PLAYOFF ────────────────────────────────────────────────────────────
    is_playoff = float(last["IS_PLAYOFF"]) if "IS_PLAYOFF" in last.index else float("nan")
    res.append(is_playoff)

    return res
