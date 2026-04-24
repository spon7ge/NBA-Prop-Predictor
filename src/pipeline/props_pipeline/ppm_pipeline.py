import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp
from src.utils.team_info import nameDict, projectedStartingFive, team3StarsPerTeam


def ppm_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]

    # ── PLAYER_ENCODED ────────────────────────────────────────────────────────
    res.append(int(pdf['PLAYER_ENCODED'].iloc[-1]))

    # ── POSITION_ENCODED ──────────────────────────────────────────────────────
    res.append(int(pdf['POSITION_ENCODED'].iloc[-1]))

    # ── TEAM_MIN_RANK_L10 / TEAM_USG_RANK_L10 ─────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]

    min_rank_l10 = gameday["MIN_roll10"].rank(ascending=False, method="dense")
    team_min_rank_l10 = float(min_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_min_rank_l10 if pd.notna(team_min_rank_l10) else float("nan"))

    usg_rank_l10 = gameday["USG_PCT_roll10"].rank(ascending=False, method="dense")
    team_usg_rank_l10 = float(usg_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l10 if pd.notna(team_usg_rank_l10) else float("nan"))

    # ── Season scalars ────────────────────────────────────────────────────────
    ppm_series = pdf["PTS_PER_MIN"].astype(float)
    ppm_season_std = float(ppm_series.std())
    ppm_season_avg = float(ppm_series.mean())

    # ── PPM_SEASON_STD ────────────────────────────────────────────────────────
    res.append(ppm_season_std if pd.notna(ppm_season_std) else float("nan"))

    # ── PTS_PER_MIN_season_avg ────────────────────────────────────────────────
    res.append(ppm_season_avg if pd.notna(ppm_season_avg) else float("nan"))

    # ── PPM_ROLE_Z_SCORE ──────────────────────────────────────────────────────
    ppm_10_ewm = float(ppm_series.ewm(span=10, adjust=False).mean().iloc[-1])
    if pd.isna(ppm_season_std) or ppm_season_std <= 0:
        ppm_role_z = 0.0
    else:
        ppm_role_z = (ppm_10_ewm - ppm_season_avg) / ppm_season_std
    res.append(ppm_role_z)

    # ── MIN_TREND (per training: MIN_5_ewm - 3PM_PER_MIN_10_ewm) ──────────────
    min_5_ewm = float(pdf["MIN"].astype(float).ewm(span=5, adjust=False).mean().iloc[-1])
    tpm_10_ewm = float(pdf["3PM_PER_MIN"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])
    min_trend = min_5_ewm - tpm_10_ewm
    res.append(min_trend if pd.notna(min_trend) else float("nan"))

    # ── USG_TREND (USG_PCT_5_ewm - USG_PCT_10_ewm) ────────────────────────────
    usg_series = pdf["USG_PCT"].astype(float)
    usg_5_ewm = float(usg_series.ewm(span=5, adjust=False).mean().iloc[-1])
    usg_10_ewm = float(usg_series.ewm(span=10, adjust=False).mean().iloc[-1])
    usg_trend = usg_5_ewm - usg_10_ewm
    res.append(usg_trend if pd.notna(usg_trend) else float("nan"))

    # ── TS_PCT_X_USG_PCT ──────────────────────────────────────────────────────
    ts_pct_season_avg = float(pdf["TS_PCT"].astype(float).mean())
    usg_pct_season_avg = float(usg_series.mean())
    res.append(ts_pct_season_avg * usg_pct_season_avg)

    # ── Opponent-side allowed stats ───────────────────────────────────────────
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])

    def _opp_allowed_pct(num_col: str, den_col: str) -> float:
        num = float(opp_team[num_col].astype(float).sum())
        den = float(opp_team[den_col].astype(float).sum())
        return (num / den) if den > 0 else float("nan")

    opp_fg_pct_allowed = _opp_allowed_pct("OPP_FGM", "OPP_FGA")
    opp_fg3_pct_allowed = _opp_allowed_pct("OPP_FG3M", "OPP_FG3A")
    opp_pts_allowed = float(opp_team["OPP_PTS"].astype(float).mean())
    opp_pfd_allowed = float(opp_team["OPP_PFD"].astype(float).mean())

    # ── Player per-minute EWMs (span=10) ──────────────────────────────────────
    def _ewm10_last(col: str) -> float:
        return float(pdf[col].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])

    pts_per_min_10_ewm = _ewm10_last("PTS_PER_MIN")
    fga_per_min_10_ewm = _ewm10_last("FGA_PER_MIN")
    fgm_per_min_10_ewm = _ewm10_last("FGM_PER_MIN")
    fta_per_min_10_ewm = _ewm10_last("FTA_PER_MIN")
    tpa_per_min_10_ewm = _ewm10_last("3PA_PER_MIN")
    tpm_per_min_10_ewm = tpm_10_ewm

    # ── Interactions (order matches PPM_FEATURES) ─────────────────────────────
    res.append(pts_per_min_10_ewm * opp_pts_allowed)     # PTS_PER_MIN_X_OPP_PTS_ALLOWED
    res.append(fga_per_min_10_ewm * opp_fg_pct_allowed)  # FGA_PER_MIN_X_OPP_FG%_ALLOWED
    res.append(fgm_per_min_10_ewm * opp_fg_pct_allowed)  # FGM_PER_MIN_X_OPP_FG%_ALLOWED
    res.append(fta_per_min_10_ewm * opp_pfd_allowed)     # FTA_PER_MIN_X_OPP_PFD_ALLOWED
    res.append(tpa_per_min_10_ewm * opp_fg3_pct_allowed) # 3PA_PER_MIN_X_OPP_FG3%_ALLOWED
    res.append(tpm_per_min_10_ewm * opp_fg3_pct_allowed) # 3PM_PER_MIN_X_OPP_FG3%_ALLOWED

    # ── PACE_DIFFERENTIAL (team_pace_roll10 - opp_pace_roll10) ────────────────
    player_team = last["TEAM_ABBREVIATION"]
    player_team_df = df[df["TEAM_ABBREVIATION"] == player_team].sort_values("GAME_DATE")
    player_team_df = player_team_df.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    team_pace_roll10 = float(player_team_df["TEAM_PACE"].tail(10).mean())
    opp_pace_roll10 = float(opp_team["TEAM_PACE"].tail(10).mean())
    res.append(team_pace_roll10 - opp_pace_roll10)

    return res
