import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp


def ppm_pipeline(df, name, current_date):
    pdf = df[df['PLAYER_NAME'] == name].sort_values('GAME_DATE').copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]

    # ── Opponent stats ────────────────────────────────────────────────────────
    opp_abbr, _ = findOpp(name, pdf, current_date, max_days_ahead=3)
    opp_team = df[df["TEAM_ABBREVIATION"] == opp_abbr].sort_values("GAME_DATE")
    opp_team = opp_team.drop_duplicates(subset=["TEAM_ID", "GAME_ID"])
    opp_def_rating = float(opp_team["DEF_RATING"].astype(float).mean()) if "DEF_RATING" in opp_team.columns else float("nan")
    opp_pts_allowed = float(opp_team["OPP_PTS"].astype(float).mean()) if "OPP_PTS" in opp_team.columns else float("nan")

    # ── Player per-minute EWMs (span=10) ──────────────────────────────────────
    def _ewm10_last(col: str) -> float:
        return float(pdf[col].astype(float).ewm(span=10, adjust=False).mean().iloc[-1])

    ufga_per_min_10_ewm = _ewm10_last("UFGA_PER_MIN") if "UFGA_PER_MIN" in pdf.columns else float("nan")
    cfga_per_min_10_ewm = _ewm10_last("CFGA_PER_MIN") if "CFGA_PER_MIN" in pdf.columns else float("nan")

    # ── UFGA_PER_MIN_X_OPP_DEF_RATING ────────────────────────────────────────
    res.append(ufga_per_min_10_ewm * opp_def_rating)

    # ── CFGA_PER_MIN_X_OPP_DEF_RATING ────────────────────────────────────────
    res.append(cfga_per_min_10_ewm * opp_def_rating)

    # ── PPM_SEASON_MEAN ───────────────────────────────────────────────────────
    ppm_series = pdf["PTS_PER_MIN"].astype(float)
    ppm_season_mean = float(ppm_series.mean())
    res.append(ppm_season_mean if pd.notna(ppm_season_mean) else float("nan"))

    # ── TEAM_USG_RANK_L10 ─────────────────────────────────────────────────────
    gameday = df[(df["TEAM_ID"] == last["TEAM_ID"]) & (df["GAME_DATE"] == last["GAME_DATE"])]
    usg_rank_l10 = gameday["USG_PCT_roll10"].rank(ascending=False, method="dense")
    team_usg_rank_l10 = float(usg_rank_l10[gameday["PLAYER_NAME"] == name].iloc[0])
    res.append(team_usg_rank_l10 if pd.notna(team_usg_rank_l10) else float("nan"))

    # ── PTS_PER_MIN_X_OPP_PTS_ALLOWED ────────────────────────────────────────
    pts_per_min_10_ewm = _ewm10_last("PTS_PER_MIN")
    res.append(pts_per_min_10_ewm * opp_pts_allowed)

    # ── TS_PCT_X_USG_PCT ──────────────────────────────────────────────────────
    ts_pct_season_avg = float(pdf["TS_PCT"].astype(float).mean())
    usg_pct_season_avg = float(pdf["USG_PCT"].astype(float).mean())
    res.append(ts_pct_season_avg * usg_pct_season_avg)

    # ── FT_PCT_season_avg ─────────────────────────────────────────────────────
    ft_pct_season_avg = float(pdf["FT_PCT"].astype(float).mean()) if "FT_PCT" in pdf.columns else float("nan")
    res.append(ft_pct_season_avg if pd.notna(ft_pct_season_avg) else float("nan"))

    # ── PTS_season_avg ────────────────────────────────────────────────────────
    pts_season_avg = float(pdf["PTS"].astype(float).mean()) if "PTS" in pdf.columns else float("nan")
    res.append(pts_season_avg if pd.notna(pts_season_avg) else float("nan"))

    return res
