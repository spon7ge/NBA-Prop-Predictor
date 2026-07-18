"""NBA points-per-minute live features → ``ppm_nba_model_*.joblib``."""

from __future__ import annotations

import pandas as pd

from src.live_pipeline.common import (
    ensure_per_min,
    ewm_hl,
    ordered_features,
    player_history,
    season_avg,
    team_stat_ewm,
)
from src.utils.helper_functions import findOpp

# Exact order from ppm_nba_model_2026-04-12.joblib
FEATURE_COLS = [
    "opp_def_rating_ewm_hl10",
    "adv_poss_ewm_hl5",
    "track_ufga_per_min_ewm_hl10",
    "opp_pace_ewm_hl10",
    "team_pace_ewm_hl10",
    "base_fta_per_min_ewm_hl10",
    "base_fga_per_min_ewm_hl10",
    "base_pts_per_min_ewm_hl5",
    "base_fg3a_per_min_ewm_hl10",
    "ts_pct_x_usg_pct",
    "track_tchs_per_min_ewm_hl10",
    "track_cfga_per_min_ewm_hl10",
    "base_pts_per_min_season_avg",
]


def ppm_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "nba"):
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_per_min(
        pdf, ["pts", "fga", "fg3a", "fta", "tchs", "cfga", "ufga"]
    )
    team_abbr = (
        pdf["team_abbreviation"].iloc[-1]
        if "team_abbreviation" in pdf.columns
        else None
    )
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3, league=league)

    ts_season = season_avg(pdf, "ts_pct")
    usg_season = season_avg(pdf, "usg_pct")
    ts_x_usg = (
        float(ts_season * usg_season)
        if pd.notna(ts_season) and pd.notna(usg_season)
        else float("nan")
    )

    values = {
        "opp_def_rating_ewm_hl10": team_stat_ewm(
            df, opp_abbr, "team_def_rating", date, halflife=10
        ),
        "adv_poss_ewm_hl5": ewm_hl(pdf["poss"], 5) if "poss" in pdf.columns else float("nan"),
        "track_ufga_per_min_ewm_hl10": (
            ewm_hl(pdf["ufga_per_min"], 10) if "ufga_per_min" in pdf.columns else float("nan")
        ),
        "opp_pace_ewm_hl10": team_stat_ewm(
            df, opp_abbr, "team_pace", date, halflife=10
        ),
        "team_pace_ewm_hl10": team_stat_ewm(
            df, team_abbr, "team_pace", date, halflife=10
        ),
        "base_fta_per_min_ewm_hl10": ewm_hl(pdf["fta_per_min"], 10),
        "base_fga_per_min_ewm_hl10": ewm_hl(pdf["fga_per_min"], 10),
        "base_pts_per_min_ewm_hl5": ewm_hl(pdf["pts_per_min"], 5),
        "base_fg3a_per_min_ewm_hl10": ewm_hl(pdf["fg3a_per_min"], 10),
        "ts_pct_x_usg_pct": ts_x_usg,
        "track_tchs_per_min_ewm_hl10": (
            ewm_hl(pdf["tchs_per_min"], 10) if "tchs_per_min" in pdf.columns else float("nan")
        ),
        "track_cfga_per_min_ewm_hl10": (
            ewm_hl(pdf["cfga_per_min"], 10) if "cfga_per_min" in pdf.columns else float("nan")
        ),
        "base_pts_per_min_season_avg": season_avg(pdf, "pts_per_min"),
    }
    return ordered_features(FEATURE_COLS, values)
