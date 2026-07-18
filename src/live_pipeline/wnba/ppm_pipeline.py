"""WNBA points-per-minute live features → ``ppm_wnba_model_*.joblib``."""

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

# Exact order from ppm_wnba_model_2026-07-12.joblib
FEATURE_COLS = [
    "base_ft_pct_season_avg",
    "adv_usg_pct_ewm_hl10",
    "base_pts_per_min_season_avg",
    "base_fga_per_min_ewm_hl10",
    "base_fta_per_min_ewm_hl10",
    "opp_def_rating_ewm_hl10",
    "adv_efg_pct_season_avg",
]


def ppm_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "wnba"):
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_per_min(pdf, ["pts", "fga", "fta"])
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3, league=league)

    values = {
        "base_ft_pct_season_avg": season_avg(pdf, "ft_pct"),
        "adv_usg_pct_ewm_hl10": (
            ewm_hl(pdf["usg_pct"], 10) if "usg_pct" in pdf.columns else float("nan")
        ),
        "base_pts_per_min_season_avg": season_avg(pdf, "pts_per_min"),
        "base_fga_per_min_ewm_hl10": ewm_hl(pdf["fga_per_min"], 10),
        "base_fta_per_min_ewm_hl10": ewm_hl(pdf["fta_per_min"], 10),
        "opp_def_rating_ewm_hl10": team_stat_ewm(
            df, opp_abbr, "team_def_rating", date, halflife=10
        ),
        "adv_efg_pct_season_avg": season_avg(pdf, "efg_pct"),
    }
    return ordered_features(FEATURE_COLS, values)
