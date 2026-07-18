"""NBA rebounds-per-minute live features → ``rpm_nba_model_*.joblib``."""

from __future__ import annotations

import pandas as pd

from src.live_pipeline.common import (
    ensure_per_min,
    ewm_hl,
    ordered_features,
    player_history,
    position_encoded,
    roll_mean,
    season_avg,
    team_stat_ewm,
)
from src.utils.helper_functions import findOpp

# Exact order from rpm_nba_model_2026-04-12.joblib
FEATURE_COLS = [
    "base_reb_per_min_season_avg",
    "adv_reb_pct_season_avg",
    "track_rbc_per_min_ewm_hl10",
    "position_encoded",
    "adv_reb_pct_ewm_hl10",
    "base_reb_per_min_ewm_hl10",
    "track_orbc_per_min_ewm_hl10",
    "opp_pace_ewm_hl10",
    "track_drbc_per_min_ewm_hl10",
    "reb_per_min_roll10",
]


def rpm_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "nba"):
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_per_min(pdf, ["reb", "orbc", "drbc", "rbc"])
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3, league=league)

    values = {
        "base_reb_per_min_season_avg": season_avg(pdf, "reb_per_min"),
        "adv_reb_pct_season_avg": season_avg(pdf, "reb_pct"),
        "track_rbc_per_min_ewm_hl10": (
            ewm_hl(pdf["rbc_per_min"], 10) if "rbc_per_min" in pdf.columns else float("nan")
        ),
        "position_encoded": position_encoded(pdf),
        "adv_reb_pct_ewm_hl10": (
            ewm_hl(pdf["reb_pct"], 10) if "reb_pct" in pdf.columns else float("nan")
        ),
        "base_reb_per_min_ewm_hl10": ewm_hl(pdf["reb_per_min"], 10),
        "track_orbc_per_min_ewm_hl10": (
            ewm_hl(pdf["orbc_per_min"], 10) if "orbc_per_min" in pdf.columns else float("nan")
        ),
        "opp_pace_ewm_hl10": team_stat_ewm(
            df, opp_abbr, "team_pace", date, halflife=10
        ),
        "track_drbc_per_min_ewm_hl10": (
            ewm_hl(pdf["drbc_per_min"], 10) if "drbc_per_min" in pdf.columns else float("nan")
        ),
        "reb_per_min_roll10": roll_mean(pdf["reb_per_min"].astype(float), 10),
    }
    return ordered_features(FEATURE_COLS, values)
