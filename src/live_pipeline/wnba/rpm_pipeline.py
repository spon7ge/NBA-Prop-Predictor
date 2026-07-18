"""WNBA rebounds-per-minute live features → ``rpm_wnba_model_*.joblib``."""

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

# Exact order from rpm_wnba_model_2026-07-16.joblib
FEATURE_COLS = [
    "base_reb_per_min_season_avg",
    "base_reb_per_min_ewm_hl5",
    "base_oreb_per_min_season_avg",
    "base_oreb_per_min_ewm_hl5",
    "base_dreb_per_min_season_avg",
    "base_dreb_per_min_ewm_hl5",
    "adv_reb_pct_ewm_hl10",
    "opp_reb_pct_ewm_hl20",
    "opp_oreb_pct_ewm_hl10",
    "opp_fg3a_ewm_hl20",
    "team_fg3a_ewm_hl20",
    "team_reb_ewm_hl10",
]


def rpm_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "wnba"):
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_per_min(pdf, ["reb", "oreb", "dreb"])
    team_abbr = (
        pdf["team_abbreviation"].iloc[-1]
        if "team_abbreviation" in pdf.columns
        else None
    )
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3, league=league)

    values = {
        "base_reb_per_min_season_avg": season_avg(pdf, "reb_per_min"),
        "base_reb_per_min_ewm_hl5": ewm_hl(pdf["reb_per_min"], 5),
        "base_oreb_per_min_season_avg": season_avg(pdf, "oreb_per_min"),
        "base_oreb_per_min_ewm_hl5": ewm_hl(pdf["oreb_per_min"], 5),
        "base_dreb_per_min_season_avg": season_avg(pdf, "dreb_per_min"),
        "base_dreb_per_min_ewm_hl5": ewm_hl(pdf["dreb_per_min"], 5),
        "adv_reb_pct_ewm_hl10": (
            ewm_hl(pdf["reb_pct"], 10) if "reb_pct" in pdf.columns else float("nan")
        ),
        "opp_reb_pct_ewm_hl20": team_stat_ewm(
            df, opp_abbr, "team_reb_pct", date, halflife=20
        ),
        "opp_oreb_pct_ewm_hl10": team_stat_ewm(
            df, opp_abbr, "team_oreb_pct", date, halflife=10
        ),
        "opp_fg3a_ewm_hl20": team_stat_ewm(
            df, opp_abbr, "team_fg3a", date, halflife=20
        ),
        "team_fg3a_ewm_hl20": team_stat_ewm(
            df, team_abbr, "team_fg3a", date, halflife=20
        ),
        "team_reb_ewm_hl10": team_stat_ewm(
            df, team_abbr, "team_reb", date, halflife=10
        ),
    }
    return ordered_features(FEATURE_COLS, values)
