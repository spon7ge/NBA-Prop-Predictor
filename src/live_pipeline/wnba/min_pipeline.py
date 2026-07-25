"""WNBA minutes live features → ``min_wnba_model_*.joblib``.

Feature order matches ``models/wnba/min_wnba_model.ipynb`` / saved bundle
``feature_names`` (includes ``base_min_season_avg``).

``starting`` prefers RotoWire projected starters
(``projectedStartingFiveWnba``); falls back to last completed-game start flag.
"""

from __future__ import annotations

import pandas as pd

from src.live_pipeline.common import (
    ensure_starting,
    ewm_hl,
    lag1,
    ordered_features,
    parse_minutes,
    player_history,
    season_avg,
    starter_roll_pct,
    team_rank_l10,
    trend_5v20,
)
from src.utils.team_info import is_projected_starter

# Exact order from models/wnba/min_wnba_model.ipynb MIN_FEATURES
FEATURE_COLS = [
    "starting",
    "starter_roll10_pct",
    "team_min_rank_l10",
    "team_usg_pct_rank_l10",
    "base_min_season_avg",
    "track_minutes_ewm_hl10",
    "base_min_trend_5v20",
    "base_min_lag1",
]


def _team_abbr(pdf: pd.DataFrame) -> str | None:
    for col in ("team_abbreviation", "TEAM_ABBREVIATION", "team_abbr"):
        if col in pdf.columns:
            val = pdf[col].iloc[-1]
            if pd.notna(val) and str(val).strip():
                return str(val).strip().upper()
    return None


def _starting_flag(pdf: pd.DataFrame, name: str, *, league: str = "wnba") -> float:
    """1/0 for tonight's projected start when available; else last-game history."""
    historical = float(pdf["starting"].iloc[-1])
    projected = is_projected_starter(name, _team_abbr(pdf), league=league)
    if projected is None:
        return historical
    return 1.0 if projected else 0.0


def min_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "wnba"):
    """Build WNBA minutes feature vector for the next game."""
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_starting(pdf)
    team_id = pdf["team_id"].iloc[-1] if "team_id" in pdf.columns else None

    if "minutes" in pdf.columns:
        track_min = parse_minutes(pdf["minutes"])
    else:
        track_min = pd.Series(dtype=float)

    min_col = "min" if "min" in pdf.columns else "minutes"
    usg_col = "usg_pct" if "usg_pct" in pdf.columns else None

    values = {
        "starting": _starting_flag(pdf, name, league=league),
        "starter_roll10_pct": starter_roll_pct(pdf, 10),
        "team_min_rank_l10": team_rank_l10(df, name, team_id, date, "min", halflife=10),
        "team_usg_pct_rank_l10": (
            team_rank_l10(df, name, team_id, date, usg_col, halflife=10)
            if usg_col
            else float("nan")
        ),
        "base_min_season_avg": season_avg(pdf, min_col),
        "track_minutes_ewm_hl10": ewm_hl(track_min, 10) if len(track_min) else float("nan"),
        "base_min_trend_5v20": trend_5v20(pdf[min_col].astype(float)),
        "base_min_lag1": lag1(pdf, min_col),
    }
    return ordered_features(FEATURE_COLS, values)
