"""Shared minutes live feature builder (NBA + WNBA).

Matches ``min_nba_model_*.joblib`` ``feature_names``.
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
    starter_roll_pct,
    team_rank_l10,
    trend_5v20,
)

# Exact order from min_nba_model_2026-04-12.joblib
FEATURE_COLS = [
    "starting",
    "starter_roll10_pct",
    "team_min_rank_l10",
    "team_usg_pct_rank_l10",
    "track_minutes_ewm_hl10",
    "base_min_trend_5v20",
    "base_min_lag1",
]


def min_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "nba"):
    """Build minutes feature vector for the next game.

    ``league`` is accepted for API symmetry; minutes features are league-agnostic.
    """
    del league  # shared model / features
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_starting(pdf)
    team_id = pdf["team_id"].iloc[-1] if "team_id" in pdf.columns else None

    # Tracking minutes (text) when present; else NaN for track feature
    if "minutes" in pdf.columns:
        track_min = parse_minutes(pdf["minutes"])
    else:
        track_min = pd.Series(dtype=float)

    usg_col = "usg_pct" if "usg_pct" in pdf.columns else None

    values = {
        # Pre-game lineup unknown → last completed start flag
        "starting": float(pdf["starting"].iloc[-1]),
        "starter_roll10_pct": starter_roll_pct(pdf, 10),
        "team_min_rank_l10": team_rank_l10(df, name, team_id, date, "min", halflife=10),
        "team_usg_pct_rank_l10": (
            team_rank_l10(df, name, team_id, date, usg_col, halflife=10)
            if usg_col
            else float("nan")
        ),
        "track_minutes_ewm_hl10": ewm_hl(track_min, 10) if len(track_min) else float("nan"),
        "base_min_trend_5v20": trend_5v20(pdf["min"].astype(float)),
        "base_min_lag1": lag1(pdf, "min"),
    }
    return ordered_features(FEATURE_COLS, values)
