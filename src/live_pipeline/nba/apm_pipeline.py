"""NBA assists-per-minute live features → ``apm_nba_model_*.joblib``."""

from __future__ import annotations

import pandas as pd

from src.live_pipeline.common import (
    ensure_per_min,
    ewm_hl,
    ordered_features,
    player_history,
    position_encoded,
)

# Exact order from apm_nba_model_2026-04-12.joblib
FEATURE_COLS = [
    "base_ast_per_min_ewm_hl10",
    "track_pass_per_min_ewm_hl10",
    "adv_ast_pct_ewm_hl10",
    "position_encoded",
    "adv_poss_ewm_hl10",
]


def apm_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "nba"):
    del league
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_per_min(pdf, ["ast", "pass"])

    values = {
        "base_ast_per_min_ewm_hl10": ewm_hl(pdf["ast_per_min"], 10),
        "track_pass_per_min_ewm_hl10": (
            ewm_hl(pdf["pass_per_min"], 10) if "pass_per_min" in pdf.columns else float("nan")
        ),
        "adv_ast_pct_ewm_hl10": (
            ewm_hl(pdf["ast_pct"], 10) if "ast_pct" in pdf.columns else float("nan")
        ),
        "position_encoded": position_encoded(pdf),
        "adv_poss_ewm_hl10": (
            ewm_hl(pdf["poss"], 10) if "poss" in pdf.columns else float("nan")
        ),
    }
    return ordered_features(FEATURE_COLS, values)
