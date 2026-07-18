"""WNBA assists-per-minute live features → ``apm_wnba_model_*.joblib``."""

from __future__ import annotations

import pandas as pd

from src.live_pipeline.common import (
    ensure_per_min,
    ewm_hl,
    ordered_features,
    player_history,
    team_rank_l10,
)

# Exact order from apm_wnba_model_2026-07-12.joblib
FEATURE_COLS = [
    "base_ast_per_min_ewm_hl10",
    "adv_ast_pct_ewm_hl10",
    "adv_poss_ewm_hl10",
    "adv_usg_pct_ewm_hl10",
    "team_ast_per_min_rank_l10",
]


def apm_pipeline(df: pd.DataFrame, name: str, date: str, *, league: str = "wnba"):
    del league
    pdf = player_history(df, name, date)
    if pdf is None:
        return None

    pdf = ensure_per_min(pdf, ["ast"])
    team_id = pdf["team_id"].iloc[-1] if "team_id" in pdf.columns else None

    # Rank teammates by ast_per_min EWM (same spirit as training team ranks)
    if "ast_per_min" not in df.columns and {"ast", "min"}.issubset(df.columns):
        work = df.copy()
        min_s = work["min"].replace(0, float("nan")).astype(float)
        work["ast_per_min"] = work["ast"].astype(float) / min_s
    else:
        work = df

    values = {
        "base_ast_per_min_ewm_hl10": ewm_hl(pdf["ast_per_min"], 10),
        "adv_ast_pct_ewm_hl10": (
            ewm_hl(pdf["ast_pct"], 10) if "ast_pct" in pdf.columns else float("nan")
        ),
        "adv_poss_ewm_hl10": (
            ewm_hl(pdf["poss"], 10) if "poss" in pdf.columns else float("nan")
        ),
        "adv_usg_pct_ewm_hl10": (
            ewm_hl(pdf["usg_pct"], 10) if "usg_pct" in pdf.columns else float("nan")
        ),
        "team_ast_per_min_rank_l10": team_rank_l10(
            work, name, team_id, date, "ast_per_min", halflife=10
        ),
    }
    return ordered_features(FEATURE_COLS, values)
