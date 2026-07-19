"""
Shared prop scoring helpers (hit / miss_reason) used by the JSONL reconciler
and the live-prop grader.
"""

from __future__ import annotations

import pandas as pd

_STAT_COL = {
    "PTS": "PTS",
    "AST": "AST",
    "REB": "REB",
    "TOV": "TOV",
    "FTA": "FTA",
    "3PM": "FG3M",
    "BLK": "BLK",
    "STL": "STL",
}


def normalize_game_date_series(s: pd.Series) -> pd.Series:
    """Parse game dates robustly (plain YYYY-MM-DD or ISO timestamps)."""
    return pd.to_datetime(s, errors="coerce", utc=False).dt.tz_localize(None)


def rows_for_calendar_date(
    base_df: pd.DataFrame,
    date_str: str,
    *,
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """Rows whose calendar day matches ``date_str`` (``YYYY-MM-DD``)."""
    if date_col not in base_df.columns:
        return pd.DataFrame()
    cal = normalize_game_date_series(base_df[date_col]).dt.strftime("%Y-%m-%d")
    return base_df.loc[cal == date_str].copy()


def player_game_rows(game_day: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Match PLAYER_NAME with stripped whitespace (prediction vs gamelog spellings)."""
    if game_day.empty or "PLAYER_NAME" not in game_day.columns:
        return pd.DataFrame()
    want = str(player_name).strip()
    mask = game_day["PLAYER_NAME"].astype(str).str.strip() == want
    return game_day.loc[mask]


def derive_side(row) -> str:
    q50 = float(row.get("STAT_Q50", 0) or 0)
    line = float(row.get("LINE", 0) or 0)
    if q50 > line:
        return "over"
    if q50 < line:
        return "under"
    po = float(row.get("P_OVER", 0.5) or 0.5)
    pu = float(row.get("P_UNDER", 0.5) or 0.5)
    return "over" if po >= pu else "under"


def score_prediction(pred: pd.Series, game_day: pd.DataFrame) -> tuple:
    """
    Returns (actual_stat, actual_min, hit, miss_reason).
    miss_reason is one of:
        dnp            — player not in gamelog (injury / rest)
        blowout        — actual < STAT_Q10 (minutes collapsed)
        sharp_line     — market IMP_PROB was 45–55% (coin flip)
        model_miss_low — predicted over, finished under line but above Q10
        model_miss_high— predicted under, finished over line
        clean_hit      — hit with margin > 10% of line
        squeaker       — hit with margin ≤ 10% of line
    """
    name = pred["PLAYER_NAME"]
    market = str(pred.get("MARKET", "PTS"))
    line = float(pred.get("LINE", 0))
    side = str(pred.get("SIDE", "over"))

    player_game = player_game_rows(game_day, name)

    if player_game.empty:
        return None, None, False, "dnp"

    stat_col = _STAT_COL.get(market, market)
    row = player_game.iloc[0]

    actual_stat = float(row[stat_col]) if stat_col in row.index else None
    actual_min = float(row["MIN"]) if "MIN" in row.index else None

    if actual_stat is None:
        return None, actual_min, False, "dnp"

    hit = (actual_stat > line) if side == "over" else (actual_stat < line)

    if not hit:
        q10 = float(pred.get("STAT_Q10", line))
        if actual_stat < q10:
            reason = "blowout"
        else:
            # Only tag sharp_line when a real market implied prob is present.
            # Live props lack IMP_PROB_OVER — defaulting to 0.5 would mislabel every miss.
            imp_raw = pred.get("IMP_PROB_OVER", None)
            sharp = False
            if imp_raw is not None:
                try:
                    if pd.notna(imp_raw):
                        imp_over = float(imp_raw)
                        sharp = 0.45 <= imp_over <= 0.55
                except (TypeError, ValueError):
                    sharp = False
            if sharp:
                reason = "sharp_line"
            elif side == "over":
                reason = "model_miss_low"
            else:
                reason = "model_miss_high"
    else:
        margin = abs(actual_stat - line)
        reason = "clean_hit" if line == 0 or margin / line > 0.10 else "squeaker"

    return actual_stat, actual_min, hit, reason


# Back-compat aliases used by log.py
_derive_side = derive_side
_score_prediction = score_prediction
_player_game_rows = player_game_rows
_rows_for_calendar_date = rows_for_calendar_date
