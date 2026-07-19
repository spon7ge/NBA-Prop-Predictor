"""
Grade live prop predictions against silver box scores.

Joins ``ml.{league}_live_prop_predictions`` (latest ``run_at`` per ``game_date``)
to ``silver.{league}_player_gamelogs`` and scores with ``log._derive_side`` /
``log._score_prediction``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.pipeline.build_slates import silver_to_base_df
from src.utils.db import read_df, upsert_live_prop_grades
from src.utils.prop_scoring import (
    derive_side,
    rows_for_calendar_date,
    score_prediction,
)


def _pred_table(league: str) -> str:
    return f"{league}_live_prop_predictions"


def _silver_table(league: str) -> str:
    return f"{league}_player_gamelogs"


def _dates_to_grade(
    *,
    game_date: date | None,
    lookback_days: int,
) -> list[date]:
    if game_date is not None:
        return [game_date]
    # Midnight DAG grades last night by default; lookback covers catch-up.
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=max(lookback_days, 1) - 1)
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _load_latest_preds(league: str, game_date: date) -> pd.DataFrame:
    table = _pred_table(league)
    d = game_date.isoformat()
    preds = read_df(
        table,
        schema="ml",
        where=(
            "game_date = %(d)s AND run_at = ("
            "  SELECT MAX(run_at) FROM ml." + table + " WHERE game_date = %(d)s"
            ")"
        ),
        params={"d": d},
        eq={"game_date": d},
    )
    if preds.empty or "run_at" not in preds.columns:
        return preds
    # REST fallback may return every run for the date — keep the latest only.
    latest = preds["run_at"].max()
    return preds.loc[preds["run_at"] == latest].copy()



def _load_silver_day(league: str, game_date: date) -> pd.DataFrame:
    d = game_date.isoformat()
    return read_df(
        _silver_table(league),
        schema="silver",
        where="game_date = %(d)s",
        params={"d": d},
        eq={"game_date": d},
    )


def _pred_row_for_score(row: pd.Series) -> pd.Series:
    """Map snake_case live-prop columns → uppercase schema expected by scorers."""
    return pd.Series(
        {
            "PLAYER_NAME": row.get("player_name"),
            "MARKET": row.get("market", "PTS"),
            "LINE": row.get("line", 0),
            "STAT_Q10": row.get("stat_q10"),
            "STAT_Q50": row.get("stat_q50"),
            "P_OVER": row.get("p_over", 0.5),
            "P_UNDER": row.get("p_under", 0.5),
            # Intentionally omit IMP_PROB_OVER (not on live props).
        }
    )


def grade_predictions_for_date(
    league: str,
    game_date: date,
    *,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Score latest live props for one slate date. Returns graded DataFrame."""
    if league not in ("nba", "wnba"):
        raise ValueError(f"Unknown league {league!r}")

    preds = _load_latest_preds(league, game_date)
    if preds.empty:
        print(f"  [{league}] {game_date}: no predictions")
        return pd.DataFrame()

    silver = _load_silver_day(league, game_date)
    if silver.empty:
        print(f"  [{league}] {game_date}: no silver gamelogs yet — skip")
        return pd.DataFrame()

    game_day = silver_to_base_df(silver)
    game_day = rows_for_calendar_date(game_day, game_date.isoformat())
    if game_day.empty:
        print(f"  [{league}] {game_date}: silver present but no matching GAME_DATE rows")
        return pd.DataFrame()

    graded_at = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for _, raw in preds.iterrows():
        score_row = _pred_row_for_score(raw)
        side = derive_side(score_row)
        score_row["SIDE"] = side
        actual_stat, actual_min, hit, miss_reason = score_prediction(
            score_row, game_day
        )
        line = float(raw["line"]) if pd.notna(raw.get("line")) else None
        abs_err = None
        if actual_stat is not None and line is not None:
            abs_err = abs(float(actual_stat) - line)

        rows.append(
            {
                "graded_at": graded_at,
                "run_at": raw.get("run_at"),
                "game_date": game_date,
                "player_name": raw.get("player_name"),
                "team_abbr": raw.get("team_abbr"),
                "opponent_abbr": raw.get("opponent_abbr"),
                "market": raw.get("market"),
                "bookmaker": raw.get("bookmaker"),
                "line": line,
                "side": side,
                "stat_q10": raw.get("stat_q10"),
                "stat_q50": raw.get("stat_q50"),
                "p_over": raw.get("p_over"),
                "p_under": raw.get("p_under"),
                "actual_stat": actual_stat,
                "actual_min": actual_min,
                "hit": bool(hit),
                "miss_reason": miss_reason,
                "abs_error": abs_err,
            }
        )

    out = pd.DataFrame(rows)
    n_hit = int(out["hit"].sum()) if not out.empty else 0
    n_dnp = int((out["miss_reason"] == "dnp").sum()) if not out.empty else 0
    n_scored = len(out) - n_dnp
    print(
        f"  [{league}] {game_date}: graded {len(out)} props "
        f"({n_hit}/{n_scored} hit excl. DNP; {n_dnp} DNP)"
    )

    if not dry_run and not out.empty:
        upsert_live_prop_grades(out, league=league)
        print(f"  [{league}] upserted → ml.{league}_live_prop_grades")

    return out


def run_grade_live_props(
    *,
    league: str,
    game_date: date | None = None,
    lookback_days: int = 1,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Grade one or more slate dates for a league. Returns concatenated grades."""
    dates = _dates_to_grade(game_date=game_date, lookback_days=lookback_days)
    frames: list[pd.DataFrame] = []
    for d in dates:
        frame = grade_predictions_for_date(league, d, dry_run=dry_run)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
