"""Load scraper snapshots into Supabase odds tables."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd

from src.odds.snapshot_rows import prizepicks_projections_to_rows, underdog_picks_to_rows
from src.utils.db import upsert_df

_PRIZEPICKS_CONFLICT_COLS = [
    "league",
    "player_name",
    "stat_type",
    "odds_type",
    "line_score",
    "scraped_at",
]

_UNDERDOG_CONFLICT_COLS = [
    "league",
    "player_name",
    "stat_name",
    "side",
    "line_score",
    "scraped_at",
]


def _skip_db(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip().lower() in {"1", "true", "yes"}


def _coerce_float_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_prizepicks_snapshot(
    projections: list[dict],
    *,
    league: str,
    scraped_at: datetime | None = None,
) -> int:
    if _skip_db("PRIZEPICKS_SKIP_DB"):
        return 0

    scraped_at = scraped_at or datetime.now(timezone.utc)
    rows = prizepicks_projections_to_rows(projections, league=league, scraped_at=scraped_at)
    if not rows:
        return 0

    df = _coerce_float_columns(pd.DataFrame(rows), ["line_score"])
    upsert_df(
        "wnba_prizepicks",
        df,
        schema="odds",
        conflict_cols=_PRIZEPICKS_CONFLICT_COLS,
        lineage_col="fetched_at",
    )
    return len(rows)


def load_underdog_snapshot(
    picks: list[dict],
    *,
    league: str,
    scraped_at: datetime | None = None,
) -> int:
    if _skip_db("UNDERDOG_SKIP_DB"):
        return 0

    scraped_at = scraped_at or datetime.now(timezone.utc)
    rows = underdog_picks_to_rows(picks, league=league, scraped_at=scraped_at)
    if not rows:
        return 0

    df = _coerce_float_columns(pd.DataFrame(rows), ["line_score", "payout_multiplier"])
    upsert_df(
        "wnba_underdogs",
        df,
        schema="odds",
        conflict_cols=_UNDERDOG_CONFLICT_COLS,
        lineage_col="fetched_at",
    )
    return len(rows)
