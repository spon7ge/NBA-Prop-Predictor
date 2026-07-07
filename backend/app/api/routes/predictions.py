"""GET /api/predictions

Returns prop lines from gold.gold_prop_history enriched with pre-game rolling
context. All data is read from Supabase — no external API calls are made here.
"""
from __future__ import annotations

import datetime

from fastapi import APIRouter, Query

from app.core import db
from app.schemas.prediction import PropPrediction

router = APIRouter(tags=["predictions"])

_SQL = """
SELECT
    bookmaker,
    market_category,
    player_id,
    player_name,
    player_name_raw,
    normalized_name,
    side,
    game_date,
    line,
    odds,
    prop_source,
    last_update_at,
    player_team_abbrev,
    home_team_abbrev,
    away_team_abbrev,
    game_season_year,
    min_roll5,
    pts_per_min_roll5,
    reb_per_min_roll5,
    ast_per_min_roll5,
    min_roll10,
    pts_per_min_roll10,
    team_min_rank_l10,
    team_usg_rank_l10,
    expected_pace,
    opp_def_rating_roll10,
    team_spread,
    game_total
FROM gold.gold_prop_history
WHERE game_date = %(game_date)s
  AND (%(bookmaker)s IS NULL OR lower(bookmaker) = lower(%(bookmaker)s))
  AND (%(market)s    IS NULL OR lower(market_category) = lower(%(market)s))
  AND (%(source)s    IS NULL OR prop_source = %(source)s)
  AND (%(side)s      IS NULL OR side = %(side)s)
ORDER BY player_name, market_category, side
LIMIT %(limit)s
"""


@router.get("/predictions", response_model=list[PropPrediction])
def get_predictions(
    date: str | None = Query(
        default=None,
        description="Slate date in YYYY-MM-DD format. Defaults to today.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    bookmaker: str | None = Query(default=None, description="Filter by bookmaker name (case-insensitive)."),
    market: str | None = Query(default=None, description="Filter by market category, e.g. player_points."),
    source: str | None = Query(default=None, description="dfs or us — prop source category."),
    side: str | None = Query(default=None, description="over or under."),
    limit: int = Query(default=500, ge=1, le=2000, description="Max rows to return."),
) -> list[PropPrediction]:
    """Return prop lines with pre-game rolling context for a given slate date.

    Reads exclusively from **gold.gold_prop_history** — no external API calls.
    """
    target_date = date or str(datetime.date.today())

    rows = db.query(
        _SQL,
        {
            "game_date": target_date,
            "bookmaker": bookmaker,
            "market": market,
            "source": source,
            "side": side,
            "limit": limit,
        },
    )
    return [PropPrediction(**row) for row in rows]
