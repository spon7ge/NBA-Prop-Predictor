"""GET /api/games/{date}

Returns all games on a given date, optionally enriched with prop lines.
All data is read from Supabase — no external API calls are made here.
"""
from __future__ import annotations

import datetime
import re

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.game import Game, GameWithProps
from app.schemas.prediction import PropPrediction

router = APIRouter(tags=["games"])

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_GAMES_SQL = """
SELECT
    game_date,
    game_id,
    event_id,
    home_team_abbrev,
    away_team_abbrev,
    season_year,
    source
FROM silver.silver_games
WHERE game_date = %(game_date)s
ORDER BY home_team_abbrev
"""

_PROPS_SQL = """
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
  AND (%(home)s IS NULL OR player_team_abbrev = %(home)s OR player_team_abbrev = %(away)s)
ORDER BY player_name, market_category, side
"""


def _validate_date(date_str: str) -> str:
    if not _DATE_RE.match(date_str):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format '{date_str}'. Use YYYY-MM-DD.",
        )
    return date_str


@router.get("/games/{date}", response_model=list[Game])
def get_games(date: str) -> list[Game]:
    """Return all games on *date* (YYYY-MM-DD).

    Reads from **silver.silver_games**. No external API calls are made.
    """
    _validate_date(date)
    rows = db.query(_GAMES_SQL, {"game_date": date})
    return [Game(**r) for r in rows]


@router.get("/games/{date}/with-props", response_model=list[GameWithProps])
def get_games_with_props(
    date: str,
    bookmaker: str | None = Query(default=None, description="Filter props by bookmaker."),
    market: str | None = Query(default=None, description="Filter props by market category."),
) -> list[GameWithProps]:
    """Return all games on *date* with their prop lines attached.

    Each game object includes a `props` list from **gold.gold_prop_history**.
    Reads only from Supabase — no external API calls are made.
    """
    _validate_date(date)
    game_rows = db.query(_GAMES_SQL, {"game_date": date})
    if not game_rows:
        return []

    results: list[GameWithProps] = []
    for game_row in game_rows:
        home = game_row["home_team_abbrev"]
        away = game_row["away_team_abbrev"]

        prop_sql = _PROPS_SQL
        prop_params: dict = {"game_date": date, "home": home, "away": away}

        if bookmaker:
            prop_sql += " AND lower(bookmaker) = lower(%(bookmaker)s)"
            prop_params["bookmaker"] = bookmaker
        if market:
            prop_sql += " AND lower(market_category) = lower(%(market)s)"
            prop_params["market"] = market

        prop_rows = db.query(prop_sql, prop_params)
        results.append(
            GameWithProps(
                **game_row,
                props=[PropPrediction(**p) for p in prop_rows],
            )
        )

    return results


@router.get("/games/today", response_model=list[Game])
def get_todays_games() -> list[Game]:
    """Shortcut — returns today's games without specifying a date."""
    today = str(datetime.date.today())
    rows = db.query(_GAMES_SQL, {"game_date": today})
    return [Game(**r) for r in rows]
