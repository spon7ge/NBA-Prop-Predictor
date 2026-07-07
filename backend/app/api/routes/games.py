"""GET /api/games/* — schedule and slate endpoints (DB-only)."""
from __future__ import annotations

import datetime
import re

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.game import Game, GameSlate, GameWithPredictions, GameWithProps
from app.schemas.ml_prediction import MLPrediction
from app.schemas.prop import PropLine

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

_PROPS_ALL_SQL = """
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
ORDER BY player_name, market_category, side
"""

_PROPS_BY_MATCHUP_SQL = """
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
  AND (%(home)s IS NULL OR player_team_abbrev IN (%(home)s, %(away)s))
ORDER BY player_name, market_category, side
"""

_PREDICTIONS_SQL = """
SELECT
    prop,
    game_id,
    player_id,
    prediction,
    predicted_at,
    game_date,
    player_name,
    model_path
FROM ml.predictions
WHERE game_date = %(game_date)s
  AND (%(prop)s IS NULL OR prop = %(prop)s)
ORDER BY player_name NULLS LAST, prop
LIMIT %(limit)s
"""


def _validate_date(date_str: str) -> str:
    if not _DATE_RE.match(date_str):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format '{date_str}'. Use YYYY-MM-DD.",
        )
    return date_str


@router.get("/games/today", response_model=list[Game])
def get_todays_games() -> list[Game]:
    """Shortcut — returns today's games without specifying a date."""
    today = str(datetime.date.today())
    rows = db.query(_GAMES_SQL, {"game_date": today})
    return [Game(**r) for r in rows]


@router.get("/games/{date}", response_model=list[Game])
def get_games(date: str) -> list[Game]:
    """Return all games on *date* (YYYY-MM-DD) from **silver.silver_games**."""
    _validate_date(date)
    rows = db.query(_GAMES_SQL, {"game_date": date})
    return [Game(**r) for r in rows]


@router.get("/games/{date}/predictions", response_model=list[MLPrediction])
def get_game_predictions(
    date: str,
    prop: str | None = Query(default=None),
    limit: int = Query(default=2000, ge=1, le=5000),
) -> list[MLPrediction]:
    """All ML predictions for a slate date from **ml.predictions**."""
    _validate_date(date)
    rows = db.query(
        _PREDICTIONS_SQL,
        {"game_date": date, "prop": prop.lower() if prop else None, "limit": limit},
    )
    return [MLPrediction(**row) for row in rows]


@router.get("/games/{date}/props", response_model=list[PropLine])
def get_game_props(
    date: str,
    bookmaker: str | None = Query(default=None),
    market: str | None = Query(default=None),
) -> list[PropLine]:
    """All prop lines for a slate date from **gold.gold_prop_history**."""
    _validate_date(date)
    sql = _PROPS_ALL_SQL
    params: dict = {"game_date": date}
    if bookmaker:
        sql += " AND lower(bookmaker) = lower(%(bookmaker)s)"
        params["bookmaker"] = bookmaker
    if market:
        sql += " AND lower(market_category) = lower(%(market)s)"
        params["market"] = market
    rows = db.query(sql, params)
    return [PropLine(**row) for row in rows]


@router.get("/games/{date}/slate", response_model=GameSlate)
def get_game_slate(date: str) -> GameSlate:
    """Combined games + props + predictions for a full slate view."""
    _validate_date(date)
    games = get_games(date)
    props = get_game_props(date)
    predictions = get_game_predictions(date)
    return GameSlate(
        game_date=datetime.date.fromisoformat(date),
        games=games,
        props=props,
        predictions=predictions,
    )


@router.get("/games/{date}/with-props", response_model=list[GameWithProps])
def get_games_with_props(
    date: str,
    bookmaker: str | None = Query(default=None),
    market: str | None = Query(default=None),
) -> list[GameWithProps]:
    """Games on *date* with prop lines grouped per matchup."""
    _validate_date(date)
    game_rows = db.query(_GAMES_SQL, {"game_date": date})
    if not game_rows:
        return []

    results: list[GameWithProps] = []
    for game_row in game_rows:
        home = game_row["home_team_abbrev"]
        away = game_row["away_team_abbrev"]

        prop_sql = _PROPS_BY_MATCHUP_SQL
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
                props=[PropLine(**p) for p in prop_rows],
            )
        )

    return results


@router.get("/games/{date}/with-predictions", response_model=list[GameWithPredictions])
def get_games_with_predictions(
    date: str,
    prop: str | None = Query(default=None),
) -> list[GameWithPredictions]:
    """Games on *date* with ML predictions attached."""
    _validate_date(date)
    game_rows = db.query(_GAMES_SQL, {"game_date": date})
    if not game_rows:
        return []

    all_preds = get_game_predictions(date, prop=prop)
    by_game: dict[str, list[MLPrediction]] = {}
    for row in all_preds:
        by_game.setdefault(row.game_id, []).append(row)

    return [
        GameWithPredictions(
            **game_row,
            predictions=by_game.get(game_row.get("game_id") or "", []),
        )
        for game_row in game_rows
    ]
