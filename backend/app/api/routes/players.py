"""GET /api/player/{player_id} and /api/players — DB-only player endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.feature import PlayerListResponse, PlayerSummary
from app.schemas.player import (
    MLPredictionSummary,
    PlayerGame,
    PlayerProfile,
    RollingAvg5,
    RollingAvg10,
)

router = APIRouter(tags=["players"])

_PROFILE_SQL = """
SELECT
    player_id,
    player_name,
    normalized_name,
    team_abbreviation,
    team_name,
    career_game_count
FROM silver.silver_players
WHERE player_id = %(player_id)s
"""

_GAMES_SQL = """
SELECT
    game_id,
    game_date,
    matchup,
    opp_team_abbreviation,
    wl,
    season_year,
    season_type,
    starting,
    is_home,
    min,
    pts,
    reb,
    ast,
    tov,
    stl,
    blk,
    fgm,
    fga,
    fg3a,
    fta,
    plus_minus,
    usg_pct,
    ts_pct,
    efg_pct,
    net_rating,
    pts_per_min,
    reb_per_min,
    ast_per_min
FROM gold.gold_player_game_stats
WHERE player_id = %(player_id)s
ORDER BY game_date DESC
LIMIT %(n)s
"""

_ROLL5_SQL = """
SELECT
    min_roll5,
    pts_roll5,
    reb_roll5,
    ast_roll5,
    tov_roll5,
    usg_pct_roll5,
    plus_minus_roll5,
    pts_per_min_roll5,
    reb_per_min_roll5,
    ast_per_min_roll5
FROM gold.gold_player_rolling_avg_5
WHERE player_id = %(player_id)s
ORDER BY game_date DESC
LIMIT 1
"""

_ROLL10_SQL = """
SELECT
    min_roll10,
    pts_roll10,
    reb_roll10,
    ast_roll10,
    tov_roll10,
    usg_pct_roll10,
    plus_minus_roll10,
    pts_per_min_roll10,
    min_season_mean,
    starter_roll10_pct,
    consec_starts,
    team_min_rank_l10,
    team_usg_rank_l10,
    min_rate_of_change
FROM gold.gold_player_rolling_avg_10
WHERE player_id = %(player_id)s
ORDER BY game_date DESC
LIMIT 1
"""

_PREDICTIONS_SQL = """
SELECT
    prop,
    game_id,
    prediction,
    predicted_at,
    game_date
FROM ml.predictions
WHERE player_id = %(player_id)s
ORDER BY predicted_at DESC
LIMIT %(n)s
"""

_SEARCH_SQL = """
SELECT
    player_id,
    player_name,
    normalized_name,
    team_abbreviation,
    team_name,
    career_game_count
FROM silver.silver_players
WHERE (%(q)s IS NULL OR normalized_name ILIKE %(pattern)s OR player_name ILIKE %(pattern)s)
  AND (%(team)s IS NULL OR team_abbreviation = %(team)s)
ORDER BY player_name
LIMIT %(limit)s
"""


@router.get("/players", response_model=PlayerListResponse)
def search_players(
    q: str | None = Query(default=None, description="Name search (case-insensitive)."),
    team: str | None = Query(default=None, description="Team tricode, e.g. LAL"),
    limit: int = Query(default=25, ge=1, le=100),
) -> PlayerListResponse:
    """Search the player directory from **silver.silver_players**."""
    pattern = f"%{q}%" if q else None
    rows = db.query(
        _SEARCH_SQL,
        {"q": q, "pattern": pattern, "team": team.upper() if team else None, "limit": limit},
    )
    players = [PlayerSummary(**row) for row in rows]
    return PlayerListResponse(count=len(players), players=players)


@router.get("/player/{player_id}", response_model=PlayerProfile)
def get_player(
    player_id: int,
    recent_n: int = Query(default=10, ge=1, le=82),
    include_predictions: bool = Query(default=True),
) -> PlayerProfile:
    """Player profile, recent games, rolling context, and latest ML predictions."""
    params = {"player_id": player_id}

    profile_row = db.query_one(_PROFILE_SQL, params)
    if profile_row is None:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found.")

    game_rows = db.query(_GAMES_SQL, {"player_id": player_id, "n": recent_n})
    roll5_row = db.query_one(_ROLL5_SQL, params)
    roll10_row = db.query_one(_ROLL10_SQL, params)

    predictions: list[MLPredictionSummary] = []
    if include_predictions:
        pred_rows = db.query(_PREDICTIONS_SQL, {"player_id": player_id, "n": 20})
        predictions = [MLPredictionSummary(**row) for row in pred_rows]

    return PlayerProfile(
        **profile_row,
        recent_games=[PlayerGame(**g) for g in game_rows],
        rolling_avg_5=RollingAvg5(**roll5_row) if roll5_row else None,
        rolling_avg_10=RollingAvg10(**roll10_row) if roll10_row else None,
        predictions=predictions,
    )
