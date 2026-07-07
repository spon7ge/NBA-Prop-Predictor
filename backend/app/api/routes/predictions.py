"""GET /api/predictions — ML model outputs from ml.predictions."""
from __future__ import annotations

import datetime

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.ml_prediction import MLPrediction, PlayerPredictions

router = APIRouter(tags=["predictions"])

_VALID_PROPS = {"min", "ppm", "rpm", "apm"}

_BASE_SQL = """
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
WHERE (%(game_date)s IS NULL OR game_date = %(game_date)s)
  AND (%(prop)s       IS NULL OR prop = %(prop)s)
  AND (%(player_id)s  IS NULL OR player_id = %(player_id)s)
  AND (%(game_id)s    IS NULL OR game_id = %(game_id)s)
ORDER BY predicted_at DESC, player_name NULLS LAST, prop
LIMIT %(limit)s
"""


def _validate_prop(prop: str | None) -> str | None:
    if prop is None:
        return None
    normalized = prop.lower()
    if normalized not in _VALID_PROPS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prop '{prop}'. Valid: {sorted(_VALID_PROPS)}",
        )
    return normalized


@router.get("/predictions", response_model=list[MLPrediction])
def list_predictions(
    date: str | None = Query(
        default=None,
        description="Filter by game_date (YYYY-MM-DD).",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    prop: str | None = Query(default=None, description="min | ppm | rpm | apm"),
    player_id: int | None = Query(default=None),
    game_id: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[MLPrediction]:
    """Return ML model predictions from **ml.predictions** (no external API calls)."""
    rows = db.query(
        _BASE_SQL,
        {
            "game_date": date,
            "prop": _validate_prop(prop),
            "player_id": player_id,
            "game_id": game_id,
            "limit": limit,
        },
    )
    return [MLPrediction(**row) for row in rows]


@router.get("/predictions/player/{player_id}", response_model=PlayerPredictions)
def predictions_for_player(
    player_id: int,
    date: str | None = Query(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    prop: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> PlayerPredictions:
    """Latest ML predictions for one player."""
    rows = db.query(
        _BASE_SQL,
        {
            "game_date": date,
            "prop": _validate_prop(prop),
            "player_id": player_id,
            "game_id": None,
            "limit": limit,
        },
    )
    player_name = rows[0]["player_name"] if rows else None
    if player_name is None:
        profile = db.query_one(
            "SELECT player_name FROM silver.silver_players WHERE player_id = %(player_id)s",
            {"player_id": player_id},
        )
        player_name = profile["player_name"] if profile else None

    return PlayerPredictions(
        player_id=player_id,
        player_name=player_name,
        predictions=[MLPrediction(**row) for row in rows],
    )


@router.get("/predictions/today", response_model=list[MLPrediction])
def predictions_today(
    prop: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[MLPrediction]:
    """Shortcut for today's slate predictions."""
    today = str(datetime.date.today())
    return list_predictions(date=today, prop=prop, player_id=None, game_id=None, limit=limit)
