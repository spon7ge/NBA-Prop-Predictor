"""GET /api/features/{prop} — ML feature rows from ml.features_* tables."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.feature import MLFeatureRow

router = APIRouter(tags=["features"])

_TABLES = {
    "base": "features",
    "min": "features_min",
    "ppm": "features_ppm",
    "rpm": "features_rpm",
    "apm": "features_apm",
}


@router.get("/features/{prop}", response_model=list[MLFeatureRow])
def list_features(
    prop: str,
    date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    player_id: int | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> list[MLFeatureRow]:
    """Return ML input features for model training / debugging."""
    table = _TABLES.get(prop.lower())
    if table is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prop '{prop}'. Valid: {sorted(_TABLES)}",
        )

    sql = f"""
    SELECT *
    FROM ml.{table}
    WHERE (%(game_date)s IS NULL OR game_date = %(game_date)s)
      AND (%(player_id)s IS NULL OR player_id = %(player_id)s)
    ORDER BY game_date DESC, player_name NULLS LAST
    LIMIT %(limit)s
    """
    rows = db.query(
        sql,
        {"game_date": date, "player_id": player_id, "limit": limit},
    )
    return [MLFeatureRow(**row) for row in rows]
