"""GET /api/models/{model_id}/accuracy — backtesting metrics per model version.

Reads from gold.gold_prediction_accuracy.
All arithmetic is done in the DB; this route is purely read-only.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.accuracy import ModelAccuracy, PropAccuracy

router = APIRouter(tags=["accuracy"])

_VALID_PROPS = {"min", "ppm", "rpm", "apm"}

# Aggregate gold_prediction_accuracy by (model_id, prop).
# hit_rate is computed only over rows that have a book_line (nullif keeps
# rows where hit IS NULL out of the avg without breaking the other metrics).
_ACCURACY_SQL = """
SELECT
    model_id::text                                              AS model_id,
    prop,
    count(*)                                                    AS n_games,
    count(*)    FILTER (WHERE hit IS NOT NULL)                  AS n_with_book_line,
    round(avg(hit)::numeric, 4)                                 AS hit_rate,
    round(avg(q50_below_prediction)::numeric, 4)                AS q50_calibration,
    round(avg(abs_error)::numeric, 6)                           AS mae,
    round(avg(signed_error)::numeric, 6)                        AS signed_bias,
    max(game_date)                                              AS scored_through
FROM gold.gold_prediction_accuracy
WHERE model_id = %(model_id)s::uuid
  AND (%(prop)s   IS NULL OR prop      = %(prop)s)
  AND (%(since)s  IS NULL OR game_date >= %(since)s::date)
GROUP BY model_id, prop
ORDER BY prop
"""


def _validate_prop(prop: str | None) -> str | None:
    if prop is None:
        return None
    normalized = prop.lower()
    if normalized not in _VALID_PROPS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid prop '{prop}'. Valid options: {sorted(_VALID_PROPS)}",
        )
    return normalized


@router.get(
    "/models/{model_id}/accuracy",
    response_model=ModelAccuracy,
    summary="Backtesting accuracy for a model version",
    description=(
        "Returns hit rate, Q50 calibration, MAE, and signed bias for a specific "
        "model version, broken down by prop type. Aggregated from "
        "**gold.gold_prediction_accuracy** (populated daily by the Airflow pipeline). "
        "Games whose actuals have not yet landed are excluded automatically."
    ),
)
def model_accuracy(
    model_id: str,
    prop: str | None = Query(
        default=None,
        description="Filter to one prop type: min | ppm | rpm | apm.",
    ),
    since: str | None = Query(
        default=None,
        description="Earliest game_date to include (YYYY-MM-DD).",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
) -> ModelAccuracy:
    rows = db.query(
        _ACCURACY_SQL,
        {
            "model_id": model_id,
            "prop": _validate_prop(prop),
            "since": since,
        },
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No scored predictions found for model_id='{model_id}'. "
                "The model may not exist, may not have run yet, or actuals "
                "may not have landed for the requested date range."
            ),
        )

    breakdown = [PropAccuracy(**row) for row in rows]
    return ModelAccuracy(model_id=model_id, breakdown=breakdown)
