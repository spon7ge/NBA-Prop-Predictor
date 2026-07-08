"""Pydantic schemas for model accuracy / backtesting metrics."""
from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class PropAccuracy(BaseModel):
    """Aggregated backtesting metrics for one (model_id, prop) pair."""

    model_id: str = Field(description="UUID of the model version from ml.model_registry.")
    prop: str = Field(description="Model target: min, ppm, rpm, or apm.")
    n_games: int = Field(description="Number of scored predictions (games with landed actuals).")
    n_with_book_line: int = Field(
        description="Subset of n_games where a book line was available for hit-rate computation."
    )
    hit_rate: float | None = Field(
        default=None,
        description=(
            "Fraction of predictions where the model's predicted direction "
            "(over/under the book line) matched the actual outcome. "
            "Null when no book lines are available."
        ),
    )
    q50_calibration: float | None = Field(
        default=None,
        description=(
            "Fraction of actuals that fell below the median (Q50) prediction. "
            "A perfectly calibrated model produces 0.50."
        ),
    )
    mae: float | None = Field(
        default=None,
        description=(
            "Mean absolute error in model-native units "
            "(per-minute rates for ppm/rpm/apm; minutes for min)."
        ),
    )
    signed_bias: float | None = Field(
        default=None,
        description=(
            "Mean signed error (prediction − actual) in model-native units. "
            "Positive = model systematically over-predicts."
        ),
    )
    scored_through: datetime.date | None = Field(
        default=None,
        description="Latest game_date included in these metrics.",
    )

    model_config = {"from_attributes": True}


class ModelAccuracy(BaseModel):
    """All prop-level accuracy breakdowns for a single model version."""

    model_id: str
    breakdown: list[PropAccuracy] = Field(
        description="One entry per prop type that has scored predictions."
    )
