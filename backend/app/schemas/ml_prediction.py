"""Pydantic schemas for ML model predictions (ml.predictions)."""
from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class MLPrediction(BaseModel):
    prop: str = Field(description="Model target: min, ppm, rpm, or apm.")
    game_id: str
    player_id: int
    prediction: float = Field(description="Median quantile (q0.50) model output.")
    predicted_at: datetime.datetime
    game_date: datetime.date | None = None
    player_name: str | None = None
    model_path: str | None = None

    model_config = {"from_attributes": True}


class PlayerPredictions(BaseModel):
    player_id: int
    player_name: str | None = None
    predictions: list[MLPrediction] = []
