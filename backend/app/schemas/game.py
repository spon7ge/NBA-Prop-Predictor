"""Pydantic schemas for /games/{date} endpoint."""
from __future__ import annotations

import datetime

from pydantic import BaseModel

from app.schemas.ml_prediction import MLPrediction
from app.schemas.prop import PropLine


class Game(BaseModel):
    game_date: datetime.date
    game_id: str | None = None
    event_id: int | None = None
    home_team_abbrev: str
    away_team_abbrev: str
    season_year: str | None = None
    source: str | None = None

    model_config = {"from_attributes": True}


class GameWithProps(Game):
    props: list[PropLine] = []


class GameWithPredictions(Game):
    predictions: list[MLPrediction] = []


class GameSlate(BaseModel):
    game_date: datetime.date
    games: list[Game]
    props: list[PropLine] = []
    predictions: list[MLPrediction] = []
