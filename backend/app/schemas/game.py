"""Pydantic schemas for /games/{date} endpoint.

Source table: silver.silver_games
"""
from __future__ import annotations

import datetime

from pydantic import BaseModel

from app.schemas.prediction import PropPrediction


class Game(BaseModel):
    """One row from silver_games."""
    game_date: datetime.date
    game_id: str | None = None
    event_id: int | None = None
    home_team_abbrev: str
    away_team_abbrev: str
    season_year: str | None = None
    source: str | None = None

    model_config = {"from_attributes": True}


class GameWithProps(Game):
    """Game enriched with all prop lines for that matchup."""
    props: list[PropPrediction] = []
