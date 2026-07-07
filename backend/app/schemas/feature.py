"""Pydantic schemas for ML feature rows (ml.features*)."""
from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class MLFeatureRow(BaseModel):
    """Shared base features from ml.features (subset exposed via API)."""
    game_id: str
    player_id: int
    player_name: str | None = None
    game_date: datetime.date | None = None
    season_year: str | None = None
    team_abbreviation: str | None = None
    opp_team_abbreviation: str | None = None
    is_home: int | None = None
    days_rest: int | None = None
    is_b2b: int | None = None
    min_roll5: float | None = None
    pts_roll5: float | None = None
    reb_roll5: float | None = None
    ast_roll5: float | None = None
    usg_pct_roll5: float | None = None
    min_roll10: float | None = None
    usg_pct_roll10: float | None = None
    opp_def_rating_roll10: float | None = None
    expected_pace: float | None = None

    model_config = {"from_attributes": True, "extra": "allow"}


class PlayerSummary(BaseModel):
    player_id: int
    player_name: str
    normalized_name: str
    team_abbreviation: str | None = None
    team_name: str | None = None
    career_game_count: int | None = None

    model_config = {"from_attributes": True}


class PlayerListResponse(BaseModel):
    count: int
    players: list[PlayerSummary]
