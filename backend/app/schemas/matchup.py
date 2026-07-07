"""Pydantic schemas for matchup / schedule context features."""
from __future__ import annotations

import datetime

from pydantic import BaseModel


class MatchupFeatures(BaseModel):
    game_id: str
    player_id: int
    player_name: str
    game_date: datetime.date
    season_year: str | None = None
    team_abbreviation: str | None = None
    opp_team_abbreviation: str | None = None
    is_home: int
    is_b2b: int
    days_rest: int | None = None
    game_number: int | None = None
    team_pace_roll10: float | None = None
    team_def_rating_roll10: float | None = None
    opp_def_rating_roll10: float | None = None
    opp_pace_roll10: float | None = None
    expected_pace: float | None = None
    pace_differential: float | None = None
    team_spread: float | None = None
    game_total: float | None = None

    model_config = {"from_attributes": True}
