from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GameStatus = Literal["scheduled", "live", "halftime", "final"]

_RESPONSE_CONFIG = ConfigDict(json_schema_serialization_defaults_required=True)


class MlbTeam(BaseModel):
    model_config = _RESPONSE_CONFIG

    abbrev: str
    name: str
    score: int | None = None
    record: str | None = None
    logo_url: str | None = None


class MlbGame(BaseModel):
    model_config = _RESPONSE_CONFIG

    id: str
    mlb_game_pk: str
    league: Literal["mlb"] = "mlb"
    status: GameStatus
    status_label: str
    away: MlbTeam
    home: MlbTeam
    start_time_et: str
    venue: str | None = None
    venue_city: str | None = None


class MlbScoreboardResponse(BaseModel):
    model_config = _RESPONSE_CONFIG

    date: str = Field(description="YYYY-MM-DD in America/New_York")
    games: list[MlbGame]
    fetched_at: str
