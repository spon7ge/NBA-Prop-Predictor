from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_RESPONSE_CONFIG = ConfigDict(json_schema_serialization_defaults_required=True)


class MlbOddsGame(BaseModel):
    model_config = _RESPONSE_CONFIG

    home_abbrev: str
    away_abbrev: str
    spread_team_abbrev: str | None = None
    spread_line: float | None = None
    total: float | None = None
    game_date: str | None = None
    sportsbook: str | None = None


class MlbOddsResponse(BaseModel):
    model_config = _RESPONSE_CONFIG

    as_of: str
    sportsbook: str = "draftkings"
    games: list[MlbOddsGame] = Field(default_factory=list)
    error: str | None = None
