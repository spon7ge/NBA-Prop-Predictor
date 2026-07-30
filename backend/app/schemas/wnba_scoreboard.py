from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GameStatus = Literal["scheduled", "live", "halftime", "final"]


class WnbaTeam(BaseModel):
    abbrev: str
    name: str
    score: int | None = None


class WnbaGame(BaseModel):
    id: str
    league: Literal["wnba"] = "wnba"
    status: GameStatus
    status_label: str
    away: WnbaTeam
    home: WnbaTeam
    start_time_et: str


class WnbaScoreboardResponse(BaseModel):
    date: str = Field(description="YYYY-MM-DD in America/New_York")
    games: list[WnbaGame]
    fetched_at: str
