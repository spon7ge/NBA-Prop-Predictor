from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.wnba_scoreboard import GameStatus

__all__ = [
    "GameDetailLatestPlay",
    "GameDetailPlay",
    "GameDetailShot",
    "GameDetailTeam",
    "GameDetailTeamStat",
    "GameDetailWinProbability",
    "GameDetailWinProbabilityPoint",
    "GameStatus",
    "WnbaGameDetail",
]


class GameDetailTeam(BaseModel):
    id: str
    abbrev: str
    name: str
    score: int | None
    color: str


class GameDetailShot(BaseModel):
    id: str
    team_id: str
    player_name: str
    made: bool
    x: float
    y: float
    period: int
    clock: str


class GameDetailPlay(BaseModel):
    id: str
    team_id: str | None
    period: int
    clock: str
    text: str
    scoring: bool
    away_score: int
    home_score: int
    shooting: bool


class GameDetailLatestPlay(BaseModel):
    id: str
    clock: str
    period: int
    text: str
    team_id: str | None


class GameDetailWinProbabilityPoint(BaseModel):
    id: str
    period: int
    clock: str
    away_score: int
    home_score: int
    away_win_pct: int
    home_win_pct: int
    team_id: str | None


class GameDetailTeamStat(BaseModel):
    key: str
    label: str
    away_value: int
    home_value: int


class GameDetailWinProbability(BaseModel):
    summary: str | None
    timeline: list[GameDetailWinProbabilityPoint]
    team_stats: list[GameDetailTeamStat]


class WnbaGameDetail(BaseModel):
    espn_event_id: str
    league: Literal["wnba"] = "wnba"
    status: GameStatus
    status_label: str
    venue: str | None
    away: GameDetailTeam
    home: GameDetailTeam
    fg_made: int
    fg_attempted: int
    latest_play: GameDetailLatestPlay | None
    shots: list[GameDetailShot]
    plays: list[GameDetailPlay]
    win_probability: GameDetailWinProbability | None
    fetched_at: str
