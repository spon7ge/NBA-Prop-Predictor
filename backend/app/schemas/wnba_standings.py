from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ConferenceKey = Literal["east", "west"]


class WnbaStandingsRow(BaseModel):
    rank: int
    team_id: str
    abbrev: str
    name: str
    logo_url: str | None
    wins: int
    losses: int
    wl: str
    pct: str
    gb: str
    home: str
    away: str
    l10: str
    diff: str
    streak: str


class WnbaStandingsConference(BaseModel):
    key: ConferenceKey
    label: str
    teams: list[WnbaStandingsRow]


class WnbaStandingsResponse(BaseModel):
    season: int
    conferences: list[WnbaStandingsConference]
