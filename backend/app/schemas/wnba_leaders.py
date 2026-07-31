from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

LeaderCategoryKey = Literal[
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "three_pointers",
]


class WnbaLeaderRow(BaseModel):
    rank: int
    player_id: str
    name: str
    team_abbrev: str
    gp: int
    value: str


class WnbaLeaderCategory(BaseModel):
    key: LeaderCategoryKey
    label: str
    stat: str
    leaders: list[WnbaLeaderRow]


class WnbaLeadersResponse(BaseModel):
    season: int
    pace: Literal["per_game"] = "per_game"
    categories: list[WnbaLeaderCategory]
