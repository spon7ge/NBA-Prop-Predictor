from __future__ import annotations

from pydantic import BaseModel


class WnbaPlayerAverages(BaseModel):
    pts: str
    reb: str
    ast: str
    fg_pct: str
    fg3_pct: str


class WnbaPlayerGame(BaseModel):
    game_id: str
    game_date: str
    matchup: str
    min: str
    pts: str
    fg: str
    three_pt: str
    ft: str
    reb: str
    ast: str
    to: str
    stl: str
    blk: str


class WnbaPlayerResponse(BaseModel):
    player_id: str
    name: str
    position: str | None
    team_name: str
    team_abbrev: str
    headshot_url: str | None
    season: int
    averages: WnbaPlayerAverages
    games: list[WnbaPlayerGame]
    source_label: str = "stats.wnba.com"
