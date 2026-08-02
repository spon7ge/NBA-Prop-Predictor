from __future__ import annotations

from pydantic import BaseModel


class WnbaFuturesEntry(BaseModel):
    team_id: str
    abbrev: str
    name: str
    logo_url: str | None
    odds_american: str


class WnbaFuturesMarket(BaseModel):
    id: str
    name: str
    display_name: str
    provider: str
    entries: list[WnbaFuturesEntry]


class WnbaFuturesResponse(BaseModel):
    season: int
    as_of: str
    markets: list[WnbaFuturesMarket]
    error: str | None = None
