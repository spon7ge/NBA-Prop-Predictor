from __future__ import annotations

from pydantic import BaseModel, Field


class WnbaPropBookQuote(BaseModel):
    line: float
    odds_american: int


class WnbaPropLine(BaseModel):
    player_name: str
    team_abbrev: str | None = None
    logo_url: str | None = None
    stat: str
    market_type: str
    side: str
    model_prediction: float | None = None
    over_under_pct: float | None = None
    ev: float | None = None
    fanduel: WnbaPropBookQuote | None = None
    draftkings: WnbaPropBookQuote | None = None


class WnbaPropsResponse(BaseModel):
    as_of: str
    sportsbooks: list[str] = Field(default_factory=lambda: ["fanduel", "draftkings"])
    props: list[WnbaPropLine] = Field(default_factory=list)
    error: str | None = None
