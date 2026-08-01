from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PROP_SPORTSBOOKS = (
    "fanduel",
    "draftkings",
    "caesars",
    "betmgm",
    "pinnacle",
    "bet365",
    "prizepicks",
    "underdog",
    "betr",
    "novig",
    "sleeper",
    "pick6",
)

_RESPONSE_CONFIG = ConfigDict(json_schema_serialization_defaults_required=True)


class WnbaPropBookQuote(BaseModel):
    model_config = _RESPONSE_CONFIG

    line: float
    odds_american: int | None = None


class WnbaPropLine(BaseModel):
    model_config = _RESPONSE_CONFIG

    player_name: str
    team_abbrev: str | None = None
    logo_url: str | None = None
    stat: str
    market_type: str
    side: str
    # Tip metadata from Parlay (used to flip to tomorrow's slate when today is final).
    game_date: str | None = None
    commence_time: str | None = None
    model_prediction: float | None = None
    over_under_pct: float | None = None
    ev: float | None = None
    fanduel: WnbaPropBookQuote | None = None
    draftkings: WnbaPropBookQuote | None = None
    caesars: WnbaPropBookQuote | None = None
    betmgm: WnbaPropBookQuote | None = None
    pinnacle: WnbaPropBookQuote | None = None
    bet365: WnbaPropBookQuote | None = None
    prizepicks: WnbaPropBookQuote | None = None
    underdog: WnbaPropBookQuote | None = None
    betr: WnbaPropBookQuote | None = None
    novig: WnbaPropBookQuote | None = None
    sleeper: WnbaPropBookQuote | None = None
    pick6: WnbaPropBookQuote | None = None


class WnbaPropsResponse(BaseModel):
    model_config = _RESPONSE_CONFIG

    as_of: str
    sportsbooks: list[str] = Field(default_factory=lambda: list(PROP_SPORTSBOOKS))
    props: list[WnbaPropLine] = Field(default_factory=list)
    error: str | None = None
