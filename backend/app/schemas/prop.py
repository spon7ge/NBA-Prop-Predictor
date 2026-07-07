"""Pydantic schemas for sportsbook prop lines (silver/gold)."""
from __future__ import annotations

import datetime

from pydantic import BaseModel


class PropLine(BaseModel):
    """One prop quote from silver.silver_props or gold.gold_prop_history."""
    bookmaker: str
    market_category: str
    player_id: int | None = None
    player_name: str
    player_name_raw: str | None = None
    normalized_name: str | None = None
    side: str
    game_date: datetime.date
    line: float
    odds: int | None = None
    prop_source: str | None = None
    last_update_at: datetime.datetime | None = None
    player_team_abbrev: str | None = None
    home_team_abbrev: str | None = None
    away_team_abbrev: str | None = None
    game_season_year: str | None = None
    min_roll5: float | None = None
    pts_per_min_roll5: float | None = None
    reb_per_min_roll5: float | None = None
    ast_per_min_roll5: float | None = None
    min_roll10: float | None = None
    pts_per_min_roll10: float | None = None
    team_min_rank_l10: int | None = None
    team_usg_rank_l10: int | None = None
    expected_pace: float | None = None
    opp_def_rating_roll10: float | None = None
    team_spread: float | None = None
    game_total: float | None = None

    model_config = {"from_attributes": True}
