"""Pydantic schemas for live prop predictions (ml.live_prop_predictions)."""
from __future__ import annotations

import datetime

from pydantic import BaseModel, computed_field


class ModelOutput(BaseModel):
    p_over: float | None = None
    p_under: float | None = None
    lean: str | None = None
    min_q10: float | None = None
    min_q50: float | None = None
    min_q90: float | None = None
    stat_q10: float | None = None
    stat_q50: float | None = None
    stat_q90: float | None = None


class GameContext(BaseModel):
    opp_def_rating: float | None = None
    opp_def_rating_rank: int | None = None
    opp_pace: float | None = None
    team_def_rating: float | None = None
    team_pace: float | None = None
    game_total: float | None = None
    team_spread: float | None = None


class Form(BaseModel):
    over_l5: float | None = None
    over_l10: float | None = None
    over_l15: float | None = None


class VsOpp(BaseModel):
    n_games: int | None = None
    avg_stat: float | None = None
    over_rate_at_line: float | None = None


class LivePropPick(BaseModel):
    """One enriched prop pick from ml.live_prop_predictions."""

    platform: str
    player: str
    team_abbr: str | None = None
    opponent_abbr: str | None = None
    is_home: bool | None = None
    market: str
    line: float | None = None
    game_date: datetime.date | None = None
    league: str | None = None
    run_at: datetime.datetime | None = None

    model: ModelOutput
    game_context: GameContext
    form: Form
    vs_opp: VsOpp

    model_config = {"from_attributes": True}


class LivePropsResponse(BaseModel):
    """Envelope for the GET /api/live-props response."""

    generated_at: datetime.datetime
    league: str
    game_date: str
    n_picks: int
    picks: list[LivePropPick]
