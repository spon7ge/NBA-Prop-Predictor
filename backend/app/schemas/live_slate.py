"""Pydantic schemas for live multi-leg parlays (ml.*_live_slates)."""
from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field


class LiveSlateParlay(BaseModel):
    """One greedy parlay row (FlatParlayRow-compatible; LEGS nested)."""

    model_config = {"extra": "allow"}

    PARLAY_PROB: float | None = None
    EV: float | None = None
    EV_DOLLARS: float | None = None
    KELLY: float | None = None
    KELLY_QUARTER: float | None = None
    N_LEGS: int | None = None
    LEGS: list[dict[str, Any]] = Field(default_factory=list)
    STRATEGY_TIER: int | None = None
    COMBO_PROFILE: str | None = None
    ANCHOR_WIN_PROB: float | None = None
    ANCHOR_NAME: str | None = None


class LiveSlatesResponse(BaseModel):
    """Envelope for GET /api/live-slates."""

    generated_at: datetime.datetime
    league: str
    game_date: str
    run_at: datetime.datetime | None = None
    count: int
    # leg_count → book → parlays  (keys are strings in JSON: "2","3","5","6")
    slates: dict[str, dict[str, list[LiveSlateParlay]]]
