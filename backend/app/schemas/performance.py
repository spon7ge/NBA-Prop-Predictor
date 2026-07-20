"""Pydantic schemas for live prop grading / performance metrics."""
from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class HitRateBucket(BaseModel):
    """Hit rate for one slice (overall, market, book, or side)."""

    key: str = Field(description="Slice label, e.g. overall | PTS | PrizePicks | over")
    hits: int
    n: int = Field(description="Graded props excluding DNPs")
    hit_rate: float | None = None
    dnps: int = 0


class DailyHitRate(BaseModel):
    game_date: datetime.date
    hits: int
    n: int
    hit_rate: float | None = None


class BookDailyTrend(BaseModel):
    """Daily hit-rate series for one bookmaker."""

    bookmaker: str
    points: list[DailyHitRate] = Field(default_factory=list)


class GradedPick(BaseModel):
    """One graded prop for the Results feed."""

    game_date: datetime.date
    player_name: str
    team_abbr: str | None = None
    market: str
    bookmaker: str
    line: float | None = None
    side: str
    stat_q50: float | None = None
    p_over: float | None = None
    actual_stat: float | None = None
    hit: bool
    miss_reason: str
    abs_error: float | None = None


class GradedLeg(BaseModel):
    """One leg of a Top Legs parlay, joined to a graded prop when possible."""

    player_name: str
    team_abbr: str | None = None
    market: str
    line: float | None = None
    side: str
    actual_stat: float | None = None
    hit: bool | None = Field(
        default=None,
        description="True/False when graded; null if DNP or no matching grade.",
    )
    miss_reason: str | None = None


class GradedParlay(BaseModel):
    """A Top Legs parlay with per-leg outcomes."""

    game_date: datetime.date
    bookmaker: str
    n_legs: int
    legs_hit: int = Field(description="Legs that hit (excl. DNP / ungraded)")
    legs_scored: int = Field(description="Legs with a non-DNP grade")
    legs_pending: int = Field(description="Legs without a grade or DNP")
    cashed: bool | None = Field(
        default=None,
        description="True if every scored leg hit and none pending; False on any miss; null if incomplete.",
    )
    parlay_prob: float | None = None
    ev: float | None = None
    legs: list[GradedLeg]


class ParlaySummary(BaseModel):
    """Aggregate Top Legs results for the lookback window."""

    cashed: int = 0
    decided: int = Field(
        default=0,
        description="Parlays with no pending legs (fully graded, excl. all-DNP).",
    )
    cash_rate: float | None = None
    legs_hit: int = 0
    legs_scored: int = 0
    leg_hit_rate: float | None = None


class PerformanceResponse(BaseModel):
    """Envelope for GET /api/performance."""

    generated_at: datetime.datetime
    league: str
    days: int
    last_night: HitRateBucket
    last_n_days: HitRateBucket
    by_market: list[HitRateBucket]
    by_book: list[HitRateBucket]
    by_side: list[HitRateBucket]
    trend: list[DailyHitRate]
    trend_by_book: list[BookDailyTrend] = Field(
        default_factory=list,
        description="Per-bookmaker daily hit-rate series for the chart.",
    )
    brier_score: float | None = Field(
        default=None,
        description="Mean squared error of p_over vs over/under outcome (excl. DNP).",
    )
    recent_picks: list[GradedPick]
    parlay_summary: ParlaySummary = Field(default_factory=ParlaySummary)
    graded_parlays: list[GradedParlay] = Field(default_factory=list)
