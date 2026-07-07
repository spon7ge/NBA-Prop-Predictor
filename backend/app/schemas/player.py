"""Pydantic schemas for /player/{player_id} endpoint.

Sources:
  - silver.silver_players  (profile)
  - gold.gold_player_game_stats  (recent games)
  - gold.gold_player_rolling_avg_5  (L5 context)
  - gold.gold_player_rolling_avg_10  (L10 context)
"""
from __future__ import annotations

import datetime

from pydantic import BaseModel


class PlayerGame(BaseModel):
    """One row from gold_player_game_stats."""
    game_id: str
    game_date: datetime.date
    matchup: str | None = None
    opp_team_abbreviation: str | None = None
    wl: str | None = None
    season_year: str | None = None
    season_type: str | None = None
    starting: int
    is_home: int
    # Counting stats
    min: float
    pts: float
    reb: float
    ast: float
    tov: float
    stl: float
    blk: float
    fgm: float | None = None
    fga: float | None = None
    fg3a: float | None = None
    fta: float | None = None
    plus_minus: float | None = None
    # Efficiency
    usg_pct: float | None = None
    ts_pct: float | None = None
    efg_pct: float | None = None
    net_rating: float | None = None
    # Per-minute rates
    pts_per_min: float | None = None
    reb_per_min: float | None = None
    ast_per_min: float | None = None

    model_config = {"from_attributes": True}


class RollingAvg5(BaseModel):
    """L5 rolling averages for a player."""
    min_roll5: float | None = None
    pts_roll5: float | None = None
    reb_roll5: float | None = None
    ast_roll5: float | None = None
    tov_roll5: float | None = None
    usg_pct_roll5: float | None = None
    plus_minus_roll5: float | None = None
    pts_per_min_roll5: float | None = None
    reb_per_min_roll5: float | None = None
    ast_per_min_roll5: float | None = None


class RollingAvg10(BaseModel):
    """L10 rolling averages + model features for a player."""
    min_roll10: float | None = None
    pts_roll10: float | None = None
    reb_roll10: float | None = None
    ast_roll10: float | None = None
    tov_roll10: float | None = None
    usg_pct_roll10: float | None = None
    plus_minus_roll10: float | None = None
    pts_per_min_roll10: float | None = None
    min_season_mean: float | None = None
    starter_roll10_pct: float | None = None
    consec_starts: int | None = None
    team_min_rank_l10: int | None = None
    team_usg_rank_l10: int | None = None
    min_rate_of_change: float | None = None


class MLPredictionSummary(BaseModel):
    """Latest model output for one prop type."""
    prop: str
    game_id: str
    prediction: float
    predicted_at: datetime.datetime
    game_date: datetime.date | None = None

    model_config = {"from_attributes": True}


class PlayerProfile(BaseModel):
    """Full player response — profile + recent games + rolling context."""
    player_id: int
    player_name: str
    normalized_name: str
    team_abbreviation: str | None = None
    team_name: str | None = None
    career_game_count: int | None = None
    recent_games: list[PlayerGame] = []
    rolling_avg_5: RollingAvg5 | None = None
    rolling_avg_10: RollingAvg10 | None = None
    predictions: list["MLPredictionSummary"] = []

    model_config = {"from_attributes": True}
