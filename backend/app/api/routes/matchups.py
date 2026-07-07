"""GET /api/matchups/{date} — pre-game context from gold.gold_matchup_features."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.core import db
from app.schemas.matchup import MatchupFeatures

router = APIRouter(tags=["matchups"])

_SQL = """
SELECT
    game_id,
    player_id,
    player_name,
    game_date,
    season_year,
    team_abbreviation,
    opp_team_abbreviation,
    is_home,
    is_b2b,
    days_rest,
    game_number,
    team_pace_roll10,
    team_def_rating_roll10,
    opp_def_rating_roll10,
    opp_pace_roll10,
    expected_pace,
    pace_differential,
    team_spread,
    game_total
FROM gold.gold_matchup_features
WHERE game_date = %(game_date)s
  AND (%(player_id)s IS NULL OR player_id = %(player_id)s)
  AND (%(team)s IS NULL OR team_abbreviation = %(team)s)
ORDER BY team_abbreviation, player_name
LIMIT %(limit)s
"""


@router.get("/matchups/{date}", response_model=list[MatchupFeatures])
def list_matchups(
    date: str,
    player_id: int | None = Query(default=None),
    team: str | None = Query(default=None, description="Team tricode filter."),
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[MatchupFeatures]:
    """Return opponent/schedule context for every player-game on a date."""
    rows = db.query(
        _SQL,
        {
            "game_date": date,
            "player_id": player_id,
            "team": team.upper() if team else None,
            "limit": limit,
        },
    )
    return [MatchupFeatures(**row) for row in rows]
