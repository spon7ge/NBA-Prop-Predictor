"""GET /api/live-props — enriched prop predictions from ml.live_prop_predictions."""
from __future__ import annotations

import datetime

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.live_prop import (
    Form,
    GameContext,
    LivePropPick,
    LivePropsResponse,
    ModelOutput,
    VsOpp,
)

router = APIRouter(tags=["live-props"])

_VALID_LEAGUES = {"nba", "wnba"}

def _build_sql(league: str) -> str:
    """Return a parameterised SQL query for the league-specific table."""
    table = f"ml.{league}_live_prop_predictions"
    return f"""
SELECT
    run_at,
    game_date,
    player_name,
    team_abbr,
    opponent_abbr,
    is_home,
    market,
    bookmaker,
    line,
    stat_q10,
    stat_q50,
    stat_q90,
    min_q10,
    min_q50,
    min_q90,
    p_over,
    p_under,
    opp_def_rating,
    opp_def_rating_rank,
    opp_pace,
    team_def_rating,
    team_pace,
    game_total,
    team_spread,
    over_l5,
    over_l10,
    over_l15,
    vs_opp_n_games,
    vs_opp_avg_stat,
    vs_opp_over_rate
FROM {table}
WHERE
    game_date = %(game_date)s
    AND (%(market)s    IS NULL OR market    = %(market)s)
    AND (%(bookmaker)s IS NULL OR lower(bookmaker) = lower(%(bookmaker)s))
    AND (%(player)s    IS NULL OR lower(player_name) LIKE lower(%(player)s))
    AND run_at = (
        SELECT MAX(run_at)
        FROM {table}
        WHERE game_date = %(game_date)s
    )
ORDER BY player_name, market, bookmaker
LIMIT %(limit)s
"""


def _lean(p_over: float | None, p_under: float | None) -> str | None:
    if p_over is None or p_under is None:
        return None
    return "OVER" if p_over >= p_under else "UNDER"


def _row_to_pick(row: dict, league: str) -> LivePropPick:
    p_over  = row.get("p_over")
    p_under = row.get("p_under")
    return LivePropPick(
        platform=row.get("bookmaker", ""),
        player=row.get("player_name", ""),
        team_abbr=row.get("team_abbr"),
        opponent_abbr=row.get("opponent_abbr"),
        is_home=row.get("is_home"),
        market=row.get("market", ""),
        line=row.get("line"),
        game_date=row.get("game_date"),
        league=league,
        run_at=row.get("run_at"),
        model=ModelOutput(
            p_over=p_over,
            p_under=p_under,
            lean=_lean(p_over, p_under),
            min_q10=row.get("min_q10"),
            min_q50=row.get("min_q50"),
            min_q90=row.get("min_q90"),
            stat_q10=row.get("stat_q10"),
            stat_q50=row.get("stat_q50"),
            stat_q90=row.get("stat_q90"),
        ),
        game_context=GameContext(
            opp_def_rating=row.get("opp_def_rating"),
            opp_def_rating_rank=row.get("opp_def_rating_rank"),
            opp_pace=row.get("opp_pace"),
            team_def_rating=row.get("team_def_rating"),
            team_pace=row.get("team_pace"),
            game_total=row.get("game_total"),
            team_spread=row.get("team_spread"),
        ),
        form=Form(
            over_l5=row.get("over_l5"),
            over_l10=row.get("over_l10"),
            over_l15=row.get("over_l15"),
        ),
        vs_opp=VsOpp(
            n_games=row.get("vs_opp_n_games"),
            avg_stat=row.get("vs_opp_avg_stat"),
            over_rate_at_line=row.get("vs_opp_over_rate"),
        ),
    )


@router.get("/live-props", response_model=LivePropsResponse)
def list_live_props(
    date: str | None = Query(
        default=None,
        description="Slate date YYYY-MM-DD. Defaults to today.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    league: str = Query(default="nba", description="nba | wnba"),
    market: str | None = Query(default=None, description="PTS | AST | REB"),
    bookmaker: str | None = Query(
        default=None,
        description="Filter by bookmaker, e.g. PrizePicks, Underdog, Betr",
    ),
    player: str | None = Query(
        default=None,
        description="Partial player name filter (case-insensitive).",
    ),
    limit: int = Query(default=1000, ge=1, le=5000),
) -> LivePropsResponse:
    """Return the most recently generated enriched prop picks.

    Each pick includes:
    - **model** — stat quantiles (q10/q50/q90), p_over, p_under, lean
    - **game_context** — opp_def_rating, opp_pace, game_total, spread
    - **form** — over-rate vs the line in the last 5 / 10 / 15 games
    - **vs_opp** — historical stats against tonight's opponent
    """
    league_lc = league.lower()
    if league_lc not in _VALID_LEAGUES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid league '{league}'. Valid: {sorted(_VALID_LEAGUES)}",
        )

    target_date = date or str(datetime.date.today())
    player_like = f"%{player}%" if player else None

    rows = db.query(
        _build_sql(league_lc),
        {
            "game_date": target_date,
            "market":    market,
            "bookmaker": bookmaker,
            "player":    player_like,
            "limit":     limit,
        },
    )

    picks = [_row_to_pick(r, league_lc) for r in rows]

    return LivePropsResponse(
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        league=league_lc,
        game_date=target_date,
        n_picks=len(picks),
        picks=picks,
    )
