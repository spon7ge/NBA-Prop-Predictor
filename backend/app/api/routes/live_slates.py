"""GET /api/live-slates — greedy multi-leg parlays from ml.*_live_slates."""
from __future__ import annotations

import datetime

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.live_slate import LiveSlateParlay, LiveSlatesResponse

router = APIRouter(tags=["live-slates"])

_VALID_LEAGUES = {"nba", "wnba"}
_VALID_BOOKS = {"prizepicks", "underdog", "draftkings", "betr"}
_LEG_COUNTS = (2, 3, 5, 6)


def _empty_slates() -> dict[str, dict[str, list[LiveSlateParlay]]]:
    return {
        str(n): {b: [] for b in sorted(_VALID_BOOKS)}
        for n in _LEG_COUNTS
    }


def _build_sql(league: str) -> str:
    table = f"ml.{league}_live_slates"
    return f"""
SELECT
    run_at,
    game_date,
    bookmaker,
    n_legs,
    parlays
FROM {table}
WHERE
    game_date = %(game_date)s
    AND (%(bookmaker)s IS NULL OR lower(bookmaker) = lower(%(bookmaker)s))
    AND (%(n_legs)s IS NULL OR n_legs = %(n_legs)s)
    AND run_at = (
        SELECT MAX(run_at)
        FROM {table}
        WHERE game_date = %(game_date)s
    )
ORDER BY n_legs, bookmaker
"""


@router.get("/live-slates", response_model=LiveSlatesResponse)
def list_live_slates(
    date: str | None = Query(
        default=None,
        description="Slate date YYYY-MM-DD. Defaults to today.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    league: str = Query(default="nba", description="nba | wnba"),
    book: str | None = Query(
        default=None,
        description="Optional book filter: prizepicks | underdog | draftkings | betr",
    ),
    n_legs: int | None = Query(
        default=None,
        description="Optional leg count filter: 2 | 3 | 5 | 6",
        ge=2,
        le=6,
    ),
) -> LiveSlatesResponse:
    """Return the most recent greedy multi-leg parlays for Top Legs.

    Response ``slates`` is nested as ``{ "2": { "prizepicks": [...], ... }, ... }``.
    """
    league_lc = league.lower()
    if league_lc not in _VALID_LEAGUES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid league '{league}'. Valid: {sorted(_VALID_LEAGUES)}",
        )

    book_lc = book.lower() if book else None
    if book_lc is not None and book_lc not in _VALID_BOOKS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid book '{book}'. Valid: {sorted(_VALID_BOOKS)}",
        )

    if n_legs is not None and n_legs not in _LEG_COUNTS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid n_legs '{n_legs}'. Valid: {list(_LEG_COUNTS)}",
        )

    target_date = date or str(datetime.date.today())
    rows = db.query(
        _build_sql(league_lc),
        {
            "game_date": target_date,
            "bookmaker": book_lc,
            "n_legs": n_legs,
        },
    )

    slates = _empty_slates()
    run_at: datetime.datetime | None = None
    count = 0

    for row in rows:
        if run_at is None:
            run_at = row.get("run_at")
        bk = str(row.get("bookmaker") or "").lower()
        legs = int(row.get("n_legs") or 0)
        raw_parlays = row.get("parlays") or []
        if not isinstance(raw_parlays, list):
            continue
        if str(legs) not in slates:
            continue
        if bk not in slates[str(legs)]:
            # Accept unknown books without crashing the envelope shape
            slates[str(legs)][bk] = []
        parsed = [LiveSlateParlay.model_validate(p) for p in raw_parlays if isinstance(p, dict)]
        slates[str(legs)][bk] = parsed
        count += len(parsed)

    return LiveSlatesResponse(
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        league=league_lc,
        game_date=target_date,
        run_at=run_at,
        count=count,
        slates=slates,
    )
