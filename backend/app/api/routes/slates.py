"""GET /api/slates/{book} — latest prop lines per bookmaker from silver.silver_props."""
from __future__ import annotations

import datetime

from fastapi import APIRouter, HTTPException, Query

from app.core import db

router = APIRouter(tags=["slates"])

_BOOK_ALIASES = {
    "prizepicks": ["prizepicks", "prize picks"],
    "underdog": ["underdog", "underdog fantasy"],
    "draftkings": ["draftkings", "draftkings pick6", "dk pick6"],
    "betr": ["betr", "betr dfs"],
}

_PROPS_SQL = """
SELECT
    bookmaker,
    market_category,
    player_id,
    player_name,
    player_name_raw,
    normalized_name,
    side,
    game_date,
    line,
    odds,
    last_update_at,
    prop_source
FROM silver.silver_props
WHERE game_date = %(game_date)s
  AND lower(bookmaker) = ANY(%(bookmakers)s)
ORDER BY player_name, market_category, side
LIMIT %(limit)s
"""


def _bookmaker_names(book: str) -> list[str]:
    key = book.lower()
    names = _BOOK_ALIASES.get(key)
    if names is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown book '{book}'. Valid: {', '.join(_BOOK_ALIASES)}",
        )
    return [n.lower() for n in names]


@router.get("/slates/{book}")
def get_slate(
    book: str,
    date: str | None = Query(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Slate date. Defaults to today.",
    ),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict:
    """Return the latest prop lines for a DFS book from **silver.silver_props**."""
    target_date = date or str(datetime.date.today())
    rows = db.query(
        _PROPS_SQL,
        {
            "game_date": target_date,
            "bookmakers": _bookmaker_names(book),
            "limit": limit,
        },
    )
    return {
        "book": book.lower(),
        "game_date": target_date,
        "count": len(rows),
        "props": rows,
    }
