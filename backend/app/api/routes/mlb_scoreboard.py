from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Response

from app.schemas.mlb_scoreboard import MlbScoreboardResponse
from app.services.mlb_scoreboard import get_scoreboard_for_date, get_today_scoreboard

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mlb"])

_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/mlb/scoreboard", response_model=MlbScoreboardResponse)
async def mlb_scoreboard_by_date(
    response: Response,
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> MlbScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_scoreboard_for_date(date)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("MLB scoreboard unavailable for %s: %s", date, exc)
        raise HTTPException(
            status_code=502,
            detail="MLB scoreboard is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc


@router.get("/mlb/scoreboard/today", response_model=MlbScoreboardResponse)
async def mlb_scoreboard_today(response: Response) -> MlbScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_today_scoreboard()
    except HTTPException:
        raise
    except Exception as exc:
        # Upstream or payload failure must surface as an uncacheable 502 so
        # clients never latch onto an error for the length of a cache TTL.
        logger.warning("MLB scoreboard unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="MLB scoreboard is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
