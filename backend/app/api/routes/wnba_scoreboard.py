from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_scoreboard import WnbaScoreboardResponse
from app.services.wnba_scoreboard import get_today_scoreboard

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])

_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/scoreboard/today", response_model=WnbaScoreboardResponse)
async def wnba_scoreboard_today(response: Response) -> WnbaScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_today_scoreboard()
    except HTTPException:
        raise
    except Exception as exc:
        # Any upstream or payload failure must surface as an uncacheable 502 so
        # clients never latch onto an error for the length of a cache TTL.
        logger.warning("WNBA scoreboard unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA scoreboard is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
