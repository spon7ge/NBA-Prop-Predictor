from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_scoreboard import WnbaScoreboardResponse
from app.services.wnba_scoreboard import get_today_scoreboard

router = APIRouter(tags=["wnba"])


@router.get("/wnba/scoreboard/today", response_model=WnbaScoreboardResponse)
async def wnba_scoreboard_today(response: Response) -> WnbaScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_today_scoreboard()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
            headers={"Cache-Control": "no-store"},
        ) from exc
