from __future__ import annotations

import logging

from fastapi import APIRouter, Response

from app.schemas.wnba_odds import WnbaOddsResponse
from app.services.sharp_odds import get_today_odds

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])


@router.get("/wnba/odds/today", response_model=WnbaOddsResponse)
async def wnba_odds_today(response: Response) -> WnbaOddsResponse:
    response.headers["Cache-Control"] = "no-store"
    return await get_today_odds()
