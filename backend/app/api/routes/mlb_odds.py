from __future__ import annotations

from fastapi import APIRouter, Response

from app.schemas.mlb_odds import MlbOddsResponse
from app.services.mlb_odds import get_today_odds

router = APIRouter(tags=["mlb"])


@router.get("/mlb/odds/today", response_model=MlbOddsResponse)
async def mlb_odds_today(response: Response) -> MlbOddsResponse:
    response.headers["Cache-Control"] = "no-store"
    return await get_today_odds()
