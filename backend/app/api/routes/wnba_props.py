from __future__ import annotations

from fastapi import APIRouter, Response

from app.schemas.wnba_props import WnbaPropsResponse
from app.services.parlay_props import get_today_props

router = APIRouter(tags=["wnba"])


@router.get("/wnba/props/today", response_model=WnbaPropsResponse)
async def wnba_props_today(response: Response) -> WnbaPropsResponse:
    response.headers["Cache-Control"] = "no-store"
    return await get_today_props()
