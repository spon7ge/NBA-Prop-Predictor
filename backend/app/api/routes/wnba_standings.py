from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_standings import WnbaStandingsResponse
from app.services.wnba_standings import get_wnba_standings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/standings", response_model=WnbaStandingsResponse)
async def wnba_standings(response: Response) -> WnbaStandingsResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_wnba_standings()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA standings unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA standings are temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
