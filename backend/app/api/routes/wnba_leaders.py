from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_leaders import WnbaLeadersResponse
from app.services.wnba_leaders import get_wnba_leaders

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/leaders", response_model=WnbaLeadersResponse)
async def wnba_leaders(response: Response) -> WnbaLeadersResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_wnba_leaders()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA leaders unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA leaders are temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
