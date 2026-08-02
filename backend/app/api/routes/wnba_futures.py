from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_futures import WnbaFuturesResponse
from app.services.wnba_futures import get_wnba_futures

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/futures", response_model=WnbaFuturesResponse)
async def wnba_futures(response: Response) -> WnbaFuturesResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_wnba_futures()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA futures unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA futures are temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
