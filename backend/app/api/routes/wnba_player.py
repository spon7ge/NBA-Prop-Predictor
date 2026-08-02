from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_player import WnbaPlayerResponse
from app.services.wnba_player import get_wnba_player

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/player/{player_id}", response_model=WnbaPlayerResponse)
async def wnba_player(player_id: str, response: Response) -> WnbaPlayerResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_wnba_player(player_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA player unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA player is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
