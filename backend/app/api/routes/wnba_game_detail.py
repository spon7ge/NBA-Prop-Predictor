from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_game_detail import WnbaGameDetail
from app.services.wnba_game_detail import get_game_detail

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])

_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/games/{espn_event_id}", response_model=WnbaGameDetail)
async def wnba_game_detail(
    espn_event_id: str, response: Response
) -> WnbaGameDetail:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_game_detail(espn_event_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail="Game not found",
            headers=_NO_STORE,
        ) from exc
    except Exception as exc:
        logger.warning("WNBA game detail unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA game detail is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
