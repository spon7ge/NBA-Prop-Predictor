"""Thin HTTP client for ParlayAPI."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import PARLAY_API_KEY

PARLAY_BASE_URL = "https://parlay-api.com/v1"
DEFAULT_TIMEOUT_SECONDS = 12.0


def require_parlay_api_key() -> str:
    if not PARLAY_API_KEY:
        raise RuntimeError("PARLAY_API_KEY is not configured")
    return PARLAY_API_KEY


async def parlay_get(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """
    GET a ParlayAPI path (e.g. ``/sports/basketball_wnba/props``).

    Auth uses query ``apiKey`` (Parlay docs) plus ``X-API-Key`` header.
    """
    api_key = require_parlay_api_key()
    query = dict(params or {})
    query["apiKey"] = api_key
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    url = f"{PARLAY_BASE_URL}{path}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.get(url, headers=headers, params=query)
        try:
            res.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = (exc.response.text or "")[:500]
            raise httpx.HTTPStatusError(
                f"{exc} body={body!r}",
                request=exc.request,
                response=exc.response,
            ) from None
        return res.json()
