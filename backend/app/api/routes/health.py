"""GET /api/health — liveness + DB connectivity."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core import db

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Basic health check with a lightweight DB ping."""
    try:
        row = db.query_one("SELECT current_database() AS db_name")
        if row is None:
            return {"status": "degraded", "db": "empty response"}
        return {
            "status": "ok",
            "db": "connected",
            "database": row["db_name"],
        }
    except HTTPException as exc:
        return {"status": "degraded", "db": str(exc.detail)}
