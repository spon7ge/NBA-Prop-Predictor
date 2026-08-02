"""Export FastAPI OpenAPI schema to a stable JSON file for frontend codegen."""

from __future__ import annotations

import json
from pathlib import Path

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "frontend" / "openapi.json"

# Paths used by frontend/src/lib/api.ts fetch helpers (with /api prefix as mounted).
REQUIRED_WNBA_PATHS = (
    "/api/wnba/scoreboard/today",
    "/api/wnba/games/{espn_event_id}",
    "/api/wnba/leaders",
    "/api/wnba/standings",
    "/api/wnba/odds/today",
    "/api/wnba/props/today",
    "/api/wnba/futures",
    "/api/wnba/player/{player_id}",
)

REQUIRED_MLB_PATHS = (
    "/api/mlb/scoreboard/today",
)

REQUIRED_FRONTEND_PATHS = REQUIRED_WNBA_PATHS + REQUIRED_MLB_PATHS


def export_openapi(path: Path | None = None) -> Path:
    """Write sorted OpenAPI JSON to ``path`` (default: frontend/openapi.json)."""
    out = path or DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    return out
