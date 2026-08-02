from __future__ import annotations

import json
from pathlib import Path

from app.openapi_export import REQUIRED_FRONTEND_PATHS, export_openapi


def test_export_openapi_includes_frontend_paths(tmp_path: Path) -> None:
    out = tmp_path / "openapi.json"
    written = export_openapi(out)
    assert written == out
    spec = json.loads(out.read_text(encoding="utf-8"))
    paths = spec["paths"]
    for path in REQUIRED_FRONTEND_PATHS:
        assert path in paths, f"missing OpenAPI path: {path}"
