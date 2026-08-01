#!/usr/bin/env python3
"""CLI: dump FastAPI OpenAPI to frontend/openapi.json."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.openapi_export import DEFAULT_OUT, export_openapi  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    written = export_openapi(out)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
