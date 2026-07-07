import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import BOOK_FILE_BASE, DATA_DIR, VALID_LEG_COUNTS

router = APIRouter(tags=["slates"])


def _slate_path(book: str, legs: int) -> Path:
    base = BOOK_FILE_BASE.get(book)
    if base is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown book '{book}'. Valid: {', '.join(BOOK_FILE_BASE)}",
        )
    if legs not in VALID_LEG_COUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid leg count {legs}. Valid: {sorted(VALID_LEG_COUNTS)}",
        )
    filename = f"{base}.json" if legs == 2 else f"{base}_{legs}leg.json"
    return DATA_DIR / "ev_analysis" / filename


def _read_json(path: Path) -> object:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/slates/{book}")
def get_slate(book: str, legs: int = 2) -> object:
    return _read_json(_slate_path(book, legs))


@router.get("/enriched")
def get_enriched() -> object:
    path = DATA_DIR / "enriched" / "dfs_enriched_latest.json"
    return _read_json(path)
