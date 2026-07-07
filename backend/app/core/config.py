import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (two levels above backend/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")

REPO_ROOT = _REPO_ROOT
DATA_DIR = REPO_ROOT / "data" / "props"

# Supabase / PostgreSQL — used by app/core/db.py
SUPABASE_DB_URL: str | None = os.environ.get("SUPABASE_DB_URL")

BOOK_FILE_BASE: dict[str, str] = {
    "prizepicks": "prizepicks",
    "underdog": "underdog",
    "draftkings": "draftKings",
    "betr": "betr",
}

VALID_LEG_COUNTS = {2, 3, 5, 6}

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
