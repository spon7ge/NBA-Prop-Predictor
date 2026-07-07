from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "props"

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
