"""Run dbt with Supabase credentials from .env (SUPABASE_DB_URL).

Usage (from repo root):
    python scripts/run_dbt.py debug
    python scripts/run_dbt.py run --select bronze
    python scripts/run_dbt.py test --select bronze
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBT_DIR = PROJECT_ROOT / "dbt"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _set_dbt_env_from_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Expected postgresql:// URL, got scheme={parsed.scheme!r}")

    os.environ["DBT_HOST"] = parsed.hostname or ""
    os.environ["DBT_PORT"] = str(parsed.port or 5432)
    os.environ["DBT_USER"] = unquote(parsed.username or "")
    os.environ["DBT_PASSWORD"] = unquote(parsed.password or "")
    os.environ["DBT_DBNAME"] = (parsed.path or "/postgres").lstrip("/") or "postgres"

    qs = parse_qs(parsed.query)
    sslmode = (qs.get("sslmode") or ["require"])[0]
    os.environ["DBT_SSLMODE"] = sslmode


def _dbt_executable() -> str:
    venv_dbt = PROJECT_ROOT / "nba_model" / "Scripts" / "dbt.exe"
    if venv_dbt.is_file():
        return str(venv_dbt)
    return "dbt"


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        print("SUPABASE_DB_URL is not set in .env", file=sys.stderr)
        return 1

    _set_dbt_env_from_url(url)
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        args = ["run"]

    cmd = [_dbt_executable(), *args]
    print(">", " ".join(cmd), f"(cwd={DBT_DIR})")
    result = subprocess.run(
        cmd,
        cwd=DBT_DIR,
        env=os.environ.copy(),
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
