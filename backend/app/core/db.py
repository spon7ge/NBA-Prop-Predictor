"""Database connection utilities.

All reads go through gold.* and silver.* tables in Supabase (PostgreSQL).
No NBA API or Odds API calls — the backend is read-only against the DB.
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any

import psycopg2
import psycopg2.extras
from fastapi import HTTPException


def _dsn() -> str:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise HTTPException(
            status_code=503,
            detail="SUPABASE_DB_URL is not configured on the server.",
        )
    return url


@contextlib.contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a short-lived psycopg2 connection, closing it on exit."""
    conn = psycopg2.connect(dsn=_dsn())
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> list[dict]:
    """Execute *sql* and return all rows as plain dicts.

    Raises HTTP 503 if the DB is unreachable, HTTP 500 for query errors.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
    except HTTPException:
        raise
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"DB connection error: {exc}") from exc
    except psycopg2.Error as exc:
        raise HTTPException(status_code=500, detail=f"DB query error: {exc}") from exc


def query_one(sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> dict | None:
    """Execute *sql* and return the first row as a dict, or None if empty."""
    rows = query(sql, params)
    return rows[0] if rows else None
