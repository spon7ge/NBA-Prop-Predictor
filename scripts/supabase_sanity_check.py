"""Smoke test: insert one row into raw.games and read it back.

Run from repo root:
    python scripts/supabase_sanity_check.py

Prerequisites:
  1. .env has SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY
  2. Either:
     a) Expose ``raw`` in Supabase → Project Settings → API → Exposed schemas, OR
     b) Set SUPABASE_DB_URL for direct Postgres (SQLAlchemy path — no API expose needed)

Deletes the test row on success unless --keep is passed.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import inspect, text

from src.utils.db import get_engine, get_supabase, raw_table

TEST_GAME_ID = "sanity-check-0000000001"
TEST_NOTES = "supabase_sanity_check.py smoke test"


def _build_row() -> dict:
    """Minimal row — adjust keys if your raw.games columns differ."""
    return {
        "game_id": TEST_GAME_ID,
        "notes": TEST_NOTES,
        "inserted_at": datetime.now(timezone.utc).isoformat(),
    }


def _row_for_table(table_columns: set[str]) -> dict:
    row = _build_row()
    if "id" in table_columns and "game_id" not in table_columns:
        row = {"id": str(uuid.uuid4()), "notes": TEST_NOTES}
    return {k: v for k, v in row.items() if k in table_columns}


def _via_sqlalchemy(keep: bool) -> None:
    engine = get_engine()
    insp = inspect(engine)
    columns = {c["name"] for c in insp.get_columns("games", schema="raw")}
    if not columns:
        raise RuntimeError("raw.games has no columns or table does not exist")

    row = _row_for_table(columns)
    if not row:
        raise RuntimeError(f"Could not map test row to columns: {sorted(columns)}")

    cols_sql = ", ".join(row.keys())
    vals_sql = ", ".join(f":{k}" for k in row.keys())
    select_key = "game_id" if "game_id" in row else list(row.keys())[0]
    select_val = row[select_key]

    with engine.begin() as conn:
        conn.execute(text(f"INSERT INTO raw.games ({cols_sql}) VALUES ({vals_sql})"), row)
        fetched = conn.execute(
            text(f"SELECT * FROM raw.games WHERE {select_key} = :v LIMIT 1"),
            {"v": select_val},
        ).mappings().first()
        if not fetched:
            raise RuntimeError("Insert succeeded but SELECT returned no row")
        print("SQLAlchemy OK — row read back:")
        print(dict(fetched))
        if not keep:
            conn.execute(
                text(f"DELETE FROM raw.games WHERE {select_key} = :v"),
                {"v": select_val},
            )
            print("(test row deleted)")


def _via_supabase(keep: bool) -> None:
    row = _build_row()
    client = get_supabase(service_role=True)
    inserted = raw_table("games").insert(row).execute()
    if not inserted.data:
        raise RuntimeError(f"Insert returned no data: {inserted}")

    game_id = row["game_id"]
    fetched = raw_table("games").select("*").eq("game_id", game_id).limit(1).execute()
    if not fetched.data:
        raise RuntimeError("Insert succeeded but select returned no row")

    print("Supabase REST OK — row read back:")
    print(fetched.data[0])

    if not keep:
        raw_table("games").delete().eq("game_id", game_id).execute()
        print("(test row deleted)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supabase raw.games smoke test")
    parser.add_argument("--keep", action="store_true", help="Leave the test row in the table")
    parser.add_argument(
        "--method",
        choices=("auto", "sqlalchemy", "rest"),
        default="auto",
        help="Connection method (default: sqlalchemy if SUPABASE_DB_URL set, else REST)",
    )
    args = parser.parse_args()

    if args.method == "sqlalchemy" or (args.method == "auto" and __import__("os").getenv("SUPABASE_DB_URL")):
        try:
            _via_sqlalchemy(args.keep)
            return
        except RuntimeError as exc:
            if args.method == "sqlalchemy":
                raise
            print(f"SQLAlchemy path skipped: {exc}")

    if args.method in ("rest", "auto"):
        try:
            _via_supabase(args.keep)
            return
        except Exception as exc:
            msg = str(exc)
            if "PGRST106" in msg or "Invalid schema" in msg:
                print(
                    "\nThe ``raw`` schema is not exposed to the REST API.\n"
                    "Fix: Supabase Dashboard → Project Settings → API → Exposed schemas → add ``raw``\n"
                    "Or set SUPABASE_DB_URL in .env and re-run (uses direct Postgres).\n"
                )
            raise

    raise RuntimeError("No working connection method. Check .env and Supabase settings.")


if __name__ == "__main__":
    main()
