import math
import os
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from supabase import create_client, Client

load_dotenv()

# ── supabase-py client (PostgREST) ────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    # Use the service role key for backend scripts — it bypasses RLS and has
    # full access to all schemas including raw.  Fall back to the anon key if
    # the service role key is not set (read-only / public operations only).
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    return create_client(url, key)


def upsert(table: str, schema: str, rows: list[dict], on_conflict: str) -> None:
    """Small-batch upsert via supabase-py / PostgREST."""
    if not rows:
        return
    client = get_client()
    (
        client.schema(schema)
        .table(table)
        .upsert(rows, on_conflict=on_conflict)
        .execute()
    )


# ── SQLAlchemy engine (direct Postgres wire) ──────────────────────────────────

@lru_cache(maxsize=1)
def get_engine():
    """Return a SQLAlchemy engine pointed at SUPABASE_DB_URL.

    Set SUPABASE_DB_URL in .env (see .env.example).
    Uses the Supabase connection pooler (port 6543) for best compatibility.
    """
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. "
            "Add it to .env — see .env.example for the format."
        )
    return create_engine(url, pool_pre_ping=True)


# Conflict column defaults — one entry per raw.* table.
_RAW_CONFLICT_COLS: dict[str, list[str]] = {
    # game-log tables (NBAGameLogs.fetch)
    "player_base":     ["game_id", "player_id"],
    "player_adv":      ["game_id", "player_id"],
    "team_base":       ["game_id", "team_id"],
    "team_adv":        ["game_id", "team_id"],
    "start_positions": ["game_id", "player_id"],
    # prop-line tables (NBAPropFinder)
    "props_dfs": ["bookmaker", "category", "name", "over_under", "commence_time"],
    "props_us":  ["bookmaker", "category", "name", "over_under", "commence_time"],
}


def _clean_val(v):
    """Convert pandas NA / numpy sentinels to JSON-safe Python values."""
    if v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _upsert_df_postgres(
    table: str,
    records: list[dict],
    schema: str,
    conflict_cols: list[str],
    cols: list[str],
    batch_size: int,
) -> None:
    col_list = ", ".join(f'"{c}"' for c in cols)
    params = ", ".join(f":{c}" for c in cols)
    conflict = ", ".join(f'"{c}"' for c in conflict_cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

    sql = text(
        f"INSERT INTO {schema}.{table} ({col_list}) "
        f"VALUES ({params}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )

    engine = get_engine()
    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            conn.execute(sql, records[i : i + batch_size])


def _upsert_df_supabase(
    table: str,
    records: list[dict],
    schema: str,
    conflict_cols: list[str],
    batch_size: int,
) -> None:
    on_conflict = ",".join(conflict_cols)
    for i in range(0, len(records), batch_size):
        upsert(table, schema, records[i : i + batch_size], on_conflict=on_conflict)


def upsert_df(
    table: str,
    df: pd.DataFrame,
    schema: str = "raw",
    conflict_cols: list[str] | None = None,
    batch_size: int = 500,
) -> None:
    """Upsert a DataFrame into a Postgres/Supabase table.

    - Column names are lowercased to match Postgres convention.
    - A ``fetched_at`` timestamp is stamped on every row.
    - NaN / NaT become NULL.
    - Rows are sent in batches of ``batch_size`` via ``executemany``.

    The table must already exist (run scripts/migrations/001_raw_gamelogs.sql).
    Unknown DataFrame columns that have no matching table column are silently
    ignored by Postgres if they are not in the INSERT list — but the INSERT list
    is built from the DataFrame, so the table must contain every column in the
    DataFrame.  Any extra columns in the *table* that are absent from the
    DataFrame will just keep their existing value (DO UPDATE only touches
    columns present in the INSERT).
    """
    if df.empty:
        return

    if conflict_cols is None:
        conflict_cols = _RAW_CONFLICT_COLS.get(table)
        if conflict_cols is None:
            raise ValueError(
                f"No default conflict_cols known for table '{table}'. "
                "Pass them explicitly via conflict_cols=."
            )

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df["fetched_at"] = datetime.now(timezone.utc)

    records = [
        {k: _clean_val(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    cols = list(df.columns)

    # Prefer direct Postgres (faster for large frames). Fall back to supabase-py
    # when SUPABASE_DB_URL is missing or the pooler connection string is wrong.
    via = "postgres"
    try:
        _upsert_df_postgres(table, records, schema, conflict_cols, cols, batch_size)
    except Exception as exc:
        if not os.environ.get("SUPABASE_URL"):
            raise
        print(f"  → Postgres wire failed ({exc.__class__.__name__}); using supabase-py")
        _upsert_df_supabase(table, records, schema, conflict_cols, batch_size)
        via = "supabase-py"

    print(f"  ✓ raw.{table} — {len(records):,} rows upserted ({via})")


if __name__ == "__main__":
    test_client = get_client()
    result = test_client.schema("raw").table("team_base").select("*").limit(1).execute()
    print("Connected to raw.team_base. Row count probe:", len(result.data), "rows returned (0 = table empty, not an error)")