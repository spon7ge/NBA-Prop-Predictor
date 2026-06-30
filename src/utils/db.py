import math
import os
import re
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
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
    Port 5432 (direct) or 6543 (pooler) both work; prefer 5432 for bulk upserts.
    """
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. "
            "Add it to .env — see .env.example for the format."
        )
    return create_engine(url, pool_pre_ping=True)


# Column renames applied before silver upload (merge artifact cleanup).
_SILVER_COLUMN_RENAME: dict[str, str] = {
    "OPP_OPP_ABBREVIATION_base": "OPP_TEAM_ABBREVIATION",
    "OPP_OPP_NAME_base": "OPP_TEAM_NAME",
    "OPP_OPP_ABBREVIATION": "OPP_TEAM_ABBREVIATION",
    "OPP_OPP_NAME": "OPP_TEAM_NAME",
}

# Conflict column defaults — one entry per raw.* / silver.* table.
_RAW_CONFLICT_COLS: dict[str, list[str]] = {
    # game-log tables (NBAGameLogs.fetch)
    "player_base":     ["game_id", "player_id"],
    "player_adv":      ["game_id", "player_id"],
    "team_base":       ["game_id", "team_id"],
    "team_adv":        ["game_id", "team_id"],
    "start_positions": ["game_id", "player_id"],
    # play-by-play (NBAPlayByPlay / PlayByPlayV3 parquet)
    "pbp": ["game_id", "action_id"],
    # prop-line tables (NBAPropFinder)
    "props_dfs": ["bookmaker", "category", "name", "over_under", "commence_time"],
    "props_us":  ["bookmaker", "category", "name", "over_under", "commence_time"],
    # silver layer (build_gamelogs_silver)
    "player_gamelogs": ["game_id", "player_id"],
}


def _normalize_col(name: str) -> str:
    """SCREAMING_SNAKE or camelCase → postgres snake_case (``gameId`` → ``game_id``)."""
    if "_" in name:
        return name.lower()
    s = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        name,
    ).lower()
    # NBA stats: FG3M/FG3A → fg3m/fg3a (not fg3_m/fg3_a)
    return re.sub(r"(\d)_([a-z])", r"\1\2", s)


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


def _df_to_tuples(df: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    """Convert a prepared DataFrame to column names + row tuples (faster than to_dict)."""
    cols = list(df.columns)
    rows = [
        tuple(_clean_val(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    return cols, rows


def _upsert_df_postgres(
    table: str,
    rows: list[tuple],
    schema: str,
    conflict_cols: list[str],
    cols: list[str],
    batch_size: int,
) -> None:
    col_list = ", ".join(f'"{c}"' for c in cols)
    conflict = ", ".join(f'"{c}"' for c in conflict_cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    updates = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

    sql = (
        f"INSERT INTO {schema}.{table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
    )

    engine = get_engine()
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        total = len(rows)
        for i in range(0, total, batch_size):
            batch = rows[i : i + batch_size]
            execute_values(cur, sql, batch, page_size=len(batch))
            done = min(i + batch_size, total)
            if done == total or done % (batch_size * 5) == 0:
                print(f"    … {done:,}/{total:,} rows", flush=True)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def _upsert_df_supabase(
    table: str,
    records: list[dict],
    schema: str,
    conflict_cols: list[str],
    batch_size: int,
) -> None:
    on_conflict = ",".join(conflict_cols)
    total = len(records)
    for i in range(0, total, batch_size):
        upsert(table, schema, records[i : i + batch_size], on_conflict=on_conflict)
        done = min(i + batch_size, total)
        if done == total or done % (batch_size * 5) == 0:
            print(f"    … {done:,}/{total:,} rows", flush=True)


def upsert_df(
    table: str,
    df: pd.DataFrame,
    schema: str = "raw",
    conflict_cols: list[str] | None = None,
    batch_size: int = 2000,
    *,
    lineage_col: str | None = "fetched_at",
) -> None:
    """Upsert a DataFrame into a Postgres/Supabase table.

    - Column names are normalized to Postgres snake_case (``GAME_ID``, ``gameId`` → ``game_id``).
    - A lineage timestamp column (default ``fetched_at``) is stamped on every row.
    - NaN / NaT become NULL.
    - Rows are sent in batches of ``batch_size`` via ``execute_values`` (Postgres)
      or PostgREST upserts (supabase-py fallback).

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
    df.columns = [_normalize_col(c) for c in df.columns]
    df = _align_df_to_table(df, schema=schema, table=table)
    if lineage_col:
        df[lineage_col] = datetime.now(timezone.utc)

    cols, rows = _df_to_tuples(df)

    # Prefer direct Postgres (faster for large frames). Fall back to supabase-py
    # when SUPABASE_DB_URL is missing or the pooler connection string is wrong.
    via = "postgres"
    try:
        _upsert_df_postgres(table, rows, schema, conflict_cols, cols, batch_size)
    except Exception as exc:
        if not os.environ.get("SUPABASE_URL"):
            raise
        if exc.__class__.__name__ in ("UndefinedColumn", "UndefinedTable", "DataError"):
            raise
        print(
            f"  → Postgres wire failed ({exc.__class__.__name__}); "
            "using supabase-py (much slower — set SUPABASE_DB_URL for bulk loads)"
        )
        records = [dict(zip(cols, row)) for row in rows]
        _upsert_df_supabase(table, records, schema, conflict_cols, batch_size)
        via = "supabase-py"

    print(f"  ✓ {schema}.{table} — {len(rows):,} rows upserted ({via})") 


def prepare_silver_df(
    df: pd.DataFrame,
    *,
    season_type: str,
) -> pd.DataFrame:
    """Normalize a ``build_gamelogs_silver`` frame for ``silver.player_gamelogs`` upload."""
    out = df.copy()
    rename = {k: v for k, v in _SILVER_COLUMN_RENAME.items() if k in out.columns}
    if rename:
        out = out.rename(columns=rename)
    out["SEASON_TYPE"] = season_type
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date
    return out


@lru_cache(maxsize=32)
def _table_columns(schema: str, table: str) -> frozenset[str]:
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %(schema)s AND table_name = %(table)s
    """
    cols = pd.read_sql(q, get_engine(), params={"schema": schema, "table": table})[
        "column_name"
    ]
    return frozenset(cols.astype(str).str.lower())


def _align_df_to_table(
    df: pd.DataFrame,
    *,
    schema: str,
    table: str,
) -> pd.DataFrame:
    """Keep only columns that exist in the target table (snake_case names)."""
    table_cols = _table_columns(schema, table)
    out = df.copy()
    out.columns = [_normalize_col(c) for c in out.columns]
    keep = [c for c in out.columns if c in table_cols]
    extra = sorted(set(out.columns) - table_cols)
    if extra:
        print(f"  → dropping {len(extra)} column(s) not in {schema}.{table}: {extra[:8]}{'…' if len(extra) > 8 else ''}")
    return out[keep]


def upsert_silver(
    df: pd.DataFrame,
    *,
    season_type: str,
    table: str = "player_gamelogs",
    batch_size: int = 2000,
) -> None:
    """Upsert a silver game-log frame into ``silver.player_gamelogs``.

    Run ``scripts/migrations/003_silver_gamelogs.sql`` in Supabase before the first upload.
    """
    upload = prepare_silver_df(df, season_type=season_type)
    upload = _align_df_to_table(
        upload,
        schema="silver",
        table=table,
    )
    upsert_df(
        table,
        upload,
        schema="silver",
        conflict_cols=["game_id", "player_id"],
        batch_size=batch_size,
        lineage_col="built_at",
    )


def read_df(
    table: str,
    schema: str = "raw",
    *,
    where: str | None = None,
    params: dict | list | tuple | None = None,
) -> pd.DataFrame:
    """Read a Postgres/Supabase table into a DataFrame."""
    q = f'SELECT * FROM "{schema}"."{table}"'
    if where:
        q += f" WHERE {where}"
    return pd.read_sql(q, get_engine(), params=params)


if __name__ == "__main__":
    test_client = get_client()
    result = test_client.schema("raw").table("team_base").select("*").limit(1).execute()
    print("Connected to raw.team_base. Row count probe:", len(result.data), "rows returned (0 = table empty, not an error)")