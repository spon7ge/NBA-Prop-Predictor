import math
import os
import re
from datetime import date, datetime, timezone
from functools import lru_cache
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import execute_values
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
    Port 5432 (direct / session pooler) or 6543 (transaction pooler) both work;
    prefer 5432 for bulk upserts and streamed reads (``read_df_streamed``).
    """
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL is not set. "
            "Add it to .env — see .env.example for the format."
        )
    return create_engine(url, pool_pre_ping=True)


# Conflict column defaults for raw.* (and other) upsert targets.
_RAW_CONFLICT_COLS: dict[str, list[str]] = {
    # NBA game-log tables (GameLogs.fetch)
    "nba_player_base":     ["game_id", "player_id"],
    "nba_player_adv":      ["game_id", "player_id"],
    "nba_team_base":       ["game_id", "team_id"],
    "nba_team_adv":        ["game_id", "team_id"],
    "nba_player_tracking": ["game_id", "player_id"],
    # WNBA game-log tables
    "wnba_player_base": ["game_id", "player_id"],
    "wnba_player_adv":  ["game_id", "player_id"],
    "wnba_team_base":   ["game_id", "team_id"],
    "wnba_team_adv":    ["game_id", "team_id"],
    "wnba_start_positions": ["game_id", "player_id"],
    # play-by-play
    "pbp": ["game_id", "action_id"],
    # prop-line tables
    "nba_props_dfs": ["bookmaker", "category", "name", "over_under", "commence_time"],
    "nba_props_us":  ["bookmaker", "category", "name", "over_under", "commence_time"],
    "wnba_props_dfs": ["bookmaker", "category", "name", "over_under", "commence_time", "data_pulled_at", "line"],
    "wnba_props_us":  ["bookmaker", "category", "name", "over_under", "commence_time", "data_pulled_at", "line"],
    # silver
    "nba_player_gamelogs": ["game_id", "player_id"],
    "wnba_player_gamelogs": ["game_id", "player_id"],
    # gold / ml
    "nba_player_min_model": ["game_id", "player_id"],
    "nba_player_ppm_model": ["game_id", "player_id"],
    "nba_player_apm_model": ["game_id", "player_id"],
    "nba_player_rpm_model": ["game_id", "player_id"],
    "wnba_player_min_model": ["game_id", "player_id"],
    "wnba_player_ppm_model": ["game_id", "player_id"],
    "wnba_player_apm_model": ["game_id", "player_id"],
    "wnba_player_rpm_model": ["game_id", "player_id"],
    # legacy unprefixed gold names (pre-league split)
    "player_min_model": ["game_id", "player_id"],
    "player_ppm_model": ["game_id", "player_id"],
    "player_apm_model": ["game_id", "player_id"],
    "player_rpm_model": ["game_id", "player_id"],
    "predictions": ["prop", "game_id", "player_id"],
    # live prop predictions (per-league tables)
    "nba_live_prop_predictions":  ["run_at", "game_date", "player_name", "market", "bookmaker"],
    "wnba_live_prop_predictions": ["run_at", "game_date", "player_name", "market", "bookmaker"],
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
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        fv = float(v)
        if math.isnan(fv):
            return None
        # Whole-number floats → int (player_id / team_id must not be "203827.0").
        if fv == int(fv):
            return int(fv)
        return fv
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
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
    if lineage_col:
        df[lineage_col] = datetime.now(timezone.utc)

    pk_cols = [c for c in conflict_cols if c in df.columns]
    if pk_cols:
        before = len(df)
        df = df.dropna(subset=pk_cols)
        dropped = before - len(df)
        if dropped:
            print(f"  → dropped {dropped} row(s) with null PK ({', '.join(pk_cols)})")

    # Prefer direct Postgres (faster for large frames). Fall back to supabase-py
    # when SUPABASE_DB_URL is missing or the wire connection fails. Column
    # alignment uses information_schema and must stay inside the Postgres try so
    # REST fallback still works when port 5432/6543 is unreachable.
    via = "postgres"
    try:
        aligned = _align_df_to_table(df, schema=schema, table=table)
        cols, rows = _df_to_tuples(aligned)
        _upsert_df_postgres(table, rows, schema, conflict_cols, cols, batch_size)
    except Exception as exc:
        if not os.environ.get("SUPABASE_URL"):
            raise
        if exc.__class__.__name__ in ("UndefinedColumn", "UndefinedTable", "DataError"):
            raise
        print(
            f"  → Postgres wire failed ({exc.__class__.__name__}); "
            "using supabase-py (much slower — fix SUPABASE_DB_URL for bulk loads)"
        )
        cols, rows = _df_to_tuples(df)
        records = [dict(zip(cols, row)) for row in rows]
        _upsert_df_supabase(table, records, schema, conflict_cols, batch_size)
        via = "supabase-py"

    print(f"  ✓ {schema}.{table} — {len(rows):,} rows upserted ({via})") 


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
    if not table_cols:
        raise RuntimeError(
            f"No columns found for {schema}.{table} — "
            "table is missing or migration was not applied. "
            "Check information_schema / run the matching db/migrations/*.sql."
        )
    out = df.copy()
    out.columns = [_normalize_col(c) for c in out.columns]
    keep = [c for c in out.columns if c in table_cols]
    extra = sorted(set(out.columns) - table_cols)
    if extra:
        print(f"  → dropping {len(extra)} column(s) not in {schema}.{table}: {extra[:8]}{'…' if len(extra) > 8 else ''}")
    return out[keep]


def upsert_gold(
    df: pd.DataFrame,
    *,
    table: str | None = None,
    league: str = "nba",
    batch_size: int = 2000,
) -> None:
    """Upsert the MIN quantile model gold frame into ``gold.{league}_player_min_model``."""
    from src.pipeline.gold import gold_table, prepare_gold_min_df

    table = table or gold_table("min", league=league)  # type: ignore[arg-type]
    upload = prepare_gold_min_df(df, league=league)  # type: ignore[arg-type]
    upload = _align_df_to_table(upload, schema="gold", table=table)
    upsert_df(
        table,
        upload,
        schema="gold",
        conflict_cols=["game_id", "player_id"],
        batch_size=batch_size,
        lineage_col="built_at",
    )


def upsert_ppm_gold(
    df: pd.DataFrame,
    *,
    table: str | None = None,
    league: str = "nba",
    batch_size: int = 2000,
) -> None:
    """Upsert the PPM quantile model gold frame into ``gold.{league}_player_ppm_model``."""
    from src.pipeline.gold import gold_table, prepare_gold_ppm_df

    table = table or gold_table("ppm", league=league)  # type: ignore[arg-type]
    upload = prepare_gold_ppm_df(df)
    upload = _align_df_to_table(upload, schema="gold", table=table)
    upsert_df(
        table,
        upload,
        schema="gold",
        conflict_cols=["game_id", "player_id"],
        batch_size=batch_size,
        lineage_col="built_at",
    )


def upsert_apm_gold(
    df: pd.DataFrame,
    *,
    table: str | None = None,
    league: str = "nba",
    batch_size: int = 2000,
) -> None:
    """Upsert the APM quantile model gold frame into ``gold.{league}_player_apm_model``."""
    from src.pipeline.gold import gold_table, prepare_gold_apm_df

    table = table or gold_table("apm", league=league)  # type: ignore[arg-type]
    upload = prepare_gold_apm_df(df)
    upload = _align_df_to_table(upload, schema="gold", table=table)
    upsert_df(
        table,
        upload,
        schema="gold",
        conflict_cols=["game_id", "player_id"],
        batch_size=batch_size,
        lineage_col="built_at",
    )


def upsert_rpm_gold(
    df: pd.DataFrame,
    *,
    table: str | None = None,
    league: str = "nba",
    batch_size: int = 2000,
) -> None:
    """Upsert the RPM quantile model gold frame into ``gold.{league}_player_rpm_model``."""
    from src.pipeline.gold import gold_table, prepare_gold_rpm_df

    table = table or gold_table("rpm", league=league)  # type: ignore[arg-type]
    upload = prepare_gold_rpm_df(df)
    upload = _align_df_to_table(upload, schema="gold", table=table)
    upsert_df(
        table,
        upload,
        schema="gold",
        conflict_cols=["game_id", "player_id"],
        batch_size=batch_size,
        lineage_col="built_at",
    )


def upsert_prop_gold(
    prop: str,
    df: pd.DataFrame,
    *,
    league: str = "nba",
    batch_size: int = 2000,
) -> None:
    """Dispatch gold upsert for one prop × league."""
    from src.pipeline.gold import gold_table

    prop = prop.lower()
    upserts = {
        "min": upsert_gold,
        "ppm": upsert_ppm_gold,
        "apm": upsert_apm_gold,
        "rpm": upsert_rpm_gold,
    }
    if prop not in upserts:
        raise ValueError(f"Unknown prop {prop!r}; expected one of {sorted(upserts)}")
    upserts[prop](
        df,
        table=gold_table(prop, league=league),  # type: ignore[arg-type]
        league=league,
        batch_size=batch_size,
    )


def upsert_ml_predictions(
    df: pd.DataFrame,
    *,
    table: str = "predictions",
    batch_size: int = 2000,
) -> None:
    """Upsert model predictions into ``ml.predictions``.

    Expects snake_case columns from ``prepare_predictions_upload``.
    Run ``db/migrations/008_ml_predictions.sql`` before the first upload.
    """
    upload = _align_df_to_table(
        df,
        schema="ml",
        table=table,
    )
    upsert_df(
        table,
        upload,
        schema="ml",
        conflict_cols=["prop", "game_id", "player_id"],
        batch_size=batch_size,
        lineage_col=None,
    )


def upsert_live_prop_predictions(
    df: pd.DataFrame,
    *,
    league: str,
    batch_size: int = 500,
) -> None:
    """Upsert enriched live prop predictions into ``ml.{league}_live_prop_predictions``.

    Run ``db/migrations/016_ml_live_prop_predictions.sql`` before the first call.
    Conflict key: (run_at, game_date, player_name, market, bookmaker).

    Parameters
    ----------
    df:
        DataFrame of enriched picks (snake_case columns).
    league:
        ``'nba'`` or ``'wnba'`` — selects the target table.
    """
    if league not in ("nba", "wnba"):
        raise ValueError(f"Unknown league {league!r}; expected 'nba' or 'wnba'")
    table = f"{league}_live_prop_predictions"
    upload = _align_df_to_table(df, schema="ml", table=table)
    upsert_df(
        table,
        upload,
        schema="ml",
        conflict_cols=["run_at", "game_date", "player_name", "market", "bookmaker"],
        batch_size=batch_size,
        lineage_col=None,
    )


def insert_model_registry(
    *,
    model_id: str,
    prop_type: str,
    trained_at,
    feature_set_version: str | None,
    training_season: str | None,
    validation_metrics: dict | None,
    joblib_path: str | None,
) -> None:
    """Record a new training run and mark it as the active model for ``prop_type``.

    Deactivates all previous entries for the same prop so exactly one is_active
    row exists per prop at any time.
    """
    import json as _json

    trained_at_str = trained_at.isoformat() if hasattr(trained_at, "isoformat") else trained_at
    metrics_json = _json.dumps(validation_metrics) if validation_metrics is not None else None

    engine = get_engine()
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE ml.model_registry SET is_active = FALSE WHERE prop_type = %s",
            (prop_type,),
        )
        cur.execute(
            """
            INSERT INTO ml.model_registry
                (model_id, prop_type, trained_at, feature_set_version, training_season,
                 validation_metrics, joblib_path, is_active)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, TRUE)
            ON CONFLICT (model_id) DO UPDATE SET
                prop_type           = EXCLUDED.prop_type,
                trained_at          = EXCLUDED.trained_at,
                feature_set_version = EXCLUDED.feature_set_version,
                training_season     = EXCLUDED.training_season,
                validation_metrics  = EXCLUDED.validation_metrics,
                joblib_path         = EXCLUDED.joblib_path,
                is_active           = EXCLUDED.is_active
            """,
            (
                model_id,
                prop_type,
                trained_at_str,
                feature_set_version,
                training_season,
                metrics_json,
                joblib_path,
            ),
        )
        conn.commit()
        print(f"  ✓ ml.model_registry — {prop_type} → {model_id} (active)")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def get_active_model_registry_entry(prop_type: str) -> dict | None:
    """Return the active model_registry row for ``prop_type``, or ``None``.

    Returns a dict with keys: model_id, prop_type, trained_at, feature_set_version,
    training_season, validation_metrics, joblib_path, is_active.
    """
    q = """
        SELECT model_id::text, prop_type, trained_at, feature_set_version,
               training_season, validation_metrics, joblib_path, is_active
        FROM ml.model_registry
        WHERE prop_type = %(prop_type)s AND is_active = TRUE
        ORDER BY trained_at DESC
        LIMIT 1
    """
    try:
        df = pd.read_sql(q, get_engine(), params={"prop_type": prop_type})
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as exc:
        print(f"  → Could not query ml.model_registry ({exc.__class__.__name__}): {exc}")
        return None


def _is_timeout_error(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "QueryCanceled":
        return True
    msg = str(exc).lower()
    return "statement timeout" in msg or "canceling statement" in msg


def read_df(
    table: str,
    schema: str = "raw",
    *,
    where: str | None = None,
    params: dict | list | tuple | None = None,
    eq: dict[str, object] | None = None,
    like: dict[str, str] | None = None,
    in_: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Read a Postgres/Supabase table into a DataFrame.

    Tries direct Postgres first. On connection/auth/timeout failure, falls back
    to paginated supabase-py when possible (``eq`` / ``like`` / ``in_``, or a
    full-table read with no SQL ``where``).
    """
    q = f'SELECT * FROM "{schema}"."{table}"'
    if where:
        q += f" WHERE {where}"
    try:
        return pd.read_sql(q, get_engine(), params=params)
    except Exception as exc:
        can_rest = bool(os.environ.get("SUPABASE_URL")) and (
            eq is not None or like is not None or in_ is not None or where is None
        )
        if not can_rest:
            raise
        print(
            f"  → Postgres read failed ({exc.__class__.__name__}"
            f"{'; timeout' if _is_timeout_error(exc) else ''}); "
            f"using supabase-py for {schema}.{table}"
        )
        return _read_df_rest(table, schema=schema, eq=eq, like=like, in_=in_)


def read_df_streamed(
    table: str,
    schema: str = "raw",
    *,
    where: str | None = None,
    params: dict | None = None,
    chunksize: int = 10_000,
    statement_timeout_ms: int = 120_000,
) -> pd.DataFrame:
    """Server-side streamed table read (named cursor via ``stream_results``).

    Unlike plain ``pd.read_sql(..., chunksize=...)`` on an engine (which often
    still buffers the full result in psycopg2), this opens a connection with
    ``stream_results=True`` so Postgres sends rows incrementally.

    Prefer the **session** pooler / direct port (5432) for bulk pulls — the
    transaction pooler (6543) can misbehave with long streamed reads and
    ``SET LOCAL`` session settings.

    ``where`` should use SQLAlchemy bind syntax (``:season``), not ``%(season)s``.
    """
    q = f'SELECT * FROM "{schema}"."{table}"'
    if where:
        q += f" WHERE {where}"

    frames: list[pd.DataFrame] = []
    # Set timeout *before* enabling stream_results — otherwise psycopg2 wraps
    # SET LOCAL in DECLARE ... CURSOR FOR SET LOCAL ..., which is a syntax error.
    with get_engine().connect() as conn:
        if statement_timeout_ms and statement_timeout_ms > 0:
            conn.execute(text(f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}"))
        streamed = conn.execution_options(stream_results=True)
        for chunk in pd.read_sql(text(q), streamed, params=params or {}, chunksize=chunksize):
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _read_df_rest(
    table: str,
    *,
    schema: str = "raw",
    eq: dict[str, object] | None = None,
    like: dict[str, str] | None = None,
    in_: dict[str, list] | None = None,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Paginated table read via PostgREST (for when direct Postgres is unavailable)."""
    client = get_client()

    def _base_query():
        q = client.schema(schema).table(table).select("*")
        for col, val in (eq or {}).items():
            q = q.eq(col, val)
        for col, val in (like or {}).items():
            q = q.like(col, val)
        return q

    if in_:
        frames: list[pd.DataFrame] = []
        for col, values in in_.items():
            chunk_size = 150
            for i in range(0, len(values), chunk_size):
                chunk = values[i : i + chunk_size]
                offset = 0
                while True:
                    q = _base_query().in_(col, chunk).range(offset, offset + page_size - 1)
                    batch = q.execute().data
                    if not batch:
                        break
                    frames.append(pd.DataFrame(batch))
                    if len(batch) < page_size:
                        break
                    offset += page_size
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    offset = 0
    frames = []
    while True:
        batch = _base_query().range(offset, offset + page_size - 1).execute().data
        if not batch:
            break
        frames.append(pd.DataFrame(batch))
        if len(batch) < page_size:
            break
        offset += page_size
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    test_client = get_client()
    result = test_client.schema("raw").table("team_base").select("*").limit(1).execute()
    print("Connected to raw.team_base. Row count probe:", len(result.data), "rows returned (0 = table empty, not an error)")