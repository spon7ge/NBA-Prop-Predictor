"""Materialize bronze/silver/gold/ml transforms directly to Supabase.

Replaces the dbt CLI: reads model SQL from ``dbt/models/``, expands macros and
refs, then creates views (bronze/silver) or refreshes tables (gold/ml) via
Postgres wire (``SUPABASE_DB_URL``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from sqlalchemy import text

from src.utils.db import get_engine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DBT_MODELS = _PROJECT_ROOT / "dbt" / "models"
_DBT_MACROS = _PROJECT_ROOT / "dbt" / "macros"

Materialization = Literal["view", "table"]

# model_name → (schema, materialization, relative path under dbt/models/)
MODELS: dict[str, tuple[str, Materialization, str]] = {
    "bronze_player_props": ("bronze", "view", "bronze/bronze_player_props.sql"),
    "bronze_games": ("bronze", "view", "bronze/bronze_games.sql"),
    "silver_players": ("silver", "view", "silver/silver_players.sql"),
    "silver_games": ("silver", "view", "silver/silver_games.sql"),
    "silver_player_gamelogs": ("silver", "view", "silver/silver_player_gamelogs.sql"),
    "silver_props": ("silver", "view", "silver/silver_props.sql"),
    "int_player_game_features": ("gold", "table", "gold/int_player_game_features.sql"),
    "gold_player_game_stats": ("gold", "table", "gold/gold_player_game_stats.sql"),
    "gold_player_rolling_avg_5": ("gold", "table", "gold/gold_player_rolling_avg_5.sql"),
    "gold_player_rolling_avg_10": ("gold", "table", "gold/gold_player_rolling_avg_10.sql"),
    "gold_matchup_features": ("gold", "table", "gold/gold_matchup_features.sql"),
    "gold_prop_history": ("gold", "table", "gold/gold_prop_history.sql"),
    "features": ("ml", "table", "ml/features.sql"),
    "features_min": ("ml", "table", "ml/features_min.sql"),
    "features_ppm": ("ml", "table", "ml/features_ppm.sql"),
    "features_rpm": ("ml", "table", "ml/features_rpm.sql"),
    "features_apm": ("ml", "table", "ml/features_apm.sql"),
    "gold_prediction_accuracy": ("gold", "table", "gold/gold_prediction_accuracy.sql"),
}

RUN_ORDER: tuple[str, ...] = tuple(MODELS.keys())

_LAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "bronze": tuple(n for n, (s, _, _) in MODELS.items() if s == "bronze"),
    "silver": tuple(n for n, (s, _, _) in MODELS.items() if s == "silver"),
    "gold": tuple(n for n, (s, _, _) in MODELS.items() if s == "gold"),
    "ml": tuple(n for n, (s, _, _) in MODELS.items() if s == "ml"),
}


def _load_macros() -> dict[str, str]:
    macros: dict[str, str] = {}
    for path in sorted(_DBT_MACROS.glob("*.sql")):
        raw = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"\{%-?\s*macro\s+(\w+)\([^)]*\)\s*-?%\}(.*?)\{%-?\s*endmacro\s*-?%\}",
            raw,
            flags=re.DOTALL,
        ):
            name, body = match.group(1), match.group(2).strip()
            if name != "generate_schema_name":
                macros[name] = body
    return macros


def _resolve_ref(model_name: str) -> str:
    if model_name not in MODELS:
        raise KeyError(f"Unknown dbt ref: {model_name!r}")
    schema, _, _ = MODELS[model_name]
    return f"{schema}.{model_name}"


def _compile_sql(model_name: str, *, macros: dict[str, str] | None = None) -> str:
    if model_name not in MODELS:
        raise KeyError(f"Unknown model: {model_name!r}")

    _, _, rel_path = MODELS[model_name]
    sql = (_DBT_MODELS / rel_path).read_text(encoding="utf-8")
    macros = macros or _load_macros()

    sql = re.sub(
        r"\{\{\s*source\(\s*'(\w+)'\s*,\s*'(\w+)'\s*\)\s*\}\}",
        lambda m: f"{m.group(1)}.{m.group(2)}",
        sql,
    )
    sql = re.sub(
        r"\{\{\s*ref\(\s*'(\w+)'\s*\)\s*\}\}",
        lambda m: _resolve_ref(m.group(1)),
        sql,
    )

    macro_pattern = re.compile(r"\{\{\s*(\w+)\((.*?)\)\s*\}\}", re.DOTALL)
    for _ in range(20):
        changed = False

        def _replace_macro(match: re.Match[str]) -> str:
            nonlocal changed
            name, arg = match.group(1), match.group(2).strip()
            if name not in macros:
                return match.group(0)
            changed = True
            return macros[name].replace("{{ column }}", arg)

        sql = macro_pattern.sub(_replace_macro, sql)
        if not changed:
            break

    if "{{" in sql:
        leftovers = re.findall(r"\{\{[^}]+\}\}", sql)
        raise RuntimeError(
            f"Unresolved Jinja in {model_name}: {leftovers[:5]}"
        )

    return sql.strip().rstrip(";")


def _materialize_sql(
    schema: str,
    model_name: str,
    materialization: Materialization,
    select_sql: str,
    *,
    conn,
) -> None:
    qualified = f'"{schema}"."{model_name}"'
    if materialization == "view":
        ddl = f"CREATE OR REPLACE VIEW {qualified} AS\n{select_sql}"
        conn.execute(text(ddl))
        return

    conn.execute(text(f"DROP TABLE IF EXISTS {qualified} CASCADE"))
    conn.execute(text(f"CREATE TABLE {qualified} AS\n{select_sql}"))


def materialize_model(model_name: str, *, conn=None) -> None:
    """Build or refresh one transform model in Supabase."""
    schema, materialization, _ = MODELS[model_name]
    select_sql = _compile_sql(model_name)
    own_conn = conn is None
    engine = get_engine()
    if own_conn:
        conn = engine.connect()
    try:
        _materialize_sql(schema, model_name, materialization, select_sql, conn=conn)
        conn.commit()
        print(f"  ✓ {schema}.{model_name} ({materialization})")
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()


def _resolve_select(select: str) -> set[str]:
    """Expand layer aliases (``ml``, ``gold``, …) and explicit model names."""
    wanted: set[str] = set()
    for token in select.split(","):
        key = token.strip()
        if not key:
            continue
        if key in _LAYER_ALIASES:
            wanted.update(_LAYER_ALIASES[key])
        elif key in MODELS:
            wanted.add(key)
        else:
            raise ValueError(f"Unknown model or layer: {key!r}")
    return wanted


def _transitive_dependencies(model_names: set[str]) -> set[str]:
    needed = set(model_names)
    changed = True
    while changed:
        changed = False
        for model_name in list(needed):
            sql = _compile_sql(model_name)
            for dep in re.findall(r"(?:bronze|silver|gold|ml)\.(\w+)", sql):
                if dep in MODELS and dep not in needed:
                    needed.add(dep)
                    changed = True
    return needed


def run_transforms(
    *,
    select: str | None = None,
    skip: set[str] | None = None,
) -> None:
    """Materialize all (or selected) transform models in dependency order."""
    skip = skip or set()
    if select:
        wanted = _transitive_dependencies(_resolve_select(select))
        models = [m for m in RUN_ORDER if m in wanted and m not in skip]
    else:
        models = [m for m in RUN_ORDER if m not in skip]

    if not models:
        print("No transform models to run.")
        return

    print(f"Materializing {len(models)} model(s) to Supabase…")
    engine = get_engine()
    with engine.connect() as conn:
        for model_name in models:
            materialize_model(model_name, conn=conn)


def assert_ml_predictions_model_id() -> None:
    """Fail when ml.predictions contains rows with a null model_id."""
    q = """
        SELECT prop, game_id, player_id, predicted_at
        FROM ml.predictions
        WHERE model_id IS NULL
        LIMIT 5
    """
    import pandas as pd

    bad = pd.read_sql(q, get_engine())
    if not bad.empty:
        sample = bad.to_dict(orient="records")
        raise AssertionError(
            f"ml.predictions has {len(bad)} row(s) with null model_id (sample: {sample})"
        )
    print("  ✓ ml.predictions — no null model_id rows")
