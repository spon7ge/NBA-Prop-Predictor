# dbt — Bronze → Silver data cleaning

Light cleanup layer on top of `raw.*` tables in Supabase/Postgres.

## Setup

```bash
# From repo root (uses nba_model venv if active)
pip install -r dbt/requirements.txt
```

Ensure `.env` has `SUPABASE_DB_URL` (same as other scripts).

Apply the bronze schema migration once in Supabase SQL Editor:

```
db/migrations/006_bronze_schema.sql
```

## Run

```bash
python scripts/run_dbt.py debug          # test connection
python scripts/run_dbt.py run            # build all models
python scripts/run_dbt.py run --select bronze   # bronze only
python scripts/run_dbt.py test --select bronze  # run column tests
```

`scripts/run_dbt.py` parses `SUPABASE_DB_URL` and sets `DBT_*` env vars for `dbt/profiles.yml`.

## Models (Phase 4)

| Model | Source | What it does |
|-------|--------|--------------|
| `bronze.bronze_games` | `raw.team_base`, `raw.games` | Dedupe NBA games; cast `game_date`; parse home/away from matchup; union Odds-API events |
| `bronze.bronze_player_props` | `raw.props_dfs` ∪ `raw.props_us` | Rename columns (`player_name`, `market_category`, `side`); cast dates, lines, odds |

Both are **views** in the `bronze` schema — no heavy transforms (that’s Silver).

## Project layout

```
dbt/
  dbt_project.yml
  profiles.yml          # gitignored — copy from profiles.yml.example if needed
  models/
    sources/_raw_sources.yml
    bronze/
      bronze_games.sql
      bronze_player_props.sql
      _bronze.yml
```

## Next (Silver)

Silver models will read from `bronze.*` — e.g. join props to games, merge gamelog tables into `silver.player_gamelogs` via dbt instead of Python-only.
