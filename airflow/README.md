# HoopVista Airflow Scheduler

Apache Airflow orchestrates the Supabase + dbt + ML pipeline:

| Step | DAG task | Command |
|------|----------|---------|
| 1. Ingest APIs | `ingest_apis.*` | NBA stats, odds/props, Rotowire |
| 2. Load raw tables | `load_raw_tables` | `scripts/upload_silver.py` → `silver.player_gamelogs` |
| 3. dbt run | `dbt_run` | `scripts/run_dbt.py run` |
| 4. Generate predictions | `generate_predictions` | `scripts/generate_predictions.py` |

A separate weekly DAG (`hoopvista_train_models`) retrains quantile models.

## Prerequisites

1. Repo-root `.env` with at least `SUPABASE_DB_URL` and `API_KEY` (see `airflow/.env.example`).
2. Supabase migrations applied (`db/migrations/001`–`008`).
3. Saved model bundles in `src/models/saved_models/` (run `train_model.py` once, or wait for the weekly DAG).

## Quick start (Docker)

```bash
# From repo root — ensure .env exists
cp airflow/.env.example .env   # then edit credentials

cd airflow
docker compose up airflow-init
docker compose up -d
```

- UI: http://localhost:8080 (default login `airflow` / `airflow`)
- DAGs: `hoopvista_daily_pipeline`, `hoopvista_train_models`

The repo is mounted at `/opt/airflow/hoopvista` inside the container.

## Local Airflow (no Docker)

```bash
pip install -r airflow/requirements.txt
playwright install chromium

export AIRFLOW_HOME=$PWD/airflow/runtime
export HOOPVISTA_REPO_ROOT=$PWD/..
airflow db init

# Point dags folder at this repo's dags/
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

airflow standalone
```

## Pipeline commands (manual)

Each step can be run outside Airflow via `scripts/run_pipeline_step.sh`:

```bash
# 1. Ingest
bash scripts/run_pipeline_step.sh python src/utils/nbaPlayerLogs.py --season 2025-26 --season-type "Regular Season" --db-upsert
bash scripts/run_pipeline_step.sh python scripts/PropFinder.py
bash scripts/run_pipeline_step.sh python src/scrapers/rotowire_scraper.py --season 2025

# 2. Load silver from raw
bash scripts/run_pipeline_step.sh python scripts/upload_silver.py --season 2025-26

# 3. dbt
bash scripts/run_pipeline_step.sh python scripts/run_dbt.py run

# 4. Predictions
bash scripts/run_pipeline_step.sh python scripts/generate_predictions.py --prop all --game-date 2026-05-12
```

## Configuration

Environment variables (docker-compose or `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOOPVISTA_REPO_ROOT` | `/opt/airflow/hoopvista` | Repo mount path in container |
| `HOOPVISTA_NBA_SEASON` | `2025-26` | NBA API season |
| `HOOPVISTA_SEASON_TYPE` | `Regular Season` | Regular Season / Playoffs |
| `HOOPVISTA_ROTOWIRE_SEASON` | `2025` | Rotowire archive year |

Optional Airflow Variables (override env): `hoopvista_repo_root`, `hoopvista_nba_season`, `hoopvista_season_type`, `hoopvista_rotowire_season`.

## Schedules

- **`hoopvista_daily_pipeline`**: `0 6,14,22 * * *` (3× daily — adjust timezone in Airflow config)
- **`hoopvista_train_models`**: `0 8 * * 1` (Mondays)

## DAG structure

```
start
  └─ ingest_apis (parallel)
       ├─ nba_stats
       ├─ odds_props
       └─ rotowire
  └─ load_raw_tables
  └─ dbt_run
  └─ dbt_test (ml tests)
  └─ generate_predictions
end
```

## Notes

- **Rotowire** requires Playwright + Chromium (included in the Docker image).
- **PropFinder** upserts directly to `raw.props_us` / `raw.props_dfs`.
- **Silver upload** must run after raw ingest; dbt silver models read `silver.player_gamelogs`.
- Predictions use `{{ ds }}` (Airflow logical date) as `--game-date`.
