# HoopVista Airflow Scheduler

Apache Airflow orchestrates the daily ingest → silver → live ML pipeline:

| Step | DAG task | Command |
|------|----------|---------|
| 1. Ingest APIs | `ingest_apis.*` | NBA + WNBA `fetch_raw.py --raw-only`, `PropFinder.py` |
| 2. Build silver | `build_silver.*` | NBA + WNBA `fetch_raw.py --silver-only` |
| 3. Live props | `live_ml.live_props.*` | `run_live_props.py` → `ml.*_live_prop_predictions` |
| 4. Live slates | `live_ml.live_slates.*` | `run_live_slates.py` → `ml.*_live_slates` (Top Legs) |

Gold feature tables and model retraining are **manual** (not scheduled).

## Prerequisites

1. Repo-root `.env` with at least `SUPABASE_DB_URL` and `API_KEY` (see `airflow/.env.example`).
2. Supabase migrations applied (`db/migrations/`), including `017_ml_live_slates.sql`.
3. Saved model bundles under `src/models/saved_models/` for each league × prop.

## Quick start (Docker)

```bash
# From repo root — ensure .env exists
cp airflow/.env.example .env   # then edit credentials

cd airflow
docker compose up airflow-init
docker compose up -d
```

- UI: http://localhost:8080 (default login `airflow` / `airflow`)
- DAG: `hoopvista_daily_pipeline`

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
bash scripts/run_pipeline_step.sh python scripts/fetch_raw.py --league nba --season 2025-26 --season-type "Regular Season" --raw-only --sequential
bash scripts/run_pipeline_step.sh python scripts/fetch_raw.py --league wnba --season 2026 --season-type "Regular Season" --raw-only --sequential
bash scripts/run_pipeline_step.sh python scripts/PropFinder.py --league wnba

# 2. Silver
bash scripts/run_pipeline_step.sh python scripts/fetch_raw.py --league nba --season 2025-26 --season-type "Regular Season" --silver-only
bash scripts/run_pipeline_step.sh python scripts/fetch_raw.py --league wnba --season 2026 --season-type "Regular Season" --silver-only

# 3. Live props (All Players)
bash scripts/run_pipeline_step.sh python scripts/run_live_props.py --league wnba
bash scripts/run_pipeline_step.sh python scripts/run_live_props.py --league nba

# 4. Live slates (Top Legs)
bash scripts/run_pipeline_step.sh python scripts/run_live_slates.py --league wnba
bash scripts/run_pipeline_step.sh python scripts/run_live_slates.py --league nba
```

When NBA is in season, use `--league nba` or `--league all` for PropFinder.

## Configuration

Environment variables (docker-compose or `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOOPVISTA_REPO_ROOT` | `/opt/airflow/hoopvista` | Repo mount path in container |
| `HOOPVISTA_NBA_SEASON` | `2025-26` | NBA API season |
| `HOOPVISTA_WNBA_SEASON` | `2026` | WNBA API season |
| `HOOPVISTA_SEASON_TYPE` | `Regular Season` | Regular Season / Playoffs |
| `HOOPVISTA_PROPFINDER_LEAGUE` | `wnba` | PropFinder `--league` (`nba` / `wnba` / `all`) |
| `HOOPVISTA_LIVE_LEAGUES` | `nba,wnba` | Comma-separated leagues for live props + slates |

## Schedule

- **`hoopvista_daily_pipeline`**: `0 6,14,22 * * *` (3× daily — adjust timezone in Airflow config)

## DAG structure

```
start
  └─ ingest_apis (parallel)
       ├─ nba_stats   (--raw-only)
       ├─ wnba_stats  (--raw-only)
       └─ odds_props  (PropFinder)
  └─ build_silver (parallel)
       ├─ nba         (--silver-only)
       └─ wnba        (--silver-only)
  └─ live_ml
       ├─ live_props (parallel per league)
       │    ├─ nba / wnba  → ml.*_live_prop_predictions
       └─ live_slates (parallel per league; after live_props)
            ├─ nba / wnba  → ml.*_live_slates
end
```

## Notes

- **PropFinder** upserts to `raw.nba_props_*` / `raw.wnba_props_*` depending on `--league`.
- **Silver** must run after raw ingest (`fetch_raw.py --silver-only`).
- **Live props** powers All Players (`GET /api/live-props`).
- **Live slates** powers Top Legs (`GET /api/live-slates`); runs after live props.
- Gold + train stay out of Airflow — rebuild features and retrain models when you choose.
