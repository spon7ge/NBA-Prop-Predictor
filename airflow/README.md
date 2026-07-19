# HoopVista Airflow Scheduler

Two DAGs (timezone ``HOOPVISTA_TZ``, default **America/Los_Angeles**):

| DAG | Schedule | What runs |
|-----|----------|-----------|
| **`hoopvista_daily_stats`** | `0 0 * * *` (midnight) | `fetch` → `clean` (NBA + WNBA game logs → silver) |
| **`hoopvista_live_odds`** | `0 8,12,15 * * *` (8am, 12pm, 3pm) | PropFinder ∥ starters → live props → live slates |

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
- Unpause **`hoopvista_daily_stats`** and **`hoopvista_live_odds`**
- Pause/delete the old `hoopvista_daily_pipeline` if it still appears

The repo is mounted at `/opt/airflow/hoopvista` inside the container.

## Local Airflow (no Docker)

```bash
pip install -r airflow/requirements.txt
playwright install chromium

export AIRFLOW_HOME=$PWD/airflow/runtime
export HOOPVISTA_REPO_ROOT=$PWD/..
airflow db init

export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DEFAULT_TIMEZONE=America/Los_Angeles

airflow standalone
```

## Pipeline commands (manual)

```bash
# Midnight path — stats
bash scripts/run_pipeline_step.sh python -m src.pipeline.fetch --league nba --season 2025-26 --season-type "Regular Season" --sequential
bash scripts/run_pipeline_step.sh python -m src.pipeline.fetch --league wnba --season 2026 --season-type "Regular Season" --sequential
bash scripts/run_pipeline_step.sh python -m src.pipeline.clean --league nba --season 2025-26 --season-type "Regular Season"
bash scripts/run_pipeline_step.sh python -m src.pipeline.clean --league wnba --season 2026 --season-type "Regular Season"

# Daytime path — odds + starters + live ML
bash scripts/run_pipeline_step.sh python scripts/PropFinder.py --league wnba
bash scripts/run_pipeline_step.sh python -m src.scrapers.rotowire_starters_scraper --league wnba --update
bash scripts/run_pipeline_step.sh python scripts/run_live_props.py --league wnba
bash scripts/run_pipeline_step.sh python scripts/run_live_slates.py --league wnba
```

When NBA is in season, use `--league nba` or `--league all` for PropFinder.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOOPVISTA_REPO_ROOT` | `/opt/airflow/hoopvista` | Repo mount path in container |
| `HOOPVISTA_NBA_SEASON` | `2025-26` | NBA API season |
| `HOOPVISTA_WNBA_SEASON` | `2026` | WNBA API season |
| `HOOPVISTA_SEASON_TYPE` | `Regular Season` | Regular Season / Playoffs |
| `HOOPVISTA_PROPFINDER_LEAGUE` | `wnba` | PropFinder `--league` (`nba` / `wnba` / `all`) |
| `HOOPVISTA_LIVE_LEAGUES` | `nba,wnba` | Leagues for live props + slates |
| `HOOPVISTA_TZ` | `America/Los_Angeles` | Cron timezone for both DAGs |

## DAG structure

**Midnight — `hoopvista_daily_stats`**

```
start
  └─ fetch_raw (parallel)     python -m src.pipeline.fetch
  └─ build_silver (parallel)  python -m src.pipeline.clean
end
```

**8am / 12pm / 3pm — `hoopvista_live_odds`**

```
start
  ├─ odds_props               PropFinder.py
  └─ scrape_starters          rotowire_starters_scraper --update → team_info.py
  └─ live_ml
       ├─ live_props → ml.*_live_prop_predictions
       └─ live_slates → ml.*_live_slates
end
```

## Writes by step

| Step | Script | Writes |
|------|--------|--------|
| Fetch | `python -m src.pipeline.fetch` | `raw.*` gamelogs / teams |
| Silver | `python -m src.pipeline.clean` | `silver.*_player_gamelogs` |
| Odds | `scripts/PropFinder.py` | `raw.*_props_us` / `raw.*_props_dfs` |
| Starters | `python -m src.scrapers.rotowire_starters_scraper --update` | `src/utils/team_info.py` projected starters |
| Live props | `scripts/run_live_props.py` | `ml.{nba,wnba}_live_prop_predictions` |
| Live slates | `scripts/run_live_slates.py` | `ml.{nba,wnba}_live_slates` |

## Notes

- **PropFinder** upserts to `raw.nba_props_*` / `raw.wnba_props_*`.
- **Starters** refresh RotoWire projected lineups into `team_info.py` before live minutes features run (league from `HOOPVISTA_LIVE_LEAGUES`).
- **Silver** updates once nightly; daytime live ML uses that silver + fresh odds.
- **Live props** → All Players (`GET /api/live-props`).
- **Live slates** → Top Legs (`GET /api/live-slates`).
- Gold + train stay out of Airflow (batch: dbt gold → `ml.features_*` → train → `ml.predictions`).
