# End-to-end workflow

This is a concise companion to the root [README](../README.md). For setup, tool choices, and architecture diagrams, start there.

## Daily pipeline (automated)

Airflow DAG `hoopvista_daily_pipeline` runs:

1. **Ingest APIs** — NBA game logs, odds/props (US + DFS regions), Rotowire metadata
2. **Load silver** — merge five `raw.*` tables + positions + Rotowire into `silver.player_gamelogs`
3. **dbt run** — bronze → silver views → gold tables → ml feature tables
4. **Generate predictions** — quantile models → `ml.predictions`

Weekly: `hoopvista_train_models` retrains MIN/PPM/RPM/APM joblib bundles.

## Manual / Docker equivalent

```bash
docker compose --profile local-db up -d postgres api
docker compose --profile etl run --rm etl full
```

Or step-by-step: `ingest` → `silver` → `dbt` → `predict` (see [docker/README](../docker/README.md)).

## Read path (dashboard)

FastAPI reads **only Postgres** — no live API calls at request time:

`ml.predictions` + `gold.gold_prop_history` + `silver.*` → React dashboard (`/api/games/{date}/slate`).

## Write path (ETL)

External APIs → Python upserts → `raw.*` → Python silver merge → dbt transforms → ML scripts → `ml.predictions`.
