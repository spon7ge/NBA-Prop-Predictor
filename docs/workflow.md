# End-to-end workflow

Companion to the root [README](../README.md). Start there for architecture, tool choices, and Docker setup.

## Daily pipeline (Airflow)

DAG **`hoopvista_daily_pipeline`** (`0 6,14,22 * * *`):

```
start
  → ingest_apis   (nba_stats ∥ wnba_stats ∥ odds_props)
  → build_silver  (nba ∥ wnba)
  → live_ml
       → live_props  (nba ∥ wnba)   → ml.*_live_prop_predictions
       → live_slates (nba ∥ wnba)   → ml.*_live_slates
  → end
```

| Step | Script | Writes |
|------|--------|--------|
| Ingest stats | `scripts/fetch_raw.py --raw-only` | `raw.*` gamelogs / teams |
| Ingest odds | `scripts/PropFinder.py` | `raw.*_props_us` / `raw.*_props_dfs` |
| Silver | `scripts/fetch_raw.py --silver-only` | `silver.*_player_gamelogs` |
| Live props | `scripts/run_live_props.py` | `ml.{nba,wnba}_live_prop_predictions` |
| Live slates | `scripts/run_live_slates.py` | `ml.{nba,wnba}_live_slates` |

Configure leagues with `HOOPVISTA_LIVE_LEAGUES` (default `nba,wnba`) and PropFinder with `HOOPVISTA_PROPFINDER_LEAGUE`.

## Manual / Docker equivalent

```bash
docker compose --profile local-db up -d postgres api
docker compose --profile etl run --rm etl full

python scripts/run_live_props.py --league wnba
python scripts/run_live_slates.py --league wnba
```

## Read path (dashboard)

FastAPI reads **only Postgres** — no live NBA/odds calls at request time:

| UI | Endpoint | Table |
|----|----------|-------|
| All Players | `GET /api/live-props` | `ml.*_live_prop_predictions` |
| Top Legs | `GET /api/live-slates` | `ml.*_live_slates` |
| Research / historical | `GET /api/games/{date}/slate` | games + gold props + `ml.predictions` |

## Batch path (manual)

dbt gold + `ml.features_*` → `train_model.py` → `generate_predictions.py` → `ml.predictions`.

Not scheduled in the daily Airflow DAG — run when you retrain or rebuild features.
