# Docker — Phase 12

Containerized HoopVista stack: **Postgres**, **FastAPI API**, and **ETL pipeline** (ingestion, dbt, ML predictions).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│  postgres   │◄────│     api      │     │  etl (one-shot / scheduled) │
│  (local)    │     │  FastAPI     │     │  ingest → silver → dbt → ML │
└─────────────┘     └──────────────┘     └─────────────────────────────┘
       ▲                    ▲                          ▲
       │                    │                          │
       └────────────────────┴──────────────────────────┘
                    SUPABASE_DB_URL
              (local Postgres or Supabase)
```

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `postgres:15-alpine` | 5432 | Local dev DB; auto-runs `db/migrations/*.sql` |
| `api` | `docker/Dockerfile.api` | 8000 | Read-only FastAPI (`/api/health`, predictions, props) |
| `etl` | `docker/Dockerfile.etl` | — | Ingest APIs, load silver, dbt, generate predictions |

## Quick start (local Postgres)

```bash
# 1. Configure env
cp .env.docker.example .env
# Add API_KEY for odds ingestion (optional for API-only demo)

# 2. Start DB + API (includes local Postgres via profile)
docker compose --profile local-db up -d postgres api

# 3. Verify API
curl http://localhost:8000/api/health
open http://localhost:8000/docs

# 4. Run ETL steps (postgres must be running from step 2)
docker compose --profile etl run --rm etl dbt          # transforms only
docker compose --profile etl run --rm etl full         # full pipeline
docker compose --profile etl run --rm etl ingest       # NBA + odds + Rotowire
docker compose --profile etl run --rm etl predict      # ml.predictions for today
```

## External Supabase

Skip local Postgres and point at your Supabase project:

```bash
# .env — set your Supabase connection string
SUPABASE_DB_URL=postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres?sslmode=require

docker compose -f docker-compose.yml -f docker-compose.supabase.yml up -d api
docker compose --profile etl run --rm etl full
```

Apply migrations once in Supabase SQL Editor (`db/migrations/001`–`008`) before first ETL run.

## ETL commands

The ETL container entrypoint (`docker/etl-entrypoint.sh`):

| Command | Steps |
|---------|-------|
| `ingest` | NBA stats → `raw.*`, odds → `raw.props_*`, Rotowire CSV |
| `silver` | `scripts/upload_silver.py` → `silver.player_gamelogs` |
| `dbt` | `dbt run` + `dbt test --select ml` |
| `predict` | `generate_predictions.py --prop all` |
| `full` | All of the above (default) |
| `shell` | Interactive bash inside container |

Override slate date:

```bash
GAME_DATE=2026-05-12 docker compose --profile etl run --rm etl predict
```

## Build individually

```bash
docker build -f docker/Dockerfile.api -t hoopvista-api .
docker build -f docker/Dockerfile.etl -t hoopvista-etl .
```

## Volumes & mounts

| Mount | Service | Purpose |
|-------|---------|---------|
| `pgdata` | postgres | Persistent local database |
| `./db/migrations` | postgres | Schema init on first boot |
| `./data` | etl | Scraper CSV/JSON output |
| `./src/models/saved_models` | etl | Pre-trained `.joblib` models |
| `./.env` | etl | API keys and DB URL |

Train models before first `predict` (host or container):

```bash
docker compose --profile etl run --rm etl shell
python scripts/train_model.py --prop all --season-year 2025-26
```

## Compose profiles

| Profile | Services | Use case |
|---------|----------|----------|
| `local-db` | `postgres` + `api` | Local dev stack (recommended) |
| `etl` | `etl` | Run pipeline jobs |
| *(default)* | `api` only | Requires external `SUPABASE_DB_URL` or combine with `docker-compose.supabase.yml` |

## Related

- **Airflow scheduling**: `airflow/` (optional orchestration on top of ETL image commands)
- **Frontend dev**: `cd frontend && npm run dev` (proxies `/api` → `localhost:8000`)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| API `503` / dbt `postgres` host not found | Start postgres first; set `SUPABASE_DB_URL` to `host.docker.internal:5433` in `.env` (see `.env.docker.example`) |
| dbt SSL error locally | Use `?sslmode=disable` in local URL |
| ETL ingest fails on odds | Set `API_KEY` in `.env` |
| Predict step empty | Train models first; run `dbt run --select ml` |
| Rotowire fails | Playwright + Chromium included in ETL image; check network |
