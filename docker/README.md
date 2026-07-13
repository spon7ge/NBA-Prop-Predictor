# Docker

Containerized HoopVista stack: **Postgres**, **FastAPI API**, and **ETL pipeline** (ingest → silver).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│  postgres   │◄────│     api      │     │  etl (one-shot / scheduled) │
│  (local)    │     │  FastAPI     │     │  ingest → silver            │
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
| `etl` | `docker/Dockerfile.etl` | — | Fetch NBA/WNBA raw + PropFinder, then clean → silver |

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
docker compose --profile etl run --rm etl full         # ingest → silver
docker compose --profile etl run --rm etl ingest       # NBA + odds → raw.*
docker compose --profile etl run --rm etl silver       # raw.* → silver.*
```

## External Supabase

Skip local Postgres and point at your Supabase project:

```bash
# .env — set your Supabase connection string
SUPABASE_DB_URL=postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres?sslmode=require

docker compose -f docker-compose.yml -f docker-compose.supabase.yml up -d api
docker compose --profile etl run --rm etl full
```

Apply migrations once in Supabase SQL Editor (`db/migrations/`) before first ETL run.

## ETL commands

The ETL container entrypoint (`docker/etl-entrypoint.sh`):

| Command | Steps |
|---------|-------|
| `ingest` | NBA + WNBA stats → `raw.*`, odds → `raw.*_props_*` via PropFinder |
| `silver` | NBA + WNBA `fetch_raw.py --silver-only` → `silver.*` |
| `full` | ingest → silver (default) |
| `shell` | Interactive bash inside container |

PropFinder league (default `wnba` while NBA is out of season):

```bash
HOOPVISTA_PROPFINDER_LEAGUE=all docker compose --profile etl run --rm etl ingest
```

Gold feature builds and model training are manual (not part of this entrypoint).

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
| API `503` / DB host not found | Start postgres first; set `SUPABASE_DB_URL` to `host.docker.internal:5433` in `.env` (see `.env.docker.example`) |
| SSL error locally | Use `?sslmode=disable` in local URL |
| ETL ingest fails on odds | Set `API_KEY` in `.env` |
| PropFinder empty (NBA offseason) | Use `--league wnba` / `HOOPVISTA_PROPFINDER_LEAGUE=wnba` |
