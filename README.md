# HoopVista

**NBA player-prop research platform** — ingest multi-source odds and game data, transform through a medallion pipeline in Postgres, train quantile ML models, and serve predictions through a FastAPI + React dashboard.

Built as a portfolio-grade data engineering + ML project: API ingestion, incremental upserts, dbt layering, Airflow orchestration, Docker deployment, and a live dashboard with model vs. sharp vs. consensus odds comparison.

> **Disclaimer:** Educational project only. Sports betting involves risk. Gamble responsibly.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Why these tools](#why-these-tools)
- [Engineering highlights](#engineering-highlights)
- [Screenshots](#screenshots)
- [Quick start (Docker)](#quick-start-docker)
- [Project structure](#project-structure)
- [Further reading](#further-reading)

---

## What it does

HoopVista helps evaluate **NBA DFS and sportsbook player props** (PTS, REB, AST):

| Surface | Description |
|---------|-------------|
| **Top Legs** | EV-ranked parlays by book (PrizePicks, Underdog, DraftKings Pick 6, Betr) and leg count |
| **All Players** | Searchable slate — model lean, sharp lines, multi-book consensus, rolling hit rates |
| **API** | Read-only FastAPI over Postgres — predictions, props, player profiles, slate bundles |

The **write path** (ETL) fetches external data daily. The **read path** (API + frontend) never calls NBA or odds APIs at request time — everything is pre-computed in the database.

---

## Architecture

### End-to-end pipeline

```mermaid
flowchart TB
  subgraph sources [External sources]
    NBA[NBA Stats API]
    ODDS[The Odds API]
    RW[Rotowire scraper]
    DFS[DFS site scrapers]
  end

  subgraph ingest [1 — Ingest]
    PY1[nbaPlayerLogs.py]
    PY2[PropFinder.py]
    PY3[rotowire_scraper.py]
  end

  subgraph raw [2 — Raw Postgres]
    R1[raw.player_base / adv]
    R2[raw.team_base / adv]
    R3[raw.start_positions]
    R4[raw.props_us / props_dfs]
  end

  subgraph silver_py [3 — Silver merge Python]
    SM[upload_silver.py]
    SG[silver.player_gamelogs]
  end

  subgraph dbt_layer [4 — dbt transforms]
    BR[bronze views]
    SV[silver views]
    GD[gold tables]
    ML[ml.features_*]
  end

  subgraph ml_pipe [5 — ML]
    TR[train_model.py]
    PR[generate_predictions.py]
    PRED[ml.predictions]
  end

  subgraph serve [6 — Serve]
    API[FastAPI :8000]
    FE[React dashboard :5173]
  end

  subgraph orch [Orchestration]
    AF[Airflow DAGs]
    DC[Docker Compose ETL]
  end

  NBA --> PY1
  ODDS --> PY2
  RW --> PY3
  DFS -.-> PY2

  PY1 --> R1 & R2 & R3
  PY2 --> R4

  R1 & R2 & R3 --> SM --> SG
  SG --> BR
  R4 --> BR
  BR --> SV --> GD --> ML
  ML --> TR --> PR --> PRED
  GD --> PR

  AF & DC -.-> ingest
  PRED --> API
  GD --> API
  API --> FE
```

### Medallion layers (dbt)

```mermaid
flowchart LR
  subgraph raw_schema [raw — append/upsert]
    direction TB
    rb[player/team logs]
    rp[props_us + props_dfs]
  end

  subgraph bronze_schema [bronze — views]
    direction TB
    bg[bronze_games]
    bp[bronze_player_props]
  end

  subgraph silver_schema [silver — views]
    direction TB
    sp[silver_players]
    sg[silver_games]
    spr[silver_props]
    sgl[silver_player_gamelogs]
  end

  subgraph gold_schema [gold — tables]
    direction TB
    gs[gold_player_game_stats]
    gr[gold_player_rolling_avg_*]
    gm[gold_matchup_features]
    gph[gold_prop_history]
  end

  subgraph ml_schema [ml — tables]
    direction TB
    mf[features / features_min / ppm / rpm / apm]
    mp[predictions]
  end

  rb --> bg
  rp --> bp
  bg --> sg
  bp --> spr
  rb --> sp
  sgl --> gs --> mf --> mp
  spr --> gph
```

| Layer | Materialization | Owner | Purpose |
|-------|-----------------|-------|---------|
| **raw** | Tables | Python upserts | Immutable API landing — one table per endpoint/source |
| **bronze** | Views | dbt | Rename, cast, union DFS + US props |
| **silver** | Views | dbt + Python | Entity resolution — players, games, deduped props, merged gamelogs |
| **gold** | Tables | dbt | Analytics-ready — rolling averages, matchup context, prop history |
| **ml** | Tables | dbt + Python | Feature matrices + prediction outputs |

### Docker services

```mermaid
flowchart LR
  PG[(Postgres\n:5433 local)]
  API[api\nFastAPI :8000]
  ETL[etl\none-shot jobs]
  AF[Airflow\noptional :8080]

  ETL -->|SUPABASE_DB_URL| PG
  API -->|read-only| PG
  AF -.->|schedules| ETL
```

---

## Why these tools

### Postgres via Supabase (not a cloud warehouse)

| Choice | Rationale |
|--------|-----------|
| **Postgres** | Row-level upserts on `(game_id, player_id)` and prop composite keys; low-latency reads for the API; single database for raw → ml |
| **Supabase** | Managed Postgres + connection pooling + optional PostgREST; free tier for portfolio; standard `postgresql://` URL works everywhere |
| **Not Snowflake/BigQuery** | Dataset is ~millions of rows, not billions; need OLTP-style upserts and sub-100ms API queries, not petabyte analytics |

Local Docker Postgres uses the **same schemas and migrations** as Supabase — swap `SUPABASE_DB_URL` only.

### Apache Airflow

| Choice | Rationale |
|--------|-----------|
| **Airflow** | Explicit DAG for ingest → silver → dbt → predict; retries, scheduling (3× daily + weekly retrain), observability |
| **Not cron alone** | Pipeline has dependencies, partial failures, and mixed runtimes (Playwright, dbt, XGBoost) — Airflow tracks task state and logs |
| **Not Prefect/Dagster** | Airflow is industry-standard on resumes; DAG mirrors the Docker ETL entrypoint 1:1 |

See [`airflow/README.md`](airflow/README.md) and DAG `hoopvista_daily_pipeline`.

### dbt medallion layering

| Choice | Rationale |
|--------|-----------|
| **bronze/silver as views** | Cheap to rebuild; raw is source of truth |
| **gold/ml as tables** | Expensive joins and rolling windows — materialized for API + training |
| **Tests in dbt** | 135+ data tests (unique keys, accepted values, relationships) catch bad ingests before predictions run |
| **SQL in repo** | Lineage documented; `dbt docs` generates the graph for README screenshots |

Python owns **API ingestion and silver merge** (complex multi-table joins); dbt owns **declarative transforms** downstream.

---

## Engineering highlights

### Nested JSON flattening

- **NBA BoxScorePlayerTrackV3** returns nested camelCase JSON → flattened to `SCREAMING_SNAKE` → normalized to Postgres `snake_case` in `upsert_df()` (`src/utils/db.py`).
- **The Odds API** returns nested `{sport → events → markets → outcomes}` → flattened in `NBAPropFinder.create_map()` into tabular rows before upsert to `raw.props_us` / `raw.props_dfs`.
- **DFS enriched exports** (legacy file path) nest `model`, `sharp`, `consensus` objects → frontend/API mapper (`frontend/src/lib/mapSlate.ts`) projects them into flat columns for the dashboard.

### Multi-location extraction

Props and lines come from **many sources**, unified in Postgres:

| Source | Region / channel | Raw table |
|--------|------------------|-----------|
| The Odds API | `us,eu` (sharp books) | `raw.props_us` |
| The Odds API | `us_dfs` (PrizePicks, Underdog, …) | `raw.props_dfs` |
| Odds-API.io | DraftKings, FanDuel, Circa, BetOnline | file scrapers → optional upload |
| Site scrapers | PrizePicks, Underdog, Pinnacle, DK Pick 6 | `data/props/*` JSON |

dbt **unions** DFS + US in `bronze_player_props`, then silver dedupes to latest line per `(player, market, book, side)`.

### Backfill vs. incremental loading

| Pattern | Where | How |
|---------|-------|-----|
| **Incremental upsert** | `raw.*` game logs, props | `ON CONFLICT` on natural keys; `fetched_at` lineage column |
| **Checkpointed backfill** | `raw.start_positions` | Per-game NBA API calls; CSV checkpoint skips completed `game_id`s; batch upsert |
| **Full re-merge** | `silver.player_gamelogs` | `upload_silver.py` rebuilds from all raw tables for a season (idempotent upsert) |
| **dbt incremental** | gold tables | Full refresh on `dbt run` (views upstream stay cheap) |
| **Predictions** | `ml.predictions` | Upsert on `(prop, game_id, player_id)` for today's slate |

Season backfill: run `nbaPlayerLogs.py --db-upsert --season 2024-25` once, then daily incremental for current season.

---

## Screenshots

> Add PNGs under [`docs/screenshots/`](docs/screenshots/) — see that folder's README for capture steps.

| Pipeline | Dashboard |
|----------|-----------|
| ![Airflow DAG](docs/screenshots/airflow-dag-success.svg) | ![All Players dashboard](docs/screenshots/dashboard-all-players.svg) |
| *Airflow: `hoopvista_daily_pipeline` all green* | *Model / Sharp / Consensus columns* |

| dbt | API |
|-----|-----|
| ![dbt lineage](docs/screenshots/dbt-lineage.svg) | ![FastAPI docs](docs/screenshots/api-docs.svg) |
| *Lineage for `ml.features` or `gold_prop_history`* | *Swagger UI at `/docs`* |

Replace SVG placeholders with PNGs — see [`docs/screenshots/README.md`](docs/screenshots/README.md).

---

## Quick start (Docker)

A new clone should reach a running API in **under 5 minutes** (excluding image build time).

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- Git

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USER/NBA-Prop-Predictor.git
cd NBA-Prop-Predictor

cp .env.docker.example .env
```

Edit `.env`:

```bash
# Required for local Docker stack
SUPABASE_DB_URL=postgresql://hoopvista:hoopvista@host.docker.internal:5433/hoopvista?sslmode=disable
POSTGRES_PORT=5433
HOOPVISTA_SEASON_TYPE="Regular Season"

# Optional — needed for live odds ingest
API_KEY=your_the_odds_api_key
```

> **Production / Supabase:** set `SUPABASE_DB_URL` to your Supabase connection string, apply `db/migrations/*.sql` in the SQL editor, and use `docker-compose.supabase.yml` (see [docker/README.md](docker/README.md)).

### 2. Start database + API

```bash
docker compose --profile local-db up -d postgres api
```

Wait for healthy status:

```bash
docker compose ps
curl http://localhost:8000/api/health
```

Open **http://localhost:8000/docs** for interactive API docs.

### 3. Run the ETL pipeline

Postgres must be running first.

```bash
# Full pipeline: ingest → silver → dbt → predictions
docker compose --profile etl build etl
docker compose --profile etl run --rm etl full

# Or individual steps:
docker compose --profile etl run --rm etl ingest
docker compose --profile etl run --rm etl silver
docker compose --profile etl run --rm etl dbt
docker compose --profile etl run --rm etl predict
```

Before first `predict`, train models once (or copy `.joblib` files into `src/models/saved_models/`):

```bash
docker compose --profile etl run --rm etl shell
python scripts/train_model.py --prop all --season-year 2025-26
exit
```

### 4. Start the dashboard (optional)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — Vite proxies `/api` → `http://localhost:8000`.

### 5. Airflow (optional scheduler)

```bash
cd airflow
docker compose up airflow-init
docker compose up -d
```

Open **http://localhost:8080** (`airflow` / `airflow`), enable `hoopvista_daily_pipeline`.

---

## Project structure

```
NBA-Prop-Predictor/
├── backend/           # FastAPI (read-only Postgres)
├── frontend/          # React + Vite dashboard
├── dbt/               # bronze → silver → gold → ml models
├── airflow/           # DAGs + scheduler Docker stack
├── docker/            # Dockerfiles, ETL entrypoint
├── db/migrations/     # Postgres schema (raw, silver, gold, ml)
├── scripts/           # ETL helpers (run_dbt, upload_silver, train/predict)
├── src/
│   ├── scrapers/      # Odds + site scrapers
│   ├── features/      # ML feature definitions
│   ├── utils/         # db, silver merge, bronze fetch, ml
│   └── models/saved_models/  # Trained .joblib bundles
├── data/              # Local CSV/JSON exports (legacy + cache)
└── docker-compose.yml # postgres + api + etl profiles
```

---

## Further reading

| Doc | Contents |
|-----|----------|
| [docs/workflow.md](docs/workflow.md) | Daily write/read paths |
| [docker/README.md](docker/README.md) | Compose profiles, ETL commands, troubleshooting |
| [airflow/README.md](airflow/README.md) | DAG structure and scheduling |
| [dbt/README.md](dbt/README.md) | Model catalog and dbt commands |
| [backend/README.md](backend/README.md) | API endpoint reference |
| [frontend/README.md](frontend/README.md) | Dashboard dev setup |
| [explainer.md](explainer.md) | Plain-English project overview |

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Ingestion | Python, nba-api, Playwright, The Odds API |
| Storage | PostgreSQL (Supabase or local Docker) |
| Transform | dbt-postgres, pandas |
| ML | XGBoost quantile regression, joblib |
| Orchestration | Apache Airflow, Docker Compose |
| API | FastAPI, psycopg2 |
| Frontend | React 19, TypeScript, TanStack Query, Vite |

---

This project was made for **educational purposes only**. Sports betting involves risk and this model is not intended to compete with professional bookmakers. Gamble responsibly.
