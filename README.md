# HoopVista

**NBA & WNBA player-prop research platform** — ingest multi-source odds and game data, transform through a medallion pipeline in Postgres, run quantile ML models, and serve live predictions + greedy multi-leg parlays through a FastAPI + React dashboard.

Built as a portfolio-grade data engineering + ML project: API ingestion, incremental upserts, dbt layering, Airflow orchestration, Docker deployment, and a live dashboard comparing model lean vs. sharp vs. consensus.

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

HoopVista evaluates **NBA and WNBA DFS / sportsbook player props** (PTS, REB, AST):

| Surface | Description | Data source |
|---------|-------------|-------------|
| **Top Legs** | EV-ranked 2/3/5/6-leg parlays by book (PrizePicks, Underdog, DraftKings Pick 6, Betr) | `GET /api/live-slates` → `ml.*_live_slates` |
| **All Players** | Searchable slate — model lean, form hit rates, vs-opp, def ranks | `GET /api/live-props` → `ml.*_live_prop_predictions` |
| **API** | Read-only FastAPI over Postgres — health, live props, live slates, players, games | silver / gold / ml schemas |

The **write path** (ETL + live ML) fetches external data on a schedule. The **read path** (API + frontend) never calls NBA or odds APIs at request time — everything is pre-computed in the database.

---

## Architecture

### End-to-end pipeline

Someone should be able to follow this diagram and understand the full system without opening code.

```mermaid
flowchart TB
  subgraph sources [External sources]
    NBA[NBA / WNBA Stats API]
    ODDS[The Odds API]
  end

  subgraph ingest [1 — Ingest Airflow]
    FR[src.pipeline.fetch]
    PF[PropFinder.py]
  end

  subgraph raw [2 — Raw Postgres]
    RGL[raw.*_player_* / team_*]
    RPR[raw.*_props_us / props_dfs]
  end

  subgraph silver_py [3 — Silver]
    SIL[src.pipeline.clean]
    SGL[silver.*_player_gamelogs]
  end

  subgraph live [4 — Live ML Airflow]
    LP[run_live_props.py]
    LS[run_live_slates.py]
    MLP[ml.*_live_prop_predictions]
    MLS[ml.*_live_slates]
  end

  subgraph batch [5 — Batch ML manual / dbt]
    DBT[dbt bronze→silver→gold→ml.features]
    TR[train_model.py]
    PR[generate_predictions.py]
    PRED[ml.predictions]
  end

  subgraph serve [6 — Serve]
    API[FastAPI :8000]
    FE[React dashboard :5173]
  end

  NBA --> FR
  ODDS --> PF
  FR --> RGL
  PF --> RPR
  RGL --> SIL --> SGL
  RPR --> LP
  SGL --> LP
  LP --> MLP
  LP --> LS
  SGL --> LS
  RPR --> LS
  LS --> MLS
  SGL --> DBT --> TR --> PR --> PRED
  MLP --> API
  MLS --> API
  PRED --> API
  API --> FE
```

**Airflow (automated):** midnight `fetch` → `clean`; daytime (8am / 12pm / 3pm PT) PropFinder ∥ starters → live props → live slates.

**Batch path (manual):** dbt gold/features → train → `ml.predictions` (historical / research slate).

![Pipeline overview](assets/screenshots/architecture-pipeline.svg)

> Static diagram above (also editable as SVG). Optional: redraw in [Excalidraw](https://excalidraw.com/) or draw.io and export PNG into `assets/screenshots/architecture-pipeline.png`.

### Medallion layers

```mermaid
flowchart LR
  subgraph raw_schema [raw — append/upsert]
    rb[NBA/WNBA game logs]
    rp[props_us + props_dfs]
  end

  subgraph bronze_schema [bronze — dbt views]
    bg[bronze_games]
    bp[bronze_player_props]
  end

  subgraph silver_schema [silver]
    sgl[player_gamelogs Python]
    spr[silver_props dbt]
  end

  subgraph gold_schema [gold — dbt tables]
    gs[player_game_stats]
    gr[rolling averages]
    gph[prop_history]
  end

  subgraph ml_schema [ml]
    mf[features_*]
    mp[predictions]
    mlp[live_prop_predictions]
    mls[live_slates]
  end

  rb --> sgl
  rp --> bp --> spr
  sgl --> gs --> mf --> mp
  spr --> gph
  sgl --> mlp --> mls
  rp --> mlp
```

| Layer | Materialization | Owner | Purpose |
|-------|-----------------|-------|---------|
| **raw** | Tables | Python upserts | Immutable API landing — one table per endpoint/source × league |
| **bronze** | Views | dbt | Rename, cast, union DFS + US props |
| **silver** | Tables + views | Python + dbt | Entity resolution — merged gamelogs, deduped props |
| **gold** | Tables | dbt | Analytics-ready — rolling averages, matchup context, prop history |
| **ml** | Tables | dbt + Python | Feature matrices, batch predictions, **live props**, **live parlays** |

### Docker services

```mermaid
flowchart LR
  PG[(Postgres :5433\nor Supabase)]
  API[api FastAPI :8000]
  ETL[etl one-shot]
  AF[Airflow :8080]
  FE[frontend :5173]

  ETL -->|write| PG
  AF -->|schedules ETL + live ML| PG
  API -->|read-only| PG
  FE -->|/api proxy| API
```

---

## Why these tools

### Postgres via Supabase (not a cloud warehouse)

| Choice | Rationale |
|--------|-----------|
| **Postgres** | Row-level upserts on `(game_id, player_id)` and prop composite keys; low-latency reads for the API; single database from raw → live ml |
| **Supabase** | Managed Postgres + pooling; free tier for portfolio demos; standard `postgresql://` URL works in Docker, Airflow, and local scripts |
| **Not Snowflake / BigQuery** | Dataset is millions of rows, not billions; need OLTP-style upserts and sub-100ms API queries, not petabyte warehouse analytics |

Local Docker Postgres uses the **same schemas and migrations** as Supabase — swap `SUPABASE_DB_URL` only.

### Apache Airflow

| Choice | Rationale |
|--------|-----------|
| **Airflow** | Two DAGs: midnight stats (`fetch`/`clean`); daytime odds + live ML (PropFinder → props → slates) |
| **Not cron alone** | Dependencies, partial failures, and mixed runtimes (Playwright PropFinder, nba-api, XGBoost) need stateful orchestration |
| **Not Prefect / Dagster** | Airflow remains the industry default on data-eng resumes; DAG mirrors CLI scripts 1:1 |

See [`airflow/README.md`](airflow/README.md) — `hoopvista_daily_stats` + `hoopvista_live_odds`.

### dbt medallion layering

| Choice | Rationale |
|--------|-----------|
| **bronze / silver as views** | Cheap to rebuild; raw stays source of truth |
| **gold / ml.features as tables** | Expensive joins and rolling windows — materialized for API + training |
| **Tests in dbt** | Unique keys, accepted values, relationships catch bad ingests before models run |
| **SQL in repo** | Lineage is documented; `dbt docs` produces the graph for screenshots |

Python owns **API ingestion, silver merge, and live inference**. dbt owns **declarative analytics transforms** downstream of raw/silver.

---

## Engineering highlights

### Nested JSON flattening

- **NBA / WNBA box scores** — nested camelCase JSON → flattened → Postgres `snake_case` via `upsert_df()` ([`src/utils/db.py`](src/utils/db.py)).
- **The Odds API** — `{events → bookmakers → markets → outcomes}` → tabular rows in PropFinder before upsert to `raw.*_props_us` / `raw.*_props_dfs`.
- **Live slate parlays** — nested `LEGS` (model / form / game_context) stored as JSONB in `ml.*_live_slates`; frontend normalizes for Top Legs cards.

### Multi-location extraction

Props and lines come from **many sources**, unified in Postgres by league:

| Source | Region / channel | Raw tables |
|--------|------------------|-------------|
| The Odds API | US / EU sharp books | `raw.nba_props_us`, `raw.wnba_props_us` |
| The Odds API | US DFS (PrizePicks, Underdog, …) | `raw.nba_props_dfs`, `raw.wnba_props_dfs` |
| nba-api Schedule + box scores | NBA (`00`) / WNBA (`10`) | `raw.*_player_*`, `raw.*_team_*` |

dbt **unions** DFS + US in bronze props; live ML reads the latest Odds pull via `load_latest_odds(league, region)`.

### Backfill vs. incremental loading

| Pattern | Where | How |
|---------|-------|-----|
| **Incremental upsert** | `raw.*` game logs, props | `ON CONFLICT` on natural keys; `fetched_at` lineage |
| **Checkpointed backfill** | start positions | Per-game API calls; CSV checkpoint skips finished `game_id`s |
| **Full re-merge** | `silver.*_player_gamelogs` | Season rebuild from raw (idempotent upsert) |
| **dbt refresh** | gold / ml.features | `dbt run` materializes tables; upstream views stay cheap |
| **Live overwrite by run** | `ml.*_live_prop_predictions`, `ml.*_live_slates` | Latest `run_at` per `game_date` served by the API |
| **Batch predictions** | `ml.predictions` | Upsert on `(prop, game_id, player_id)` |

---

## Screenshots

> Capture PNGs under [`assets/screenshots/`](assets/screenshots/) — see that folder's README for steps. SVG placeholders ship until you replace them.

| Pipeline | Dashboard |
|----------|-----------|
| ![Airflow DAG](assets/screenshots/airflow-dag-success.svg) | ![All Players](assets/screenshots/dashboard-all-players.svg) |
| *Airflow: ingest → silver → live_props → live_slates* | *All Players — model lean + form* |

| Top Legs | dbt / API |
|----------|-----------|
| ![Top Legs](assets/screenshots/dashboard-top-legs.svg) | ![dbt lineage](assets/screenshots/dbt-lineage.svg) |
| *Greedy parlays by book × legs × league* | *Lineage for gold / ml.features* |

---

## Quick start (Docker)

Goal: clone → `docker compose` → healthy API → optional ETL / live ML / dashboard. No code reading required.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- Git
- Node 20+ (optional, for the React UI)
- Odds API key (optional for ingest; required for live props/slates)
- Python 3.11+ venv (optional if you run live scripts on the host; Airflow can run them instead)

### 1. Clone and configure

```bash
git clone https://github.com/spon7ge/NBA-Prop-Predictor.git
cd NBA-Prop-Predictor

cp .env.docker.example .env
# Edit .env — set API_KEY for odds / live ML
```

Minimum `.env` for local Docker:

```bash
SUPABASE_DB_URL=postgresql://hoopvista:hoopvista@host.docker.internal:5433/hoopvista?sslmode=disable
POSTGRES_PORT=5433
HOOPVISTA_NBA_SEASON=2025-26
HOOPVISTA_WNBA_SEASON=2026
HOOPVISTA_SEASON_TYPE="Regular Season"
HOOPVISTA_PROPFINDER_LEAGUE=wnba   # or nba / all
HOOPVISTA_LIVE_LEAGUES=nba,wnba
API_KEY=your_the_odds_api_key     # needed for PropFinder + live ML
```

> **Supabase instead of local Postgres:** set `SUPABASE_DB_URL` to your project connection string, apply `db/migrations/*.sql` in the SQL editor, then use `docker-compose.supabase.yml` (see [docker/README.md](docker/README.md)).

### 2. Bring the stack up

```bash
# Postgres + FastAPI (profiles gate optional services)
docker compose --profile local-db up -d postgres api

docker compose ps
curl http://localhost:8000/api/health
```

Open **http://localhost:8000/docs** for Swagger (`/live-props`, `/live-slates`, …).

### 3. Ingest + silver (ETL)

```bash
docker compose --profile etl build etl
docker compose --profile etl run --rm etl full    # ingest → silver
```

### 4. Live predictions (All Players + Top Legs)

With models under `src/models/saved_models/` and fresh odds in `raw.*`:

```bash
# Host Python (activate your venv first), or trigger via Airflow live_ml tasks
python scripts/run_live_props.py --league wnba
python scripts/run_live_props.py --league nba

python scripts/run_live_slates.py --league wnba
python scripts/run_live_slates.py --league nba
```

Verify:

```bash
curl "http://localhost:8000/api/live-props?league=wnba"
curl "http://localhost:8000/api/live-slates?league=wnba"
```

### 5. Dashboard

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — Vite proxies `/api` → `:8000`. Use the **League** dropdown on Top Legs / All Players for WNBA, NBA, or All.

### 6. Airflow (optional scheduler)

```bash
cd airflow
docker compose up airflow-init
docker compose up -d
```

- UI: **http://localhost:8080** — login `airflow` / `airflow`
- Unpause `hoopvista_daily_stats` (midnight fetch/clean) and `hoopvista_live_odds` (8am/12pm/3pm odds + starters + live ML)
- Pause/delete legacy `hoopvista_daily_pipeline` if it still shows in the UI

---

## Project structure

```
NBA-Prop-Predictor/
├── backend/                 # FastAPI — read-only Postgres (/live-props, /live-slates, …)
├── frontend/                # React + Vite — Top Legs + All Players
├── dbt/                     # bronze → silver → gold → ml.features
├── airflow/                 # DAG + scheduler Docker stack
├── docker/                  # API / ETL Dockerfiles
├── db/migrations/           # Postgres schemas (incl. 016 live props, 017 live slates)
├── scripts/
│   ├── PropFinder.py        # Odds → raw.*_props_*
│   ├── run_live_props.py    # → ml.*_live_prop_predictions
│   ├── run_live_slates.py   # → ml.*_live_slates
│   └── train_model.py / generate_predictions.py
├── src/
│   ├── pipeline/
│   │   ├── fetch.py         # raw.* game logs (CLI: python -m src.pipeline.fetch)
│   │   ├── clean.py         # silver.* merge (CLI: python -m src.pipeline.clean)
│   │   ├── predict.py / build_slates.py / features
│   ├── live_pipeline/       # common + nba/ + wnba/ (apm/ppm/rpm; min in common)
│   ├── scrapers/            # Odds + site scrapers
│   └── models/saved_models/ # .joblib quantile bundles
└── docker-compose.yml       # postgres + api + etl profiles
```

---

## Further reading

| Doc | Contents |
|-----|----------|
| [docker/README.md](docker/README.md) | Compose profiles, ETL commands |
| [airflow/README.md](airflow/README.md) | DAG schedules, writes, env vars |
| [dbt/README.md](dbt/README.md) | Model catalog |
| [backend/README.md](backend/README.md) | API read path + endpoint reference |
| [frontend/README.md](frontend/README.md) | Dashboard setup |
| [assets/screenshots/README.md](assets/screenshots/README.md) | How to capture README images |

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Ingestion | Python, nba-api, Playwright, The Odds API |
| Storage | PostgreSQL (Supabase or local Docker) |
| Transform | dbt-postgres, pandas |
| ML | XGBoost quantile regression, joblib |
| Live inference | `predict_rate` + `enrich_dfs_picks` + greedy slates |
| Orchestration | Apache Airflow 2.10, Docker Compose |
| API | FastAPI, psycopg2 |
| Frontend | React 19, TypeScript, TanStack Query, Vite |

---

This project was made for **educational purposes only**. Sports betting involves risk and this model is not intended to compete with professional bookmakers. Gamble responsibly.
