# HoopVista API

FastAPI backend for the HoopVista dashboard. **All endpoints read from Supabase only** — no NBA API or Odds API calls are made in this service.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Ensure repo-root `.env` has `SUPABASE_DB_URL`.

## Run

```bash
# from repo root — root must be on PYTHONPATH so `src.scrapers` imports work
PYTHONPATH=.:backend uvicorn app.main:app --reload --port 8000
```

WNBA game detail may call ESPN + RotoWire for scheduled projected starters.

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## Dashboard read path

FastAPI serves the React app from Postgres only (no live NBA/odds calls at request time):

| UI | Endpoint | Table |
|----|----------|-------|
| All Players | `GET /api/live-props` | `ml.*_live_prop_predictions` |
| Top Legs | `GET /api/live-slates` | `ml.*_live_slates` |
| Research / historical | `GET /api/games/{date}/slate` | games + gold props + `ml.predictions` |

## Endpoints

### Live dashboard

| Route | Source table | Description |
|-------|--------------|-------------|
| `GET /api/live-props` | `ml.*_live_prop_predictions` | All Players — model lean + form |
| `GET /api/live-slates` | `ml.*_live_slates` | Top Legs — greedy parlays by book |

### Core

| Route | Source table | Description |
|-------|--------------|-------------|
| `GET /api/predictions` | `ml.predictions` | ML model outputs (min/ppm/rpm/apm) |
| `GET /api/predictions/today` | `ml.predictions` | Today's predictions |
| `GET /api/predictions/player/{id}` | `ml.predictions` | Predictions for one player |
| `GET /api/player/{id}` | silver + gold + ml | Profile, game log, rolling stats, predictions |
| `GET /api/games/{date}` | `silver.silver_games` | Schedule for a date |
| `GET /api/games/today` | `silver.silver_games` | Today's schedule |

### Slate / lines / context

| Route | Source table | Description |
|-------|--------------|-------------|
| `GET /api/props` | `gold.gold_prop_history` | Prop lines + rolling context |
| `GET /api/games/{date}/props` | `gold.gold_prop_history` | Props for a slate date |
| `GET /api/games/{date}/predictions` | `ml.predictions` | Predictions for a slate date |
| `GET /api/games/{date}/slate` | multiple | Games + props + predictions bundle |
| `GET /api/games/{date}/with-props` | silver + gold | Games grouped with prop lines |
| `GET /api/games/{date}/with-predictions` | silver + ml | Games grouped with ML predictions |
| `GET /api/slates/{book}` | `silver.silver_props` | Latest lines for a DFS book |
| `GET /api/matchups/{date}` | `gold.gold_matchup_features` | Rest, pace, opponent def context |

### Discovery / ML inputs

| Route | Source table | Description |
|-------|--------------|-------------|
| `GET /api/players?q=` | `silver.silver_players` | Player search |
| `GET /api/features/{prop}` | `ml.features_*` | Model input features (base/min/ppm/rpm/apm) |
| `GET /api/health` | Postgres ping | Liveness + DB connectivity |

## Query parameters

**`/api/predictions`**

- `date` — YYYY-MM-DD (game_date filter)
- `prop` — `min` | `ppm` | `rpm` | `apm`
- `player_id`, `game_id`, `limit`

**`/api/props`**

- `date`, `bookmaker`, `market`, `source`, `side`, `player_id`, `limit`

**`/api/player/{id}`**

- `recent_n` — number of recent games (default 10)
- `include_predictions` — attach latest ML outputs (default true)

## Frontend dev

With the API running on port 8000, start the React app from `frontend/`:

```bash
npm run dev
```

Vite proxies `/api` requests to the backend.
