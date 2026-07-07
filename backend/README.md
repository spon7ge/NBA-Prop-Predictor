# HoopVista API

FastAPI backend for the HoopVista dashboard.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## Endpoints

| Route | Description |
|-------|-------------|
| `GET /api/health` | Health check |
| `GET /api/slates/{book}?legs=2` | Parlay slate JSON (`prizepicks`, `underdog`, `draftkings`, `betr`) |
| `GET /api/enriched` | Latest enriched player picks |

Data is read from repo-root `data/props/` (same files the pipeline exports).

## Frontend dev

With the API running on port 8000, start the React app from `frontend/`:

```bash
npm run dev
```

Vite proxies `/api` requests to the backend.
