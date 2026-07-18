# Screenshot assets for README

Add PNG captures here, then reference them from the root `README.md`. SVG placeholders are checked in until you replace them.

| File | What to capture |
|------|-----------------|
| `architecture-pipeline.svg` | Static pipeline diagram (already in repo; optional redraw in Excalidraw/draw.io) |
| `airflow-dag-success.png` | Airflow UI → `hoopvista_daily_pipeline` → Graph with `live_ml.live_props` + `live_ml.live_slates` green |
| `dbt-lineage.png` | dbt docs → lineage for `ml.features` or `gold_prop_history` |
| `dashboard-all-players.png` | Frontend → All Players (league filter + model/form columns) |
| `dashboard-top-legs.png` | Frontend → Top Legs (Book / League / Legs dropdowns + parlay cards) |
| `api-docs.png` | FastAPI Swagger at `http://localhost:8000/docs` showing `/live-props` and `/live-slates` |

## How to generate

**dbt lineage**
```bash
python scripts/run_dbt.py docs generate
cd dbt && dbt docs serve --port 8081
# Open http://localhost:8081 → select a model → View Lineage
```

**Airflow**
```bash
cd airflow && docker compose up -d
# http://localhost:8080 (airflow / airflow)
# Unpause hoopvista_daily_pipeline → Trigger → wait for success
```

**Dashboard**
```bash
docker compose --profile local-db up -d postgres api
python scripts/run_live_props.py --league wnba
python scripts/run_live_slates.py --league wnba
cd frontend && npm run dev
# http://localhost:5173 — capture All Players and Top Legs
```

Export at ~1400px wide for readable README rendering. Prefer PNG over SVG for real UI screenshots.
