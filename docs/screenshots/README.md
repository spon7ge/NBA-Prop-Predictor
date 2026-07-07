# Screenshot assets for README

Add PNG captures here, then reference them from the root `README.md`.

| File | What to capture |
|------|-----------------|
| `airflow-dag-success.png` | Airflow UI → `hoopvista_daily_pipeline` → Graph or Grid view with all tasks green |
| `dbt-lineage.png` | dbt docs → lineage graph for `ml.features` or `gold_prop_history` (run `dbt docs generate && dbt docs serve`) |
| `dashboard-all-players.png` | Frontend → All Players view with model/sharp/consensus columns |
| `dashboard-top-legs.png` | Frontend → Top Legs parlay cards |
| `api-docs.png` | FastAPI Swagger at `http://localhost:8000/docs` |

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
# Open http://localhost:8080 → trigger hoopvista_daily_pipeline
```

**Dashboard**
```bash
docker compose --profile local-db up -d postgres api
cd frontend && npm run dev
# Open http://localhost:5173
```

Export at ~1400px wide for readable README rendering.
