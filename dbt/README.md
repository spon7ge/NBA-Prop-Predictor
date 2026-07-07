# dbt — Bronze → Silver data cleaning

Light cleanup layer on top of `raw.*` tables in Supabase/Postgres.

## Setup

```bash
# From repo root (uses nba_model venv if active)
pip install -r dbt/requirements.txt
```

Ensure `.env` has `SUPABASE_DB_URL` (same as other scripts).

Apply the bronze schema migration once in Supabase SQL Editor:

```
db/migrations/006_bronze_schema.sql
```

## Run

```bash
python scripts/run_dbt.py debug          # test connection
python scripts/run_dbt.py run            # build all models
python scripts/run_dbt.py run --select bronze   # bronze only
python scripts/run_dbt.py test --select bronze  # run column tests
```

`scripts/run_dbt.py` parses `SUPABASE_DB_URL` and sets `DBT_*` env vars for `dbt/profiles.yml`.

## Models (Phase 4)

| Model | Source | What it does |
|-------|--------|--------------|
| `bronze.bronze_games` | `raw.team_base`, `raw.games` | Dedupe NBA games; cast `game_date`; parse home/away from matchup; union Odds-API events |
| `bronze.bronze_player_props` | `raw.props_dfs` ∪ `raw.props_us` | Rename columns (`player_name`, `market_category`, `side`); cast dates, lines, odds |

Both are **views** in the `bronze` schema — no heavy transforms (that’s Silver).

## Silver models (Phase 4b)

| Model | Source | What it does |
|-------|--------|--------------|
| `silver.silver_players` | `raw.player_base` | One row per `player_id`; canonical name; `normalized_name`; latest team; name aliases |
| `silver.silver_games` | `bronze.bronze_games` | Dedupe games; map full team names → tricodes |
| `silver.silver_props` | `bronze.bronze_player_props` | Join players; standardize markets; dedupe to latest line |

```bash
python scripts/run_dbt.py run --select silver
python scripts/run_dbt.py test --select silver
```

## Project layout

```
dbt/
  dbt_project.yml
  profiles.yml          # gitignored — copy from profiles.yml.example if needed
  models/
    sources/_raw_sources.yml
    bronze/
      bronze_games.sql
      bronze_player_props.sql
      _bronze.yml
    silver/
      silver_players.sql
      silver_games.sql
      silver_props.sql
      _silver.yml
```

## ML models (Phase 7)

| Model | Source | What it does |
|-------|--------|--------------|
| `ml.features` | `int_player_game_features` | Shared ML inputs: L5/L10 rolls, usage, home/away, rest days, opponent def rating |
| `ml.features_min` | `features` + MIN model cols | MIN quantile model (`src/features/min_features.py`) |
| `ml.features_ppm` | `features` + PPM model cols | PPM quantile model (`src/features/ppm_features.py`) |
| `ml.features_rpm` | `features` + RPM model cols | RPM quantile model (`src/features/rpm_features.py`) |
| `ml.features_apm` | `features` + APM model cols | APM quantile model (`src/features/apm_features.py`) |

Apply the ML schema migration once:

```
db/migrations/007_ml_schema.sql
```

```bash
python scripts/run_dbt.py run --select ml
python scripts/run_dbt.py test --select ml
```

Load from Python:

```python
from src.utils.ml import read_ml_features
min_df = read_ml_features("min", season_year="2025-26")
```

## ML prediction pipeline (Phase 8)

Train from `ml.features_*` and save a joblib bundle:

```bash
python scripts/train_model.py --prop min
python scripts/train_model.py --prop all --season-year 2025-26
```

Generate predictions and write to `ml.predictions`:

```bash
# Apply db/migrations/008_ml_predictions.sql once
python scripts/generate_predictions.py --prop min
python scripts/generate_predictions.py --prop all --game-date 2026-05-12
```

Each row includes `player_id`, `game_id`, `prediction` (median quantile), and `predicted_at`.

## Next (Gold)

Gold models will read from `silver.*` for model-ready features (MIN/PPM pipelines).
