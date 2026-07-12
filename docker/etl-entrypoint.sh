#!/usr/bin/env bash
# ETL container entrypoint — run pipeline steps or an interactive shell.
set -euo pipefail

cd /app

# Env vars are injected by docker compose (env_file + environment). Do not `source`
# .env here — unquoted values like "Regular Season" break bash.

NBA_SEASON="${HOOPVISTA_NBA_SEASON:-2025-26}"
SEASON_TYPE="${HOOPVISTA_SEASON_TYPE:-Regular Season}"
ROTOWIRE_SEASON="${HOOPVISTA_ROTOWIRE_SEASON:-${NBA_SEASON%%-*}}"
GAME_DATE="${GAME_DATE:-$(date +%F)}"

run_ingest() {
  echo "── 1/4 Ingest APIs ──"
  python scripts/fetch_raw.py \
    --league nba \
    --season "$NBA_SEASON" \
    --season-type "$SEASON_TYPE" \
    --raw-only \
    --sequential
  python scripts/PropFinder.py
  python src/scrapers/rotowire_scraper.py --season "$ROTOWIRE_SEASON"
}

run_silver() {
  echo "── 2/4 Merge raw → silver (pos + Rotowire) ──"
  python scripts/fetch_raw.py \
    --league nba \
    --season "$NBA_SEASON" \
    --season-type "$SEASON_TYPE" \
    --silver-only
}

run_transforms() {
  echo "── 3/4 Materialize transforms → Supabase ──"
  python scripts/run_transforms.py
  python scripts/run_transforms.py --test ml
}

run_predict() {
  echo "── 4/4 Generate predictions ──"
  python scripts/generate_predictions.py \
    --prop all \
    --game-date "$GAME_DATE" \
    --season-type "$SEASON_TYPE"
}

run_full() {
  run_ingest
  run_silver
  run_transforms
  run_predict
}

usage() {
  cat <<EOF
HoopVista ETL — usage:
  etl-entrypoint.sh ingest   Fetch NBA stats, odds, Rotowire → raw.*
  etl-entrypoint.sh silver   Merge raw.* → silver (pos + Rotowire odds)
  etl-entrypoint.sh transforms Materialize bronze/silver/gold/ml → Supabase
  etl-entrypoint.sh predict  Write ml.predictions for GAME_DATE
  etl-entrypoint.sh full     Run all steps (default)
  etl-entrypoint.sh shell    Interactive bash
EOF
}

cmd="${1:-full}"
shift || true

case "$cmd" in
  ingest) run_ingest "$@" ;;
  silver) run_silver "$@" ;;
  transforms) run_transforms "$@" ;;
  dbt) echo "Note: 'dbt' is deprecated; use 'transforms'." >&2; run_transforms "$@" ;;
  predict) run_predict "$@" ;;
  full) run_full "$@" ;;
  shell) exec bash "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
