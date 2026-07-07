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
  python src/utils/nbaPlayerLogs.py \
    --season "$NBA_SEASON" \
    --season-type "$SEASON_TYPE" \
    --db-upsert \
    --sequential
  python scripts/PropFinder.py
  python src/scrapers/rotowire_scraper.py --season "$ROTOWIRE_SEASON"
}

run_silver() {
  echo "── 2/4 Load silver tables ──"
  python scripts/upload_silver.py \
    --season "$NBA_SEASON" \
    --season-type "$SEASON_TYPE"
}

run_dbt() {
  echo "── 3/4 dbt run ──"
  python scripts/run_dbt.py run
  python scripts/run_dbt.py test --select ml
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
  run_dbt
  run_predict
}

usage() {
  cat <<EOF
HoopVista ETL — usage:
  etl-entrypoint.sh ingest   Fetch NBA stats, odds, Rotowire → raw.*
  etl-entrypoint.sh silver   Merge raw.* → silver.player_gamelogs
  etl-entrypoint.sh dbt      Run dbt models + ml tests
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
  dbt) run_dbt "$@" ;;
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
