#!/usr/bin/env bash
# ETL container entrypoint — run pipeline steps or an interactive shell.
set -euo pipefail

cd /app

# Env vars are injected by docker compose (env_file + environment). Do not `source`
# .env here — unquoted values like "Regular Season" break bash.

NBA_SEASON="${HOOPVISTA_NBA_SEASON:-2025-26}"
WNBA_SEASON="${HOOPVISTA_WNBA_SEASON:-2026}"
SEASON_TYPE="${HOOPVISTA_SEASON_TYPE:-Regular Season}"
PROPFINDER_LEAGUE="${HOOPVISTA_PROPFINDER_LEAGUE:-wnba}"

run_ingest() {
  echo "── 1/2 Ingest APIs ──"
  python -m src.pipeline.fetch \
    --league nba \
    --season "$NBA_SEASON" \
    --season-type "$SEASON_TYPE" \
    --sequential
  python -m src.pipeline.fetch \
    --league wnba \
    --season "$WNBA_SEASON" \
    --season-type "$SEASON_TYPE" \
    --sequential
  python scripts/PropFinder.py --league "$PROPFINDER_LEAGUE"
}

run_silver() {
  echo "── 2/2 Merge raw → silver ──"
  python -m src.pipeline.clean \
    --league nba \
    --season "$NBA_SEASON" \
    --season-type "$SEASON_TYPE"
  python -m src.pipeline.clean \
    --league wnba \
    --season "$WNBA_SEASON" \
    --season-type "$SEASON_TYPE"
}

run_full() {
  run_ingest
  run_silver
}

usage() {
  cat <<EOF
HoopVista ETL — usage:
  etl-entrypoint.sh ingest   Fetch NBA/WNBA stats + odds → raw.*
  etl-entrypoint.sh silver   Merge raw.* → silver.*
  etl-entrypoint.sh full     ingest → silver (default)
  etl-entrypoint.sh shell    Interactive bash

Gold feature builds and model training are manual (not part of this entrypoint).
EOF
}

cmd="${1:-full}"
shift || true

case "$cmd" in
  ingest) run_ingest "$@" ;;
  silver) run_silver "$@" ;;
  full) run_full "$@" ;;
  shell) exec bash "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
