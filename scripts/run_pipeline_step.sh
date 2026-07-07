#!/usr/bin/env bash
# Run a pipeline command from repo root with .env loaded.
set -euo pipefail

REPO_ROOT="${HOOPVISTA_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/load_dotenv.sh"
load_dotenv "$REPO_ROOT/.env"

if [[ $# -eq 0 ]]; then
  echo "Usage: run_pipeline_step.sh <command> [args...]" >&2
  exit 1
fi

exec "$@"
