#!/bin/bash
# Apply SQL migrations on first Postgres startup (local dev stack).
set -euo pipefail

MIGRATIONS_DIR="/docker-migrations"

if [[ ! -d "$MIGRATIONS_DIR" ]]; then
  echo "No migrations directory mounted; skipping schema init."
  exit 0
fi

echo "Applying HoopVista DB migrations from $MIGRATIONS_DIR ..."

for f in $(find "$MIGRATIONS_DIR" -maxdepth 1 -name '*.sql' | sort); do
  echo "  → $(basename "$f")"
  psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -f "$f"
done

echo "Migrations complete."
