#!/usr/bin/env bash
# Safely export variables from a .env file (handles spaces and special chars).
load_dotenv() {
  local env_file="${1:-}"
  [[ -n "$env_file" && -f "$env_file" ]] || return 0

  if ! command -v python3 >/dev/null 2>&1; then
    echo "load_dotenv: python3 required to parse $env_file" >&2
    return 1
  fi

  eval "$(
    python3 - "$env_file" <<'PY'
import shlex
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
try:
    from dotenv import dotenv_values
except ImportError:
    sys.exit(0)

for key, value in dotenv_values(env_file).items():
    if not key or value is None:
        continue
    print(f"export {key}={shlex.quote(str(value))}")
PY
  )"
}
