"""Materialize bronze/silver/gold/ml transforms directly to Supabase.

Replaces ``scripts/run_dbt.py``. Reads SQL from ``dbt/models/`` and writes
views/tables via ``SUPABASE_DB_URL``.

Usage (from repo root):
    python scripts/run_transforms.py
    python scripts/run_transforms.py --select ml
    python scripts/run_transforms.py --test ml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize transforms to Supabase.")
    p.add_argument(
        "--select",
        default=None,
        help="Comma-separated model names (default: all, in dependency order)",
    )
    p.add_argument(
        "--test",
        default=None,
        help="Run post-materialization checks (currently: ml)",
    )
    return p.parse_args()


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.transforms import assert_ml_predictions_model_id, run_transforms

    args = _parse_args()

    if args.test == "ml":
        assert_ml_predictions_model_id()
        return 0

    run_transforms(select=args.select)
    if args.test:
        print(f"Unknown test suite: {args.test!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
