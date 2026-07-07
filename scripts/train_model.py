"""
Train a quantile XGB model from ml.features_* tables and save a joblib bundle.

Examples (run from repository root):

  python scripts/train_model.py --prop min
  python scripts/train_model.py --prop ppm --season-year 2025-26
  python scripts/train_model.py --prop all --season-type "Regular Season"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train quantile models from ml.features_* tables.")
    p.add_argument(
        "--prop",
        required=True,
        choices=["min", "ppm", "rpm", "apm", "all"],
        help="Prop model to train",
    )
    p.add_argument("--season-year", default=None, help="Optional season filter, e.g. 2025-26")
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help="Regular Season | Playoffs | PlayIn",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Trailing fraction held out for early stopping",
    )
    p.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "src" / "models" / "saved_models"),
        help="Directory for saved joblib bundles",
    )
    p.add_argument("--output", default=None, help="Optional explicit output .joblib path")
    return p.parse_args()


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.ml import train_prop_model

    args = _parse_args()
    props = ["min", "ppm", "rpm", "apm"] if args.prop == "all" else [args.prop]

    for prop in props:
        print(f"\nTraining {prop.upper()} model...")
        save_path = train_prop_model(
            prop,
            season_year=args.season_year,
            season_type=args.season_type,
            val_fraction=args.val_fraction,
            models_dir=args.models_dir,
            save_path=args.output if args.prop != "all" else None,
        )
        print(f"Saved {prop.upper()} model → {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
