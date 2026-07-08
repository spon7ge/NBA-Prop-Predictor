"""
Generate ML predictions from ml.features_* tables and write to ml.predictions.

Pipeline:
  1. Read features from DB
  2. Load saved quantile model (median = prediction)
  3. Upsert rows into ml.predictions

Examples (run from repository root):

  python scripts/generate_predictions.py --prop min
  python scripts/generate_predictions.py --prop all --game-date 2026-05-12
  python scripts/generate_predictions.py --prop ppm --season-year 2025-26
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions and write to ml.predictions.")
    p.add_argument(
        "--prop",
        required=True,
        choices=["min", "ppm", "rpm", "apm", "all"],
        help="Prop model to run",
    )
    p.add_argument("--season-year", default=None, help="Optional season filter, e.g. 2025-26")
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help="Regular Season | Playoffs | PlayIn",
    )
    p.add_argument("--game-date", default=None, help="Optional YYYY-MM-DD filter")
    p.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit .joblib path (otherwise latest bundle is used)",
    )
    p.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "src" / "models" / "saved_models"),
        help="Directory containing saved joblib bundles",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run predictions but do not write to the database",
    )
    return p.parse_args()


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.db import get_active_model_registry_entry, upsert_ml_predictions
    from src.utils.ml import (
        predict_quantiles,
        prepare_predictions_upload,
        read_ml_features,
    )

    args = _parse_args()
    props = ["min", "ppm", "rpm", "apm"] if args.prop == "all" else [args.prop]

    total_rows = 0
    for prop in props:
        print(f"\nLoading features for {prop.upper()}...")
        features = read_ml_features(
            prop,
            season_year=args.season_year,
            season_type=args.season_type,
            game_date=args.game_date,
        )
        if features.empty:
            print(f"  No feature rows found for {prop}; skipping.")
            continue

        print(f"  {len(features):,} rows loaded")

        # Resolve model path: explicit flag > registry > directory scan (legacy).
        model_path = args.model_path
        active_model_id: str | None = None

        if model_path is None:
            registry_entry = get_active_model_registry_entry(prop)
            if registry_entry is not None:
                active_model_id = registry_entry["model_id"]
                model_path = registry_entry.get("joblib_path")
                print(f"  Registry: model_id={active_model_id}, path={model_path}")
            else:
                print(
                    f"  WARNING: no active model found in ml.model_registry for '{prop}'. "
                    "Falling back to directory scan. Run train_model.py to register a model."
                )

        preds = predict_quantiles(
            prop,
            features,
            model_path=model_path,
            models_dir=args.models_dir,
        )

        # Stamp the registry model_id onto every prediction row.
        # predict_quantiles already sets MODEL_ID from bundle["model_id"] when the
        # bundle was saved by the new train_prop_model; override with the registry
        # value if we got one from the DB (handles both old and new bundles).
        if active_model_id is not None:
            preds["MODEL_ID"] = active_model_id

        upload = prepare_predictions_upload(preds)
        print(f"  {len(upload):,} predictions generated")

        if args.dry_run:
            print(upload.head(min(5, len(upload))).to_string(index=False))
            total_rows += len(upload)
            continue

        upsert_ml_predictions(upload)
        total_rows += len(upload)

    print(f"\nDone — {total_rows:,} prediction rows processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
