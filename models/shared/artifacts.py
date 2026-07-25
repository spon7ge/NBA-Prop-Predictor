"""Save / load quantile model bundles and run inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = _REPO_ROOT / "models" / "saved_models"


def save_model_bundle(
    models_ho: dict[str, Any],
    *,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    wf_results: list[dict[str, Any]],
    ho_metrics: dict[str, Any],
    features: list[str],
    artifact_stem: str,
    naive_holdout_results: dict[str, Any] | None = None,
    save_dir: str | Path | None = None,
) -> Path:
    """Persist quantile models + metrics under ``models/saved_models/``."""
    out_dir = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for r in wf_results:
        fold_metrics.append({
            k: v for k, v in r.items()
            if k not in ("train_mask", "val_mask")
        })

    bundle = {
        "quantile_models": models_ho,
        "feature_names": list(features),
        "fold_metrics": fold_metrics,
        "holdout_metrics": ho_metrics,
        "naive_baseline": naive_holdout_results,
        "train_end": train_df["game_date"].max(),
        "val_end": holdout_df["game_date"].max(),
    }

    save_path = out_dir / f"{artifact_stem}_{holdout_df['game_date'].max().date()}.joblib"
    joblib.dump(bundle, save_path)
    print(f"Saved to {save_path}")
    return save_path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a saved quantile model bundle."""
    bundle = joblib.load(path)
    print(f"Loaded {path}")
    return bundle


def predict_quantiles(
    models: dict[str, Any],
    feature_names: list[str],
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict Q10 / Q50 / Q90 for rows with columns matching ``feature_names``."""
    assert list(player_features.columns) == list(feature_names), (
        f"Feature mismatch — expected {feature_names}"
    )
    return pd.DataFrame({
        "q10": models["q_0.10"].predict(player_features),
        "q50": models["q_0.50"].predict(player_features),
        "q90": models["q_0.90"].predict(player_features),
    })
