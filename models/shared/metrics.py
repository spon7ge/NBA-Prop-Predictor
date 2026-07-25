"""Scoring helpers for quantile prop models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Default minutes bands (override per league/prop if needed).
DEFAULT_MIN_TIERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "<10 min": lambda a: a < 10,
    "10-20 min": lambda a: (a >= 10) & (a < 20),
    "20-30 min": lambda a: (a >= 20) & (a < 30),
    "30+ min": lambda a: a >= 30,
}


def pinball_50(y_true, y_pred, alpha: float = 0.50) -> float:
    """Negative pinball loss at ``alpha`` (higher is better for sklearn scorers)."""
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return -float(
        np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual))
    )


def score_quantile_fold(
    actual,
    preds: Mapping[str, np.ndarray],
    *,
    fold_label: str,
    starting=None,
    models: Mapping[str, Any] | None = None,
    tiers: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    lower_key: str = "q_0.10",
    median_key: str = "q_0.50",
    upper_key: str = "q_0.90",
    verbose: bool = True,
) -> dict[str, Any]:
    """Score one fold: MAE / R² / 80% coverage + optional role and tier slices."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(preds[lower_key], dtype=float)
    median = np.asarray(preds[median_key], dtype=float)
    upper = np.asarray(preds[upper_key], dtype=float)

    mae = float(mean_absolute_error(actual, median))
    r2 = float(r2_score(actual, median))
    coverage = float(np.mean((actual >= lower) & (actual <= upper)))

    metrics: dict[str, Any] = {
        "fold": fold_label,
        "n": int(len(actual)),
        "mae": mae,
        "r2": r2,
        "coverage_80pct": coverage,
    }
    if models is not None:
        metrics["best_iters"] = {
            k: getattr(models[k], "best_iteration", None) for k in models
        }

    if verbose:
        print(f"\n{fold_label}")
        print(
            f"  Overall  | n={len(actual):5d} | MAE: {mae:.3f} | "
            f"R²: {r2:.3f} | 80% Coverage: {coverage:.1%}"
        )

    if starting is not None:
        starting = np.asarray(starting)
        for role, mask in (("Starters", starting == 1), ("Bench", starting == 0)):
            if mask.sum() == 0:
                continue
            r_mae = float(mean_absolute_error(actual[mask], median[mask]))
            r_cov = float(
                np.mean(
                    (actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])
                )
            )
            metrics[f"mae_{role}"] = r_mae
            metrics[f"coverage_{role}"] = r_cov
            if verbose:
                print(
                    f"  {role:10s} | n={mask.sum():5d} | "
                    f"MAE: {r_mae:.3f} | Coverage: {r_cov:.1%}"
                )

    for tier, fn in (tiers or DEFAULT_MIN_TIERS).items():
        mask = np.asarray(fn(actual), dtype=bool)
        if mask.sum() == 0:
            continue
        t_mae = float(mean_absolute_error(actual[mask], median[mask]))
        t_cov = float(
            np.mean(
                (actual[mask] >= lower[mask]) & (actual[mask] <= upper[mask])
            )
        )
        metrics[f"mae_{tier}"] = t_mae
        metrics[f"coverage_{tier}"] = t_cov
        if verbose:
            print(
                f"  {tier:10s} | n={mask.sum():5d} | "
                f"MAE: {t_mae:.3f} | Coverage: {t_cov:.1%}"
            )

    return metrics
