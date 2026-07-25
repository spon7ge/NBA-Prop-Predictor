"""Leakage-safe naive predictors and paired hypothesis tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.shared.metrics import score_quantile_fold

DEFAULT_ALPHA = 0.05
DEFAULT_QUANTILES = [0.10, 0.50, 0.90]


def _quantile_linear_pipeline(quantile: float, *, l1_alpha: float = 0.0) -> Pipeline:
    """Impute → scale → linear quantile regressor (fit on train fold only)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            QuantileRegressor(
                quantile=quantile,
                alpha=l1_alpha,
                solver="highs",
            ),
        ),
    ])


def fit_quantile_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    *,
    quantiles: Sequence[float] | None = None,
    l1_alpha: float = 0.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Train one linear QuantileRegressor per quantile; return models + val preds.

    Report-only baseline between frozen naive and XGB. Not saved as a ship artifact.
    """
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    models: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for q in quantiles:
        pipe = _quantile_linear_pipeline(q, l1_alpha=l1_alpha)
        pipe.fit(X_train, y_train)
        key = f"q_{q:.2f}"
        models[key] = pipe
        preds[key] = np.asarray(pipe.predict(X_val), dtype=float)
    return models, preds


def run_quantile_linear_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    train_df: pd.DataFrame,
    *,
    holdout_df: pd.DataFrame,
    wf_results: list[dict[str, Any]],
    features: Sequence[str],
    target_col: str,
    naive_primary: str,
    xgb_preds_last: Mapping[str, np.ndarray],
    xgb_preds_ho: Mapping[str, np.ndarray],
    last_fold: dict[str, Any],
    role_col: str = "starting",
    quantiles: Sequence[float] | None = None,
    alpha: float = DEFAULT_ALPHA,
    l1_alpha: float = 0.0,
    holdout_label: str = "holdout",
) -> dict[str, Any]:
    """Walk-forward + holdout ladder: naive vs linear quantile vs XGB.

    Reuses ``wf_results`` train/val masks (no new split). Fits a fresh linear
    quantile bundle per fold and once on the full train pool for holdout.
    """
    features = list(features)
    quantiles = list(quantiles or DEFAULT_QUANTILES)

    print("── Quantile linear baseline (report-only) ───────────────────────────")
    print(f"  Features : {len(features)}")
    print(f"  Quantiles: {quantiles}")
    print(f"  L1 alpha : {l1_alpha}")

    wf_by_fold: list[dict[str, Any]] = []
    models_last = preds_last = None

    for r in wf_results:
        train_mask = r["train_mask"]
        val_mask = r["val_mask"]
        X_tr, X_val = X.loc[train_mask, features], X.loc[val_mask, features]
        y_tr, y_val = y.loc[train_mask], y.loc[val_mask]
        starting_val = train_df.loc[val_mask, role_col].values

        models, preds = fit_quantile_linear(
            X_tr, y_tr, X_val, quantiles=quantiles, l1_alpha=l1_alpha,
        )
        metrics = score_quantile_fold(
            y_val.values, preds,
            fold_label=f"Linear | {r['fold']}",
            starting=starting_val,
            verbose=False,
        )
        y_val_arr = y_val.values
        naive_p = train_df.loc[val_mask, naive_primary].values
        mae_naive, n_naive = naive_mae(y_val_arr, naive_p)
        row = {
            "fold": r["fold"],
            "mae_naive": mae_naive,
            "mae_linear": metrics["mae"],
            "mae_xgb": r["mae"],
            "coverage_80pct": metrics["coverage_80pct"],
            "n_naive": n_naive,
            "n": metrics["n"],
            "delta_linear_naive": metrics["mae"] - mae_naive,
            "delta_xgb_linear": r["mae"] - metrics["mae"],
        }
        wf_by_fold.append(row)
        models_last = models
        preds_last = preds

    print(f"\nThree-way MAE by walk-forward fold:")
    print(
        f"  {'Fold':<45} {'Naive':>8} {'Linear':>8} {'XGB':>8} "
        f"{'ΔL−N':>8} {'ΔX−L':>8} {'Cov80':>8}"
    )
    for row in wf_by_fold:
        print(
            f"  {row['fold']:<45} {row['mae_naive']:8.3f} {row['mae_linear']:8.3f} "
            f"{row['mae_xgb']:8.3f} {row['delta_linear_naive']:+8.3f} "
            f"{row['delta_xgb_linear']:+8.3f} {row['coverage_80pct']:7.1%}"
        )

    y_last = train_df.loc[last_fold["val_mask"], target_col].values
    naive_last = train_df.loc[last_fold["val_mask"], naive_primary].values
    linear_last = np.asarray(preds_last["q_0.50"])  # type: ignore[index]
    xgb_last = np.asarray(xgb_preds_last["q_0.50"])

    wf_linear_vs_naive = compare_model_vs_naive(
        y_last, linear_last, naive_last,
        "WF last fold: linear vs naive",
        alpha=alpha,
    )
    wf_xgb_vs_linear = compare_model_vs_naive(
        y_last, xgb_last, linear_last,
        "WF last fold: XGB vs linear",
        alpha=alpha,
    )

    print(f"\n{'─' * 60}")
    print("Walk-forward last-fold pairwise tests:")
    for test in (wf_linear_vs_naive, wf_xgb_vs_linear):
        print(f"  {test['label']}")
        print(
            f"    MAE challenger={test['mae_model']:.3f}  "
            f"baseline={test['mae_naive']:.3f}  "
            f"Δ={test['delta_mae']:+.3f}  p={test['p_value']:.4g}  "
            f"{'REJECT' if test['reject_h0'] else 'fail to reject'}"
        )

    # Holdout: fit on full train pool (no early stopping for linear)
    X_ho = holdout_df[features]
    y_ho = holdout_df[target_col]
    models_ho, preds_ho = fit_quantile_linear(
        X[features], y, X_ho, quantiles=quantiles, l1_alpha=l1_alpha,
    )
    starting_ho = holdout_df[role_col].values
    ho_metrics = score_quantile_fold(
        y_ho.values, preds_ho,
        fold_label=f"Linear | {holdout_label}",
        starting=starting_ho,
        verbose=True,
    )

    naive_ho = holdout_df[naive_primary].values
    mae_naive_ho, _ = naive_mae(y_ho.values, naive_ho)
    mae_xgb_ho = float(
        mean_absolute_error(y_ho.values, np.asarray(xgb_preds_ho["q_0.50"]))
    )

    print(f"\n{holdout_label} three-way MAE:")
    print(f"  Naive  : {mae_naive_ho:.3f}")
    print(f"  Linear : {ho_metrics['mae']:.3f}  (cov80={ho_metrics['coverage_80pct']:.1%})")
    print(f"  XGB    : {mae_xgb_ho:.3f}")

    ho_linear_vs_naive = compare_model_vs_naive(
        y_ho.values, preds_ho["q_0.50"], naive_ho,
        f"{holdout_label}: linear vs naive",
        alpha=alpha,
    )
    ho_xgb_vs_linear = compare_model_vs_naive(
        y_ho.values, np.asarray(xgb_preds_ho["q_0.50"]), preds_ho["q_0.50"],
        f"{holdout_label}: XGB vs linear",
        alpha=alpha,
    )

    print(f"\n{holdout_label} pairwise tests:")
    for test in (ho_linear_vs_naive, ho_xgb_vs_linear):
        print(f"  {test['label']}")
        print(
            f"    MAE challenger={test['mae_model']:.3f}  "
            f"baseline={test['mae_naive']:.3f}  "
            f"Δ={test['delta_mae']:+.3f}  p={test['p_value']:.4g}  "
            f"{'REJECT' if test['reject_h0'] else 'fail to reject'}"
        )

    return {
        "wf_by_fold": wf_by_fold,
        "models_last": models_last,
        "preds_last": preds_last,
        "models_ho": models_ho,
        "preds_ho": preds_ho,
        "ho_metrics": ho_metrics,
        "mae_naive_ho": mae_naive_ho,
        "mae_xgb_ho": mae_xgb_ho,
        "wf_linear_vs_naive": wf_linear_vs_naive,
        "wf_xgb_vs_linear": wf_xgb_vs_linear,
        "ho_linear_vs_naive": ho_linear_vs_naive,
        "ho_xgb_vs_linear": ho_xgb_vs_linear,
    }


def ensure_naive_cols(
    frame: pd.DataFrame,
    *,
    target_col: str,
    primary_col: str,
    secondary_col: str,
    player_col: str = "player_id",
    season_col: str = "season_year",
    date_col: str = "game_date",
) -> pd.DataFrame:
    """Ensure shift-then-expand season avg and lag-1 columns exist.

    Only builds missing columns. Existing engineered features are left as-is
    (preferred — they should come from ``src.pipeline.features``).
    """
    out = frame.copy()
    need_lag = secondary_col not in out.columns
    need_avg = primary_col not in out.columns
    if not need_lag and not need_avg:
        return out

    ordered = out.sort_values([player_col, season_col, date_col])
    if need_lag:
        lag = ordered.groupby(player_col, sort=False)[target_col].shift(1)
        out[secondary_col] = lag.reindex(out.index)
    if need_avg:
        shifted = ordered.groupby(
            [player_col, season_col], sort=False
        )[target_col].shift(1)
        season_avg = (
            shifted.groupby(
                [ordered[player_col], ordered[season_col]], sort=False
            )
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        out[primary_col] = season_avg.reindex(out.index)
    return out


def naive_mae(y_true: np.ndarray, y_hat: np.ndarray) -> tuple[float, int]:
    """MAE on rows where the naive prediction is finite."""
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    mask = np.isfinite(y_hat) & np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan"), 0
    return float(mean_absolute_error(y_true[mask], y_hat[mask])), int(mask.sum())


def compare_model_vs_naive(
    y_true,
    model_pred,
    naive_pred,
    label: str,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired one-sided Wilcoxon on |error|: H1 = model better than naive.

    Reject H0 only if model MAE < naive MAE and p < alpha.
    """
    y_true = np.asarray(y_true, dtype=float)
    model_pred = np.asarray(model_pred, dtype=float)
    naive_pred = np.asarray(naive_pred, dtype=float)

    mask = np.isfinite(model_pred) & np.isfinite(naive_pred) & np.isfinite(y_true)
    n = int(mask.sum())
    if n < 20:
        return {
            "label": label,
            "n": n,
            "mae_model": float("nan"),
            "mae_naive": float("nan"),
            "delta_mae": float("nan"),
            "p_value": float("nan"),
            "reject_h0": False,
        }

    err_m = np.abs(y_true[mask] - model_pred[mask])
    err_n = np.abs(y_true[mask] - naive_pred[mask])
    mae_m = float(err_m.mean())
    mae_n = float(err_n.mean())
    delta = mae_m - mae_n

    diff = err_n - err_m  # positive => model better
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        p_value = float(
            wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue
        )

    return {
        "label": label,
        "n": n,
        "mae_model": mae_m,
        "mae_naive": mae_n,
        "delta_mae": delta,
        "p_value": p_value,
        "reject_h0": bool((mae_m < mae_n) and (p_value < alpha)),
    }


def run_naive_comparison(
    wf_results: list[dict[str, Any]],
    train_df: pd.DataFrame,
    *,
    target_col: str,
    naive_primary: str,
    naive_secondary: str,
    y_val_last,
    preds_last: dict[str, np.ndarray],
    last_fold: dict[str, Any],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Compare model MAE vs frozen naive baselines on walk-forward folds."""
    print("── Naive baselines on walk-forward validation folds ─────────────────")
    print(f"  Primary H0 predictor : {naive_primary}")
    print(f"  Secondary (report)   : {naive_secondary}")
    print(f"  Alpha                : {alpha}")

    naive_wf_primary: list[dict[str, Any]] = []
    naive_wf_secondary: list[dict[str, Any]] = []

    for r in wf_results:
        val_mask = r["val_mask"]
        y_val = train_df.loc[val_mask, target_col].values
        naive_p = train_df.loc[val_mask, naive_primary].values
        naive_s = train_df.loc[val_mask, naive_secondary].values
        mae_p, n_p = naive_mae(y_val, naive_p)
        mae_s, n_s = naive_mae(y_val, naive_s)
        naive_wf_primary.append({
            "fold": r["fold"],
            "mae_model": r["mae"],
            "mae_naive": mae_p,
            "delta_mae": r["mae"] - mae_p,
            "n_naive": n_p,
            "n_model": r["n"],
        })
        naive_wf_secondary.append({
            "fold": r["fold"],
            "mae_model": r["mae"],
            "mae_naive": mae_s,
            "delta_mae": r["mae"] - mae_s,
            "n_naive": n_s,
            "n_model": r["n"],
        })

    print(f"\nPrimary naive ({naive_primary}) vs model MAE by fold:")
    print(f"  {'Fold':<45} {'Model':>8} {'Naive':>8} {'ΔMAE':>8} {'n':>7}")
    for row in naive_wf_primary:
        print(
            f"  {row['fold']:<45} {row['mae_model']:8.3f} {row['mae_naive']:8.3f} "
            f"{row['delta_mae']:+8.3f} {row['n_naive']:7d}"
        )

    print(f"\nSecondary naive ({naive_secondary}) vs model MAE by fold:")
    print(f"  {'Fold':<45} {'Model':>8} {'Naive':>8} {'ΔMAE':>8} {'n':>7}")
    for row in naive_wf_secondary:
        print(
            f"  {row['fold']:<45} {row['mae_model']:8.3f} {row['mae_naive']:8.3f} "
            f"{row['delta_mae']:+8.3f} {row['n_naive']:7d}"
        )

    y_last = (
        y_val_last.values if hasattr(y_val_last, "values") else np.asarray(y_val_last)
    )
    model_last = np.asarray(preds_last["q_0.50"])
    naive_last_p = train_df.loc[last_fold["val_mask"], naive_primary].values
    naive_last_s = train_df.loc[last_fold["val_mask"], naive_secondary].values

    wf_test_primary = compare_model_vs_naive(
        y_last, model_last, naive_last_p, f"WF last fold vs {naive_primary}",
        alpha=alpha,
    )
    wf_test_secondary = compare_model_vs_naive(
        y_last, model_last, naive_last_s, f"WF last fold vs {naive_secondary}",
        alpha=alpha,
    )

    print(f"\n{'─' * 60}")
    print("Walk-forward last-fold hypothesis test (primary naive):")
    print(f"  n paired        : {wf_test_primary['n']:,}")
    print(f"  MAE model       : {wf_test_primary['mae_model']:.3f}")
    print(f"  MAE naive       : {wf_test_primary['mae_naive']:.3f}")
    print(f"  ΔMAE (m−n)      : {wf_test_primary['delta_mae']:+.3f}")
    print(f"  Wilcoxon p      : {wf_test_primary['p_value']:.4g}")
    print(
        f"  Reject H0?      : "
        f"{'YES — model beats naive' if wf_test_primary['reject_h0'] else 'NO — fail to reject H0'}"
    )

    return {
        "naive_wf_primary": naive_wf_primary,
        "naive_wf_secondary": naive_wf_secondary,
        "wf_test_primary": wf_test_primary,
        "wf_test_secondary": wf_test_secondary,
    }


def evaluate_holdout_vs_naive(
    holdout_df: pd.DataFrame,
    preds_ho: dict[str, np.ndarray],
    *,
    target_col: str,
    naive_primary: str,
    naive_secondary: str,
    naive_wf_primary: list[dict[str, Any]] | None = None,
    wf_test_primary: dict[str, Any] | None = None,
    alpha: float = DEFAULT_ALPHA,
    holdout_label: str = "holdout",
) -> dict[str, Any]:
    """Holdout hypothesis test: model median vs frozen naive baselines."""
    print(f"{'─' * 55}")
    print(f"{holdout_label} — model vs frozen naive baselines")
    print(f"  H0: model does not beat naive | H1: model beats naive (α={alpha})")

    y_ho_arr = holdout_df[target_col].values
    model_ho_med = preds_ho["q_0.50"]
    naive_ho_p = holdout_df[naive_primary].values
    naive_ho_s = holdout_df[naive_secondary].values

    ho_test_primary = compare_model_vs_naive(
        y_ho_arr, model_ho_med, naive_ho_p,
        f"{holdout_label} vs {naive_primary}",
        alpha=alpha,
    )
    ho_test_secondary = compare_model_vs_naive(
        y_ho_arr, model_ho_med, naive_ho_s,
        f"{holdout_label} vs {naive_secondary}",
        alpha=alpha,
    )

    for test, name in [
        (ho_test_primary, f"PRIMARY ({naive_primary})"),
        (ho_test_secondary, f"secondary ({naive_secondary})"),
    ]:
        print(f"\n  {name}")
        print(f"    n paired   : {test['n']:,}")
        print(f"    MAE model  : {test['mae_model']:.3f}")
        print(f"    MAE naive  : {test['mae_naive']:.3f}")
        print(f"    ΔMAE (m−n) : {test['delta_mae']:+.3f}")
        print(f"    Wilcoxon p : {test['p_value']:.4g}")
        print(
            f"    Decision   : "
            f"{'REJECT H0 — model significantly better' if test['reject_h0'] else 'FAIL TO REJECT H0'}"
        )

    return {
        "primary_col": naive_primary,
        "secondary_col": naive_secondary,
        "alpha": alpha,
        "primary": ho_test_primary,
        "secondary": ho_test_secondary,
        "wf_primary_by_fold": naive_wf_primary,
        "wf_last_fold_test": wf_test_primary,
    }
