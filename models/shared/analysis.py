"""Feature analysis helpers (ablation, correlation)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def rmse_median_ablation(
    feature_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    xgb_params: dict,
) -> float:
    """Fit median quantile model on ``feature_cols``; return val RMSE."""
    m = XGBRegressor(**dict(xgb_params), quantile_alpha=0.50)
    m.fit(
        X_train[feature_cols], y_train,
        eval_set=[(X_val[feature_cols], y_val)],
        verbose=False,
    )
    pred = m.predict(X_val[feature_cols])
    return float(np.sqrt(mean_squared_error(y_val, pred)))


def run_feature_ablation(
    features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    xgb_params: dict,
) -> pd.DataFrame:
    """Leave-one-out feature ablation on the median model (val RMSE)."""
    baseline_rmse = rmse_median_ablation(
        features, X_train, y_train, X_val, y_val, xgb_params=xgb_params,
    )
    print(f"Baseline RMSE (all features): {baseline_rmse:.4f}")

    rows: list[dict[str, Any]] = []
    for feat in features:
        cols = [c for c in features if c != feat]
        ablated_rmse = rmse_median_ablation(
            cols, X_train, y_train, X_val, y_val, xgb_params=xgb_params,
        )
        delta = ablated_rmse - baseline_rmse
        rows.append({
            "feature": feat,
            "ablated_rmse": round(ablated_rmse, 4),
            "delta_rmse": round(delta, 4),
        })

    ablation_df = (
        pd.DataFrame(rows)
        .sort_values("delta_rmse", ascending=False)
        .reset_index(drop=True)
    )
    ablation_df.index += 1
    print(ablation_df.to_string())
    return ablation_df


def analyze_correlations(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
    *,
    plot: bool = True,
    title: str = "Feature Correlation Matrix",
) -> list[str]:
    """Identify highly correlated feature pairs; optionally plot a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title(title)
        plt.show()

    print(f"--- High Correlation Report (Threshold > {threshold}) ---")
    high_corr_pairs = []
    for col in upper.columns:
        connected = upper.index[upper[col] > threshold].tolist()
        for c in connected:
            high_corr_pairs.append((col, c, upper.loc[c, col]))

    if not high_corr_pairs:
        print("No features exceed the correlation threshold.")
    else:
        for f1, f2, val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"{f1} <-> {f2}: {val:.4f}")

    return to_drop
