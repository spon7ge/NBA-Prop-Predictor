"""Shared train/eval helpers for prop models.

Prop-specific constants (features, holdout season, XGB params, naive cols)
live in each notebook. Import orchestration helpers from the modules below.
"""

from models.shared.analysis import analyze_correlations, run_feature_ablation
from models.shared.artifacts import load_model_bundle, predict_quantiles, save_model_bundle
from models.shared.baselines import (
    compare_model_vs_naive,
    ensure_naive_cols,
    evaluate_holdout_vs_naive,
    fit_quantile_linear,
    run_naive_comparison,
    run_quantile_linear_baseline,
)
from models.shared.metrics import pinball_50, score_quantile_fold
from models.shared.splits import date_walk_forward_folds, prepare_splits, season_holdout_split
from models.shared.train import (
    evaluate_holdout,
    fit_quantile_models,
    run_timeseries_cv,
    run_walk_forward,
    tune_xgb_quantile,
)

__all__ = [
    "analyze_correlations",
    "compare_model_vs_naive",
    "date_walk_forward_folds",
    "ensure_naive_cols",
    "evaluate_holdout",
    "evaluate_holdout_vs_naive",
    "fit_quantile_linear",
    "fit_quantile_models",
    "load_model_bundle",
    "pinball_50",
    "predict_quantiles",
    "prepare_splits",
    "run_feature_ablation",
    "run_naive_comparison",
    "run_quantile_linear_baseline",
    "run_timeseries_cv",
    "run_walk_forward",
    "save_model_bundle",
    "score_quantile_fold",
    "season_holdout_split",
    "tune_xgb_quantile",
]
