# when i implement ast and rebound use poisson distribution

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal, Poisson 
from ngboost.scores import MLE
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit


def fit_ngboost(train_df: pd.DataFrame,
                val_df: pd.DataFrame | None,
                features: list[str],
                target_col: str,
                distribution: type = Normal):

    # Ensure data is sorted by date for time series consistency
    if 'GAME_DATE' in train_df.columns:
        train_df = train_df.sort_values('GAME_DATE').reset_index(drop=True)
    if val_df is not None and len(val_df) > 0 and 'GAME_DATE' in val_df.columns:
        val_df = val_df.sort_values('GAME_DATE').reset_index(drop=True)
    
    X_tr = train_df[features]
    y_tr = train_df[target_col].to_numpy()

    X_va = None
    y_va = None
    if val_df is not None and len(val_df) > 0:
        X_va = val_df[features]
        y_va = val_df[target_col].to_numpy()
    
    # For LogNormal distribution
    if distribution == LogNormal and np.any(y_tr <= 0):
        print(f"Warning: LogNormal requires positive targets. Shifting by {abs(np.min(y_tr)) + 1:.2f}")
        y_min = np.min(y_tr)
        shift = abs(y_min) + 1 if y_min <= 0 else 0
        y_tr = y_tr + shift
        if y_va is not None:
            y_va = y_va + shift
    
    # Clean infinity and large values from features
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median())
    if X_va is not None:
        X_va = X_va.replace([np.inf, -np.inf], np.nan)
        X_va = X_va.fillna(X_tr.median())  # Use train median for validation

    base_est = DecisionTreeRegressor(max_depth=4)

    # Enable early stopping if validation data is available
    early_stopping_params = {}
    if val_df is not None and len(val_df) > 0:
        early_stopping_params = {'early_stopping_rounds': 20}
    
    ngb = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
        minibatch_frac=0.7,
        verbose=False,
        **early_stopping_params,
        Base=base_est
    )

    if X_va is not None:
        ngb.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va)
    else:
        ngb.fit(X_tr, y_tr)

    return ngb


def predict_mean_variance(model: NGBRegressor, df: pd.DataFrame, features: list[str]):
    """Return per-row mean and variance from NGBoost predictive distribution."""
    X = df[features]
    dist = model.pred_dist(X)
    mean = dist.loc
    var = dist.scale ** 2
    return mean, var


def predict_interval(model: NGBRegressor, df: pd.DataFrame, features: list[str], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper prediction intervals at 1-alpha."""
    X = df[features]
    dist = model.pred_dist(X)
    lower = dist.ppf(alpha / 2)
    upper = dist.ppf(1 - alpha / 2)
    return lower, upper

def evaluate_calibration(model: NGBRegressor,
                         df: pd.DataFrame,
                         features: list[str],
                         target_col: str,
                         alpha: float = 0.05) -> tuple[float, float]:

    lower, upper = predict_interval(model, df, features, alpha)
    y_actual = df[target_col].values
    coverage = ((y_actual >= lower) & (y_actual <= upper)).mean()
    expected_coverage = 1 - alpha
    return coverage, expected_coverage


def calibrate_predictions(model: NGBRegressor,
                          train_df: pd.DataFrame,
                          features: list[str],
                          target_col: str,
                          method: str = 'isotonic') -> IsotonicRegression | dict[str, float]:
    """Calibrate model predictions using isotonic regression or simple bias correction.
    
    Args:
        model: Trained NGBRegressor
        train_df: Training data for fitting calibration
        features: List of feature names
        target_col: Name of target column
        method: 'isotonic' for isotonic regression or 'bias' for simple bias correction
        
    Returns:
        Calibration object (IsotonicRegression or dict with calibration params)
    """
    X_train = train_df[features]
    y_train = train_df[target_col].values
    
    # Get predictions from model
    pred_mean, _ = predict_mean_variance(model, train_df, features)
    pred_mean = pred_mean.values if hasattr(pred_mean, 'values') else pred_mean
    
    if method == 'isotonic':
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip', increasing=True)
        iso_reg.fit(pred_mean, y_train)
        return iso_reg
    elif method == 'bias':
        # Simple bias correction
        bias = np.mean(y_train - pred_mean)
        return {'bias': bias, 'method': 'bias'}
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def predict_calibrated_mean(model: NGBRegressor,
                            calibrator,
                            df: pd.DataFrame,
                            features: list[str],
                            method: str = 'isotonic') -> np.ndarray:
    """Get calibrated predictions from model.
    
    Args:
        model: Trained NGBRegressor
        calibrator: Calibration object from calibrate_predictions
        df: DataFrame to make predictions on
        features: List of feature names
        method: Calibration method used ('isotonic' or 'bias')
        
    Returns:
        Calibrated mean predictions
    """
    pred_mean, _ = predict_mean_variance(model, df, features)
    pred_mean = pred_mean.values if hasattr(pred_mean, 'values') else pred_mean
    
    if method == 'isotonic':
        return calibrator.predict(pred_mean)
    elif method == 'bias':
        return pred_mean + calibrator['bias']
    else:
        raise ValueError(f"Unknown calibration method: {method}")


# def predict_mean_with_tier_calibration(
#     model: NGBRegressor,
#     df: pd.DataFrame,
#     features: list[str],
#     player_avg_col: str = 'PTS_ROLLING_AVG_40',
#     star_threshold: float = 20.0,
#     role_threshold: float = 10.0,
#     star_correction: float = 1.30,
#     role_correction: float = 0.81,
#     bench_correction: float = -0.63,
#     return_variance: bool = False  # NEW
# ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:  # NEW

#     # Get raw predictions
#     pred_mean, pred_var = predict_mean_variance(model, df, features)
#     pred_mean = pred_mean.values if hasattr(pred_mean, 'values') else pred_mean
#     pred_var = pred_var.values if hasattr(pred_var, 'values') else pred_var
    
#     # Get player average
#     if player_avg_col not in df.columns:
#         raise ValueError(f"Column '{player_avg_col}' not found in dataframe")
    
#     player_avg = df[player_avg_col].values
    
#     # Apply tier-specific corrections (only to mean, not variance)
#     corrections = np.where(
#         player_avg > star_threshold, star_correction,
#         np.where(player_avg >= role_threshold, role_correction, bench_correction)
#     )
    
#     calibrated_mean = pred_mean + corrections
    
#     if return_variance:
#         return calibrated_mean, pred_var
#     return calibrated_mean


def _nll_score(estimator: NGBRegressor, X, y):
    """BayesSearchCV scorer: maximize log-likelihood (higher is better)."""
    dist = estimator.pred_dist(X)
    return float(dist.logpdf(y).mean())


def fit_ngboost_bayes(train_df: pd.DataFrame,
                      val_df: pd.DataFrame | None,
                      features: list[str],
                      target_col: str,
                      n_iter: int = 30,
                      random_state: int = 42,
                      fast_mode: bool = False,
                      distribution: type = Normal):

    # Ensure data is sorted by date for time series consistency
    if 'GAME_DATE' in train_df.columns:
        train_df = train_df.sort_values('GAME_DATE').reset_index(drop=True)
    if val_df is not None and len(val_df) > 0 and 'GAME_DATE' in val_df.columns:
        val_df = val_df.sort_values('GAME_DATE').reset_index(drop=True)
    
    X_tr = train_df[features]
    y_tr = train_df[target_col].to_numpy()

    X_va = None
    y_va = None
    if val_df is not None and len(val_df) > 0:
        X_va = val_df[features]
        y_va = val_df[target_col].to_numpy()
    
    # For LogNormal distribution, ensure positive targets
    if distribution == LogNormal and np.any(y_tr <= 0):
        print(f"Warning: LogNormal requires positive targets. Shifting by {abs(np.min(y_tr)) + 1:.2f}")
        y_min = np.min(y_tr)
        shift = abs(y_min) + 1 if y_min <= 0 else 0
        y_tr = y_tr + shift
        if y_va is not None:
            y_va = y_va + shift
    
    # Clean infinity and large values from features
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median())
    if X_va is not None:
        X_va = X_va.replace([np.inf, -np.inf], np.nan)
        X_va = X_va.fillna(X_tr.median())  # Use train median for validation

    if fast_mode:
        n_splits = 2
        n_estimators_range = (200, 500)
        max_depth_range = (2, 5)
        early_stopping = 15
    else:
        n_splits = 3
        n_estimators_range = (300, 800)
        max_depth_range = (2, 6)
        early_stopping = 20

    base_est = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        verbose=False,
        random_state=random_state,
        Base=DecisionTreeRegressor(),
        early_stopping_rounds=early_stopping
    )

    search_spaces = {
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'n_estimators': Integer(*n_estimators_range),
        'minibatch_frac': Real(0.4, 0.8),
        'Base__max_depth': Integer(*max_depth_range)
    }

    # This ensures training always comes before validation in time
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    search = BayesSearchCV(
        estimator=base_est,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring=_nll_score,
        cv=tscv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    fit_params = {}
    if X_va is not None:
        fit_params = {'X_val': X_va, 'Y_val': y_va}

    search.fit(X_tr, y_tr, **fit_params)

    return search.best_estimator_, search.best_params_


