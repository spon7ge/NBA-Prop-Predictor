# when i implement ast and rebound use poisson distribution

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal, Poisson 
from ngboost.scores import MLE
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

def build_recent_weights(df, player_col, recent_n=15, recent_weight=5.0):
    """Build sample weights emphasizing recent games for each player."""
    w = np.ones(len(df), dtype=float)
    df_reset = df.reset_index(drop=True)
    idx = df_reset.groupby(player_col, sort=False).tail(recent_n).index
    w[idx] = recent_weight
    return w

def fit_ngboost_split(train_df: pd.DataFrame,
                     val_df: pd.DataFrame | None,
                     features: list[str],
                     target_col: str,
                     distribution: type = Normal,
                     player_col: str = 'PLAYER_ID',
                     recent_n: int = 30,
                     recent_weight: float = 3.0,
                     variance_recent_weight: float = 1.0,
                     variance_lr: float = 0.01,
                     variance_max_depth: int = 8,
                     variance_n_estimators: int = 600,
                     variance_calibration_factor: float = 0.95,
                     clip_residuals_percentile: float = 99.0):
    """
    This approach trains separate models for mean and variance:
    - Mean model uses recency weights to prioritize recent performance
    - Variance model uses equal (or slight recency) weights to capture 
      long-term volatility patterns
    """
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
        
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median())
    if X_va is not None:
        X_va = X_va.replace([np.inf, -np.inf], np.nan)
        X_va = X_va.fillna(X_tr.median())
    
    base_est_mean = DecisionTreeRegressor(
        max_depth=5, 
        min_samples_leaf=20  
    )
    base_est_scale = DecisionTreeRegressor(
        max_depth=variance_max_depth,
        min_samples_leaf=15
    )
    
    early_stopping_params = {}
    if val_df is not None and len(val_df) > 0:
        early_stopping_params = {'early_stopping_rounds': 20}
    
    w_recent = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    w_recent_val = None
    if X_va is not None:
        w_recent_val = build_recent_weights(val_df, player_col, recent_n, recent_weight)
    
    # Train mean model
    ngb_mean = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42,
        minibatch_frac=0.8,
        verbose=False,
        **early_stopping_params,
        Base=base_est_mean
    )
    
    if X_va is not None:
        ngb_mean.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va, 
                    sample_weight=w_recent, val_sample_weight=w_recent_val)
    else:
        ngb_mean.fit(X_tr, y_tr, sample_weight=w_recent)

    # Calculate residuals WITHOUT bias correction first (avoid data leakage)
    y_pred_mean = ngb_mean.predict(X_tr)
    residuals = y_tr - y_pred_mean
    squared_residuals = residuals ** 2
    
    # Clip extreme residuals to stabilize variance model
    upper_limit = np.percentile(squared_residuals, clip_residuals_percentile)
    squared_residuals_clipped = np.minimum(squared_residuals, upper_limit)

    # Build variance weights
    if variance_recent_weight == 1.0:
        w_variance = np.ones(len(train_df))
        w_variance_val = None if X_va is None else np.ones(len(val_df))
    else:
        w_variance = build_recent_weights(train_df, player_col, recent_n, variance_recent_weight)
        w_variance_val = build_recent_weights(val_df, player_col, recent_n, 
                                              variance_recent_weight) if X_va is not None else None
    
    # Train variance model
    ngb_scale = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=variance_lr,
        n_estimators=variance_n_estimators,
        random_state=42,
        minibatch_frac=1.0,
        verbose=False,
        **early_stopping_params,
        Base=base_est_scale
    )
    
    if X_va is not None:
        y_pred_mean_va = ngb_mean.predict(X_va)
        residuals_va = y_va - y_pred_mean_va
        squared_residuals_va = residuals_va ** 2
        squared_residuals_va_clipped = np.minimum(squared_residuals_va, upper_limit)
        
        ngb_scale.fit(
            X_tr, squared_residuals_clipped,
            X_val=X_va, Y_val=squared_residuals_va_clipped,
            sample_weight=w_variance, 
            val_sample_weight=w_variance_val
        )
    else:
        ngb_scale.fit(X_tr, squared_residuals_clipped, sample_weight=w_variance)
    
    # Calculate bias correction AFTER training both models
    bias_correction = 0.0
    if X_va is not None:
        mean_pred_val = ngb_mean.predict(X_va)
        bias = (mean_pred_val - y_va).mean()
        bias_correction = -bias
    
    # Store calibration parameters on the models for easy access
    ngb_mean.bias_correction_ = bias_correction
    ngb_scale.variance_calibration_factor_ = variance_calibration_factor
    
    # Return same 3 items as before
    return ngb_mean, ngb_scale, variance_calibration_factor


def fit_ngboost_full(train_df: pd.DataFrame,
                     val_df: pd.DataFrame | None,
                     features: list[str],
                     target_col: str,
                     distribution: type = Normal,
                     player_col: str = 'PLAYER_ID',
                     recent_n: int = 30,
                     recent_weight: float = 3.0,
                     learning_rate: float = 0.08,
                     max_depth: int = 4,
                     n_estimators: int = 500,
                     minibatch_frac: float = 0.8,
                     random_state: int = 42):
    """
    Fit a single NGBoost model that learns both mean and variance together.
    This is the standard NGBoost approach without splitting into separate models.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        features: List of feature names
        target_col: Name of target column
        distribution: Distribution type (Normal, Poisson)
        player_col: Column name for player ID (for recency weighting)
        recent_n: Number of recent games to weight more heavily
        recent_weight: Weight multiplier for recent games
        learning_rate: Learning rate for NGBoost
        max_depth: Max depth for base decision tree
        n_estimators: Number of boosting rounds
        minibatch_frac: Fraction of data to use per iteration
        random_state: Random seed
        
    Returns:
        Trained NGBRegressor model
    """
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
    
    # Handle infinite and NaN values
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median())
    if X_va is not None:
        X_va = X_va.replace([np.inf, -np.inf], np.nan)
        X_va = X_va.fillna(X_tr.median())
    
    # Build base estimator
    base_est = DecisionTreeRegressor(max_depth=max_depth)
    
    # Setup early stopping if validation data provided
    early_stopping_params = {}
    if val_df is not None and len(val_df) > 0:
        early_stopping_params = {'early_stopping_rounds': 20}
    
    # Build recency weights
    w_recent = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    w_recent_val = None
    if X_va is not None:
        w_recent_val = build_recent_weights(val_df, player_col, recent_n, recent_weight)
    
    # Create and train single NGBoost model
    ngb = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        minibatch_frac=minibatch_frac,
        verbose=False,
        **early_stopping_params,
        Base=base_est
    )
    
    # Fit the model
    if X_va is not None:
        ngb.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va, sample_weight=w_recent, val_sample_weight=w_recent_val)
    else:
        ngb.fit(X_tr, y_tr, sample_weight=w_recent)
    
    return ngb


def predict_mean_variance(model: NGBRegressor, df: pd.DataFrame, features: list[str]):
    """Return per-row mean and variance from NGBoost predictive distribution."""
    X = df[features]
    dist = model.pred_dist(X)
    mean = dist.loc
    var = dist.scale ** 2
    bias_correction = getattr(model, 'bias_correction_', 0.0)
    mean = mean + bias_correction
    return mean, var


def predict_mean_variance_split(mean_model: NGBRegressor, 
                                 variance_model: NGBRegressor,
                                 df: pd.DataFrame, 
                                 features: list[str],
                                 calibration_factor: float = 1.25):
    """Return per-row mean and variance from split models (mean + variance).
    
    Note: variance_model now predicts squared residuals (variance) directly,
    so the calibration_factor is applied to variance (not scale).
    """
    X = df[features]
    mean = mean_model.predict(X)
    bias_correction = getattr(mean_model, 'bias_correction_', 0.0)
    mean = mean + bias_correction
    variance = variance_model.predict(X)  # Already variance (squared residuals)
    variance = variance * calibration_factor  # Apply calibration to variance
    return mean, variance


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

def validate_production_model(mean_model, variance_model, val_df, features, target_col, calibration_factor=1.25):
    """
    Validate that your split model is properly calibrated.
    """
    from scipy.stats import norm
    
    # Get predictions
    mu, variance = predict_mean_variance_split(
        mean_model, variance_model, val_df, features, calibration_factor
    )
    sigma = np.sqrt(variance)
    
    y_actual = val_df[target_col].values
    
    # Calculate standardized residuals
    standardized = (y_actual - mu) / sigma
    
    print("=== CALIBRATION CHECK ===")
    print(f"Standardized residuals mean: {standardized.mean():.3f} (should be ~0)")
    print(f"Standardized residuals std: {standardized.std():.3f} (should be ~1)")
    
    # Check if probabilities are accurate
    for quantile in [0.25, 0.5, 0.75]:
        predicted_value = norm.ppf(quantile, loc=mu, scale=sigma)
        actual_quantile = np.percentile(y_actual, quantile * 100)
        print(f"Q{int(quantile*100):02d}: Predicted={predicted_value.mean():.1f}, "
              f"Actual={actual_quantile:.1f}")
    
    # Mean Absolute Error
    mae = np.abs(y_actual - mu).mean()
    print(f"\nMAE: {mae:.2f}")
    
    # Check coverage at different confidence levels
    for alpha in [0.1, 0.05, 0.01]:
        lower = norm.ppf(alpha/2, loc=mu, scale=sigma)
        upper = norm.ppf(1-alpha/2, loc=mu, scale=sigma)
        coverage = ((y_actual >= lower) & (y_actual <= upper)).mean()
        expected = 1 - alpha
        print(f"{int((1-alpha)*100)}% CI: Coverage={coverage:.2%}, "
              f"Expected={expected:.2%}")
    
    return standardized