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
                     variance_lr: float = 0.08,
                     variance_max_depth: int = 6,
                     variance_n_estimators: int = 300,
                     variance_calibration_factor: float = 1.25):
    """Fit mean with recency weights, variance with equal weights.
    
    This approach trains separate models for mean and variance:
    - Mean model uses recency weights to prioritize recent performance
    - Variance model uses equal (or slight recency) weights to capture 
      long-term volatility patterns
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame (optional)
        features: List of feature column names
        target_col: Name of target column
        distribution: Distribution type (default Normal)
        player_col: Column name for player ID
        recent_n: Number of recent games to weight
        recent_weight: Weight for recent games in mean model
        variance_recent_weight: Weight for recent games in variance model (1.0 = equal weights)
        variance_lr: Learning rate for variance model (default 0.08, higher than mean)
        variance_max_depth: Max tree depth for variance model (default 6, higher than mean)
        variance_n_estimators: Number of estimators for variance model (default 300)
        variance_calibration_factor: Factor to inflate variance predictions (default 1.25)
        
    Returns:
        Tuple of (mean_model, variance_model, calibration_factor)
    """
    
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
        X_va = X_va.fillna(X_tr.median())
    
    base_est_mean = DecisionTreeRegressor(max_depth=4)
    base_est_scale = DecisionTreeRegressor(max_depth=variance_max_depth)
    
    # Enable early stopping if validation data is available
    early_stopping_params = {}
    if val_df is not None and len(val_df) > 0:
        early_stopping_params = {'early_stopping_rounds': 20}
    
    # 1. Fit MEAN model with recency weights
    w_recent = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    w_recent_val = None
    if X_va is not None:
        w_recent_val = build_recent_weights(val_df, player_col, recent_n, recent_weight)
    
    ngb_mean = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=0.08,
        n_estimators=500,
        random_state=42,
        minibatch_frac=0.8,
        verbose=False,
        **early_stopping_params,
        Base=base_est_mean
    )
    
    if X_va is not None:
        ngb_mean.fit(X_tr, y_tr, X_val=X_va, Y_val=y_va, sample_weight=w_recent, val_sample_weight=w_recent_val)
    else:
        ngb_mean.fit(X_tr, y_tr, sample_weight=w_recent)
    
    # 2. Get residuals from mean model
    y_pred_mean = ngb_mean.predict(X_tr)
    residuals = y_tr - y_pred_mean
    
    # 3. Fit VARIANCE/SCALE model with EQUAL weights (or light recency)
    if variance_recent_weight == 1.0:
        w_variance = np.ones(len(train_df))  # No recency bias
        w_variance_val = None if X_va is None else np.ones(len(val_df))
    else:
        w_variance = build_recent_weights(train_df, player_col, recent_n, variance_recent_weight)
        w_variance_val = build_recent_weights(val_df, player_col, recent_n, variance_recent_weight) if X_va is not None else None
    
    ngb_scale = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=variance_lr,  # Use higher learning rate for variance
        n_estimators=variance_n_estimators,
        random_state=42,
        minibatch_frac=0.7,
        verbose=False,
        **early_stopping_params,
        Base=base_est_scale
    )
    
    # Fit on absolute residuals to model variance
    if X_va is not None:
        # For validation, we need residuals from validation set
        y_pred_mean_va = ngb_mean.predict(X_va)
        residuals_va = y_va - y_pred_mean_va
        ngb_scale.fit(X_tr, np.abs(residuals), X_val=X_va, Y_val=np.abs(residuals_va), sample_weight=w_variance, val_sample_weight=w_variance_val)
    else:
        ngb_scale.fit(X_tr, np.abs(residuals), sample_weight=w_variance)
    
    return ngb_mean, ngb_scale, variance_calibration_factor


def predict_mean_variance(model: NGBRegressor, df: pd.DataFrame, features: list[str]):
    """Return per-row mean and variance from NGBoost predictive distribution."""
    X = df[features]
    dist = model.pred_dist(X)
    mean = dist.loc
    var = dist.scale ** 2
    return mean, var


def predict_mean_variance_split(mean_model: NGBRegressor, 
                                 variance_model: NGBRegressor,
                                 df: pd.DataFrame, 
                                 features: list[str],
                                 calibration_factor: float = 1.25):
    """Return per-row mean and variance from split models (mean + variance).
    
    Args:
        mean_model: NGBoost model trained on target values (with recency weights)
        variance_model: NGBoost model trained on absolute residuals (with equal weights)
        df: DataFrame to make predictions on
        features: List of feature column names
        calibration_factor: Factor to inflate variance predictions (default 1.25)
        
    Returns:
        Tuple of (mean, variance) arrays
    """
    X = df[features]
    
    # Get mean prediction from mean model
    mean = mean_model.predict(X)
    
    # Get scale prediction from variance model (trained on |residuals|)
    variance_scale = variance_model.predict(X)
    
    # Apply calibration factor to inflate variance predictions
    variance_scale = variance_scale * calibration_factor
    
    # Convert scale to variance: variance = scale^2
    # Note: variance model was trained on |residuals|, which approximates scale
    var = variance_scale ** 2
    
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
    for quantile in [0.1, 0.25, 0.5, 0.75, 0.9]:
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

def production_predict(mean_model, variance_model, df, features, prop_line, calibration_factor=1.25):
    """
    Make production predictions with proper distribution.
    
    Returns:
    - mu: Expected points
    - sigma: Standard deviation
    - prob_over: Probability of going over the line
    """
    from scipy.stats import norm
    
    # Get predictions
    mu, variance = predict_mean_variance_split(
        mean_model, variance_model, df, features, calibration_factor
    )
    sigma = np.sqrt(variance)
    
    # Calculate probability of going over
    prob_over = 1 - norm.cdf(prop_line, loc=mu, scale=sigma)
    
    return mu, sigma, prob_over


def track_calibration_weekly(predictions_log):
    """
    Track how calibration changes over time.
    Alerts you if recalibration is needed.
    
    Args:
        predictions_log: DataFrame with columns:
            - date, predicted_mean, predicted_std, actual, calibration_used
    """
    predictions_log['week'] = pd.to_datetime(predictions_log['date']).dt.isocalendar().week
    
    weekly_calibration = []
    
    for week, group in predictions_log.groupby('week'):
        standardized = (group['actual'] - group['predicted_mean']) / group['predicted_std']
        
        weekly_calibration.append({
            'week': week,
            'n_predictions': len(group),
            'std_residuals': standardized.std(),
            'mean_residuals': standardized.mean(),
            'mae': np.abs(group['actual'] - group['predicted_mean']).mean()
        })
    
    cal_df = pd.DataFrame(weekly_calibration)
    
    print("=== CALIBRATION DRIFT MONITORING ===")
    print(f"Overall std of residuals: {cal_df['std_residuals'].mean():.3f}")
    print(f"Std variation across weeks: {cal_df['std_residuals'].std():.3f}")
    
    # Alert if calibration is drifting
    recent_std = cal_df['std_residuals'].tail(4).mean()  # Last 4 weeks
    if recent_std > 1.15:
        print(f"⚠️ ALERT: Recent calibration degrading (std={recent_std:.3f})")
        print(f"   Consider increasing calibration to {1.25 * recent_std:.2f}")
    elif recent_std < 0.90:
        print(f"⚠️ ALERT: Overconfident predictions (std={recent_std:.3f})")
        print(f"   Consider decreasing calibration to {1.25 * recent_std:.2f}")
    
    return cal_df