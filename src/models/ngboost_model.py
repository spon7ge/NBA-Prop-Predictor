import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE, LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

# Try to import NegativeBinomial, fall back to Normal if not available
try:
    from ngboost.distns import NegativeBinomial
except ImportError:
    # NegativeBinomial not available in this version of NGBOOST
    # We'll use Normal for training but can still use Negative Binomial for probability calculations
    NegativeBinomial = None

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
                     variance_calibration_factor: float = 1.25,
                     target_transform: str | None = None,
                     boost_high_scorers: bool = True,
                     high_scorer_percentile: float = 75,
                     high_scorer_boost: float = 2.0):
    """
    Fit NGBOOST model with split mean and variance models.
    
    Note: If NegativeBinomial is requested but not available, falls back to Normal.
    You can still use Negative Binomial for probability calculations in get_over_under_probabilities().
    """
    # If NegativeBinomial was requested but not available, use Normal
    if distribution is not None and distribution.__name__ == 'NegativeBinomial' and NegativeBinomial is None:
        print("Warning: NegativeBinomial not available in this NGBOOST version. Using Normal distribution.")
        print("Note: You can still use Negative Binomial for probability calculations.")
        distribution = Normal
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
        
    if target_transform == 'log':
        y_tr = np.log1p(y_tr)
        if y_va is not None:
            y_va = np.log1p(y_va)
        
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_tr = X_tr.fillna(X_tr.median())
    if X_va is not None:
        X_va = X_va.replace([np.inf, -np.inf], np.nan)
        X_va = X_va.fillna(X_tr.median())
    
    base_est_mean = DecisionTreeRegressor(
        max_depth=8, 
        min_samples_leaf=10  
    )
    base_est_scale = DecisionTreeRegressor(
        max_depth=variance_max_depth,
        min_samples_leaf=10
    )
    
    early_stopping_params = {}
    if val_df is not None and len(val_df) > 0:
        early_stopping_params = {'early_stopping_rounds': 20}
    
    w_recent = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    if boost_high_scorers:
        orig_y_tr = train_df[target_col].to_numpy()
        threshold = np.percentile(orig_y_tr, high_scorer_percentile)
        w_recent[orig_y_tr >= threshold] *= high_scorer_boost

    w_recent_val = None
    if X_va is not None:
        w_recent_val = build_recent_weights(val_df, player_col, recent_n, recent_weight)
        if boost_high_scorers:
            orig_y_va = val_df[target_col].to_numpy()
            threshold_val = np.percentile(orig_y_va, high_scorer_percentile)
            w_recent_val[orig_y_va >= threshold_val] *= high_scorer_boost
    
    # Train mean model
    ngb_mean = NGBRegressor(
        Dist=distribution,
        Score=LogScore,
        natural_gradient=True,
        learning_rate=0.02,
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
    
    # Log-transform for stability (variance is multiplicative)
    log_squared_residuals = np.log(squared_residuals + 1e-6)

    # Build variance weights
    if variance_recent_weight == 1.0:
        w_variance = np.ones(len(train_df))
        w_variance_val = None if X_va is None else np.ones(len(val_df))
    else:
        w_variance = build_recent_weights(train_df, player_col, recent_n, variance_recent_weight)
        w_variance_val = build_recent_weights(val_df, player_col, recent_n, 
                                              variance_recent_weight) if X_va is not None else None
    
    # Train variance model
    # ALWAYS use Normal distribution for variance model because we are predicting 
    # log_squared_residuals, which is a continuous variable (not count data).
    ngb_scale = NGBRegressor(
        Dist=Normal, 
        Score=LogScore,
        natural_gradient=True,
        learning_rate=variance_lr,
        n_estimators=variance_n_estimators,
        random_state=42,
        minibatch_frac=0.7,
        verbose=False,
        **early_stopping_params,
        Base=base_est_scale
    )
    
    if X_va is not None:
        y_pred_mean_va = ngb_mean.predict(X_va)
        residuals_va = y_va - y_pred_mean_va
        squared_residuals_va = residuals_va ** 2
        log_squared_residuals_va = np.log(squared_residuals_va + 1e-6)
        
        ngb_scale.fit(
            X_tr, log_squared_residuals,
            X_val=X_va, Y_val=log_squared_residuals_va,
            sample_weight=w_variance, 
            val_sample_weight=w_variance_val
        )
    else:
        ngb_scale.fit(X_tr, log_squared_residuals, sample_weight=w_variance)
    
    # Calculate bias correction AFTER training both models
    bias_correction = 0.0
    if X_va is not None:
        mean_pred_val = ngb_mean.predict(X_va)
        bias = (mean_pred_val - y_va).mean()
        bias_correction = -bias
    
    # Store calibration parameters on the models for easy access
    ngb_mean.bias_correction_ = bias_correction
    ngb_scale.variance_calibration_factor_ = variance_calibration_factor
    ngb_mean.target_transform_ = target_transform
    
    # Return same 3 items as before
    return ngb_mean, ngb_scale, variance_calibration_factor

def predict_mean_variance(model: NGBRegressor, df: pd.DataFrame, features: list[str], prediction_type: str = 'mean'):
    """Return per-row mean and variance from NGBoost predictive distribution.
    
    Args:
        prediction_type: 'mean' (expects log-normal correction), 'median' (no correction), 
                        or 'robust_mean' (clipped variance correction)
    """
    X = df[features].replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = X.fillna(X.median())
        
    dist = model.pred_dist(X)
    mean = dist.loc
    var = dist.scale ** 2
    bias_correction = getattr(model, 'bias_correction_', 0.0)
    mean = mean + bias_correction
    
    target_transform = getattr(model, 'target_transform_', None)
    if target_transform == 'log':
        # Model predicts log(y+1). Y = log(X+1) ~ N(mu, sigma^2)
        # E[X] = exp(mu + sigma^2/2) - 1
        # Median[X] = exp(mu) - 1
        mu = mean
        sigma2 = var
        
        if prediction_type == 'median':
            original_mean = np.expm1(mu)
        elif prediction_type == 'robust_mean':
            # Clip sigma2 to avoid exploding predictions if variance model is unstable
            sigma2_clipped = np.minimum(sigma2, 0.5) 
            original_mean = np.exp(mu + sigma2_clipped/2) - 1
        else: # 'mean'
            original_mean = np.exp(mu + sigma2/2) - 1
            
        original_var = (np.exp(sigma2) - 1) * np.exp(2*mu + sigma2)
        return original_mean, original_var
        
    return mean, var


def get_score_dependent_calibration_factor(predicted_mean: np.ndarray, 
                                          base_factor: float = 1.25,
                                          low_score_threshold: float = 10.0,
                                          high_score_threshold: float = 15.0,
                                          low_factor: float = 1.0,
                                          mid_factor: float = 1.25,
                                          high_factor: float = 1.6,
                                          very_high_threshold: float = None,
                                          very_high_factor: float = None) -> np.ndarray:
    if very_high_factor is None:
        very_high_factor = high_factor
    
    # Set very_high_threshold to a value beyond the range if not provided
    if very_high_threshold is None:
        very_high_threshold = high_score_threshold  # Effectively disable the very_high range
    
    # Convert to numpy array and handle None/NaN values
    # First check if input is None
    if predicted_mean is None:
        raise ValueError("predicted_mean is None")
    
    # Convert to list first if it's not already, to handle None values
    if isinstance(predicted_mean, (np.ndarray, pd.Series)):
        predicted_mean = predicted_mean.tolist()
    elif not isinstance(predicted_mean, (list, tuple)):
        predicted_mean = [predicted_mean]
    
    # Replace None with 0.0 before converting to array
    predicted_mean = [0.0 if x is None else float(x) for x in predicted_mean]
    
    # Now convert to numpy array
    predicted_mean = np.asarray(predicted_mean, dtype=float)
    
    if predicted_mean.size == 0:
        raise ValueError("predicted_mean is empty")
    
    # Replace NaN and inf values with 0
    predicted_mean = np.nan_to_num(predicted_mean, nan=0.0, posinf=0.0, neginf=0.0)
    
    factors = np.full_like(predicted_mean, mid_factor, dtype=float)
    
    # Create masks for different ranges (only where values are valid)
    valid_mask = ~np.isnan(predicted_mean) & ~np.isinf(predicted_mean)
    low_mask = valid_mask & (predicted_mean < low_score_threshold)
    very_high_mask = valid_mask & (predicted_mean >= very_high_threshold)
    high_mask = valid_mask & (predicted_mean >= high_score_threshold) & (predicted_mean < very_high_threshold)
    mid_mask = valid_mask & (predicted_mean >= low_score_threshold) & (predicted_mean < high_score_threshold)
    
    # Apply factors
    factors[low_mask] = low_factor
    factors[very_high_mask] = very_high_factor
    factors[high_mask] = high_factor
    
    # Smooth transition in mid range
    if mid_mask.any():
        mid_scores = predicted_mean[mid_mask]
        # Linear interpolation: low_factor at low_threshold, high_factor at high_threshold
        t = (mid_scores - low_score_threshold) / (high_score_threshold - low_score_threshold)
        factors[mid_mask] = low_factor + t * (high_factor - low_factor)
    
    # Smooth transition in high range (if very_high_threshold is different from high_threshold)
    if very_high_threshold > high_score_threshold and high_mask.any():
        high_scores = predicted_mean[high_mask]
        # Linear interpolation: high_factor at high_threshold, very_high_factor at very_high_threshold
        t = (high_scores - high_score_threshold) / (very_high_threshold - high_score_threshold)
        factors[high_mask] = high_factor + t * (very_high_factor - high_factor)
    
    # Apply base_factor as multiplier
    factors = factors * base_factor
    
    return factors

def predict_mean_variance_split(mean_model: NGBRegressor, 
                                 variance_model: NGBRegressor,
                                 df: pd.DataFrame, 
                                 features: list[str],
                                 calibration_factor: float = 1.25,
                                 prediction_type: str = 'mean',
                                 use_score_dependent_calibration: bool = False,
                                 score_dependent_params: dict = None):
    """Return per-row mean and variance from split models (mean + variance).
    
    Note: variance_model predicts log(variance), so we transform back via exp.
    The calibration_factor is applied to variance (not scale).
    
    Args:
        mean_model: Trained mean model
        variance_model: Trained variance model
        df: DataFrame with features
        features: List of feature names
        calibration_factor: Base factor to apply to variance (or constant if not using score-dependent)
        prediction_type: 'mean', 'median', or 'robust_mean'
        use_score_dependent_calibration: If True, use score-dependent calibration factors
        score_dependent_params: Dict with params for score-dependent calibration
            (low_score_threshold, high_score_threshold, low_factor, mid_factor, high_factor)
    
    Returns:
        mean, variance: Mean and variance predictions
    """
    X = df[features].replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = X.fillna(X.median())
        
    mean = mean_model.predict(X)
    bias_correction = getattr(mean_model, 'bias_correction_', 0.0)
    
    # Ensure bias_correction is not None
    if bias_correction is None:
        bias_correction = 0.0
    
    mean = mean + bias_correction
    
    # Validate mean - ensure no None or NaN values
    if mean is None:
        raise ValueError("Mean prediction returned None")
    
    # Convert to list first to handle None values, then to array
    if isinstance(mean, (np.ndarray, pd.Series)):
        mean = mean.tolist()
    elif not isinstance(mean, (list, tuple)):
        mean = [mean]
    
    # Replace None with 0.0 before converting to array
    mean = [0.0 if x is None else float(x) for x in mean]
    mean = np.asarray(mean, dtype=float)
    
    if np.any(np.isnan(mean)) or np.any(np.isinf(mean)):
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    
    log_var_pred = variance_model.predict(X)  # Predicts log(variance)
    variance = np.exp(log_var_pred)  # Transform back to variance
    
    # Validate variance - ensure no None or NaN values
    if variance is None:
        raise ValueError("Variance prediction returned None")
    
    # Convert to list first to handle None values, then to array
    if isinstance(variance, (np.ndarray, pd.Series)):
        variance = variance.tolist()
    elif not isinstance(variance, (list, tuple)):
        variance = [variance]
    
    # Replace None with 1.0 before converting to array
    variance = [1.0 if x is None else float(x) for x in variance]
    variance = np.asarray(variance, dtype=float)
    
    if np.any(np.isnan(variance)) or np.any(np.isinf(variance)):
        variance = np.nan_to_num(variance, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Apply calibration factor(s)
    if use_score_dependent_calibration:
        # Get predicted mean in original space for calibration
        target_transform = getattr(mean_model, 'target_transform_', None)
        if target_transform == 'log':
            # Need to transform mean back to original space to determine calibration
            temp_mean_orig = np.expm1(mean) if prediction_type == 'median' else np.exp(mean + variance/2) - 1
        else:
            temp_mean_orig = mean
        
        # Ensure temp_mean_orig is valid (no None/NaN)
        temp_mean_orig = np.asarray(temp_mean_orig)
        if np.any(np.isnan(temp_mean_orig)) or np.any(np.isinf(temp_mean_orig)):
            temp_mean_orig = np.nan_to_num(temp_mean_orig, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get score-dependent factors
        params = score_dependent_params or {}
        # Extract very_high parameters if provided
        very_high_threshold = params.pop('very_high_threshold', None)
        very_high_factor = params.pop('very_high_factor', None)
        
        cal_factors = get_score_dependent_calibration_factor(
            temp_mean_orig,
            base_factor=calibration_factor,
            very_high_threshold=very_high_threshold,
            very_high_factor=very_high_factor,
            **params
        )
        variance = variance * cal_factors
    else:
        variance = variance * calibration_factor
    
    # Check if using Negative Binomial distribution
    distribution_type = getattr(mean_model, 'Dist', None)
    is_negative_binomial = False
    if distribution_type is not None:
        # Check if it's NegativeBinomial by checking the class name
        dist_name = distribution_type.__name__ if hasattr(distribution_type, '__name__') else str(distribution_type)
        if 'NegativeBinomial' in dist_name or 'NegBin' in dist_name:
            is_negative_binomial = True
    
    # For Negative Binomial, mean and variance are already in original space
    # No log transform needed - Negative Binomial works directly with count data
    if is_negative_binomial:
        return mean, variance
    
    target_transform = getattr(mean_model, 'target_transform_', None)
    if target_transform == 'log':
        # Model predicts log(y+1). Y = log(X+1) ~ N(mu, sigma^2)
        mu = mean
        sigma2 = variance
        
        if prediction_type == 'median':
            original_mean = np.expm1(mu)
        elif prediction_type == 'robust_mean':
             # Clip sigma2 to avoid exploding predictions
            sigma2_clipped = np.minimum(sigma2, 0.5)
            original_mean = np.exp(mu + sigma2_clipped/2) - 1
        else:
            original_mean = np.exp(mu + sigma2/2) - 1
            
        original_var = (np.exp(sigma2) - 1) * np.exp(2*mu + sigma2)
        return original_mean, original_var
        
    return mean, variance


def predict_interval(model: NGBRegressor, df: pd.DataFrame, features: list[str], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper prediction intervals at 1-alpha."""
    X = df[features].replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        X = X.fillna(X.median())
        
    dist = model.pred_dist(X)
    lower = dist.ppf(alpha / 2)
    upper = dist.ppf(1 - alpha / 2)
    
    target_transform = getattr(model, 'target_transform_', None)
    if target_transform == 'log':
        lower = np.expm1(lower)
        upper = np.expm1(upper)
        
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
        iso_reg = IsotonicRegression(increasing=True)
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

def validate_production_model(mean_model, variance_model, val_df, features, target_col, 
                              calibration_factor=1.25, prediction_type='mean',
                              use_score_dependent_calibration=False, score_dependent_params=None,
                              show_by_score_range=False):
    """
    Validate that your split model is properly calibrated.
    
    Args:
        mean_model: Trained mean model
        variance_model: Trained variance model
        val_df: Validation DataFrame
        features: List of feature names
        target_col: Target column name
        calibration_factor: Base calibration factor
        prediction_type: 'mean', 'median', or 'robust_mean'
        use_score_dependent_calibration: If True, use score-dependent calibration
        score_dependent_params: Parameters for score-dependent calibration
        show_by_score_range: If True, show calibration metrics by predicted score ranges
    """
    from scipy.stats import norm
    
    # Get predictions
    mu, variance = predict_mean_variance_split(
        mean_model, variance_model, val_df, features, calibration_factor, 
        prediction_type=prediction_type,
        use_score_dependent_calibration=use_score_dependent_calibration,
        score_dependent_params=score_dependent_params
    )
    sigma = np.sqrt(variance)
    
    y_actual = val_df[target_col].values
    
    # Calculate standardized residuals
    standardized = (y_actual - mu) / sigma
    
    print(f"=== CALIBRATION CHECK ({prediction_type}) ===")
    if use_score_dependent_calibration:
        print("Using score-dependent calibration")
    print(f"Standardized residuals mean: {standardized.mean():.3f} (should be ~0)")
    print(f"Standardized residuals std: {standardized.std():.3f} (should be ~1)")
    
    # Check if probabilities are accurate
    for quantile in [0.25, 0.5, 0.75]:
        predicted_value = norm.ppf(quantile, loc=mu, scale=sigma)
        actual_quantile = np.percentile(y_actual, quantile * 100)
        error = predicted_value.mean() - actual_quantile
        print(f"Q{int(quantile*100):02d}: Predicted={predicted_value.mean():.1f}, "
              f"Actual={actual_quantile:.1f}, Error={error:+.1f}")
    
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
    
    # Show calibration by score range if requested
    if show_by_score_range:
        print("\n=== CALIBRATION BY SCORE RANGE ===")
        score_ranges = [
            (0, 8, "Low (0-8)"),
            (8, 12, "Mid-Low (8-12)"),
            (12, 16, "Mid-High (12-16)"),
            (16, 25, "High (16-25)"),
            (25, 100, "Very High (25+)")
        ]
        
        for low, high, label in score_ranges:
            mask = (mu >= low) & (mu < high)
            if mask.sum() == 0:
                continue
            
            mu_range = mu[mask]
            sigma_range = sigma[mask]
            y_range = y_actual[mask]
            std_range = standardized[mask]
            
            # Q75 check for this range
            q75_pred = norm.ppf(0.75, loc=mu_range, scale=sigma_range).mean()
            q75_actual = np.percentile(y_range, 75)
            
            print(f"\n{label} (n={mask.sum()}):")
            print(f"  Mean pred: {mu_range.mean():.2f}, Mean actual: {y_range.mean():.2f}")
            print(f"  Q75 pred: {q75_pred:.1f}, Q75 actual: {q75_actual:.1f}, Error: {q75_pred - q75_actual:+.1f}")
            print(f"  Std residuals mean: {std_range.mean():.3f}, std: {std_range.std():.3f}")
    
    return standardized


def optimize_score_dependent_calibration(
    mean_model, variance_model, val_df, features, target_col,
    base_factor=1.25, prediction_type='median',
    optimize_for='all'  # 'all', 'q75', 'q50', 'balanced', 'conservative', 'aggressive'
):
    """
    Optimize score-dependent calibration with proper weighting for betting.
    
    Args:
        optimize_for: Optimization mode
            - 'all': Balance all quantiles (Q25, Q50, Q75) - default
            - 'q75': Focus on Q75 quantile accuracy
            - 'q50': Focus on Q50 (median) accuracy
            - 'balanced': Full multi-objective optimization (coverage + residuals + quantiles)
            - 'conservative': Favor higher variances (underconfident but safer)
            - 'aggressive': Favor tighter variances (overconfident but higher edge claims)
    """
    from scipy.stats import norm
    from scipy.optimize import minimize
    
    # Get base predictions
    mu_base, variance_base = predict_mean_variance_split(
        mean_model, variance_model, val_df, features, base_factor, 
        prediction_type=prediction_type
    )
    sigma_base = np.sqrt(variance_base)
    y_actual = val_df[target_col].values
    
    # Get predicted means in original space
    target_transform = getattr(mean_model, 'target_transform_', None)
    if target_transform == 'log':
        mu_orig = np.expm1(mu_base) if prediction_type == 'median' else np.exp(mu_base + variance_base/2) - 1
    else:
        mu_orig = mu_base
    
    def objective(params):
        """
        Multi-objective optimization with proper priorities for betting.
        """
        low_thresh, high_thresh, low_fac, mid_fac, high_fac = params
        
        # Ensure reasonable bounds
        if low_thresh >= high_thresh or low_thresh < 0 or high_thresh > 30:
            return 1e6
        
        # Ensure factors are ordered sensibly
        # (Don't want low_factor > high_factor, etc.)
        if low_fac > mid_fac or mid_fac > high_fac:
            return 1e6
        
        cal_params = {
            'low_score_threshold': low_thresh,
            'high_score_threshold': high_thresh,
            'low_factor': low_fac,
            'mid_factor': mid_fac,
            'high_factor': high_fac
        }
        
        mu, variance = predict_mean_variance_split(
            mean_model, variance_model, val_df, features, base_factor,
            prediction_type=prediction_type,
            use_score_dependent_calibration=True,
            score_dependent_params=cal_params
        )
        sigma = np.sqrt(variance)
        
        # ===================================================================
        # METRIC 1: Confidence Interval Coverage (MOST CRITICAL FOR BETTING)
        # ===================================================================
        # 90% CI should contain ~90% of observations
        lower_90 = norm.ppf(0.05, loc=mu, scale=sigma)
        upper_90 = norm.ppf(0.95, loc=mu, scale=sigma)
        coverage_90 = ((y_actual >= lower_90) & (y_actual <= upper_90)).mean()
        coverage_90_error = abs(coverage_90 - 0.90) * 100  # Weight: 100x
        
        # 95% CI should contain ~95% of observations
        lower_95 = norm.ppf(0.025, loc=mu, scale=sigma)
        upper_95 = norm.ppf(0.975, loc=mu, scale=sigma)
        coverage_95 = ((y_actual >= lower_95) & (y_actual <= upper_95)).mean()
        coverage_95_error = abs(coverage_95 - 0.95) * 100  # Weight: 100x
        
        # Penalize overconfidence MORE than underconfidence
        if coverage_90 < 0.90:  # Overconfident (dangerous!)
            coverage_90_error *= 1.5
        if coverage_95 < 0.95:  # Overconfident (dangerous!)
            coverage_95_error *= 1.5
        
        # ===================================================================
        # METRIC 2: Standardized Residuals Std (CRITICAL)
        # ===================================================================
        standardized = (y_actual - mu) / (sigma + 1e-8)
        std_residuals_std = standardized.std()
        
        # Asymmetric penalty: overconfidence is worse than underconfidence
        if std_residuals_std < 1.0:
            # Overconfident (variances too small) - DANGEROUS
            std_error = (1.0 - std_residuals_std) * 50.0
        else:
            # Underconfident (variances too large) - Less dangerous, just inefficient
            std_error = (std_residuals_std - 1.0) * 25.0
        
        # ===================================================================
        # METRIC 3: Standardized Residuals Mean (IMPORTANT)
        # ===================================================================
        std_residuals_mean = abs(standardized.mean())
        mean_error = std_residuals_mean * 20.0  # Weight: 20x
        
        # ===================================================================
        # METRIC 4: Quantile Errors by Score Range (SECONDARY)
        # ===================================================================
        # Check calibration separately for different score ranges
        score_range_errors = []
        
        for score_min, score_max in [(0, 12), (12, 18), (18, 100)]:
            mask = (mu >= score_min) & (mu < score_max)
            if mask.sum() > 20:  # Need minimum sample
                mu_range = mu[mask]
                sigma_range = sigma[mask]
                y_range = y_actual[mask]
                
                # Check Q75 for this range
                q75_pred = norm.ppf(0.75, loc=mu_range, scale=sigma_range).mean()
                q75_actual = np.percentile(y_range, 75)
                score_range_errors.append(abs(q75_pred - q75_actual))
        
        quantile_error = np.mean(score_range_errors) if score_range_errors else 0
        quantile_error *= 5.0  # Weight: 5x
        
        # ===================================================================
        # COMBINED ERROR with proper priorities
        # ===================================================================
        total_error = (
            coverage_90_error +      # ~100x (most important)
            coverage_95_error +      # ~100x (most important)
            std_error +              # ~25-50x (critical)
            mean_error +             # ~20x (important)
            quantile_error          # ~5x (secondary)
        )
        
        return total_error
    
    # Initial guess based on typical patterns
    initial_params = [10.0, 16.0, 0.85, 1.0, 1.6]
    
    # Bounds with better ranges
    bounds = [
        (5.0, 15.0),    # low_score_threshold
        (12.0, 25.0),   # high_score_threshold
        (0.6, 1.1),     # low_factor (can reduce variance significantly)
        (0.85, 1.15),   # mid_factor (should be close to 1.0)
        (1.3, 3.0)      # high_factor (high scorers need much more variance)
    ]
    
    # Optimization settings based on mode
    if optimize_for == 'conservative':
        # Favor higher variances (underconfident but safer)
        initial_params = [10.0, 16.0, 0.9, 1.05, 1.8]
    elif optimize_for == 'aggressive':
        # Favor tighter variances (overconfident but higher edge claims)
        initial_params = [10.0, 16.0, 0.8, 0.95, 1.4]
    elif optimize_for in ['all', 'q75', 'q50', 'balanced']:
        # Use default balanced approach for all these modes
        # The objective function already handles quantile optimization
        pass
    # else: use defaults
    
    # Try multiple methods
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    best_result = None
    best_error = np.inf
    
    for method in methods:
        try:
            result = minimize(
                objective, initial_params, 
                method=method, bounds=bounds,
                options={'maxiter': 150, 'disp': False}
            )
            if result.success and result.fun < best_error:
                best_result = result
                best_error = result.fun
        except Exception as e:
            continue
    
    if best_result is not None and best_result.success:
        opt_params = {
            'low_score_threshold': best_result.x[0],
            'high_score_threshold': best_result.x[1],
            'low_factor': best_result.x[2],
            'mid_factor': best_result.x[3],
            'high_factor': best_result.x[4]
        }
        
        print(f"✅ Optimization successful! Error: {best_error:.3f}")
        print(f"📊 Optimized parameters:")
        for key, val in opt_params.items():
            print(f"   {key}: {val:.3f}")
        
        # Validate the result
        mu_test, var_test = predict_mean_variance_split(
            mean_model, variance_model, val_df, features, base_factor,
            prediction_type=prediction_type,
            use_score_dependent_calibration=True,
            score_dependent_params=opt_params
        )
        sigma_test = np.sqrt(var_test)
        standardized_test = (y_actual - mu_test) / sigma_test
        
        lower_90 = norm.ppf(0.05, loc=mu_test, scale=sigma_test)
        upper_90 = norm.ppf(0.95, loc=mu_test, scale=sigma_test)
        coverage_90 = ((y_actual >= lower_90) & (y_actual <= upper_90)).mean()
        
        lower_95 = norm.ppf(0.025, loc=mu_test, scale=sigma_test)
        upper_95 = norm.ppf(0.975, loc=mu_test, scale=sigma_test)
        coverage_95 = ((y_actual >= lower_95) & (y_actual <= upper_95)).mean()
        
        print(f"\n✓ Validation metrics:")
        print(f"   Std residuals mean: {standardized_test.mean():.3f} (target: ~0)")
        print(f"   Std residuals std: {standardized_test.std():.3f} (target: ~1)")
        print(f"   90% CI coverage: {coverage_90:.1%} (target: 90%)")
        print(f"   95% CI coverage: {coverage_95:.1%} (target: 95%)")
        
        # Warning if still miscalibrated
        if abs(standardized_test.std() - 1.0) > 0.1:
            print(f"\n⚠️  WARNING: Std residuals std = {standardized_test.std():.3f}")
            print(f"   Model is {'overconfident' if standardized_test.std() > 1.1 else 'underconfident'}")
        
        if abs(coverage_95 - 0.95) > 0.03:
            print(f"\n⚠️  WARNING: 95% CI coverage = {coverage_95:.1%}")
            print(f"   {'Overconfident' if coverage_95 < 0.95 else 'Underconfident'} intervals")
        
        return opt_params
    else:
        print("❌ Optimization failed, using safe default parameters")
        return {
            'low_score_threshold': 10.0,
            'high_score_threshold': 18.0,
            'low_factor': 0.9,
            'mid_factor': 1.0,
            'high_factor': 1.8
        }

def remove_highly_correlated_features(df, features_list, target_col='PTS', threshold=0.95):
    available_features = [col for col in features_list if col in df.columns]
    missing_features = [col for col in features_list if col not in df.columns]
    
    if missing_features:
        print(f"\nWARNING: {len(missing_features)} features not found in dataframe:")
        for feat in missing_features[:20]:  # Show first 20
            print(f"  - {feat}")
        if len(missing_features) > 20:
            print(f"  ... and {len(missing_features) - 20} more")
    
    if target_col in df.columns and target_col not in available_features:
        available_features.append(target_col)
    
    corr_matrix = df[available_features].corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > threshold:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_val))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    features_to_remove = set()
    
    for feat1, feat2, corr in high_corr_pairs:
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue
            
        if feat1 == target_col or feat2 == target_col:
            continue
            
        feat1_target_corr = abs(corr_matrix.loc[feat1, target_col])
        feat2_target_corr = abs(corr_matrix.loc[feat2, target_col])
        
        if feat1_target_corr >= feat2_target_corr:
            features_to_remove.add(feat2)
            print(f"REMOVED: {feat2:30} (corr with {feat1}: {corr:.3f})")
        else:
            features_to_remove.add(feat1)
            print(f"REMOVED: {feat1:30} (corr with {feat2}: {corr:.3f})")
    
    cleaned_features = [f for f in available_features if f not in features_to_remove and f != target_col]
    
    print(f"\nSUMMARY:")
    print(f"Original features: {len(features_list)}")
    print(f"Missing from dataframe: {len(missing_features)}")
    print(f"Removed (high correlation): {len(features_to_remove)}")
    print(f"Final features: {len(cleaned_features)}")
    
    return cleaned_features

# Get feature importance from mean model
def get_ngboost_feature_importance(model, features):
    """
    Extract feature importance from NGBoost model.
    NGBoost aggregates importance from all base estimators.
    """
    importances = np.zeros(len(features))
    
    # Sum importances from all base estimators
    for estimator in model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            importances += estimator.feature_importances_
    
    # Normalize
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

import shap

def analyze_ngboost_shap(mean_model, variance_model, test_df, features, sample_size=1000):
    """
    Generate SHAP analysis for NGBoost models.
    """
    # Sample data if too large
    if len(test_df) > sample_size:
        test_sample = test_df.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_df.copy()
    
    X_test = test_sample[features].fillna(0)
    
    # SHAP for mean model
    print("🔍 SHAP Analysis for Mean Model:")
    print("-" * 50)
    explainer_mean = shap.TreeExplainer(mean_model)
    shap_values_mean = explainer_mean.shap_values(X_test)
    
    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values_mean).mean(axis=0)
    importance_df_mean = pd.DataFrame({
        'feature': features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features by SHAP importance (Mean Model):")
    for idx, row in importance_df_mean.head(20).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:8.4f}")
    
    # SHAP for variance model
    print("\n🔍 SHAP Analysis for Variance Model:")
    print("-" * 50)
    explainer_var = shap.TreeExplainer(variance_model)
    shap_values_var = explainer_var.shap_values(X_test)
    
    # Mean absolute SHAP values
    mean_shap_var = np.abs(shap_values_var).mean(axis=0)
    importance_df_var = pd.DataFrame({
        'feature': features,
        'importance': mean_shap_var
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Features by SHAP importance (Variance Model):")
    for idx, row in importance_df_var.head(20).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:8.4f}")
    
    return importance_df_mean, importance_df_var, shap_values_mean, shap_values_var

