import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import MLE
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor


def fit_ngboost(train_df: pd.DataFrame,
                val_df: pd.DataFrame | None,
                features: list[str],
                target_col: str,
                learning_rate: float = 0.06,
                n_estimators: int = 500,
                max_depth: int = 3,
                random_state: int = 42,
                use_gpu: bool = False,
                base_max_depth: int | None = None,
                distribution: type = Normal):

    # Ensure data is sorted by date for time series consistency
    # Note: NGBoost doesn't care about order internally, but this ensures no leakage
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

    # Choose base learner (GPU via XGBoost, or CPU DecisionTree)
    if base_max_depth is None:
        base_max_depth = max_depth

    if use_gpu:
        base_est = XGBRegressor(
            max_depth=base_max_depth,
            tree_method='hist',
            device='cuda',
            n_estimators=1,  # single tree per stage as weak learner
            learning_rate=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            random_state=random_state,
            n_jobs=1
        )
    else:
        base_est = DecisionTreeRegressor(max_depth=base_max_depth)

    # Enable early stopping if validation data is available
    early_stopping_params = {}
    if val_df is not None and len(val_df) > 0:
        early_stopping_params = {'early_stopping_rounds': 50}
    
    ngb = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
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


def predict_mean_with_tier_calibration(model: NGBRegressor,
                                       df: pd.DataFrame,
                                       features: list[str],
                                       player_avg_col: str = 'PTS_ROLLING_AVG_15',
                                       star_threshold: float = 20.0,
                                       role_threshold: float = 10.0,
                                       star_correction: float = 1.30,
                                       role_correction: float = 0.81,
                                       bench_correction: float = -0.63) -> np.ndarray:
    """Get predictions with tier-specific bias correction based on validation findings.
    
    Applies different calibrations for star players (>20 PPG), role players (10-20 PPG),
    and bench players (<10 PPG) based on observed bias patterns.
    
    Args:
        model: Trained NGBRegressor
        df: DataFrame to make predictions on
        features: List of feature names
        player_avg_col: Column name for player's rolling average to determine tier
        star_threshold: Points per game threshold for star players
        role_threshold: Points per game threshold for role players
        star_correction: Bias correction to add for star players (positive = reduce overprediction)
        role_correction: Bias correction to add for role players
        bench_correction: Bias correction to add for bench players (negative = reduce underprediction)
        
    Returns:
        Calibrated mean predictions with tier-specific adjustments
        
    Example:
        calibrated_preds = predict_mean_with_tier_calibration(
            model, test_df, features,
            player_avg_col='PTS_ROLLING_AVG_15'
        )
    """
    # Get raw predictions
    pred_mean, _ = predict_mean_variance(model, df, features)
    pred_mean = pred_mean.values if hasattr(pred_mean, 'values') else pred_mean
    
    # Get player average to determine tier
    if player_avg_col not in df.columns:
        raise ValueError(f"Column '{player_avg_col}' not found in dataframe")
    
    player_avg = df[player_avg_col].values
    
    # Apply tier-specific corrections
    corrections = np.where(
        player_avg > star_threshold, star_correction,
        np.where(player_avg >= role_threshold, role_correction, bench_correction)
    )
    
    return pred_mean + corrections


def _nll_score(estimator: NGBRegressor, X, y):
    """BayesSearchCV scorer: maximize log-likelihood (higher is better)."""
    dist = estimator.pred_dist(X)
    return float(dist.logpdf(y).mean())


def chronological_split(df: pd.DataFrame,
                        date_col: str,
                        train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single dataframe chronologically into train/val sets.
    
    IMPORTANT: This function operates on ONE dataframe and splits it by time.
    If you already have separate train/val data (e.g., by season), don't use this!
    
    Args:
        df: Single DataFrame to split
        date_col: Name of date column
        train_frac: Fraction of data to use for training (default 0.8)
        
    Returns:
        Tuple of (train_df, val_df)
        
    Example:
        # If you have combined data and need to split it:
        combined_data = pd.concat([season_2019, season_2020, season_2021])
        train_df, val_df = chronological_split(combined_data, 'GAME_DATE')
        
        # If you already have train/val split, skip this:
        # train_df = season_2019 + season_2020
        # val_df = season_2021
        # model = fit_ngboost(train_df, val_df, features, 'PTS')  # Don't call chronological_split!
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    cut = int(len(data) * train_frac)
    train_df = data.iloc[:cut].reset_index(drop=True)
    val_df = data.iloc[cut:].reset_index(drop=True)
    return train_df, val_df


def expanding_window_cv(df: pd.DataFrame,
                         date_col: str,
                         min_train_size: int = 200,
                         step_size: int = 30) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate train/val splits using expanding window for time-series cross-validation.
    
    This simulates walk-forward validation where you incrementally add data to training
    and test on the next period. Useful for testing model robustness over time.
    
    IMPORTANT: This operates on ONE combined dataframe. Use this for walk-forward CV
    evaluation, NOT for your final model training if you already have year-based splits.
    
    Args:
        df: Combined DataFrame with all time periods
        date_col: Name of date column
        min_train_size: Minimum number of unique dates in first training set
        step_size: Number of dates to advance the validation window each split
        
    Returns:
        List of (train_df, val_df) tuples for each expanding window split
        
    Example:
        # For walk-forward cross-validation evaluation:
        all_data = pd.concat([s19, s20, s21, s22, s23, s24, s25, s26])
        cv_splits = expanding_window_cv(all_data, 'GAME_DATE', min_train_size=200)
        
        # Now loop through splits to evaluate model performance over time:
        for train, val in cv_splits:
            model = fit_ngboost(train, None, features, 'PTS')  # Note: no val for training
            # evaluate on val set
            
        # For final model training with year-based splits, skip this:
        # train_df = pd.concat([s19, s20, s21, s22, s23, s24])
        # val_df = s25
        # final_model = fit_ngboost(train_df, val_df, features, 'PTS')
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    
    dates = data[date_col].sort_values().unique()
    splits = []
    
    for i in range(min_train_size, len(dates), step_size):
        train_dates = dates[:i]
        val_dates = dates[i:i + step_size] if i + step_size < len(dates) else dates[i:]
        
        train_df = data[data[date_col].isin(train_dates)].copy()
        val_df = data[data[date_col].isin(val_dates)].copy()
        
        if len(val_df) > 0:  # Only include splits with validation data
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            splits.append((train_df, val_df))
    
    return splits


def fit_ngboost_bayes(train_df: pd.DataFrame,
                      val_df: pd.DataFrame | None,
                      features: list[str],
                      target_col: str,
                      n_iter: int = 30,
                      random_state: int = 42,
                      use_gpu: bool = False,
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

    base_learn = XGBRegressor() if use_gpu else DecisionTreeRegressor()

    base_est = NGBRegressor(
        Dist=distribution,
        Score=MLE,
        natural_gradient=True,
        verbose=False,
        random_state=random_state,
        Base=base_learn
    )

    search_spaces = {
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'n_estimators': Integer(500, 2000),
        'minibatch_frac': Real(0.5, 1.0)
    }
    # Base learner hyperparameters to tune
    if use_gpu:
        search_spaces.update({
            'Base__max_depth': Integer(2, 8)
        })
    else:
        search_spaces.update({
            'Base__max_depth': Integer(2, 8)
        })

    search = BayesSearchCV(
        estimator=base_est,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring=_nll_score,
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    fit_params = {}
    if X_va is not None:
        fit_params = {'X_val': X_va, 'Y_val': y_va}

    search.fit(X_tr, y_tr, **fit_params)

    return search.best_estimator_, search.best_params_


