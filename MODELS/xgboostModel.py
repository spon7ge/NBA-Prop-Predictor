from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def build_recent_weights(df, player_col, recent_n=15, recent_weight=5.0):
    w = np.ones(len(df), dtype=float)
    df_reset = df.reset_index(drop=True)
    idx = df_reset.groupby(player_col, sort=False).tail(recent_n).index
    w[idx] = recent_weight
    return w

def clean_feature_dtypes(X, cat_cols):
    Xc = X.copy()
    for c in Xc.columns:
        col = Xc[c] 
        if c in cat_cols:
            continue
        if col.dtype == bool:
            Xc[c] = col.astype(int)
        elif not np.issubdtype(col.dtype, np.number):
            Xc[c] = pd.to_numeric(col, errors="coerce")
    return Xc


def xgb_regression_params(base=None, use_gpu=False):
    """Default XGBoost regression parameters."""
    p = dict(
        objective="reg:pseudohubererror",  
        eval_metric="rmse",
        tree_method='hist',
        booster='gbtree',
        random_state=42,
        n_jobs=-1,
        
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=5,
        min_child_weight=4,
        
        subsample=0.75,
        colsample_bytree=0.75,
        
        gamma=0.8,
        reg_alpha=1.5,
        reg_lambda=8.0,
        
        early_stopping_rounds=75,
        verbosity=1,
    )
    if base:
        p.update(base)
    if use_gpu:
        p["tree_method"] = "hist"
        p["device"] = "cuda"
        p["n_jobs"] = 1
    return p


def tune_hyperparams(X_train, y_train, X_val, y_val, 
                     sample_weight=None, use_gpu=False, n_iter=50):
    """Tune hyperparameters for XGBoost regression model."""
    base_params = xgb_regression_params(use_gpu=use_gpu)
    base_params.pop("early_stopping_rounds", None)

    xgb = XGBRegressor(**base_params)

    param_dist = {
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "gamma": uniform(0, 2),
        "reg_alpha": uniform(0, 9),
        "reg_lambda": uniform(1, 50),
        "learning_rate": uniform(0.005, 0.02)
    }

    n_jobs = 1 if use_gpu else -1  
    
    search = BayesSearchCV(
        estimator=xgb,
        search_spaces=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        cv=[(range(len(y_train)), range(len(y_val)))],
        random_state=42,
        verbose=1
    )

    X = pd.concat([X_train, X_val], ignore_index=True)
    y = np.concatenate([y_train, y_val])
    
    if sample_weight is not None:
        val_weights = np.ones(len(y_val))
        combined_sample_weight = np.concatenate([sample_weight, val_weights])
    else:
        combined_sample_weight = None
    
    search.fit(X, y, sample_weight=combined_sample_weight)

    print(f"Best params: {search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-search.best_score_):.3f}")

    return search.best_params_


def fit_model(train_df, val_df, features, target_col, date_col, player_col,
              recent_n=30, recent_weight=3.0, use_gpu=False, 
              tune_hyperparams_flag=False, tune_iters=50):
    """Fit XGBoost regression model with optional hyperparameter tuning."""
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    
    if val_df is None or len(val_df) == 0:
        print("No validation data provided - using training data for early stopping")
        val_df = train_df.copy()
        use_train_for_val = True
    else:
        val_df = val_df.sort_values(date_col).reset_index(drop=True)
        use_train_for_val = False
    
    # Drop rows with NaN in key features
    train_df = train_df.dropna(subset=[target_col] + [f for f in features if f in train_df.columns])
    val_df = val_df.dropna(subset=[target_col] + [f for f in features if f in val_df.columns])

    X_tr = train_df[features]
    y_tr = train_df[target_col].to_numpy()
    X_va = val_df[features]
    y_va = val_df[target_col].to_numpy()

    X_tr = clean_feature_dtypes(X_tr, set())
    X_va = clean_feature_dtypes(X_va, set())
    w_tr = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    
    print(f"Training XGBoost regression model...")
    print(f"Training samples: {len(X_tr)}, Validation samples: {len(X_va)}")
    
    if tune_hyperparams_flag and not use_train_for_val:
        print(f"Tuning hyperparameters...")
        best_params = tune_hyperparams(
            X_tr, y_tr, X_va, y_va,
            sample_weight=w_tr, use_gpu=use_gpu, n_iter=tune_iters
        )
        p = xgb_regression_params(best_params, use_gpu=use_gpu)
    else:
        p = xgb_regression_params(use_gpu=use_gpu)
    
    model = XGBRegressor(**p)
    
    # Use different fitting approach based on validation data availability
    if use_train_for_val:
        # Train without early stopping when using training data as validation
        p_no_early_stop = p.copy()
        p_no_early_stop.pop("early_stopping_rounds", None)
        model = XGBRegressor(**p_no_early_stop)
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        best_iter = p["n_estimators"]
    else:
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else p["n_estimators"]
    
    p_final = xgb_regression_params(use_gpu=use_gpu)
    p_final["n_estimators"] = int(best_iter)
    p_final.pop("early_stopping_rounds", None)
    
    # Final model training on combined data
    if use_train_for_val:
        final_model = XGBRegressor(**p_final)
        final_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    else:
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        X_full = clean_feature_dtypes(full_df[features], set())
        y_full = full_df[target_col].to_numpy()
        
        w_full = np.concatenate([
            build_recent_weights(train_df, player_col, recent_n, recent_weight),
            np.ones(len(val_df))
        ])

        final_model = XGBRegressor(**p_final)
        final_model.fit(X_full, y_full, sample_weight=w_full, verbose=False)
    
    # Validation metrics
    if use_train_for_val:
        val_preds = model.predict(X_tr)
        diff = val_preds - y_tr
    else:
        val_preds = model.predict(X_va)
        diff = val_preds - y_va
    
    val_rmse = float(np.sqrt(np.mean(diff**2)))
    val_mae = float(np.mean(np.abs(diff)))
    val_r2 = float(r2_score(y_va if not use_train_for_val else y_tr, val_preds))
    
    print(f"Validation RMSE: {val_rmse:.3f}")
    print(f"Validation MAE: {val_mae:.3f}")
    print(f"Validation R²: {val_r2:.3f}")
    
    metrics = {
        'RMSE': val_rmse, 
        'MAE': val_mae, 
        'R2': val_r2, 
        'best_iteration': int(best_iter)
    }
    
    return final_model, metrics


def predict(model, test_df, features):
    """Generate predictions from trained model."""
    X_te = clean_feature_dtypes(test_df[features], set())
    preds = model.predict(X_te)
    return preds


def evaluate_model(model, test_df, features, target_col):
    """Evaluate model performance on test set."""
    X_te = clean_feature_dtypes(test_df[features], set())
    y_te = test_df[target_col].to_numpy()
    
    preds = model.predict(X_te)
    
    rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
    mae = float(np.mean(np.abs(preds - y_te)))
    r2 = float(r2_score(y_te, preds))
    
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE: {mae:.3f}")
    print(f"Test R²: {r2:.3f}")
    
    return {
        'RMSE': rmse,
        'MAE': mae, 
        'R2': r2
    }


def correlation_analysis(df, features_list, target_col='PTS'):
    # Filter to only include features that exist in the dataframe
    available_features = [col for col in features_list if col in df.columns]
    
    # Add target column if not already in features
    if target_col not in available_features:
        available_features.append(target_col)
    
    # Create correlation matrix
    corr_matrix = df[available_features].corr()
    
    # Create the plot
    plt.figure(figsize=(15, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    
    print("TOP 10 FEATURES MOST CORRELATED WITH TARGET:")
    print("="*50)
    for i, (feature, corr) in enumerate(target_corrs.head(10).items(), 1):
        print(f"{i:2}. {feature:30} {corr:.3f}")
    
    return corr_matrix, target_corrs.head(10)

def remove_highly_correlated_features(df, features_list, target_col='PTS', threshold=0.95):
    available_features = [col for col in features_list if col in df.columns]
    
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
    print(f"Removed features: {len(features_to_remove)}")
    print(f"Final features: {len(cleaned_features)}")
    
    return cleaned_features

def select_features_ngboost(
    X, y,
    top_n=40,
    n_runs=3,
    validate_both_distributions=True
):
    """
    Feature selection optimized for NGBoost (mean + variance modeling)
    """
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.model_selection import cross_val_score
    
    print("Selecting features for NGBoost (mean + variance)...")
    
    # 1. Standard feature importance (mean-focused)
    mean_importance = np.zeros(X.shape[1])
    
    for seed in range(n_runs):
        ngb = NGBRegressor(
            Dist=Normal,
            n_estimators=100,
            learning_rate=0.05,
            minibatch_frac=0.8,
            random_state=42 + seed,
            verbose=False
        )
        ngb.fit(X, y)
        
        # Handle 2D feature importances (mean + variance) or 1D (mean only)
        feature_imp = ngb.feature_importances_
        if feature_imp.ndim == 2:
            # If 2D, use mean model importances (first row)
            # Shape is typically (2, n_features) for mean and variance models
            mean_importance += feature_imp[0]  # Mean model importances
        else:
            # If 1D, use directly
            mean_importance += feature_imp
    
    mean_importance /= n_runs
    
    # 2. Variance-specific importance
    # Features important for modeling uncertainty
    if validate_both_distributions:
        print("Analyzing variance modeling importance...")
        variance_importance = compute_variance_importance(X, y, n_runs=n_runs)
    else:
        variance_importance = np.zeros(X.shape[1])
    
    # 3. Combine mean and variance importance
    # Weight both aspects (adjust weights based on your use case)
    combined_importance = 0.7 * mean_importance + 0.3 * variance_importance
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_importance': mean_importance,
        'variance_importance': variance_importance,
        'combined_importance': combined_importance
    }).sort_values('combined_importance', ascending=False)
    
    # Select top features
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    print(f"\nSelected {len(top_features)} features")
    print("\nTop 10 features (combined importance):")
    print(importance_df.head(10).to_string(index=False))
    
    # Show features that are variance-heavy
    variance_heavy = importance_df[
        (importance_df['variance_importance'] > importance_df['mean_importance']) & 
        (importance_df['variance_importance'] > 0.01)
    ].head(5)
    
    if len(variance_heavy) > 0:
        print("\n⚠️  Features especially important for uncertainty estimation:")
        for _, row in variance_heavy.iterrows():
            print(f"  - {row['feature']} (var: {row['variance_importance']:.4f}, mean: {row['mean_importance']:.4f})")
    
    return X[top_features], top_features, importance_df


def compute_variance_importance(X, y, n_runs=3):
    """
    Estimate which features are important for variance prediction
    by looking at how well features predict residual magnitudes
    """
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    
    variance_scores = np.zeros(X.shape[1])
    
    for seed in range(n_runs):
        # First, fit NGBoost to get residuals
        ngb = NGBRegressor(
            Dist=Normal,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42 + seed,
            verbose=False
        )
        ngb.fit(X, y)
        
        # Get predictions and residuals
        preds = ngb.predict(X)
        residuals = np.abs(y - preds)  # Absolute residuals as proxy for variance
        
        # Fit XGBoost to predict residual magnitude
        # Features important here help model heteroskedasticity
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42 + seed,
            verbosity=0
        )
        xgb.fit(X, residuals)
        variance_scores += xgb.feature_importances_
    
    return variance_scores / n_runs


def validate_feature_set_ngboost(X, y, features, cv=5):
    """
    Validate feature set using NGBoost-specific metrics:
    - Mean prediction accuracy (RMSE)
    - Calibration of uncertainty estimates (NLL)
    """
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.model_selection import cross_validate
    
    print(f"\nValidating {len(features)} features with {cv}-fold CV...")
    
    ngb = NGBRegressor(
        Dist=Normal,
        n_estimators=300,
        learning_rate=0.05,
        verbose=False
    )
    
    # Custom scorer for negative log-likelihood (lower is better)
    def ngboost_nll_scorer(estimator, X, y):
        from ngboost.scores import LogScore
        dist = estimator.pred_dist(X)
        return -LogScore(dist).score(y).mean()  # Negative for minimization
    
    scoring = {
        'neg_rmse': 'neg_root_mean_squared_error',
        'nll': ngboost_nll_scorer  # Measures calibration
    }
    
    scores = cross_validate(
        ngb, X[features], y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )
    
    print(f"RMSE: {-scores['test_neg_rmse'].mean():.4f} (+/- {scores['test_neg_rmse'].std():.4f})")
    print(f"NLL:  {scores['test_nll'].mean():.4f} (+/- {scores['test_nll'].std():.4f})")
    print("  ↳ NLL measures uncertainty calibration (lower = better)")
    
    return scores

def get_residuals(model, test_df, features, target_col='PTS'):
    """Get residuals analysis for regression model."""
    X_test = test_df[features].copy()
    y_test = test_df[target_col].values
    
    # Get predictions
    preds = model.predict(X_test)
    
    # Create residuals DataFrame
    residuals_df = pd.DataFrame({
        'actual': y_test,
        'predicted': preds,
        'residual': y_test - preds,
        'abs_residual': np.abs(y_test - preds),
    })
    
    # Add context columns
    context_cols = ["PLAYER_NAME", "MATCHUP", "TEAM_PTS", "OPP_PTS", "STARTING", "MIN", "GUARD", "FORWARD", "CENTER"]
    available_cols = [c for c in context_cols if c in test_df.columns]
    residuals_df = residuals_df.join(test_df[available_cols])
    
    return residuals_df

def analyze_residuals(model, test_df, features, target_col='PTS'):
    """Analyze residuals for regression model and show worst predictions."""
    
    # Get residuals
    residuals_df = get_residuals(model, test_df, features, target_col)
    
    print("RESIDUALS ANALYSIS")
    print("=" * 60)
    
    # Overall residuals statistics
    print(f"Residuals - Mean: {residuals_df['residual'].mean():.3f}, Std: {residuals_df['residual'].std():.3f}")
    print(f"Absolute Residuals - Mean: {residuals_df['abs_residual'].mean():.3f}")
    
    # Worst predictions (largest residuals)
    print(f"\nWORST PREDICTIONS:")
    print("-" * 50)
    
    worst_predictions = residuals_df.copy()
    worst_predictions = worst_predictions.sort_values('abs_residual', ascending=False)
    
    # Show top 20 worst predictions
    top_errors = worst_predictions.head(20)[['PLAYER_NAME', 'MATCHUP', 'actual', 'predicted', 'residual', 'MIN']]
    print(top_errors.to_string(index=False))
    
    return residuals_df

def analyze_shap(model, test_df, features, target_col='PTS'):
    """Generate SHAP analysis for model interpretability."""
    X_test = test_df[features].copy()
    
    print(f"\nSHAP ANALYSIS:")
    print("-" * 30)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Overall top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Features by SHAP importance:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)
    plt.title(f'SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return model