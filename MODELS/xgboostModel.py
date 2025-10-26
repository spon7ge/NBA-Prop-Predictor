from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
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


def xgb_quantile_params(quantile, base=None, use_gpu=False):
    p = dict(
        objective="reg:quantileerror",  
        eval_metric="quantile",
        tree_method='hist',
        booster='gbtree',
        random_state=42,
        n_jobs=-1,
        
        n_estimators=5000,
        learning_rate=0.01412139968434072, 
        max_depth=9,
        min_child_weight=3, 
        
        subsample=0.73338144662399,
        colsample_bytree=0.6460723242676091,
        
        gamma=0.7327236865873834,
        reg_alpha=3.4421579214044646,
        reg_lambda=50.16154429033941,
        
        early_stopping_rounds=75, 
        verbosity=1,
        
        quantile_alpha=quantile,  
    )
    if base:
        p.update(base)
    if use_gpu:
        p["tree_method"] = "gpu_hist"
        p["gpu_id"] = 0
        p["n_jobs"] = 1
    return p

def adaptive_quantile_alpha(quantile):
    """
    Adaptive quantile alpha that adjusts based on the quantile value.
    More extreme quantiles (closer to 0 or 1) get slightly adjusted alphas.
    """
    if quantile <= 0.1:
        return max(0.05, quantile - 0.02)  # Slightly lower for very low quantiles
    elif quantile >= 0.9:
        return min(0.95, quantile + 0.02)  # Slightly higher for very high quantiles
    else:
        return quantile  # Keep original for median quantiles

def xgb_quantile_params_improved(quantile, base=None, use_gpu=False):
    base_params = dict(
        objective="reg:quantileerror",  
        eval_metric="quantile",
        tree_method='hist',
        booster='gbtree',
        random_state=42,
        n_jobs=-1,
        n_estimators=5000,
        early_stopping_rounds=75, 
        verbosity=1,
        quantile_alpha=adaptive_quantile_alpha(quantile),
    )
    
    if quantile in [0.1, 0.9]:
        params = dict(
            learning_rate=0.01,  # Lower learning rate for stability
            max_depth=6,  # Shallower trees to prevent overfitting
            min_child_weight=2,  # More conservative splitting
            subsample=0.7,  # Less data to prevent overfitting
            colsample_bytree=0.7,  # Fewer features
            gamma=1.0,  # More pruning
            reg_alpha=1.0,  # More L1 regularization
            reg_lambda=10.0,  # More L2 regularization
            max_delta_step=0,  # Smaller steps
        )
    else:
        # Keep conservative settings for median
        params = dict(
            learning_rate=0.01412139968434072,
            max_depth=9,
            min_child_weight=3,
            subsample=0.73338144662399,
            colsample_bytree=0.6460723242676091,
            gamma=0.7327236865873834,
            reg_alpha=3.4421579214044646,
            reg_lambda=50.16154429033941,
        )
    
    p = {**base_params, **params}
    if base:
        p.update(base)
    if use_gpu:
        p["tree_method"] = "hist"
        p["device"] = "cuda"
        p["n_jobs"] = 1
    return p


def tune_quantile_hyperparams(X_train, y_train, X_val, y_val, quantile, 
                              sample_weight=None, use_gpu=False, n_iter=50):
    """
    Tune hyperparameters for a specific quantile model.
    """
    base_params = xgb_quantile_params_improved(quantile, use_gpu=use_gpu)
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
    
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,  
        cv=[(range(len(y_train)), range(len(y_val)))],
        verbose=1,
        random_state=42
    )

    X = pd.concat([X_train, X_val], ignore_index=True)
    y = np.concatenate([y_train, y_val])
    
    if sample_weight is not None:
        val_weights = np.ones(len(y_val))
        combined_sample_weight = np.concatenate([sample_weight, val_weights])
    else:
        combined_sample_weight = None
    
    search.fit(X, y, sample_weight=combined_sample_weight)

    print(f"Best params for q{int(quantile*100)}: {search.best_params_}")
    print(f"Best R² for q{int(quantile*100)}: {search.best_score_}")

    return search.best_params_

def fit_quantile_models(train_df, val_df, features, target_col, date_col, player_col,
                       recent_n=30, recent_weight=3.0, use_gpu=False, 
                       tune_hyperparams=False, tune_iters=50):  
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    val_df = val_df.sort_values(date_col).reset_index(drop=True)
    
    rolling_avg_cols = ['PTS_ROLLING_AVG_40', 'PTS_ROLLING_AVG_15', 'PTS_ROLLING_AVG_10']
    train_df = train_df.dropna(subset=rolling_avg_cols)
    val_df = val_df.dropna(subset=rolling_avg_cols)

    X_tr = train_df[features]
    y_tr = train_df[target_col].to_numpy()
    X_va = val_df[features]
    y_va = val_df[target_col].to_numpy()

    X_tr = clean_feature_dtypes(X_tr, set())
    X_va = clean_feature_dtypes(X_va, set())
    w_tr = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    
    quantiles = [0.1, 0.5, 0.9]  # q10, q50, q90
    models = {}
    val_metrics = {}
    
    for q in quantiles:
        print(f"\nTraining quantile model for q{int(q*100)}...")
        
        if tune_hyperparams:
            print(f"Tuning hyperparameters for q{int(q*100)}...")
            best_params = tune_quantile_hyperparams(
                X_tr, y_tr, X_va, y_va, quantile=q,
                sample_weight=w_tr, use_gpu=use_gpu, n_iter=tune_iters
            )
            p = xgb_quantile_params_improved(q, best_params, use_gpu=use_gpu)
        else:
            p = xgb_quantile_params_improved(q, use_gpu=use_gpu)
        
        model = XGBRegressor(**p)
        
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else p["n_estimators"]
        
        if tune_hyperparams:
            p_final = xgb_quantile_params_improved(q, best_params, use_gpu=use_gpu)
        else:
            p_final = xgb_quantile_params_improved(q, use_gpu=use_gpu)
        p_final["n_estimators"] = int(best_iter)
        p_final.pop("early_stopping_rounds", None)
        
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
        val_preds = model.predict(X_va)
        diff = val_preds - y_va
        val_rmse = float(np.sqrt(np.mean(diff**2)))
        val_mae = float(np.mean(np.abs(diff)))
        val_r2 = float(r2_score(y_va, val_preds))
        
        models[f'q{int(q*100)}'] = final_model
        val_metrics[f'q{int(q*100)}'] = {
            'RMSE': val_rmse, 
            'MAE': val_mae, 
            'R2': val_r2, 
            'best_iteration': int(best_iter)
        }
        
        print(f"q{int(q*100)} validation RMSE: {val_rmse:.3f}")
    
    return models, val_metrics

def predict_quantiles(models, test_df, features):
    X_te = clean_feature_dtypes(test_df[features], set())
    
    predictions = {}
    for quantile_name, model in models.items():
        preds = model.predict(X_te)
        predictions[quantile_name] = preds
    
    # Create DataFrame with all predictions
    pred_df = pd.DataFrame(predictions)
    pred_df['q10'] = predictions['q10']
    pred_df['q50'] = predictions['q50']  # Median prediction
    pred_df['q90'] = predictions['q90']
    
    # Calculate prediction intervals
    pred_df['lower_bound'] = pred_df['q10']
    pred_df['upper_bound'] = pred_df['q90']
    pred_df['prediction_interval_width'] = pred_df['q90'] - pred_df['q10']
    
    return pred_df

def evaluate_quantile_models(models, test_df, features, target_col):
    X_te = clean_feature_dtypes(test_df[features], set())
    y_te = test_df[target_col].to_numpy()
    
    results = {}
    
    for quantile_name, model in models.items():
        preds = model.predict(X_te)
        diff = preds - y_te
        
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        r2 = float(r2_score(y_te, preds))
        
        results[quantile_name] = {
            'RMSE': rmse,
            'MAE': mae, 
            'R2': r2
        }
        
        print(f"{quantile_name} Test RMSE: {rmse:.3f}")
    
    return results



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
    
    plt.title('Feature Correlation Matrix - Point Prediction', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    
    print("TOP 10 FEATURES MOST CORRELATED WITH PTS:")
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

def select_features_xgb_importance(
    X,
    y,
    top_n=40,
    n_runs=5,
    include_context=True,
    validate_permutation=False
):
    feature_scores = np.zeros(X.shape[1])

    # Run multiple XGBoost fits to stabilize importance
    for seed in range(n_runs):
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + seed,
            verbosity=0
        )
        xgb.fit(X, y)
        feature_scores += xgb.feature_importances_

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_scores / n_runs
    }).sort_values('importance', ascending=False)

    top_features = importance_df.head(top_n)['feature'].tolist()

    # Add context features that improve stability and interpretability
    if include_context:
        context_features = [

        ]
        for f in context_features:
            if f in X.columns and f not in top_features:
                top_features.append(f)

    # Optional: Permutation importance validation
    if validate_permutation:
        print("Validating with permutation importance...")
        model = XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
        model.fit(X[top_features], y)
        perm = permutation_importance(model, X[top_features], y, n_repeats=5, random_state=42)
        perm_df = pd.DataFrame({
            'feature': top_features,
            'perm_importance': perm.importances_mean
        }).sort_values('perm_importance', ascending=False)

        # Filter out weak features
        perm_df = perm_df[perm_df['perm_importance'] > 0]
        top_features = perm_df['feature'].tolist()

    print(f"Selected {len(top_features)} total features")
    print("Top 10 most important:")
    for i, f in enumerate(top_features[:10]):
        print(f"{i+1}. {f}")

    return X[top_features], top_features


def get_quantile_residuals(models, test_df, features, target_col='PTS'):
    """
    Get residuals analysis for quantile models.
    """
    import pandas as pd
    import numpy as np
    
    X_test = test_df[features].copy()
    y_test = test_df[target_col].values
    
    # Get predictions from all quantile models
    preds_q10 = models['q10'].predict(X_test)
    preds_q50 = models['q50'].predict(X_test)
    preds_q90 = models['q90'].predict(X_test)
    
    # Create residuals DataFrame
    residuals_df = pd.DataFrame({
        'actual': y_test,
        'pred_q10': preds_q10,
        'pred_q50': preds_q50,
        'pred_q90': preds_q90,
        'residual_q10': y_test - preds_q10,
        'residual_q50': y_test - preds_q50,
        'residual_q90': y_test - preds_q90,
    })
    
    # Add context columns
    context_cols = ["PLAYER_NAME", "MATCHUP", "TEAM_PTS", "OPP_PTS", "STARTING", "MIN", "GUARD", "FORWARD", "CENTER"]
    available_cols = [c for c in context_cols if c in test_df.columns]
    residuals_df = residuals_df.join(test_df[available_cols])
    
    # Add prediction interval analysis
    residuals_df['interval_width'] = preds_q90 - preds_q10
    residuals_df['within_80_interval'] = (y_test >= preds_q10) & (y_test <= preds_q90)
    residuals_df['within_50_interval'] = (y_test >= preds_q10) & (y_test <= preds_q50)
    
    return residuals_df

def analyze_quantile_residuals(models, test_df, features, target_col='PTS'):
    """
    Analyze residuals for quantile models and show worst predictions.
    """
    
    # Get residuals
    residuals_df = get_quantile_residuals(models, test_df, features, target_col)
    
    print("QUANTILE MODEL RESIDUALS ANALYSIS")
    print("=" * 60)
    
    # Overall residuals statistics
    print(f"Q10 Residuals - Mean: {residuals_df['residual_q10'].mean():.3f}, Std: {residuals_df['residual_q10'].std():.3f}")
    print(f"Q50 Residuals - Mean: {residuals_df['residual_q50'].mean():.3f}, Std: {residuals_df['residual_q50'].std():.3f}")
    print(f"Q90 Residuals - Mean: {residuals_df['residual_q90'].mean():.3f}, Std: {residuals_df['residual_q90'].std():.3f}")
    
    # Coverage analysis
    coverage_80 = residuals_df['within_80_interval'].mean()
    coverage_50 = residuals_df['within_50_interval'].mean()
    print(f"\nCoverage Analysis:")
    print(f"80% Interval Coverage: {coverage_80:.3f} (target: 0.80)")
    print(f"50% Interval Coverage: {coverage_50:.3f} (target: 0.50)")
    
    # Worst predictions (largest residuals)
    print(f"\nWORST PREDICTIONS (Q50 - Median):")
    print("-" * 50)
    
    # Sort by absolute residual for q50 (median)
    worst_predictions = residuals_df.copy()
    worst_predictions['abs_residual_q50'] = np.abs(worst_predictions['residual_q50'])
    worst_predictions = worst_predictions.sort_values('abs_residual_q50', ascending=False)
    
    # Show top 20 worst predictions
    top_errors = worst_predictions.head(20)[['PLAYER_NAME', 'MATCHUP', 'actual', 'pred_q50', 'residual_q50', 'MIN', 'GUARD', 'FORWARD', 'CENTER']]
    print(top_errors.to_string(index=False))
    
    return residuals_df

def analyze_quantile_shap(models, test_df, features, target_col='PTS'):
    X_test = test_df[features].copy()
    
    # Define position columns
    position_cols = ['GUARD', 'FORWARD', 'CENTER']
    quantiles = ['q10', 'q50', 'q90']
    for quantile in quantiles:
        if quantile not in models:
            continue
            
        print(f"\n{quantile.upper()} MODEL:")
        print("-" * 30)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(models[quantile])
        shap_values = explainer.shap_values(X_test)
        
        # Overall top features
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Features:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']:30} {row['importance']:.4f}")
        
        # Position-specific analysis
        print(f"\nBy Position:")
        for pos_col in position_cols:
            if pos_col in test_df.columns:
                pos_mask = test_df[pos_col] == 1
                if pos_mask.sum() > 0:
                    pos_shap = shap_values[pos_mask]
                    pos_mean_shap = np.abs(pos_shap).mean(axis=0)
                    
                    pos_importance = pd.DataFrame({
                        'feature': features,
                        'importance': pos_mean_shap
                    }).sort_values('importance', ascending=False)
                    
                    print(f"  {pos_col} (n={pos_mask.sum()}): {pos_importance.iloc[0]['feature']} ({pos_importance.iloc[0]['importance']:.3f})")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, max_display=15, show=False)
        plt.title(f'SHAP Summary - {quantile.upper()} Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return models


def validate_out_of_time_pinball_loss(models, test_df, features, target_col='PTS'):
    X_test = test_df[features].copy()
    y_test = test_df[target_col].values
    
    def pinball_loss(y_true, y_pred, quantile):
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    print("OUT-OF-TIME PINBALL LOSS VALIDATION")
    print("=" * 60)
    print(f"Test set size: {len(y_test)} samples")
    print(f"Date range: {test_df['GAME_DATE'].min()} to {test_df['GAME_DATE'].max()}")
    
    results = {}
    quantiles = [0.1, 0.5, 0.9]  # q10, q50, q90
    
    for q in quantiles:
        quantile_name = f'q{int(q*100)}'
        if quantile_name not in models:
            continue
            
        # Get predictions
        preds = models[quantile_name].predict(X_test)
        
        # Calculate pinball loss
        pinball = pinball_loss(y_test, preds, q)
        
        # Calculate other metrics for comparison
        mae = np.mean(np.abs(y_test - preds))
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        
        # Calculate coverage (for quantile validation)
        if q == 0.1:
            coverage = np.mean(y_test >= preds)
        elif q == 0.9:
            coverage = np.mean(y_test <= preds)
        else:  # q50
            coverage = None
        
        results[quantile_name] = {
            'pinball_loss': pinball,
            'mae': mae,
            'rmse': rmse,
            'coverage': coverage,
            'quantile': q
        }
        
        print(f"\n{quantile_name.upper()} Model:")
        print(f"  Pinball Loss: {pinball:.4f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        if coverage is not None:
            print(f"  Coverage: {coverage:.3f} (target: {q:.1f})")
    
    # Calculate average pinball loss
    avg_pinball = np.mean([results[f'q{int(q*100)}']['pinball_loss'] for q in quantiles])
    print(f"\nAverage Pinball Loss: {avg_pinball:.4f}")
    
    # Model quality assessment
    print(f"\nMODEL QUALITY ASSESSMENT:")
    print("-" * 40)
    if avg_pinball < 2.0:
        print("EXCELLENT - Very reliable uncertainty estimates!")
    elif avg_pinball < 3.0:
        print("GOOD - Solid uncertainty estimates!")
    elif avg_pinball < 4.0:
        print("FAIR - Acceptable uncertainty estimates!")
    else:
        print("POOR - Uncertainty estimates need improvement!")
    
    # Additional validation metrics
    print(f"\nADDITIONAL VALIDATION METRICS:")
    print("-" * 40)
    
    # Check if q10 < q50 < q90 (monotonicity)
    q10_preds = models['q10'].predict(X_test)
    q50_preds = models['q50'].predict(X_test)
    q90_preds = models['q90'].predict(X_test)
    
    monotonic_q10_q50 = np.mean(q10_preds <= q50_preds)
    monotonic_q50_q90 = np.mean(q50_preds <= q90_preds)
    
    print(f"Monotonicity q10 ≤ q50: {monotonic_q10_q50:.3f} (should be ~1.0)")
    print(f"Monotonicity q50 ≤ q90: {monotonic_q50_q90:.3f} (should be ~1.0)")
    
    # Prediction interval width analysis
    interval_width = q90_preds - q10_preds
    print(f"Average 80% interval width: {interval_width.mean():.2f}")
    print(f"Median 80% interval width: {np.median(interval_width):.2f}")
    
    return results

def validate_by_position_pinball_loss(models, test_df, features, target_col='PTS'):
    """
    Validate pinball loss by position to see if models work well for all positions.
    """
    import numpy as np
    
    X_test = test_df[features].copy()
    y_test = test_df[target_col].values
    
    def pinball_loss(y_true, y_pred, quantile):
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    position_cols = ['GUARD', 'FORWARD', 'CENTER']
    quantiles = [0.1, 0.5, 0.9]
    
    print("PINBALL LOSS BY POSITION")
    print("=" * 50)
    
    for pos_col in position_cols:
        if pos_col in test_df.columns:
            pos_mask = test_df[pos_col] == 1
            if pos_mask.sum() > 10:  # Only analyze if enough samples
                print(f"\n{pos_col} (n={pos_mask.sum()}):")
                
                pos_X = X_test[pos_mask]
                pos_y = y_test[pos_mask]
                
                for q in quantiles:
                    quantile_name = f'q{int(q*100)}'
                    if quantile_name in models:
                        preds = models[quantile_name].predict(pos_X)
                        pinball = pinball_loss(pos_y, preds, q)
                        print(f"  {quantile_name}: {pinball:.4f}")