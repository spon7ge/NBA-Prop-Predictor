from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns


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

def xgb_params(base=None, use_gpu=False):
    p = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
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
        verbosity=1
    )
    if base:
        p.update(base)
    if use_gpu:
        p["tree_method"] = "gpu_hist"
        p["gpu_id"] = 0
        p["n_jobs"] = 1
    return p

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

def tune_xgb_hyperparams(X_train, y_train, X_val, y_val, sample_weight=None, use_gpu=False, n_iter=50):
    base_params = xgb_params(use_gpu=use_gpu)
    base_params.pop("early_stopping_rounds", None)

    xgb = XGBRegressor(**base_params)

    param_dist = {
        "max_depth": randint(3, 10),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.5, 0.5),       # 0.5–1.0
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

    print("Best params:", search.best_params_)
    print("Best R²:", search.best_score_)

    return search.best_params_

def fit_train_val(
    train_df, val_df,
    features, target_col, date_col, player_col,
    recent_n=30, recent_weight=3.0,
    params=None, use_gpu=False, tune_hyperparams=False, tune_iters=50):

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
    
    if tune_hyperparams:
        best_params = tune_xgb_hyperparams(
        X_tr, y_tr, X_va, y_va, sample_weight=w_tr,
        use_gpu=use_gpu, n_iter=tune_iters
    )
        p = xgb_params(best_params, use_gpu=use_gpu)
    else:
        p = xgb_params(params, use_gpu=use_gpu)
    
    X_tr = clean_feature_dtypes(X_tr, set())
    X_va = clean_feature_dtypes(X_va, set())

    w_tr = build_recent_weights(train_df, player_col, recent_n, recent_weight)

    p = xgb_params(params, use_gpu=use_gpu)
    
    model = XGBRegressor(**p)
    
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else p["n_estimators"]

    p_final = xgb_params(params, use_gpu=use_gpu)
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

    val_preds = model.predict(X_va)
    diff = val_preds - y_va
    val_rmse = float(np.sqrt(np.mean(diff**2)))
    val_mae = float(np.mean(np.abs(diff)))
    val_r2 = float(r2_score(y_va, val_preds))

    return final_model, dict(RMSE=val_rmse, MAE=val_mae, R2=val_r2, best_iteration=int(best_iter)), []

def fit_quantile_models(train_df, val_df, features, target_col, date_col, player_col,
                       recent_n=30, recent_weight=3.0, use_gpu=False):  
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
        
        p = xgb_quantile_params(q, use_gpu=use_gpu)
        
        model = XGBRegressor(**p)
        
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else p["n_estimators"]
        
        p_final = xgb_quantile_params(q, use_gpu=use_gpu)
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

def evaluate_test(model, test_df, features, target_col, cat_cols):
    X_te = clean_feature_dtypes(test_df[features], set())
    y_te = test_df[target_col].to_numpy()
    
    preds = model.predict(X_te)
    diff = preds - y_te

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    r2 = float(r2_score(y_te, preds))

    residuals_df = pd.DataFrame({
        "actual": y_te,
        "predicted": preds,
        "residual": diff
    })
    
    # Join back some useful context for grouping
    context_cols = ["PLAYER_NAME", "MATCHUP", "TEAM_PTS", "OPP_PTS", "STARTING",'MIN']
    available_cols = [c for c in context_cols if c in test_df.columns]
    residuals_df = residuals_df.join(test_df[available_cols])
    
    return dict(RMSE=rmse, MAE=mae, R2=r2), residuals_df


def simple_correlation_analysis(df, features_list, target_col='PTS'):
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

            'HOME_GAME', 'BACK_TO_BACK', 'PLAYER_DAYS_REST', 
            'spread', 'TEAM_IMPLIED_PTS',
            'GUARD', 'FORWARD', 'CENTER',
            'TEAM_PACE_AVG_TO_DATE', 'TEAM_OFF_RATING_AVG_TO_DATE',
            'OPP_DEF_RATING_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE', 'OPP_TOV_AVG_TO_DATE',
            'OPP_FORWARD_DEF_RATING', 'OPP_CENTER_DEF_RATING', 'OPP_GUARD_DEF_RATING',
            'MIN_ROLLING_AVG_5', 'MIN_ROLLING_AVG_10', 'MIN_LAG_1', 
            'MIN_AVG_TO_DATE', 'MIN_VOLATILITY_10_TO_DATE', 'MIN_VOLATILITY_5_TO_DATE'
            'MIN_STD_LAST_5'
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