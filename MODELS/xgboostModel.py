from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


def build_recent_weights(df, player_col, recent_n=15, recent_weight=5.0):
    w = np.ones(len(df), dtype=float)
    df_reset = df.reset_index(drop=True)
    idx = df_reset.groupby(player_col, sort=False).tail(recent_n).index
    w[idx] = recent_weight
    return w

def clean_feature_dtypes(X, cat_cols):
    Xc = X.copy()
    for c in Xc.columns:
        if c in cat_cols:
            continue
        if Xc[c].dtype == "bool":
            Xc[c] = Xc[c].astype(int)
        elif not np.issubdtype(Xc[c].dtype, np.number):
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce")
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

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="r2",
        n_jobs=-1,
        cv=[(range(len(y_train)), range(len(y_val)))],  # train/val split manually defined
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
    
    # clean types (no categorical handling needed)
    X_tr = clean_feature_dtypes(X_tr, set())
    X_va = clean_feature_dtypes(X_va, set())

    # sample weights
    w_tr = build_recent_weights(train_df, player_col, recent_n, recent_weight)

    # Get parameters
    p = xgb_params(params, use_gpu=use_gpu)
    
    # Create model
    model = XGBRegressor(**p)
    
    # Fit with early stopping - removed early_stopping_rounds from here
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else p["n_estimators"]

    # retrain final model on train+val
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

    # validation metrics
    val_preds = model.predict(X_va)
    diff = val_preds - y_va
    val_rmse = float(np.sqrt(np.mean(diff**2)))
    val_mae = float(np.mean(np.abs(diff)))
    val_r2 = float(r2_score(y_va, val_preds))

    return final_model, dict(RMSE=val_rmse, MAE=val_mae, R2=val_r2, best_iteration=int(best_iter)), []

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