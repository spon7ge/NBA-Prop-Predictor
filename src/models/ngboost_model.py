import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
import shap
from scipy.stats import lognorm

def build_recent_weights(df, player_col, recent_n=15, recent_weight=5.0):
    w = np.ones(len(df), dtype=float)
    df_reset = df.reset_index(drop=True)
    idx = df_reset.groupby(player_col, sort=False).tail(recent_n).index
    w[idx] = recent_weight
    return w

def train_ngboost_model(train_df: pd.DataFrame,
                       val_df: pd.DataFrame | None,
                       features: list[str],
                       target_col: str,
                       player_col: str = 'PLAYER_ID',
                       recent_n: int = 20,
                       recent_weight: float = 3.0,
                       learning_rate: float = 0.02,
                       n_estimators: int = 500):
    
    train_df = train_df.dropna(subset=[target_col]).copy()
    if val_df is not None and len(val_df) > 0:
        val_df = val_df.dropna(subset=[target_col]).copy()
    
    if 'GAME_DATE' in train_df.columns:
        train_df = train_df.sort_values('GAME_DATE').reset_index(drop=True)
    if val_df is not None and len(val_df) > 0 and 'GAME_DATE' in val_df.columns:
        val_df = val_df.sort_values('GAME_DATE').reset_index(drop=True)
    
    X_tr = train_df[features].copy()
    y_tr = train_df[target_col].to_numpy()
    
    X_va = None
    y_va = None
    if val_df is not None and len(val_df) > 0:
        X_va = val_df[features].copy()
        y_va = val_df[target_col].to_numpy()
    
    y_tr = np.maximum(y_tr, 0)
    if y_va is not None:
        y_va = np.maximum(y_va, 0)
    
    y_tr_log = np.log1p(y_tr)
    if np.any(~np.isfinite(y_tr_log)):
        raise ValueError(f"Training: After log1p, found {np.sum(~np.isfinite(y_tr_log))} non-finite values")
    
    if y_va is not None:
        y_va_log = np.log1p(y_va)
        if np.any(~np.isfinite(y_va_log)):
            raise ValueError(f"Validation: After log1p, found {np.sum(~np.isfinite(y_va_log))} non-finite values")
    
    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(X_tr.median())
    if X_va is not None:
        X_va = X_va.replace([np.inf, -np.inf], np.nan).fillna(X_tr.median())
    
    w_recent = build_recent_weights(train_df, player_col, recent_n, recent_weight)
    w_recent_val = build_recent_weights(val_df, player_col, recent_n, recent_weight) if X_va is not None else None
    
    base_est = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10)
    early_stopping_params = {'early_stopping_rounds': 20} if val_df is not None and len(val_df) > 0 else {}
    
    ngb = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        natural_gradient=True,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbose=False,
        **early_stopping_params,
        Base=base_est
    )
    
    if X_va is not None:
        ngb.fit(X_tr, y_tr_log, X_val=X_va, Y_val=y_va_log, 
                sample_weight=w_recent, val_sample_weight=w_recent_val)
    else:
        ngb.fit(X_tr, y_tr_log, sample_weight=w_recent)
    
    return ngb

def predict_mean(model: NGBRegressor, df: pd.DataFrame, features: list[str], 
                return_type: str = 'mean', variance_calibration: dict = None):
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
    dist = model.pred_dist(X)
    
    mean = dist.loc
    scale = dist.scale
    var = scale ** 2
    
    if variance_calibration:
        median_pred = np.expm1(mean)
        
        low_mask = median_pred < 7
        medium_mask = (median_pred >= 7) & (median_pred < 15)
        high_mask = median_pred >= 15

        var[low_mask] *= variance_calibration.get('low', 2.5)
        var[medium_mask] *= variance_calibration.get('medium', 1.8)
        var[high_mask] *= variance_calibration.get('high', 1.5)
    
    if return_type == 'median':
        return np.expm1(mean)
    
    return np.exp(mean + var / 2) - 1

def get_distribution_params(model: NGBRegressor, df: pd.DataFrame, features: list[str], 
                           variance_calibration: dict = None):
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
    dist = model.pred_dist(X)
    
    mean_log = dist.loc
    scale_log = dist.scale
    var_log = scale_log ** 2
    
    if variance_calibration:
        median_pred = np.expm1(mean_log)
        low_mask = median_pred < 7
        medium_mask = (median_pred >= 7) & (median_pred < 15)
        high_mask = median_pred >= 15
        
        var_log[low_mask] *= variance_calibration.get('low', 2.5)
        var_log[medium_mask] *= variance_calibration.get('medium', 1.8)
        var_log[high_mask] *= variance_calibration.get('high', 1.5)
        
        scale_log = np.sqrt(var_log)
    
    return mean_log, scale_log, var_log

def estimate_residual_variance(model: NGBRegressor, 
                              df: pd.DataFrame, 
                              features: list[str], 
                              target_col: str) -> float:
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
    y_actual = df[target_col].to_numpy()
    y_actual = np.maximum(y_actual, 0)
    
    y_actual_log = np.log1p(y_actual)
    dist = model.pred_dist(X)
    pred_log = dist.loc
    
    residual = y_actual_log - pred_log
    var_log = np.var(residual)
    return var_log

def sample_lognormal_scipy(pred_log: np.ndarray, var_log: float, n_samples: int = 1000) -> np.ndarray:
    std_log = np.sqrt(var_log)
    samples = np.array([lognorm.rvs(s=std_log, scale=np.exp(pred), size=n_samples) 
                        for pred in pred_log])
    return samples - 1

def remove_highly_correlated_features(df, features_list, target_col='PTS', threshold=0.95):
    available_features = [col for col in features_list if col in df.columns]
    missing_features = [col for col in features_list if col not in df.columns]
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found")
    
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
        else:
            features_to_remove.add(feat1)
    
    cleaned_features = [f for f in available_features if f not in features_to_remove and f != target_col]
    print(f"Removed {len(features_to_remove)} highly correlated features. Final: {len(cleaned_features)}")
    
    return cleaned_features

def get_ngboost_feature_importance(model, features):
    importances = np.zeros(len(features))
    
    for estimator in model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            importances += estimator.feature_importances_
    
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

def analyze_ngboost_shap(model, test_df, features, variance_model=None, sample_size=1000):
    if len(test_df) > sample_size:
        test_sample = test_df.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_df.copy()
    
    X_test = test_sample[features].fillna(0)
    
    print("SHAP Analysis - Main Model:")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print("Top 20 Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:8.4f}")
    
    print("Bottom 10 Features:")
    for idx, row in importance_df.tail(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:8.4f}")
    
    return importance_df, shap_values
