from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
import shap
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_xgb_model(X, y, stat_line='PTS', val_fraction=0.2, playoff_weight=0.5):
    """
    Train XGBoost model with randomized search CV, time-based weights, 
    and down-weighted playoff games for a regular-season focus.
    
    Parameters:
    - X: Features DataFrame
    - y: Target Series
    - stat_line: Statistic being predicted (e.g., 'PTS', 'AST')
    - val_fraction: Fraction of data to reserve for final validation
    - playoff_weight: Multiplier for playoff games (e.g., 0.5 → playoff games get half weight)
    """
    # Split train/test
    split_idx = int(len(X) * (1 - val_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Base time-based weights
    train_weights = np.linspace(1, 3, num=len(X_train)) ** 2
    test_weights  = np.linspace(1, 3, num=len(X_test))  ** 2

    # Down-weight playoff games if present
    if 'IS_PLAYOFF' in X_train.columns:
        mask_train = X_train['IS_PLAYOFF'].astype(bool)
        mask_test  = X_test['IS_PLAYOFF'].astype(bool)
        train_weights[mask_train] *= playoff_weight
        test_weights[mask_test]   *= playoff_weight

    # Hyperparameter space
    param_grid = {
        'learning_rate': [0.01, 0.02, 0.05],
        'max_depth': [3, 4, 5],
        'n_estimators': [200, 300, 500],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'gamma': [0.1, 0.2, 0.5],
        'min_child_weight': [3, 5, 10],
        'reg_alpha': [0.1, 0.5, 1],
        'reg_lambda': [3, 5, 10]
    }

    base_model = XGBRegressor(objective='reg:squarederror', random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train, sample_weight=train_weights)
    best_model = search.best_estimator_

    # Evaluate on test set
    pred = best_model.predict(X_test)
    print(f"\nModel Performance Metrics for {stat_line}:")
    print(f"R2 Score: {r2_score(y_test, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
    print(f"\nBest Parameters: {search.best_params_}")

    saveXGBModel(best_model, stat_line)
    return best_model





def saveXGBModel(model, stat_line):
    models_dir = 'Models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, f'{stat_line}_xgb_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def loadXGBModel(stat_line):
    models_dir = 'Models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, 'Models', f'{stat_line}_xgb_model.pkl')
    model = joblib.load(model_path)
    return model
    
def getTopFeatures(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    features = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    })
    features = features.sort_values('importance', ascending=False)
    return features
