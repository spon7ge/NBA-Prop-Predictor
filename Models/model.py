from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
import shap
import pandas as pd

def train_xgb_model(X, y, stat_line='PTS', val_fraction=0.2):
    n_total = len(X)
    split_idx = int(n_total * (1 - val_fraction))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Time-based weights: more recent training games get higher weight
    train_weights = np.linspace(1, 3, num=len(X_train)) ** 2
    test_weights = np.linspace(1, 3, num=len(X_test)) ** 2

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

    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
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
