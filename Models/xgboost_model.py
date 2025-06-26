from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
import shap
import pandas as pd

def train_xgb_model(X, y,stat_line='PTS'):
    #sort by date to give more weight to recent games
    weights = np.linspace(1,3,num=len(X))**2

    param_grid = {
        'learning_rate': [0.01, 0.02, 0.05],       # Lower is better for stability
        'max_depth': [3, 4, 5],                    # Avoid overfitting on training stats
        'n_estimators': [200, 300, 500],           # More trees with lower learning rate
        'subsample': [0.6, 0.7, 0.8],              # Add randomness
        'colsample_bytree': [0.6, 0.7, 0.8],       # Avoid dependency on specific features
        'gamma': [0.1, 0.2, 0.5],                  # Avoid unnecessary splits
        'min_child_weight': [3, 5, 10],            # Avoid learning from outliers
        'reg_alpha': [0.1, 0.5, 1],                # L1 – sparsity helps with irrelevant stats
        'reg_lambda': [3, 5, 10]                   # L2 – strong penalty helps generalize
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

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)
    search.fit(X_train, y_train, sample_weight=w_train)

    best_model = search.best_estimator_
    pred = best_model.predict(X_test)
    print(f"\nModel Performance Metrics for {stat_line}:")
    print(f"R2 Score: {r2_score(y_test, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
    print(f"\nBest Parameters: {search.best_params_}")

    save_xgb_model(best_model, stat_line)
    return best_model

def save_xgb_model(model, stat_line):
    joblib.dump(model, f'Models/{stat_line}_xgb_model.pkl')
    print(f"Model saved to {stat_line}_xgb_model.pkl")

def load_xgb_model(stat_line):
    model = joblib.load(f'Models/{stat_line}_xgb_model.pkl')
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
