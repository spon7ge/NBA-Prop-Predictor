from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
import shap
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV



def train_xgb_model(data,feature_cols,target_col='PTS',n_splits=5,n_iter=60,random_state=42):
    df = data.sort_values('GAME_DATE').reset_index(drop=True)
    # features and target
    X = df[feature_cols].select_dtypes(include=[np.number]).astype(np.float32).values
    y = df[target_col].astype(np.float32).values

    # time series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # model
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
        random_state=random_state
    )

    # randomized search grid
    param_distributions = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3, 0.5, 1.0],
        "reg_alpha": [0, 0.01, 0.05, 0.1, 1, 5, 10],
        "reg_lambda": [0.1, 0.5, 1, 5, 10, 20],
        "min_child_weight": [1, 3, 5, 7, 10]
    }

    # search
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    search.fit(X, y)

    print("Best params:")
    print(search.best_params_)
    print("CV best RMSE:")
    print(np.sqrt(-search.best_score_))

    best_model = search.best_estimator_

    # evaluate best model across folds
    r2_list, mae_list, rmse_list = [], [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)

        print(f"Fold {fold} R2: {r2:.4f}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")

    print("Mean CV R2:", np.mean(r2_list))
    print("Mean CV MAE:", np.mean(mae_list))
    print("Mean CV RMSE:", np.mean(rmse_list))

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
