from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import os

def train_random_forest(X, y, n_iter=50, cv=3):
    X.sort_values(by='GAME_DATE', ascending=False, inplace=True)
    weights = np.linspace(1,3,num=len(X))**2

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 0.5],
    }

    model = RandomForestRegressor(random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Train-test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    # Fit with weights
    search.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    print(f"\n✅ Best Parameters: {search.best_params_}")
    print(f"📈 R² Score: {r2_score(y_test, preds):.3f}")
    print(f"📉 MAE: {mean_absolute_error(y_test, preds):.3f}")
    print(f"📉 MSE: {mean_squared_error(y_test, preds):.3f}")

    return best_model

def save_random_forest_model(model, stat_line):
    joblib.dump(model, f'Models/{stat_line}_rf_model.pkl')
    print(f"Model saved to {stat_line}_rf_model.pkl")