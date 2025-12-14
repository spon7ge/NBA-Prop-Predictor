import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

def create_recent_game_weights(df, player_column, recent_games=15, weight=3.0):
    weights = np.ones(len(df))
    df_with_index = df.reset_index(drop=True)
    recent_game_indices = df_with_index.groupby(player_column, sort=False).tail(recent_games).index
    weights[recent_game_indices] = weight
    return weights

def ensure_features_list(features):
    if isinstance(features, list):
        return features
    elif hasattr(features, '__iter__') and not isinstance(features, str):
        return list(features)
    else:
        return [features]

def prepare_features(X):
    X_clean = X.copy()
    for column in X_clean.columns:
        if X_clean[column].dtype == bool:
            X_clean[column] = X_clean[column].astype(int)
        elif not np.issubdtype(X_clean[column].dtype, np.number):
            X_clean[column] = pd.to_numeric(X_clean[column], errors='coerce')
    return X_clean

def train_model(train_data, val_data, features, target, date_column, player_column):
    print("Training XGBoost Model")
    
    train_data = train_data.sort_values(date_column).reset_index(drop=True)
    val_data = val_data.sort_values(date_column).reset_index(drop=True)
    
    train_data = train_data.dropna(subset=[target])
    val_data = val_data.dropna(subset=[target])
    
    X_train = train_data[features].copy()
    X_val = val_data[features].copy()
    y_train = train_data[target].values
    y_val = val_data[target].values
    
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    weights = np.ones(len(train_data))
    df_reset = train_data.reset_index(drop=True)
    recent_idx = df_reset.groupby(player_column, sort=False).tail(15).index
    weights[recent_idx] = 3.0
    
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=4,
        gamma=0.8,
        reg_alpha=1.5,
        reg_lambda=8.0,
        subsample=0.75,
        colsample_bytree=0.75,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=75,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 1500
    
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    
    print(f"Validation - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    
    combined = pd.concat([train_data, val_data], ignore_index=True)
    X_combined = combined[features].fillna(0)
    y_combined = combined[target].values
    
    train_weights = np.ones(len(train_data))
    train_reset = train_data.reset_index(drop=True)
    train_recent = train_reset.groupby(player_column, sort=False).tail(15).index
    train_weights[train_recent] = 3.0
    
    val_weights = np.ones(len(val_data))
    combined_weights = np.concatenate([train_weights, val_weights])
    
    final_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=int(best_iter),
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=4,
        gamma=0.8,
        reg_alpha=1.5,
        reg_lambda=8.0,
        subsample=0.75,
        colsample_bytree=0.75,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    
    final_model.fit(X_combined, y_combined, sample_weight=combined_weights, verbose=False)
    
    return final_model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'best_iteration': int(best_iter)}

def predict(model, test_data, features):
    X_test = test_data[features].fillna(0)
    return model.predict(X_test)

def evaluate_predictions(model, test_data, features, target):
    X_test = prepare_features(test_data[features])
    y_test = test_data[target].values
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test Results - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def analyze_prediction_errors(model, test_data, features, target):
    X_test = prepare_features(test_data[features])
    predictions = model.predict(X_test)
    actual = test_data[target].values
    
    results = pd.DataFrame({
        'actual': actual,
        'predicted': predictions,
        'error': actual - predictions,
        'abs_error': np.abs(actual - predictions)
    })
    
    info_columns = ['PLAYER_NAME', 'MATCHUP', 'MIN']
    for col in info_columns:
        if col in test_data.columns:
            results[col] = test_data[col].values
    
    print(f"Error Stats - Mean: {results['error'].mean():.3f}, Std: {results['error'].std():.3f}, MAE: {results['abs_error'].mean():.3f}")
    
    worst = results.nlargest(10, 'abs_error')
    if 'PLAYER_NAME' in worst.columns:
        print(worst[['PLAYER_NAME', 'actual', 'predicted', 'error', 'MIN']].to_string(index=False))
    else:
        print(worst[['actual', 'predicted', 'error']].to_string(index=False))
    
    return results

def get_residuals(model, test_df, features, target_col='PTS'):
    X_test = test_df[features].copy()
    y_test = test_df[target_col].values
    preds = model.predict(X_test)
    
    residuals_df = pd.DataFrame({
        'actual': y_test,
        'predicted': preds,
        'residual': y_test - preds,
        'abs_residual': np.abs(y_test - preds),
    })
    
    context_cols = ["PLAYER_NAME", "MATCHUP", "TEAM_PTS", "OPP_PTS", "STARTING", "MIN", "GUARD", "FORWARD", "CENTER"]
    available_cols = [c for c in context_cols if c in test_df.columns]
    residuals_df = residuals_df.join(test_df[available_cols])
    
    return residuals_df

def analyze_residuals(model, test_df, features, target_col='PTS'):
    residuals_df = get_residuals(model, test_df, features, target_col)
    
    print(f"Residuals - Mean: {residuals_df['residual'].mean():.3f}, Std: {residuals_df['residual'].std():.3f}")
    print(f"Absolute Residuals - Mean: {residuals_df['abs_residual'].mean():.3f}")
    
    worst_predictions = residuals_df.sort_values('abs_residual', ascending=False)
    top_errors = worst_predictions.head(20)[['PLAYER_NAME', 'MATCHUP', 'actual', 'predicted', 'residual', 'MIN']]
    print(top_errors.to_string(index=False))
    
    return residuals_df

def analyze_shap(model, test_df, features, target_col='PTS'):
    model_features = model.get_booster().feature_names
    if model_features is None:
        model_features = features
    
    missing_features = [f for f in model_features if f not in test_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing, filling with 0.0")
        for f in missing_features:
            test_df[f] = 0.0
    
    extra_features = [f for f in features if f not in model_features]
    if extra_features:
        print(f"Warning: {len(extra_features)} features in list not in model")
    
    X_test = test_df[model_features].copy()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': model_features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print("Top 25 Features by SHAP importance:")
    for idx, row in feature_importance.head(25).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    print("Bottom 25 Features by SHAP importance:")
    for idx, row in feature_importance.tail(25).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, max_display=25, show=False)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return model

def train_cascading_model(train_data, val_data, 
                          min_features, fga_features, fg3a_features, fta_features,
                          date_column='GAME_DATE', player_column='PLAYER_ID'):
    
    min_model, min_metrics = train_model(
        train_data=train_data,
        val_data=val_data,
        features=min_features,
        target='MIN',
        date_column=date_column,
        player_column=player_column
    )
    
    train_min_pred = predict(min_model, train_data, min_features)
    val_min_pred = predict(min_model, val_data, min_features)
    
    train_with_min = train_data.copy()
    train_with_min['PREDICTED_MIN'] = train_min_pred
    val_with_min = val_data.copy()
    val_with_min['PREDICTED_MIN'] = val_min_pred
    
    # Calculate interaction features for FGA model
    train_with_min['EXPECTED_PACE_x_PREDICTED_MIN'] = (
        train_with_min['EXPECTED_PACE'] * train_with_min['PREDICTED_MIN']
    )
    # Calculate FGA_PER_MIN using predicted MIN
    train_with_min['FGA_PER_MIN'] = (
        train_with_min['FGA_ROLLING_AVG_10'] / (train_with_min['PREDICTED_MIN'] + 1e-8)
    ).round(3)
    train_with_min['FGA_PER_MIN'] = train_with_min['FGA_PER_MIN'].fillna(0.0)

    val_with_min['EXPECTED_PACE_x_PREDICTED_MIN'] = (
        val_with_min['EXPECTED_PACE'] * val_with_min['PREDICTED_MIN']
    )
    # Calculate FGA_PER_MIN using predicted MIN
    val_with_min['FGA_PER_MIN'] = (
        val_with_min['FGA_ROLLING_AVG_10'] / (val_with_min['PREDICTED_MIN'] + 1e-8)
    ).round(3)
    val_with_min['FGA_PER_MIN'] = val_with_min['FGA_PER_MIN'].fillna(0.0)
    
    fga_features_with_min = fga_features + ['PREDICTED_MIN', 'FGA_PER_MIN']
    fga_model, fga_metrics = train_model(
        train_data=train_with_min,
        val_data=val_with_min,
        features=fga_features_with_min,
        target='FGA',
        date_column=date_column,
        player_column=player_column
    )
    
    train_fga_pred = predict(fga_model, train_with_min, fga_features_with_min)
    val_fga_pred = predict(fga_model, val_with_min, fga_features_with_min)
    
    train_with_min_fga = train_with_min.copy()
    train_with_min_fga['PREDICTED_FGA'] = train_fga_pred
    val_with_min_fga = val_with_min.copy()
    val_with_min_fga['PREDICTED_FGA'] = val_fga_pred
    
    # Calculate interaction features for FG3A model
    train_with_min_fga['EXPECTED_PACE_x_PREDICTED_FGA'] = (
        train_with_min_fga['EXPECTED_PACE'] * train_with_min_fga['PREDICTED_FGA']
    )
    # Calculate FG3A_PER_MIN using predicted FGA
    train_with_min_fga['FG3A_PER_MIN'] = (
        train_with_min_fga['FG3A_ROLLING_AVG_10'] / (train_with_min_fga['PREDICTED_FGA'] + 1e-8)
    ).round(3)

    val_with_min_fga['EXPECTED_PACE_x_PREDICTED_FGA'] = (
        val_with_min_fga['EXPECTED_PACE'] * val_with_min_fga['PREDICTED_FGA']
    )
    # Calculate FG3A_PER_MIN using predicted FGA
    val_with_min_fga['FG3A_PER_MIN'] = (
        val_with_min_fga['FG3A_ROLLING_AVG_10'] / (val_with_min_fga['PREDICTED_FGA'] + 1e-8)
    ).round(3)
    train_with_min_fga['FG3A_PER_MIN'] = train_with_min_fga['FG3A_PER_MIN'].fillna(0.0)
    val_with_min_fga['FG3A_PER_MIN'] = val_with_min_fga['FG3A_PER_MIN'].fillna(0.0)

    fg3a_features_with_min_fga = fg3a_features + ['PREDICTED_MIN', 'PREDICTED_FGA', 'FG3A_PER_MIN']
    fg3a_model, fg3a_metrics = train_model(
        train_data=train_with_min_fga,
        val_data=val_with_min_fga,
        features=fg3a_features_with_min_fga,
        target='FG3A',
        date_column=date_column,
        player_column=player_column
    )
    
    train_fg3a_pred = predict(fg3a_model, train_with_min_fga, fg3a_features_with_min_fga)
    val_fg3a_pred = predict(fg3a_model, val_with_min_fga, fg3a_features_with_min_fga)
    
    train_with_min_fga_fg3a = train_with_min_fga.copy()
    train_with_min_fga_fg3a['PREDICTED_FG3A'] = train_fg3a_pred
    val_with_min_fga_fg3a = val_with_min_fga.copy()
    val_with_min_fga_fg3a['PREDICTED_FG3A'] = val_fg3a_pred
    

    # Calculate interaction features for FTA model
    train_with_min_fga_fg3a['EXPECTED_PACE_x_PREDICTED_FGA_x_PREDICTED_FG3A'] = (
        train_with_min_fga_fg3a['EXPECTED_PACE'] * train_with_min_fga_fg3a['PREDICTED_FGA'] * train_with_min_fga_fg3a['PREDICTED_FG3A']
    )
    # Calculate FTA_PER_MIN using predicted FGA and FG3A
    train_with_min_fga_fg3a['FTA_PER_MIN'] = (
        train_with_min_fga_fg3a['FTA_ROLLING_AVG_10'] / (train_with_min_fga_fg3a['PREDICTED_FGA'] * train_with_min_fga_fg3a['PREDICTED_FG3A'] + 1e-8)
    ).round(3)
    train_with_min_fga_fg3a['FTA_PER_MIN'] = train_with_min_fga_fg3a['FTA_PER_MIN'].fillna(0.0)

    val_with_min_fga_fg3a['EXPECTED_PACE_x_PREDICTED_FGA_x_PREDICTED_FG3A'] = (
        val_with_min_fga_fg3a['EXPECTED_PACE'] * val_with_min_fga_fg3a['PREDICTED_FGA'] * val_with_min_fga_fg3a['PREDICTED_FG3A']
    )
    # Calculate FTA_PER_MIN using predicted FGA and FG3A
    val_with_min_fga_fg3a['FTA_PER_MIN'] = (
        val_with_min_fga_fg3a['FTA_ROLLING_AVG_10'] / (val_with_min_fga_fg3a['PREDICTED_FGA'] * val_with_min_fga_fg3a['PREDICTED_FG3A'] + 1e-8)
    ).round(3)
    val_with_min_fga_fg3a['FTA_PER_MIN'] = val_with_min_fga_fg3a['FTA_PER_MIN'].fillna(0.0)
    train_with_min_fga_fg3a['FTA_PER_MIN'] = train_with_min_fga_fg3a['FTA_PER_MIN'].fillna(0.0)
    
    fta_features_with_min_fga_fg3a = fta_features + ['PREDICTED_MIN', 'PREDICTED_FGA', 'PREDICTED_FG3A', 'FTA_PER_MIN']
    fta_model, fta_metrics = train_model(
        train_data=train_with_min_fga_fg3a,
        val_data=val_with_min_fga_fg3a,
        features=fta_features_with_min_fga_fg3a,
        target='FTA',
        date_column=date_column,
        player_column=player_column
    )
    
    print(f"\nMIN Model - RMSE: {min_metrics['RMSE']:.3f}, MAE: {min_metrics['MAE']:.3f}, R²: {min_metrics['R2']:.3f}")
    print(f"FGA Model - RMSE: {fga_metrics['RMSE']:.3f}, MAE: {fga_metrics['MAE']:.3f}, R²: {fga_metrics['R2']:.3f}")
    print(f"FG3A Model - RMSE: {fg3a_metrics['RMSE']:.3f}, MAE: {fg3a_metrics['MAE']:.3f}, R²: {fg3a_metrics['R2']:.3f}")
    print(f"FTA Model - RMSE: {fta_metrics['RMSE']:.3f}, MAE: {fta_metrics['MAE']:.3f}, R²: {fta_metrics['R2']:.3f}")
    
    result = {
        'min_model': min_model,
        'fga_model': fga_model,
        'fg3a_model': fg3a_model,
        'fta_model': fta_model,
        'metrics': {
            'MIN': min_metrics,
            'FGA': fga_metrics,
            'FG3A': fg3a_metrics,
            'FTA': fta_metrics
        },
        'train_with_min_fga_fg3a': train_with_min_fga_fg3a,
        'val_with_min_fga_fg3a': val_with_min_fga_fg3a
    }
    
    return result

def predict_cascading(models_dict, test_data, min_features, fga_features, fg3a_features, fta_features):
    min_pred = predict(models_dict['min_model'], test_data, min_features)
    
    test_with_min = test_data.copy()
    test_with_min['PREDICTED_MIN'] = min_pred
    
    # Calculate interaction features for FGA model (matching training)
    test_with_min['EXPECTED_PACE_x_PREDICTED_MIN'] = (
        test_with_min['EXPECTED_PACE'] * test_with_min['PREDICTED_MIN']
    )
    # Calculate FGA_PER_MIN using predicted MIN
    test_with_min['FGA_PER_MIN'] = (
        test_with_min['FGA_AVG_TO_DATE'] / (test_with_min['PREDICTED_MIN'] + 1e-8)
    ).round(3)
    test_with_min['FGA_PER_MIN'] = test_with_min['FGA_PER_MIN'].fillna(0.0)
    
    fga_features_with_min = fga_features + ['PREDICTED_MIN', 'FGA_PER_MIN']
    fga_pred = predict(models_dict['fga_model'], test_with_min, fga_features_with_min)
    
    test_with_min_fga = test_with_min.copy()
    test_with_min_fga['PREDICTED_FGA'] = fga_pred
    
    fg3a_features_with_min_fga = fg3a_features + ['PREDICTED_MIN', 'PREDICTED_FGA']
    fg3a_pred = predict(models_dict['fg3a_model'], test_with_min_fga, fg3a_features_with_min_fga)
    
    test_with_min_fga_fg3a = test_with_min_fga.copy()
    test_with_min_fga_fg3a['PREDICTED_FG3A'] = fg3a_pred
    
    fta_features_with_min_fga_fg3a = fta_features + ['PREDICTED_MIN', 'PREDICTED_FGA', 'PREDICTED_FG3A']
    fta_pred = predict(models_dict['fta_model'], test_with_min_fga_fg3a, fta_features_with_min_fga_fg3a)
    
    test_with_min_fga_fg3a_fta = test_with_min_fga_fg3a.copy()
    test_with_min_fga_fg3a_fta['PREDICTED_FTA'] = fta_pred
    
    result = {
        'MIN': min_pred,
        'FGA': fga_pred,
        'FG3A': fg3a_pred,
        'FTA': fta_pred,
        'test_with_min_fga_fg3a_fta': test_with_min_fga_fg3a_fta
    }
    
    return result
