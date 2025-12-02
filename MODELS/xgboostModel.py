import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

# ==============================================================================
# STEP 1: HELPER FUNCTIONS
# ==============================================================================

def create_recent_game_weights(df, player_column, recent_games=15, weight=3.0):
    """
    Give more importance to recent games when training the model.
    
    WHY? Recent games are more relevant for predicting future performance.
    A player's stats from 3 months ago matter less than last week's games.
    """
    # Start with all games having weight = 1.0
    weights = np.ones(len(df))
    
    # For each player, find their last N games and increase their weight
    df_with_index = df.reset_index(drop=True)
    
    # Get the indices of the last N games for each player
    recent_game_indices = df_with_index.groupby(player_column, sort=False).tail(recent_games).index
    
    # Increase the weight for those recent games
    weights[recent_game_indices] = weight
    
    print(f"✓ Applied higher weight ({weight}x) to last {recent_games} games per player")
    
    return weights


def ensure_features_list(features):
    """Convert features to a list if it's not already one."""
    if isinstance(features, list):
        return features
    elif hasattr(features, '__iter__') and not isinstance(features, str):
        return list(features)
    else:
        return [features]


def prepare_features(X):
    """
    Clean up feature data types to ensure XGBoost can use them.
    
    WHY? XGBoost needs numeric data. This converts any non-numeric columns.
    """
    X_clean = X.copy()
    
    for column in X_clean.columns:
        # Convert boolean (True/False) to integers (1/0)
        if X_clean[column].dtype == bool:
            X_clean[column] = X_clean[column].astype(int)
        
        # Convert any other non-numeric columns to numbers
        elif not np.issubdtype(X_clean[column].dtype, np.number):
            X_clean[column] = pd.to_numeric(X_clean[column], errors='coerce')
    
    return X_clean

# ==============================================================================
# STEP 2: TRAIN THE MODEL (THE MAIN FUNCTION)
# ==============================================================================

def train_model(train_data, val_data, features, target, date_column, player_column):
    """
    Train XGBoost model with proper NaN handling.
    """
    
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL")
    print("="*70)
    
    # Step 1: Sort by date
    print("\n📅 Step 1: Sorting by date...")
    train_data = train_data.sort_values(date_column).reset_index(drop=True)
    val_data = val_data.sort_values(date_column).reset_index(drop=True)
    
    print(f"   Training: {train_data[date_column].min()} to {train_data[date_column].max()}")
    print(f"   Validation: {val_data[date_column].min()} to {val_data[date_column].max()}")
    
    # Step 2: Remove rows where TARGET is missing (keep rows with NaN features!)
    print("\n🧹 Step 2: Removing rows with missing target...")
    train_data = train_data.dropna(subset=[target])
    val_data = val_data.dropna(subset=[target])
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Step 3: Prepare features
    print("\n📊 Step 3: Preparing features...")
    X_train = train_data[features].copy()
    X_val = val_data[features].copy()
    y_train = train_data[target].values
    y_val = val_data[target].values
    
    # Step 4: Fill NaN in features (CRITICAL!)
    print("\n🔧 Step 4: Handling NaN values in features...")
    nan_train_before = X_train.isna().sum().sum()
    nan_val_before = X_val.isna().sum().sum()
    
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    print(f"   • Filled {nan_train_before} NaN in training features")
    print(f"   • Filled {nan_val_before} NaN in validation features")
    print(f"   • Remaining NaN: {X_train.isna().sum().sum() + X_val.isna().sum().sum()}")
    
    # Step 5: Create sample weights
    print("\n⚖️  Step 5: Creating sample weights...")
    weights = np.ones(len(train_data))
    df_reset = train_data.reset_index(drop=True)
    recent_idx = df_reset.groupby(player_column, sort=False).tail(15).index
    weights[recent_idx] = 3.0
    print(f"   ✓ Applied 3x weight to last 15 games per player")
    
    # Step 6: Train model
    print("\n🤖 Step 6: Training XGBoost...")
    
    model = XGBRegressor(
        objective='reg:squarederror',  # Fixed! Was 'reg:pseudohubererror' which caused issues
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=5,
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
    print(f"   ✓ Training stopped at iteration {best_iter}")
    
    # Step 7: Evaluate
    print("\n📈 Step 7: Evaluating on validation...")
    val_preds = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)
    
    print(f"\n   VALIDATION METRICS:")
    print(f"   • RMSE: {rmse:.3f}")
    print(f"   • MAE:  {mae:.3f}")
    print(f"   • R²:   {r2:.3f}")
    
    # Check predictions are reasonable
    print(f"\n   Predictions: min={val_preds.min():.2f}, max={val_preds.max():.2f}, mean={val_preds.mean():.2f}")
    print(f"   Actual:      min={y_val.min():.2f}, max={y_val.max():.2f}, mean={y_val.mean():.2f}")
    
    # Step 8: Retrain on combined data
    print("\n🔄 Step 8: Retraining on combined data...")
    
    combined = pd.concat([train_data, val_data], ignore_index=True)
    X_combined = combined[features].fillna(0)
    y_combined = combined[target].values
    
    # Combined weights
    train_weights = np.ones(len(train_data))
    train_reset = train_data.reset_index(drop=True)
    train_recent = train_reset.groupby(player_column, sort=False).tail(15).index
    train_weights[train_recent] = 3.0
    
    val_weights = np.ones(len(val_data))
    combined_weights = np.concatenate([train_weights, val_weights])
    
    # Final model
    final_model = XGBRegressor(
        objective='reg:squarederror',  # Fixed! Was 'reg:pseudohubererror'
        n_estimators=int(best_iter),
        learning_rate=0.05,
        max_depth=6,
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
    print(f"   ✓ Final model trained on {len(combined)} games")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    return final_model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'best_iteration': int(best_iter)}


# ==============================================================================
# STEP 3: MAKE PREDICTIONS
# ==============================================================================

def predict(model, test_data, features):
    """Make predictions, handling NaN properly."""
    X_test = test_data[features].fillna(0)
    return model.predict(X_test)


# ==============================================================================
# STEP 4: EVALUATE MODEL PERFORMANCE
# ==============================================================================

def evaluate_predictions(model, test_data, features, target):
    """
    Calculate how well the model performs on test data.
    
    Args:
        model: Trained XGBoost model
        test_data: DataFrame with test games
        features: List of feature column names
        target: Target column name
    
    Returns:
        Dictionary with performance metrics
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST DATA")
    print("="*70)
    
    # Get predictions
    X_test = prepare_features(test_data[features])
    y_test = test_data[target].values
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n   TEST SET RESULTS:")
    print(f"   • RMSE: {rmse:.3f}")
    print(f"   • MAE:  {mae:.3f}")
    print(f"   • R²:   {r2:.3f}")
    print("\n" + "="*70 + "\n")
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


# ==============================================================================
# STEP 5: ANALYZE ERRORS (RESIDUALS)
# ==============================================================================

def analyze_prediction_errors(model, test_data, features, target):
    """
    Analyze where the model makes the biggest mistakes.
    
    This helps you understand:
    - Which predictions are most accurate
    - Which predictions are least accurate
    - Patterns in the errors
    
    Args:
        model: Trained XGBoost model
        test_data: DataFrame with test games
        features: List of feature column names
        target: Target column name
    
    Returns:
        DataFrame with actual values, predictions, and errors
    """
    print("\n" + "="*70)
    print("ANALYZING PREDICTION ERRORS")
    print("="*70)
    
    # Get predictions
    X_test = prepare_features(test_data[features])
    predictions = model.predict(X_test)
    actual = test_data[target].values
    
    # Create results DataFrame
    results = pd.DataFrame({
        'actual': actual,
        'predicted': predictions,
        'error': actual - predictions,
        'abs_error': np.abs(actual - predictions)
    })
    
    # Add player names and game info if available
    info_columns = ['PLAYER_NAME', 'MATCHUP', 'MIN']
    for col in info_columns:
        if col in test_data.columns:
            results[col] = test_data[col].values
    
    # Print summary statistics
    print(f"\n   ERROR STATISTICS:")
    print(f"   • Mean Error:     {results['error'].mean():.3f}")
    print(f"   • Std Dev Error:  {results['error'].std():.3f}")
    print(f"   • Mean Abs Error: {results['abs_error'].mean():.3f}")
    
    # Show worst predictions
    print(f"\n   WORST 10 PREDICTIONS:")
    print("   " + "-"*66)
    worst = results.nlargest(10, 'abs_error')
    
    if 'PLAYER_NAME' in worst.columns:
        display_cols = ['PLAYER_NAME', 'actual', 'predicted', 'error', 'MIN']
        print(worst[display_cols].to_string(index=False))
    else:
        print(worst[['actual', 'predicted', 'error']].to_string(index=False))
    
    print("\n" + "="*70 + "\n")
    
    return results

def get_residuals(model, test_df, features, target_col='PTS'):
    """Get residuals analysis for regression model."""
    X_test = test_df[features].copy()
    y_test = test_df[target_col].values
    
    # Get predictions
    preds = model.predict(X_test)
    
    # Create residuals DataFrame
    residuals_df = pd.DataFrame({
        'actual': y_test,
        'predicted': preds,
        'residual': y_test - preds,
        'abs_residual': np.abs(y_test - preds),
    })
    
    # Add context columns
    context_cols = ["PLAYER_NAME", "MATCHUP", "TEAM_PTS", "OPP_PTS", "STARTING", "MIN", "GUARD", "FORWARD", "CENTER"]
    available_cols = [c for c in context_cols if c in test_df.columns]
    residuals_df = residuals_df.join(test_df[available_cols])
    
    return residuals_df

def analyze_residuals(model, test_df, features, target_col='PTS'):
    """Analyze residuals for regression model and show worst predictions."""
    
    # Get residuals
    residuals_df = get_residuals(model, test_df, features, target_col)
    
    print("RESIDUALS ANALYSIS")
    print("=" * 60)
    
    # Overall residuals statistics
    print(f"Residuals - Mean: {residuals_df['residual'].mean():.3f}, Std: {residuals_df['residual'].std():.3f}")
    print(f"Absolute Residuals - Mean: {residuals_df['abs_residual'].mean():.3f}")
    
    # Worst predictions (largest residuals)
    print(f"\nWORST PREDICTIONS:")
    print("-" * 50)
    
    worst_predictions = residuals_df.copy()
    worst_predictions = worst_predictions.sort_values('abs_residual', ascending=False)
    
    # Show top 20 worst predictions
    top_errors = worst_predictions.head(20)[['PLAYER_NAME', 'MATCHUP', 'actual', 'predicted', 'residual', 'MIN']]
    print(top_errors.to_string(index=False))
    
    return residuals_df

def analyze_shap(model, test_df, features, target_col='PTS'):
    """Generate SHAP analysis for model interpretability."""
    
    # Get the actual features the model expects
    model_features = model.get_booster().feature_names
    if model_features is None:
        # If model doesn't have feature names, use the provided list
        model_features = features
    
    # Check which features are missing from test_df
    missing_features = [f for f in model_features if f not in test_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing from test_df:")
        for f in missing_features[:10]:  # Show first 10
            print(f"   - {f}")
        if len(missing_features) > 10:
            print(f"   ... and {len(missing_features) - 10} more")
        
        # Create missing features with zeros
        for f in missing_features:
            test_df[f] = 0.0
        print(f"   ✓ Filled missing features with 0.0")
    
    # Check which features in the list don't exist in model
    extra_features = [f for f in features if f not in model_features]
    if extra_features:
        print(f"⚠️  Warning: {len(extra_features)} features in list not in model (will be ignored)")
    
    # Use only the features the model expects, in the correct order
    X_test = test_df[model_features].copy()
    
    print(f"\nSHAP ANALYSIS:")
    print(f"   Model expects: {len(model_features)} features")
    print(f"   Test data has: {len(X_test.columns)} features")
    print("-" * 30)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Overall top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': model_features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Features by SHAP importance:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.4f}")
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, max_display=15, show=False)
    plt.title(f'SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return model


# ==============================================================================
# CASCADING MODEL: MIN -> USG_PCT -> PTS
# ==============================================================================

def train_cascading_model(train_data, val_data, 
                          min_features, usg_features, pts_features=None,
                          date_column='GAME_DATE', player_column='PLAYER_ID'):
    """
    Train a cascading model that predicts MIN -> USG_PCT -> (optionally PTS) sequentially.
    
    Process:
    1. Train MIN model using min_features
    2. Predict MIN for train/val, add as feature
    3. Train USG_PCT model using usg_features + predicted MIN
    4. Predict USG_PCT for train/val, add as feature
    5. (Optional) Train PTS model using pts_features + predicted MIN + predicted USG_PCT
    
    Args:
        pts_features: Optional. If None, PTS model training is skipped (use Negative Binomial for PTS instead)
    """
    if pts_features is None:
        print("\n" + "="*70)
        print("TRAINING CASCADING MODEL: MIN -> USG_PCT")
        print("(PTS will use Negative Binomial model separately)")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("TRAINING CASCADING MODEL: MIN -> USG_PCT -> PTS")
        print("="*70)
    
    # ==========================================================================
    # STEP 1: Train MIN model
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 1: Training MIN Model")
    print("="*70)
    
    min_model, min_metrics = train_model(
        train_data=train_data,
        val_data=val_data,
        features=min_features,
        target='MIN',
        date_column=date_column,
        player_column=player_column
    )
    
    # Predict MIN for train and val
    print("\n📊 Predicting MIN for cascading features...")
    train_min_pred = predict(min_model, train_data, min_features)
    val_min_pred = predict(min_model, val_data, min_features)
    
    # Add predicted MIN to datasets
    train_with_min = train_data.copy()
    train_with_min['PREDICTED_MIN'] = train_min_pred
    val_with_min = val_data.copy()
    val_with_min['PREDICTED_MIN'] = val_min_pred
    
    print(f"   Train MIN predictions: min={train_min_pred.min():.2f}, max={train_min_pred.max():.2f}, mean={train_min_pred.mean():.2f}")
    print(f"   Val MIN predictions:   min={val_min_pred.min():.2f}, max={val_min_pred.max():.2f}, mean={val_min_pred.mean():.2f}")
    
    # ==========================================================================
    # STEP 2: Train USG_PCT model (with predicted MIN)
    # ==========================================================================
    print("\n" + "="*70)
    print("STEP 2: Training USG_PCT Model (with PREDICTED_MIN)")
    print("="*70)
    
    # Add PREDICTED_MIN to USG features
    usg_features_with_min = usg_features + ['PREDICTED_MIN']
    
    usg_model, usg_metrics = train_model(
        train_data=train_with_min,
        val_data=val_with_min,
        features=usg_features_with_min,
        target='USG_PCT',
        date_column=date_column,
        player_column=player_column
    )
    
    # Predict USG_PCT for train and val
    print("\n📊 Predicting USG_PCT for cascading features...")
    train_usg_pred = predict(usg_model, train_with_min, usg_features_with_min)
    val_usg_pred = predict(usg_model, val_with_min, usg_features_with_min)
    
    # Add predicted USG_PCT to datasets
    train_with_min_usg = train_with_min.copy()
    train_with_min_usg['PREDICTED_USG_PCT'] = train_usg_pred
    val_with_min_usg = val_with_min.copy()
    val_with_min_usg['PREDICTED_USG_PCT'] = val_usg_pred
    
    print(f"   Train USG predictions: min={train_usg_pred.min():.2f}, max={train_usg_pred.max():.2f}, mean={train_usg_pred.mean():.2f}")
    print(f"   Val USG predictions:   min={val_usg_pred.min():.2f}, max={val_usg_pred.max():.2f}, mean={val_usg_pred.mean():.2f}")
    
    # ==========================================================================
    # STEP 3: Train PTS model (with predicted MIN + USG_PCT) - OPTIONAL
    # ==========================================================================
    if pts_features is not None:
        print("\n" + "="*70)
        print("STEP 3: Training PTS Model (with PREDICTED_MIN + PREDICTED_USG_PCT)")
        print("="*70)
        
        # Add PREDICTED_MIN and PREDICTED_USG_PCT to PTS features
        pts_features_with_cascade = pts_features + ['PREDICTED_MIN', 'PREDICTED_USG_PCT']
        
        pts_model, pts_metrics = train_model(
            train_data=train_with_min_usg,
            val_data=val_with_min_usg,
            features=pts_features_with_cascade,
            target='PTS',
            date_column=date_column,
            player_column=player_column
        )
    else:
        pts_model = None
        pts_metrics = None
        print("\n" + "="*70)
        print("STEP 3: Skipped (Using Negative Binomial for PTS)")
        print("="*70)
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("CASCADING MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\nFINAL METRICS SUMMARY:")
    print(f"\n   MIN Model:")
    print(f"      RMSE: {min_metrics['RMSE']:.3f}")
    print(f"      MAE:  {min_metrics['MAE']:.3f}")
    print(f"      R²:   {min_metrics['R2']:.3f}")
    
    print(f"\n   USG_PCT Model:")
    print(f"      RMSE: {usg_metrics['RMSE']:.3f}")
    print(f"      MAE:  {usg_metrics['MAE']:.3f}")
    print(f"      R²:   {usg_metrics['R2']:.3f}")
    
    if pts_model is not None:
        print(f"\n   PTS Model:")
        print(f"      RMSE: {pts_metrics['RMSE']:.3f}")
        print(f"      MAE:  {pts_metrics['MAE']:.3f}")
        print(f"      R²:   {pts_metrics['R2']:.3f}")
    else:
        print(f"\n   PTS Model: Skipped (use Negative Binomial model separately)")
    
    print("="*70 + "\n")
    
    result = {
        'min_model': min_model,
        'usg_model': usg_model,
        'metrics': {
            'MIN': min_metrics,
            'USG_PCT': usg_metrics
        },
        'train_with_min_usg': train_with_min_usg,  # Return this for Negative Binomial training
        'val_with_min_usg': val_with_min_usg  # Return this for Negative Binomial training
    }
    
    if pts_model is not None:
        result['pts_model'] = pts_model
        result['metrics']['PTS'] = pts_metrics
    
    return result


def predict_cascading(models_dict, test_data, min_features, usg_features, pts_features=None):
    """
    Make predictions using the cascading model approach.
    
    Args:
        models_dict: Dictionary containing 'min_model' and 'usg_model' (and optionally 'pts_model')
        test_data: Test dataframe
        min_features: List of features for MIN prediction
        usg_features: List of features for USG prediction
        pts_features: Optional. List of features for PTS prediction. If None, PTS prediction is skipped.
    """
    # Step 1: Predict MIN
    min_pred = predict(models_dict['min_model'], test_data, min_features)
    
    # Step 2: Add predicted MIN and predict USG_PCT
    test_with_min = test_data.copy()
    test_with_min['PREDICTED_MIN'] = min_pred
    usg_features_with_min = usg_features + ['PREDICTED_MIN']
    usg_pred = predict(models_dict['usg_model'], test_with_min, usg_features_with_min)
    
    result = {
        'MIN': min_pred,
        'USG_PCT': usg_pred,
        'test_with_min_usg': None  # Return this for use with Negative Binomial
    }
    
    # Step 3: Add predicted USG_PCT and predict PTS (if pts_model exists)
    if pts_features is not None and 'pts_model' in models_dict:
        test_with_min_usg = test_with_min.copy()
        test_with_min_usg['PREDICTED_USG_PCT'] = usg_pred
        pts_features_with_cascade = pts_features + ['PREDICTED_MIN', 'PREDICTED_USG_PCT']
        pts_pred = predict(models_dict['pts_model'], test_with_min_usg, pts_features_with_cascade)
        result['PTS'] = pts_pred
    else:
        # Prepare dataframe with predicted MIN and USG for Negative Binomial
        test_with_min_usg = test_with_min.copy()
        test_with_min_usg['PREDICTED_USG_PCT'] = usg_pred
        result['test_with_min_usg'] = test_with_min_usg
    
    return result
