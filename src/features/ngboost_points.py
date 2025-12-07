import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.ngboost_model import predict_mean
from src.utils.helper_functions import findOpp
from src.pipeline.pipeline_pts import build_ngboost_points_features


def load_trained_ngboost_models(model_paths=None):
    if model_paths is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        model_wrapper_path = project_root / 'src' / 'models' / 'saved' / 'pts_model_wrapper.pkl'
    else:
        # If model_paths is provided, it might be a dict with 'model_wrapper' key or individual paths
        if 'model_wrapper' in model_paths:
            model_wrapper_path = Path(model_paths['model_wrapper'])
        else:
            # Fallback: try to load individual files if model_wrapper not provided
            project_root = Path(__file__).resolve().parent.parent.parent
            model_wrapper_path = project_root / 'src' / 'models' / 'saved' / 'pts_model_wrapper.pkl'
    
    try:
        model_wrapper = joblib.load(str(model_wrapper_path))
        return model_wrapper
    except Exception as e:
        print(f"Error loading NGBOOST model wrapper: {e}")
        return None


def predict_points_ngboost(
    player_name,
    data,
    date,
    projectedStartingFive=None,
    mainStartingFive=None,
    teamStarPlayer=None,
    league_df=None,
    findOpp=None,
    predicted_minutes=None,
    predicted_usage=None,
    model_wrapper=None,
    model_paths=None,
    use_calibration=True
):
    if model_wrapper is None:
        model_wrapper = load_trained_ngboost_models(model_paths)
    
    if model_wrapper is None:
        raise ValueError("No trained NGBOOST models available. Please train the models first.")
    
    mean_model = model_wrapper['mean_model']
    features_list = model_wrapper['features']
    
    # Get variance calibration and bins from model_wrapper if available
    variance_calibration = None
    bins = None
    if use_calibration:
        variance_calibration = model_wrapper.get('variance_calibration')
        bins = model_wrapper.get('bins')
    
    feature_dict = build_ngboost_points_features(
        player_name=player_name,
        data=data,
        current_date=date,
        projectedStartingFive=projectedStartingFive,
        mainStartingFive=mainStartingFive,
        teamStarPlayer=teamStarPlayer,
        league_df=league_df,
        findOpp=findOpp,
        predicted_minutes=predicted_minutes,
        predicted_usage=predicted_usage
    )
    
    if feature_dict is None:
        return None
    
    feature_vector = [feature_dict.get(f, 0.0) for f in features_list]
    feature_df = pd.DataFrame([feature_vector], columns=features_list)
    # Don't fillna here - let predict_mean handle it consistently with training
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    
    try:
        mu = predict_mean(
            mean_model, 
            feature_df, 
            features_list, 
            return_type='median',
            variance_calibration=variance_calibration,
            bins=bins
        )
        mu = float(mu[0] if isinstance(mu, (np.ndarray, pd.Series)) else mu)
        mu = max(0.0, mu)
        
        return {
            'predicted_points': mu
        }
    except Exception as e:
        print(f"Error predicting points for {player_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

