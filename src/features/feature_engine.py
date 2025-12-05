# PRODUCTION/feature_engine/feature_engine.py

import joblib
from .min_features import MinutesFeatureBuilder
from .usg_features import UsageFeatureBuilder
from .ngboost_points import predict_points_ngboost, load_trained_ngboost_models

class FeatureEngine:

    def __init__(self, paths):
        """
        paths = {
            'min_model': '../MODELS/SAVED_MODELS/min_model.pkl',
            'usg_model': '../MODELS/SAVED_MODELS/usg_model.pkl',
            'ngboost_model_paths': {  # Optional - NGBOOST model paths
                'mean_model': '../MODELS/SAVED_MODELS/NGBOOST_PTS_MEAN_MODEL_PRODUCTION.pkl',
                'variance_model': '../MODELS/SAVED_MODELS/NGBOOST_PTS_VAR_MODEL_PRODUCTION.pkl',
                'calibration_factor': '../MODELS/SAVED_MODELS/NGBOOST_PTS_CALIBRATION_FACTOR_PRODUCTION.pkl',
                'calibration_params': '../MODELS/SAVED_MODELS/NGBOOST_PTS_CALIBRATION_PARAMS_PRODUCTION.pkl',  # Optional - for score-dependent calibration
                'features': '../MODELS/SAVED_MODELS/pts_features.pkl'
            }
        }
        """
        self.min_model = joblib.load(paths["min_model"])
        self.usg_model = joblib.load(paths["usg_model"])
        
        # Load NGBOOST models
        ngboost_paths = paths.get("ngboost_model_paths", None)
        self.ngboost_model_wrapper = load_trained_ngboost_models(ngboost_paths)

        self.minutes_builder = MinutesFeatureBuilder()
        self.usage_builder   = UsageFeatureBuilder()
        # Points prediction now uses NGBOOST model (see predict_points_ngboost)

    def predict_minutes(self, player_name, data, date, **kwargs):
        X = self.minutes_builder.build(player_name, data, date, **kwargs)
        return float(self.min_model.predict([X])[0])

    def predict_usage(self, player_name, data, date, pred_minutes, **kwargs):
        X = self.usage_builder.build(player_name, data, date, predicted_minutes=pred_minutes, **kwargs)
        return float(self.usg_model.predict([X])[0])

    def predict_points(self, player_name, data, date, pred_minutes=None, pred_usage=None, **kwargs):
        """
        Predict points using NGBOOST model.
        Uses predicted_minutes and predicted_usage if provided to adjust the prediction.
        """
        result = predict_points_ngboost(
            player_name=player_name,
            data=data,
            date=date,
            predicted_minutes=pred_minutes,
            predicted_usage=pred_usage,
            model_wrapper=self.ngboost_model_wrapper,
            **kwargs
        )
        
        if result is None:
            return None
        
        return float(result['predicted_points'])

    def project_player(self, player_name, data, date, **kwargs):
        """
        Full chain:
        MIN (XGBoost) → USG (XGBoost) → PTS (NGBOOST using predicted MIN and USG)
        """
        pred_min = self.predict_minutes(player_name, data, date, **kwargs)
        pred_usg = self.predict_usage(player_name, data, date, pred_min, **kwargs)
        
        # Use NGBOOST for points prediction, incorporating predicted minutes and usage
        pts_result = predict_points_ngboost(
            player_name=player_name,
            data=data,
            date=date,
            predicted_minutes=pred_min,
            predicted_usage=pred_usg,
            model_wrapper=self.ngboost_model_wrapper,
            **kwargs
        )
        
        if pts_result is None:
            return None
        
        pred_pts = float(pts_result['predicted_points'])

        return {
            "predicted_minutes": pred_min,
            "predicted_usage": pred_usg,
            "predicted_points": pred_pts
        }
