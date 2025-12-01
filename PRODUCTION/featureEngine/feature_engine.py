# PRODUCTION/feature_engine/feature_engine.py

import joblib
from .min_features import MinutesFeatureBuilder
from .usg_features import UsageFeatureBuilder
from .poisson_points import predict_points_poisson

class FeatureEngine:

    def __init__(self, paths):
        """
        paths = {
            'min_model': '../MODELS/SAVED_MODELS/min_model.pkl',
            'usg_model': '../MODELS/SAVED_MODELS/usg_model.pkl',
            'pts_model': '../MODELS/SAVED_MODELS/pts_model.pkl'  # Optional - not used (Poisson used instead)
        }
        """
        self.min_model = joblib.load(paths["min_model"])
        self.usg_model = joblib.load(paths["usg_model"])
        # Points model no longer used - we use Poisson instead
        # pts_model is optional for backward compatibility

        self.minutes_builder = MinutesFeatureBuilder()
        self.usage_builder   = UsageFeatureBuilder()
        # Points prediction now uses Poisson model (see predict_points_poisson)

    def predict_minutes(self, player_name, data, date, **kwargs):
        X = self.minutes_builder.build(player_name, data, date, **kwargs)
        return float(self.min_model.predict([X])[0])

    def predict_usage(self, player_name, data, date, pred_minutes, **kwargs):
        X = self.usage_builder.build(player_name, data, date, predicted_minutes=pred_minutes, **kwargs)
        return float(self.usg_model.predict([X])[0])

    def predict_points(self, player_name, data, date, pred_minutes=None, pred_usage=None, **kwargs):
        """
        Predict points using Poisson model.
        Uses predicted_minutes and predicted_usage if provided to adjust the lambda.
        """
        result = predict_points_poisson(
            player_name=player_name,
            data=data,
            date=date,
            predicted_minutes=pred_minutes,
            predicted_usage=pred_usage,
            **kwargs
        )
        
        if result is None:
            return None
        
        return float(result['predicted_points'])

    def project_player(self, player_name, data, date, **kwargs):
        """
        Full chain:
        MIN (XGBoost) → USG (XGBoost) → PTS (Poisson using predicted MIN and USG)
        """
        pred_min = self.predict_minutes(player_name, data, date, **kwargs)
        pred_usg = self.predict_usage(player_name, data, date, pred_min, **kwargs)
        
        # Use Poisson for points prediction, incorporating predicted minutes and usage
        pts_result = predict_points_poisson(
            player_name=player_name,
            data=data,
            date=date,
            predicted_minutes=pred_min,
            predicted_usage=pred_usg,
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
