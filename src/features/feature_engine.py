import joblib
import pandas as pd
from .min_features import MinutesFeatureBuilder
from .fga_features import FGAFeatureBuilder
from .fg3a_features import FG3AFeatureBuilder
from .fta_features import FTAFeatureBuilder
from .ngboost_points import predict_points_ngboost, load_trained_ngboost_models

class FeatureEngine:

    def __init__(self, paths):
        self.min_model = joblib.load(paths["min_model"])
        # Usage model is optional - if not provided, we'll calculate usage from historical data
        self.usg_model = None
        if "usg_model" in paths or "usg_model_path" in paths:
            self.usg_model = joblib.load(paths.get("usg_model", paths.get("usg_model_path")))
        self.fga_model = joblib.load(paths.get("fga_model", paths.get("fga_model_path")))
        
        # Load optional fg3a and fta models if provided
        self.fg3a_model = None
        if "fg3a_model" in paths or "fg3a_model_path" in paths:
            self.fg3a_model = joblib.load(paths.get("fg3a_model", paths.get("fg3a_model_path")))
        
        self.fta_model = None
        if "fta_model" in paths or "fta_model_path" in paths:
            self.fta_model = joblib.load(paths.get("fta_model", paths.get("fta_model_path")))
        
        # Load model wrapper - expects a dict with keys: mean_model, features, variance_calibration, bins
        if "ngboost_model_wrapper" in paths:
            model_wrapper_path = {"model_wrapper": paths["ngboost_model_wrapper"]}
        elif "ngboost_model_paths" in paths:
            # Support legacy format or new model_wrapper key
            if "model_wrapper" in paths["ngboost_model_paths"]:
                model_wrapper_path = {"model_wrapper": paths["ngboost_model_paths"]["model_wrapper"]}
            else:
                model_wrapper_path = paths["ngboost_model_paths"]
        else:
            model_wrapper_path = None
        
        self.ngboost_model_wrapper = load_trained_ngboost_models(model_wrapper_path)
        
        # Validate model_wrapper structure matches new format
        if self.ngboost_model_wrapper is not None:
            required_keys = ['mean_model', 'features', 'variance_calibration', 'bins']
            missing_keys = [key for key in required_keys if key not in self.ngboost_model_wrapper]
            if missing_keys:
                raise ValueError(f"Model wrapper missing required keys: {missing_keys}. "
                               f"Expected keys: {required_keys}")

        self.minutes_builder = MinutesFeatureBuilder()
        self.fga_builder = FGAFeatureBuilder()
        self.fg3a_builder = FG3AFeatureBuilder()
        self.fta_builder = FTAFeatureBuilder()

    def predict_minutes(self, player_name, data, date, **kwargs):
        X = self.minutes_builder.build(player_name, data, date, **kwargs)
        if X is None:
            return None
        return float(self.min_model.predict([X])[0])

    def predict_fga(self, player_name, data, date, pred_minutes, **kwargs):
        X = self.fga_builder.build(player_name, data, date, predicted_minutes=pred_minutes, **kwargs)
        if X is None:
            return None
        return float(self.fga_model.predict([X])[0])

    def predict_fg3a(self, player_name, data, date, pred_minutes, pred_fga, **kwargs):
        if self.fg3a_model is None:
            return None
        X = self.fg3a_builder.build(player_name, data, date, predicted_minutes=pred_minutes, predicted_fga=pred_fga, **kwargs)
        if X is None:
            return None
        return float(self.fg3a_model.predict([X])[0])

    def predict_fta(self, player_name, data, date, pred_minutes, pred_fga, pred_fg3a, **kwargs):
        if self.fta_model is None:
            return None
        X = self.fta_builder.build(player_name, data, date, predicted_minutes=pred_minutes, predicted_fga=pred_fga, predicted_fg3a=pred_fg3a, **kwargs)
        if X is None:
            return None
        return float(self.fta_model.predict([X])[0])

    def predict_points(self, player_name, data, date, pred_minutes=None, pred_fga=None, pred_fg3a=None, pred_fta=None, **kwargs):
        result = predict_points_ngboost(
            player_name=player_name,
            data=data,
            date=date,
            predicted_minutes=pred_minutes,
            predicted_fga=pred_fga,
            predicted_fg3a=pred_fg3a,
            predicted_fta=pred_fta,
            model_wrapper=self.ngboost_model_wrapper,
            **kwargs
        )
        
        if result is None:
            return None
        
        return float(result['predicted_points'])

    def project_player(self, player_name, data, date, **kwargs):
        pred_min = self.predict_minutes(player_name, data, date, **kwargs)
        if pred_min is None:
            return None
            
        pred_fga = self.predict_fga(player_name, data, date, pred_min, **kwargs)
        if pred_fga is None:
            return None
        
        # Predict fg3a if model is available
        pred_fg3a = None
        if self.fg3a_model is not None:
            pred_fg3a = self.predict_fg3a(player_name, data, date, pred_min, pred_fga, **kwargs)
        
        # Predict fta if model is available
        pred_fta = None
        if self.fta_model is not None:
            pred_fta = self.predict_fta(player_name, data, date, pred_min, pred_fga, pred_fg3a, **kwargs)
        
        pts_result = predict_points_ngboost(
            player_name=player_name,
            data=data,
            date=date,
            predicted_minutes=pred_min,
            predicted_fga=pred_fga,
            predicted_fg3a=pred_fg3a,
            predicted_fta=pred_fta,
            model_wrapper=self.ngboost_model_wrapper,
            **kwargs
        )
        
        if pts_result is None:
            return None
        
        pred_pts = float(pts_result['predicted_points'])

        result = {
            "predicted_minutes": pred_min,
            "predicted_fga": pred_fga,
            "predicted_points": pred_pts
        }
        
        if pred_fg3a is not None:
            result["predicted_fg3a"] = pred_fg3a
        if pred_fta is not None:
            result["predicted_fta"] = pred_fta
        
        return result
