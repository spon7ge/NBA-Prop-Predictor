import joblib
from .min_features import MinutesFeatureBuilder
from .usg_features import UsageFeatureBuilder
from .fga_features import FGAFeatureBuilder
from .ngboost_points import predict_points_ngboost, load_trained_ngboost_models

class FeatureEngine:

    def __init__(self, paths):
        self.min_model = joblib.load(paths["min_model"])
        self.usg_model = joblib.load(paths["usg_model"])
        self.fga_model = joblib.load(paths.get("fga_model", paths.get("fga_model_path")))
        
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
        self.usage_builder = UsageFeatureBuilder()
        self.fga_builder = FGAFeatureBuilder()

    def predict_minutes(self, player_name, data, date, **kwargs):
        X = self.minutes_builder.build(player_name, data, date, **kwargs)
        if X is None:
            return None
        return float(self.min_model.predict([X])[0])

    def predict_usage(self, player_name, data, date, pred_minutes, **kwargs):
        X = self.usage_builder.build(player_name, data, date, predicted_minutes=pred_minutes, **kwargs)
        if X is None:
            return None
        return float(self.usg_model.predict([X])[0])

    def predict_fga(self, player_name, data, date, pred_minutes, pred_usage, **kwargs):
        X = self.fga_builder.build(player_name, data, date, predicted_minutes=pred_minutes, predicted_usage=pred_usage, **kwargs)
        if X is None:
            return None
        return float(self.fga_model.predict([X])[0])

    def predict_points(self, player_name, data, date, pred_minutes=None, pred_usage=None, **kwargs):
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
        pred_min = self.predict_minutes(player_name, data, date, **kwargs)
        if pred_min is None:
            return None
            
        pred_usg = self.predict_usage(player_name, data, date, pred_min, **kwargs)
        if pred_usg is None:
            return None
            
        pred_fga = self.predict_fga(player_name, data, date, pred_min, pred_usg, **kwargs)
        if pred_fga is None:
            return None
        
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
            "predicted_fga": pred_fga,
            "predicted_points": pred_pts
        }
