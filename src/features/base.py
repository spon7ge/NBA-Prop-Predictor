# PRODUCTION/feature_engine/base.py

class BaseFeatureBuilder:
    """
    Base class for feature builders.
    Subclasses should define:
    - feature_names: list of feature names
    - build(): method to build features
    """
    feature_names = []
    
    def build(self, *args, **kwargs):
        """
        Build features. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build()")

