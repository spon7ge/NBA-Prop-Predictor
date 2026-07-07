"""Backward-compatible alias — prefer ``app.schemas.prop.PropLine`` for new code."""
from app.schemas.prop import PropLine as PropPrediction

__all__ = ["PropPrediction"]
