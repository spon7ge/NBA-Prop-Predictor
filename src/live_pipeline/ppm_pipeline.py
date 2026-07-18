"""Backward-compatible re-export of NBA ppm live pipeline."""

from src.live_pipeline.nba.ppm_pipeline import FEATURE_COLS, ppm_pipeline

__all__ = ["FEATURE_COLS", "ppm_pipeline"]
