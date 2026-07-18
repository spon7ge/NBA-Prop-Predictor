"""Backward-compatible re-export of NBA rpm live pipeline."""

from src.live_pipeline.nba.rpm_pipeline import FEATURE_COLS, rpm_pipeline

__all__ = ["FEATURE_COLS", "rpm_pipeline"]
