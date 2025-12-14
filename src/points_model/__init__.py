"""
NBA Points Prop Prediction Model
================================
A modular approach to predicting NBA player points props.

Pipeline Architecture:
1. Minutes Projection → Expected minutes + volatility
2. Shot Volume Model → Expected FGA, FG3A, FTA (per-minute rates)
3. Efficiency Model → FG%, FG3%, FT% with Bayesian shrinkage
4. Points Projection → Combined expected points + uncertainty
5. Edge Calculator → Compare to market, Kelly sizing
6. Bet Tracker → Track performance, CLV, calibration

Usage:
    from src.points_model import PointsPropModel
    
    model = PointsPropModel()
    model.fit(player_game_logs)
    projection = model.project(player_id, **context)
    edge = model.evaluate_prop(player_id, market_line=22.5)
    
    # Track bets
    from src.points_model import PropBetTracker, PerformanceAnalyzer
    tracker = PropBetTracker()
    tracker.add_bet_from_evaluation(evaluation, bookmaker='FanDuel', odds=-110)
"""

from .minutes_model import MinutesProjector
from .volume_model import ShotVolumeModel
from .efficiency_model import EfficiencyModel
from .points_projector import PointsProjector
from .edge_calculator import EdgeCalculator
from .matchup_adjuster import MatchupAdjuster
from .volatility_analyzer import VolatilityAnalyzer
from .main import PointsPropModel
from .bet_tracker import PropBetTracker, PerformanceAnalyzer, LineTracker, print_tracking_guide

__all__ = [
    'MinutesProjector',
    'ShotVolumeModel',
    'EfficiencyModel',
    'PointsProjector',
    'EdgeCalculator',
    'MatchupAdjuster',
    'VolatilityAnalyzer',
    'PointsPropModel',
    'PropBetTracker',
    'PerformanceAnalyzer',
    'LineTracker',
    'print_tracking_guide',
]
