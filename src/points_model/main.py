"""
Points Prop Model - Main Integration
====================================
Main model class integrating all components for end-to-end prop evaluation.

Pipeline:
1. Load player game logs
2. Fit component models (minutes, volume, efficiency)
3. Generate projections for target player/game
4. Compare to market line
5. Identify edge opportunities

Usage:
    from src.points_model import PointsPropModel
    
    model = PointsPropModel()
    model.fit(game_logs)
    
    # Get projection
    projection = model.project_points(player_id, opp_pace=102, opp_drtg=112)
    
    # Evaluate prop
    evaluation = model.evaluate_prop(player_id, market_line=22.5)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List

from .minutes_model import MinutesProjector
from .volume_model import ShotVolumeModel
from .efficiency_model import EfficiencyModel
from .points_projector import PointsProjector
from .edge_calculator import EdgeCalculator
from .matchup_adjuster import MatchupAdjuster
from .volatility_analyzer import VolatilityAnalyzer


class PointsPropModel:
    """
    Main model class for NBA points prop prediction.
    
    Integrates all component models to provide:
    - Points projections with uncertainty
    - Edge calculation vs market lines
    - Kelly criterion bet sizing
    - Volatility-based edge detection
    """
    
    def __init__(
        self,
        min_edge: float = 0.03,
        min_confidence: float = 0.55,
        shrinkage_games: int = 20
    ):
        """
        Args:
            min_edge: Minimum edge required for bet recommendation
            min_confidence: Minimum win probability threshold
            shrinkage_games: Games for full weight in efficiency model
        """
        self.minutes_model = MinutesProjector()
        self.volume_model = ShotVolumeModel()
        self.efficiency_model = EfficiencyModel(shrinkage_games=shrinkage_games)
        self.matchup_adjuster = MatchupAdjuster()
        self.edge_calculator = EdgeCalculator(
            min_edge=min_edge,
            min_confidence=min_confidence
        )
        self.volatility_analyzer = VolatilityAnalyzer()
        
        self.projector: Optional[PointsProjector] = None
        self.game_logs: Optional[pd.DataFrame] = None
        self.fitted = False
        
    def fit(self, game_logs: pd.DataFrame) -> 'PointsPropModel':
        """
        Fit all component models on historical data.
        
        Expected columns: player_id, minutes, fga, fg3a, fta, fgm, fg3m, ftm, pts
        Optional: margin, is_b2b
        
        Args:
            game_logs: DataFrame with player game log data
            
        Returns:
            self for method chaining
        """
        # Store game logs for volatility analysis
        self.game_logs = game_logs.copy()
        
        # Fit component models
        self.minutes_model.fit(game_logs)
        self.volume_model.fit(game_logs)
        self.efficiency_model.fit(game_logs)
        
        # Create integrated projector
        self.projector = PointsProjector(
            minutes_model=self.minutes_model,
            volume_model=self.volume_model,
            efficiency_model=self.efficiency_model,
            matchup_adjuster=self.matchup_adjuster
        )
        
        self.fitted = True
        return self
    
    def project_points(
        self,
        player_id: int,
        is_b2b: bool = False,
        blowout_prob: float = 0.15,
        opp_pace: float = None,
        opp_drtg: float = None,
        opp_foul_rate: float = None,
        opp_team_id: int = None,
        usage_adjustment: float = 1.0
    ) -> Optional[Dict]:
        """
        Generate full points projection with uncertainty.
        
        Args:
            player_id: Player identifier
            is_b2b: Whether game is on back-to-back
            blowout_prob: Probability of blowout
            opp_pace: Opponent pace (auto-fetched if opp_team_id provided)
            opp_drtg: Opponent defensive rating (auto-fetched if opp_team_id provided)
            opp_foul_rate: Opponent foul rate
            opp_team_id: Opponent team ID - if provided, automatically fetches pace/DRTG
            usage_adjustment: Usage multiplier
            
        Returns:
            Comprehensive projection dictionary
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before projecting")
            
        return self.projector.project(
            player_id=player_id,
            is_b2b=is_b2b,
            blowout_prob=blowout_prob,
            opp_pace=opp_pace,
            opp_drtg=opp_drtg,
            opp_foul_rate=opp_foul_rate,
            opp_team_id=opp_team_id,
            usage_adjustment=usage_adjustment
        )
    
    def evaluate_prop(
        self,
        player_id: int,
        market_line: float,
        market_juice: int = -110,
        **projection_kwargs
    ) -> Optional[Dict]:
        """
        Evaluate a specific prop bet.
        
        Args:
            player_id: Player identifier
            market_line: The prop line (e.g., 24.5)
            market_juice: Odds in American format
            **projection_kwargs: Additional args for projection
            
        Returns:
            Complete evaluation with projection and edge analysis
        """
        projection = self.project_points(player_id, **projection_kwargs)
        if projection is None:
            return None
            
        edge_analysis = self.edge_calculator.evaluate_prop(
            projection,
            market_line,
            market_juice
        )
        
        return {
            'projection': projection,
            'edge_analysis': edge_analysis,
            'player_id': player_id,
            'market_line': market_line,
        }
    
    def analyze_volatility(self, player_id: int) -> Optional[Dict]:
        """
        Run volatility analysis for a player.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Volatility analysis results
        """
        if self.game_logs is None:
            return None
            
        player_logs = self.game_logs[self.game_logs['player_id'] == player_id]
        if len(player_logs) < 10:
            return None
            
        # Fit distribution
        dist_analysis = self.volatility_analyzer.fit_distribution(player_logs)
        
        # Check for minutes volatility edge
        mins_edge = self.volatility_analyzer.detect_minutes_volatility_edge(player_logs)
        
        # Check for recency bias edge
        recency_edge = self.volatility_analyzer.detect_recency_bias_edge(player_logs)
        
        return {
            'distribution': dist_analysis,
            'minutes_volatility': mins_edge,
            'recency_bias': recency_edge,
        }
    
    def find_best_props(
        self,
        player_ids: List[int],
        market_lines: Dict[int, float],
        min_edge: float = 0.03,
        **projection_kwargs
    ) -> List[Dict]:
        """
        Find best prop opportunities from a list of players.
        
        Args:
            player_ids: List of player IDs to evaluate
            market_lines: Dict mapping player_id to market line
            min_edge: Minimum edge to include
            **projection_kwargs: Additional args for projection
            
        Returns:
            List of props with positive edge, sorted by edge magnitude
        """
        results = []
        
        for player_id in player_ids:
            if player_id not in market_lines:
                continue
                
            evaluation = self.evaluate_prop(
                player_id,
                market_lines[player_id],
                **projection_kwargs
            )
            
            if evaluation is None:
                continue
                
            edge = evaluation['edge_analysis']['edge']
            if edge >= min_edge:
                results.append(evaluation)
                
        # Sort by edge magnitude (descending)
        results.sort(key=lambda x: x['edge_analysis']['edge'], reverse=True)
        
        return results
    
    def get_player_summary(self, player_id: int) -> Optional[Dict]:
        """
        Get comprehensive summary of player parameters.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Summary of all fitted parameters for player
        """
        mins_params = self.minutes_model.get_player_params(player_id)
        volume_rates = self.volume_model.get_player_rates(player_id)
        efficiency = self.efficiency_model.get_efficiency(player_id)
        
        if not all([mins_params, volume_rates, efficiency]):
            return None
            
        return {
            'player_id': player_id,
            'minutes': {
                'baseline': mins_params['baseline'],
                'volatility': mins_params['volatility'],
                'b2b_effect': mins_params['b2b_effect'],
            },
            'volume': {
                'fga_per_min': volume_rates['fga_per_min'],
                'fg3a_per_min': volume_rates['fg3a_per_min'],
                'fta_per_min': volume_rates['fta_per_min'],
            },
            'efficiency': {
                'fg_pct': efficiency['fg_pct'],
                'fg3_pct': efficiency['fg3_pct'],
                'ft_pct': efficiency['ft_pct'],
                'shrinkage_factor': efficiency['shrinkage_factor'],
            },
            'sample_sizes': {
                'minutes': mins_params['games_played'],
                'volume': volume_rates['sample_size'],
                'efficiency': efficiency['sample_size'],
            }
        }


def print_projection_report(evaluation: Dict) -> None:
    """
    Pretty print a prop evaluation.
    
    Args:
        evaluation: Result from PointsPropModel.evaluate_prop()
    """
    proj = evaluation['projection']
    edge = evaluation['edge_analysis']
    
    print("=" * 60)
    print("POINTS PROP ANALYSIS")
    print("=" * 60)
    print(f"\nProjection: {proj['expected_points']:.1f} points")
    print(f"90% CI: [{proj['lower_90']:.1f}, {proj['upper_90']:.1f}]")
    print(f"\nComponents:")
    print(f"  Minutes: {proj['components']['minutes']['expected']:.1f}")
    print(f"  FGA: {proj['components']['volume']['fga']:.1f}")
    print(f"  FG3A: {proj['components']['volume']['fg3a']:.1f}")
    print(f"  FTA: {proj['components']['volume']['fta']:.1f}")
    print(f"\nEfficiency:")
    print(f"  FG%: {proj['components']['efficiency']['fg_pct']:.1%}")
    print(f"  FG3%: {proj['components']['efficiency']['fg3_pct']:.1%}")
    print(f"  FT%: {proj['components']['efficiency']['ft_pct']:.1%}")
    print(f"\nMarket Line: {edge['line']}")
    print(f"  P(Over): {edge['prob_over']:.1%}")
    print(f"  P(Under): {edge['prob_under']:.1%}")
    print(f"\nEdge Analysis:")
    print(f"  Over Edge: {edge['over_edge']:+.1%}")
    print(f"  Under Edge: {edge['under_edge']:+.1%}")
    print(f"  Expected Value: {edge['expected_value']:.3f}")
    print(f"\n>>> RECOMMENDATION: {edge['recommendation']}")
    if edge['kelly_bet_size'] > 0:
        print(f">>> Kelly Bet Size: {edge['kelly_bet_size']:.1%} of bankroll")
        print(f">>> Unit Sizing: {edge['unit_recommendation']}")
    print("=" * 60)
