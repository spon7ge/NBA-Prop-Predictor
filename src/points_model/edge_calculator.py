"""
Edge Calculator
===============
Compares model projections to market lines to identify +EV opportunities.

Edge Sources in Prop Markets:
1. Stale lines (not updated for news)
2. Over-reliance on recent performance
3. Ignoring matchup context
4. Minutes projection errors
5. Lineup changes not fully priced in

This module calculates:
- Probability of Over/Under
- Edge vs market implied probability
- Expected Value (EV)
- Kelly criterion bet sizing
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


class EdgeCalculator:
    """
    Calculates edge and optimal bet sizing for prop bets.
    """
    
    def __init__(self, min_edge: float = 0.03, min_confidence: float = 0.55):
        """
        Args:
            min_edge: Minimum edge required to recommend a bet
            min_confidence: Minimum win probability to consider
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        
    def calculate_edge(
        self,
        projection: Dict,
        market_line: float,
        market_juice: int = -110
    ) -> Dict:
        """
        Calculate expected value of a prop bet.
        
        Args:
            projection: Dictionary with 'expected' and 'std' keys
            market_line: The prop line (e.g., 24.5 points)
            market_juice: Odds in American format (e.g., -110)
            
        Returns:
            Comprehensive edge analysis dictionary
        """
        expected = projection.get('expected_points', projection.get('expected'))
        std = projection.get('std', projection.get('vol', expected * 0.2))
        
        # Calculate probability of going over/under (assumes normal distribution)
        z_score = (market_line - expected) / std
        prob_over = 1 - stats.norm.cdf(z_score)
        prob_under = stats.norm.cdf(z_score)
        
        # Convert juice to implied probability (breakeven)
        implied_prob = self._american_to_implied(market_juice)
        
        # Calculate edge (true probability - breakeven)
        over_edge = prob_over - implied_prob
        under_edge = prob_under - implied_prob
        
        # Calculate Expected Value
        payout_multiplier = self._american_to_decimal(market_juice) - 1
        over_ev = prob_over * payout_multiplier - (1 - prob_over)
        under_ev = prob_under * payout_multiplier - (1 - prob_under)
        
        # Determine recommendation
        best_edge = max(over_edge, under_edge)
        best_direction = 'OVER' if over_edge > under_edge else 'UNDER'
        best_prob = prob_over if best_direction == 'OVER' else prob_under
        best_ev = over_ev if best_direction == 'OVER' else under_ev
        
        if best_edge >= self.min_edge and best_prob >= self.min_confidence:
            recommendation = best_direction
        else:
            recommendation = 'NO BET'
            
        return {
            'line': market_line,
            'projection': expected,
            'std': std,
            'z_score': z_score,
            'prob_over': prob_over,
            'prob_under': prob_under,
            'implied_prob': implied_prob,
            'over_edge': over_edge,
            'under_edge': under_edge,
            'over_ev': over_ev,
            'under_ev': under_ev,
            'recommendation': recommendation,
            'confidence': best_prob,
            'edge': best_edge,
            'expected_value': best_ev,
        }
    
    def kelly_criterion(
        self,
        edge: float,
        win_prob: float,
        odds: int = -110,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate optimal bet size using fractional Kelly criterion.
        
        Full Kelly is too aggressive for sports betting; use fractional Kelly.
        
        Args:
            edge: Edge as decimal (e.g., 0.05 for 5% edge)
            win_prob: Probability of winning
            odds: American odds
            kelly_fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)
            
        Returns:
            Recommended bet size as fraction of bankroll
        """
        if edge <= 0 or win_prob <= 0:
            return 0.0
            
        # Convert to decimal odds minus 1
        b = self._american_to_decimal(odds) - 1
        
        # Kelly formula: f* = (bp - q) / b
        # where p = win prob, q = 1-p, b = decimal odds - 1
        q = 1 - win_prob
        kelly_pct = (b * win_prob - q) / b
        
        # Apply fraction and ensure non-negative
        return max(0, kelly_pct * kelly_fraction)
    
    def _american_to_implied(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
    
    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds < 0:
            return 1 + (100 / abs(american_odds))
        else:
            return 1 + (american_odds / 100)
    
    def evaluate_prop(
        self,
        projection: Dict,
        market_line: float,
        market_juice: int = -110
    ) -> Dict:
        """
        Full prop evaluation with Kelly sizing.
        
        Args:
            projection: Projection dictionary from PointsProjector
            market_line: Market line
            market_juice: American odds
            
        Returns:
            Complete evaluation with edge analysis and bet sizing
        """
        edge_analysis = self.calculate_edge(projection, market_line, market_juice)
        
        # Calculate Kelly bet size
        if edge_analysis['edge'] > 0 and edge_analysis['recommendation'] != 'NO BET':
            win_prob = (edge_analysis['prob_over'] 
                       if edge_analysis['recommendation'] == 'OVER' 
                       else edge_analysis['prob_under'])
            kelly_size = self.kelly_criterion(
                edge_analysis['edge'],
                win_prob,
                market_juice
            )
        else:
            kelly_size = 0.0
            
        edge_analysis['kelly_bet_size'] = kelly_size
        edge_analysis['unit_recommendation'] = self._kelly_to_units(kelly_size)
        
        return edge_analysis
    
    def _kelly_to_units(self, kelly_pct: float) -> str:
        """Convert Kelly percentage to unit recommendation."""
        if kelly_pct <= 0:
            return "0 units"
        elif kelly_pct < 0.01:
            return "0.5 units"
        elif kelly_pct < 0.02:
            return "1 unit"
        elif kelly_pct < 0.03:
            return "1.5 units"
        elif kelly_pct < 0.04:
            return "2 units"
        elif kelly_pct < 0.05:
            return "2.5 units"
        else:
            return "3+ units (max bet)"
    
    def batch_evaluate(
        self,
        projections: list,
        market_lines: list,
        market_juices: list = None
    ) -> list:
        """
        Evaluate multiple props at once.
        
        Args:
            projections: List of projection dictionaries
            market_lines: List of market lines
            market_juices: List of market juices (defaults to -110 for all)
            
        Returns:
            List of evaluation dictionaries
        """
        if market_juices is None:
            market_juices = [-110] * len(projections)
            
        results = []
        for proj, line, juice in zip(projections, market_lines, market_juices):
            results.append(self.evaluate_prop(proj, line, juice))
            
        return results
