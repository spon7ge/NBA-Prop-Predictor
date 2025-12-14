"""
Minutes Projection Model
========================
Minutes are the primary driver of all counting stats.

Key factors affecting minutes:
1. Season average / role (baseline)
2. Game script (blowout risk)
3. Rest patterns (back-to-backs)
4. Matchup (pace)
5. Injury/load management

This module projects expected minutes with uncertainty bounds.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


class MinutesProjector:
    """
    Projects player minutes with uncertainty quantification.
    
    Uses trimmed mean to establish baseline (handles blowout outliers)
    and calculates adjustments for situational factors.
    """
    
    def __init__(self, trim_pct: float = 0.1):
        """
        Args:
            trim_pct: Percentage to trim from each tail when calculating baseline
        """
        self.trim_pct = trim_pct
        self.player_params: Dict[int, Dict] = {}
        self.fitted = False
        
    def fit(self, game_logs: pd.DataFrame) -> 'MinutesProjector':
        """
        Fit minutes model from player game logs.
        
        Expected columns: player_id, minutes
        Optional columns: margin (final game margin), is_b2b (back-to-back flag)
        
        Args:
            game_logs: DataFrame with player game log data
            
        Returns:
            self for method chaining
        """
        required_cols = ['player_id', 'minutes']
        for col in required_cols:
            if col not in game_logs.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for player_id, group in game_logs.groupby('player_id'):
            mins = group['minutes'].values
            
            if len(mins) < 5:
                continue
                
            # Calculate robust baseline using trimmed mean
            baseline = stats.trim_mean(mins, self.trim_pct)
            
            # Calculate volatility (critical for prop betting)
            volatility = np.std(mins)
            
            # Blowout adjustment - minutes lost in blowout games
            blowout_effect = self._calculate_blowout_effect(group, mins)
            
            # Back-to-back adjustment
            b2b_effect = self._calculate_b2b_effect(group, mins)
            
            self.player_params[player_id] = {
                'baseline': baseline,
                'volatility': volatility,
                'blowout_effect': blowout_effect,
                'b2b_effect': b2b_effect,
                'games_played': len(mins),
                'min_minutes': np.min(mins),
                'max_minutes': np.max(mins),
            }
            
        self.fitted = True
        return self
    
    def _calculate_blowout_effect(self, group: pd.DataFrame, mins: np.ndarray) -> float:
        """Calculate minutes reduction in blowout games."""
        if 'margin' in group.columns:
            blowout_mask = np.abs(group['margin'].values) > 20
            if blowout_mask.sum() >= 3:
                return mins[blowout_mask].mean() - mins[~blowout_mask].mean()
        return -3.0  # Default assumption: ~3 minutes lost in blowouts
    
    def _calculate_b2b_effect(self, group: pd.DataFrame, mins: np.ndarray) -> float:
        """Calculate minutes reduction in back-to-back games."""
        if 'is_b2b' in group.columns:
            b2b_mask = group['is_b2b'].values == 1
            if b2b_mask.sum() >= 2:
                return mins[b2b_mask].mean() - mins[~b2b_mask].mean()
        return -2.0  # Default assumption: ~2 minutes reduction on B2B
    
    def project(
        self,
        player_id: int,
        is_b2b: bool = False,
        blowout_prob: float = 0.15,
        custom_adjustments: float = 0.0
    ) -> Optional[Dict]:
        """
        Project minutes with uncertainty bounds.
        
        Args:
            player_id: Player identifier
            is_b2b: Whether game is on back-to-back
            blowout_prob: Probability of blowout (0-1)
            custom_adjustments: Additional minutes adjustment (e.g., injury status)
            
        Returns:
            Dictionary with projection details or None if player not found
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before projecting")
            
        if player_id not in self.player_params:
            return None
            
        params = self.player_params[player_id]
        
        # Start with baseline
        expected = params['baseline']
        
        # Apply adjustments
        if is_b2b:
            expected += params['b2b_effect']
            
        # Expected blowout deduction (probability-weighted)
        expected += params['blowout_effect'] * blowout_prob
        
        # Custom adjustments
        expected += custom_adjustments
        
        # Ensure reasonable bounds
        expected = np.clip(expected, 0, 48)
        
        # Calculate uncertainty bounds (90% confidence interval)
        vol = params['volatility']
        lower_90 = max(0, expected - 1.645 * vol)
        upper_90 = min(48, expected + 1.645 * vol)
        
        return {
            'expected': expected,
            'lower_90': lower_90,
            'upper_90': upper_90,
            'volatility': vol,
            'baseline': params['baseline'],
            'sample_size': params['games_played'],
            'distribution': 'truncated_normal',
            'adjustments': {
                'b2b': params['b2b_effect'] if is_b2b else 0,
                'blowout': params['blowout_effect'] * blowout_prob,
                'custom': custom_adjustments,
            }
        }
    
    def get_player_params(self, player_id: int) -> Optional[Dict]:
        """Get fitted parameters for a player."""
        return self.player_params.get(player_id)
    
    def get_all_player_ids(self) -> list:
        """Get list of all fitted player IDs."""
        return list(self.player_params.keys())
