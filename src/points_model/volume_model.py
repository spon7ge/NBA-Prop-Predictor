"""
Shot Volume Model
=================
Models shot attempts (FGA, FG3A, FTA) per minute played.

Key insight: Per-minute rates are more stable than totals and allow us 
to separate volume from playing time variation.

Factors:
1. Usage rate baseline
2. Teammate availability (usage increases when stars out)
3. Opponent defense (DRTG, pace)
4. Opponent foul rate (affects FTA)
5. Game script (more shots when trailing)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class ShotVolumeModel:
    """
    Models per-minute shot attempt rates with recency weighting.
    
    Outputs expected FGA, FG3A, and FTA given projected minutes
    and situational adjustments.
    """
    
    def __init__(self, min_minutes_threshold: int = 15, recency_decay: float = 1.0):
        """
        Args:
            min_minutes_threshold: Minimum minutes to include a game in rate calculation
            recency_decay: Exponential decay factor for recency weighting (higher = more recent weight)
        """
        self.min_minutes_threshold = min_minutes_threshold
        self.recency_decay = recency_decay
        self.player_rates: Dict[int, Dict] = {}
        self.fitted = False
        
    def fit(self, game_logs: pd.DataFrame) -> 'ShotVolumeModel':
        """
        Calculate per-minute rates with trend detection.
        
        Expected columns: player_id, minutes, fga, fg3a, fta
        
        Args:
            game_logs: DataFrame with player game log data
            
        Returns:
            self for method chaining
        """
        required_cols = ['player_id', 'minutes', 'fga', 'fg3a', 'fta']
        for col in required_cols:
            if col not in game_logs.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for player_id, group in game_logs.groupby('player_id'):
            # Filter out low-minute games (not representative of typical role)
            meaningful_games = group[group['minutes'] >= self.min_minutes_threshold].copy()
            
            if len(meaningful_games) < 5:
                continue
                
            mins = meaningful_games['minutes'].values
            
            # Calculate per-minute rates
            fga_rate = meaningful_games['fga'].values / mins
            fg3a_rate = meaningful_games['fg3a'].values / mins
            fta_rate = meaningful_games['fta'].values / mins
            
            # Recency-weighted averages (most recent games weighted more heavily)
            weights = np.exp(np.linspace(-self.recency_decay, 0, len(fga_rate)))
            weights = weights / weights.sum()
            
            # Also calculate simple averages for comparison
            simple_fga_rate = np.mean(fga_rate)
            simple_fg3a_rate = np.mean(fg3a_rate)
            simple_fta_rate = np.mean(fta_rate)
            
            self.player_rates[player_id] = {
                # Recency-weighted rates
                'fga_per_min': np.average(fga_rate, weights=weights),
                'fg3a_per_min': np.average(fg3a_rate, weights=weights),
                'fta_per_min': np.average(fta_rate, weights=weights),
                # Volatility of rates
                'fga_vol': np.std(fga_rate),
                'fg3a_vol': np.std(fg3a_rate),
                'fta_vol': np.std(fta_rate),
                # Simple averages (for comparison)
                'fga_simple': simple_fga_rate,
                'fg3a_simple': simple_fg3a_rate,
                'fta_simple': simple_fta_rate,
                # Sample info
                'sample_size': len(meaningful_games),
                'avg_minutes': np.mean(mins),
            }
            
        self.fitted = True
        return self
    
    def project(
        self,
        player_id: int,
        projected_minutes: float,
        usage_adjustment: float = 1.0,
        pace_adjustment: float = 1.0,
        defense_adjustment: float = 1.0,
        foul_rate_adjustment: float = 1.0
    ) -> Optional[Dict]:
        """
        Project shot attempts given minutes projection.
        
        Args:
            player_id: Player identifier
            projected_minutes: Expected minutes from MinutesProjector
            usage_adjustment: Multiplier for usage (e.g., 1.1 when star teammate out)
            pace_adjustment: Multiplier for pace differential
            defense_adjustment: Multiplier for opponent defense
            foul_rate_adjustment: Multiplier for opponent foul rate (affects FTA)
            
        Returns:
            Dictionary with shot volume projections or None if player not found
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before projecting")
            
        if player_id not in self.player_rates:
            return None
            
        rates = self.player_rates[player_id]
        
        # Apply adjustments to rates
        # FGA and FG3A affected by pace and usage
        adj_fga_rate = rates['fga_per_min'] * usage_adjustment * pace_adjustment * defense_adjustment
        adj_fg3a_rate = rates['fg3a_per_min'] * usage_adjustment * pace_adjustment * defense_adjustment
        
        # FTA affected by usage, defense, and opponent foul rate
        adj_fta_rate = rates['fta_per_min'] * usage_adjustment * defense_adjustment * foul_rate_adjustment
        
        # Project totals
        expected_fga = adj_fga_rate * projected_minutes
        expected_fg3a = adj_fg3a_rate * projected_minutes
        expected_fta = adj_fta_rate * projected_minutes
        
        # Calculate uncertainty (propagate through minutes)
        fga_std = rates['fga_vol'] * projected_minutes
        fg3a_std = rates['fg3a_vol'] * projected_minutes
        fta_std = rates['fta_vol'] * projected_minutes
        
        return {
            'fga': {
                'expected': expected_fga,
                'std': fga_std,
                'rate': adj_fga_rate,
                'base_rate': rates['fga_per_min'],
            },
            'fg3a': {
                'expected': expected_fg3a,
                'std': fg3a_std,
                'rate': adj_fg3a_rate,
                'base_rate': rates['fg3a_per_min'],
            },
            'fta': {
                'expected': expected_fta,
                'std': fta_std,
                'rate': adj_fta_rate,
                'base_rate': rates['fta_per_min'],
            },
            'adjustments': {
                'usage': usage_adjustment,
                'pace': pace_adjustment,
                'defense': defense_adjustment,
                'foul_rate': foul_rate_adjustment,
            },
            'sample_size': rates['sample_size'],
        }
    
    def get_player_rates(self, player_id: int) -> Optional[Dict]:
        """Get fitted rates for a player."""
        return self.player_rates.get(player_id)
    
    def get_usage_trend(self, player_id: int) -> Optional[Dict]:
        """
        Compare recency-weighted vs simple average to detect trends.
        
        Returns:
            Dictionary with trend analysis or None
        """
        if player_id not in self.player_rates:
            return None
            
        rates = self.player_rates[player_id]
        
        fga_trend = (rates['fga_per_min'] - rates['fga_simple']) / rates['fga_simple']
        fg3a_trend = (rates['fg3a_per_min'] - rates['fg3a_simple']) / rates['fg3a_simple']
        fta_trend = (rates['fta_per_min'] - rates['fta_simple']) / rates['fta_simple']
        
        return {
            'fga_trend_pct': fga_trend,
            'fg3a_trend_pct': fg3a_trend,
            'fta_trend_pct': fta_trend,
            'trending_up': fga_trend > 0.05,
            'trending_down': fga_trend < -0.05,
        }
