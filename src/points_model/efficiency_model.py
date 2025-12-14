"""
Efficiency Model
================
Models shooting efficiency (FG%, FG3%, FT%) with Bayesian shrinkage.

Key insight: Efficiency is highly variable game-to-game but mean-reverts 
over time. We use:
- Career/season baseline
- Bayesian shrinkage toward league average (more shrinkage for small samples)
- NO recency weighting (efficiency is noisy and reverts to mean)

This approach is more conservative than just using season averages,
which is appropriate for prop betting where we want calibrated uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class EfficiencyModel:
    """
    Models shooting efficiency with Bayesian shrinkage toward league averages.
    
    The shrinkage factor depends on sample size - players with fewer games
    are regressed more heavily toward league average.
    """
    
    # League average efficiency baselines (update each season)
    LEAGUE_FG_PCT = 0.470
    LEAGUE_3PT_PCT = 0.360
    LEAGUE_FT_PCT = 0.780
    
    def __init__(self, shrinkage_games: int = 20):
        """
        Args:
            shrinkage_games: Number of games for full weight on player's own efficiency
                            (fewer games = more regression to league average)
        """
        self.shrinkage_games = shrinkage_games
        self.player_efficiency: Dict[int, Dict] = {}
        self.fitted = False
        
    def fit(self, game_logs: pd.DataFrame) -> 'EfficiencyModel':
        """
        Calculate efficiency baselines with Bayesian shrinkage.
        
        Expected columns: player_id, fga, fgm (or fg_pct), fg3a, fg3m (or fg3_pct), fta, ftm (or ft_pct)
        
        Args:
            game_logs: DataFrame with player game log data
            
        Returns:
            self for method chaining
        """
        required_cols = ['player_id', 'fga', 'fg3a', 'fta']
        for col in required_cols:
            if col not in game_logs.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for player_id, group in game_logs.groupby('player_id'):
            n_games = len(group)
            
            if n_games < 3:
                continue
            
            # Calculate totals for efficiency
            total_fga = group['fga'].sum()
            total_fg3a = group['fg3a'].sum()
            total_fta = group['fta'].sum()
            
            # Get made shots (handle both column naming conventions)
            total_fgm = self._get_made_total(group, 'fgm', 'fga', 'fg_pct', self.LEAGUE_FG_PCT)
            total_fg3m = self._get_made_total(group, 'fg3m', 'fg3a', 'fg3_pct', self.LEAGUE_3PT_PCT)
            total_ftm = self._get_made_total(group, 'ftm', 'fta', 'ft_pct', self.LEAGUE_FT_PCT)
            
            # Raw shooting percentages
            raw_fg_pct = total_fgm / total_fga if total_fga > 0 else self.LEAGUE_FG_PCT
            raw_fg3_pct = total_fg3m / total_fg3a if total_fg3a > 0 else self.LEAGUE_3PT_PCT
            raw_ft_pct = total_ftm / total_fta if total_fta > 0 else self.LEAGUE_FT_PCT
            
            # Calculate shrinkage factor (0 to 1)
            # More games = higher weight on player's actual performance
            shrink_factor = min(n_games / self.shrinkage_games, 1.0)
            
            # Apply Bayesian shrinkage
            shrunk_fg_pct = raw_fg_pct * shrink_factor + self.LEAGUE_FG_PCT * (1 - shrink_factor)
            shrunk_fg3_pct = raw_fg3_pct * shrink_factor + self.LEAGUE_3PT_PCT * (1 - shrink_factor)
            shrunk_ft_pct = raw_ft_pct * shrink_factor + self.LEAGUE_FT_PCT * (1 - shrink_factor)
            
            self.player_efficiency[player_id] = {
                # Shrunk (Bayesian) estimates - USE THESE
                'fg_pct': shrunk_fg_pct,
                'fg3_pct': shrunk_fg3_pct,
                'ft_pct': shrunk_ft_pct,
                # Raw estimates (for reference)
                'raw_fg_pct': raw_fg_pct,
                'raw_fg3_pct': raw_fg3_pct,
                'raw_ft_pct': raw_ft_pct,
                # Metadata
                'shrinkage_factor': shrink_factor,
                'sample_size': n_games,
                'total_fga': total_fga,
                'total_fg3a': total_fg3a,
                'total_fta': total_fta,
            }
            
        self.fitted = True
        return self
    
    def _get_made_total(
        self, 
        group: pd.DataFrame, 
        made_col: str, 
        attempt_col: str, 
        pct_col: str,
        default_pct: float
    ) -> float:
        """Helper to get made shots total from various column formats."""
        if made_col in group.columns:
            return group[made_col].sum()
        elif pct_col in group.columns:
            return (group[pct_col] * group[attempt_col]).sum()
        else:
            # Estimate from league average
            return group[attempt_col].sum() * default_pct
    
    def get_efficiency(self, player_id: int) -> Optional[Dict]:
        """
        Get efficiency estimates for a player.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Dictionary with efficiency estimates or None if not found
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before getting efficiency")
            
        return self.player_efficiency.get(player_id)
    
    def get_efficiency_with_matchup(
        self,
        player_id: int,
        opp_fg_adjustment: float = 0.0,
        opp_3pt_adjustment: float = 0.0,
        opp_ft_adjustment: float = 0.0
    ) -> Optional[Dict]:
        """
        Get efficiency estimates with opponent adjustments.
        
        Args:
            player_id: Player identifier
            opp_fg_adjustment: Adjustment to FG% based on opponent defense (e.g., -0.02 for elite defense)
            opp_3pt_adjustment: Adjustment to 3PT% based on opponent perimeter defense
            opp_ft_adjustment: Adjustment to FT% (rarely needed)
            
        Returns:
            Dictionary with adjusted efficiency estimates
        """
        base = self.get_efficiency(player_id)
        if base is None:
            return None
            
        return {
            'fg_pct': np.clip(base['fg_pct'] + opp_fg_adjustment, 0.25, 0.70),
            'fg3_pct': np.clip(base['fg3_pct'] + opp_3pt_adjustment, 0.20, 0.50),
            'ft_pct': np.clip(base['ft_pct'] + opp_ft_adjustment, 0.50, 0.98),
            'adjustments': {
                'fg': opp_fg_adjustment,
                'fg3': opp_3pt_adjustment,
                'ft': opp_ft_adjustment,
            },
            'base_efficiency': base,
        }
    
    def get_league_averages(self) -> Dict:
        """Get league average efficiency values."""
        return {
            'fg_pct': self.LEAGUE_FG_PCT,
            'fg3_pct': self.LEAGUE_3PT_PCT,
            'ft_pct': self.LEAGUE_FT_PCT,
        }
