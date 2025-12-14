"""
Points Projector
================
Combines all component models to generate final points projection.

Points Formula:
PTS = 2 × (FGA - FG3A) × FG% + 3 × FG3A × FG3% + FTA × FT%

This module integrates:
- Minutes projection
- Shot volume projection (FGA, FG3A, FTA)
- Efficiency estimates (FG%, FG3%, FT%)
- Uncertainty propagation
"""

import numpy as np
from typing import Dict, Optional

from .minutes_model import MinutesProjector
from .volume_model import ShotVolumeModel
from .efficiency_model import EfficiencyModel
from .matchup_adjuster import MatchupAdjuster


class PointsProjector:
    """
    Generates comprehensive points projections with uncertainty quantification.
    
    Uses component models for minutes, volume, and efficiency to produce
    a full projection with confidence intervals.
    """
    
    def __init__(
        self,
        minutes_model: MinutesProjector,
        volume_model: ShotVolumeModel,
        efficiency_model: EfficiencyModel,
        matchup_adjuster: MatchupAdjuster = None
    ):
        """
        Args:
            minutes_model: Fitted MinutesProjector
            volume_model: Fitted ShotVolumeModel
            efficiency_model: Fitted EfficiencyModel
            matchup_adjuster: Optional MatchupAdjuster (creates default if None)
        """
        self.minutes_model = minutes_model
        self.volume_model = volume_model
        self.efficiency_model = efficiency_model
        self.matchup_adjuster = matchup_adjuster or MatchupAdjuster()
        
    def project(
        self,
        player_id: int,
        is_b2b: bool = False,
        blowout_prob: float = 0.15,
        opp_pace: float = None,
        opp_drtg: float = None,
        opp_foul_rate: float = None,
        opp_team_id: int = None,
        usage_adjustment: float = 1.0,
        custom_minutes_adj: float = 0.0
    ) -> Optional[Dict]:
        """
        Generate full points projection with uncertainty.
        
        Args:
            player_id: Player identifier
            is_b2b: Whether game is on back-to-back
            blowout_prob: Probability of blowout (0-1)
            opp_pace: Opponent pace (defaults to league average if not provided)
            opp_drtg: Opponent defensive rating (defaults to league average if not provided)
            opp_foul_rate: Opponent foul rate (defaults to league average if not provided)
            opp_team_id: Opponent team ID - if provided, automatically fetches pace/DRTG
            usage_adjustment: Usage multiplier (e.g., 1.1 when star teammate out)
            custom_minutes_adj: Manual minutes adjustment
            
        Returns:
            Comprehensive projection dictionary or None if player not found
        """
        # Step 1: Auto-fetch opponent stats if team_id provided and manual values not set
        if opp_team_id is not None:
            if opp_pace is None:
                opp_pace = self.matchup_adjuster.get_opponent_pace(opp_team_id)
            if opp_drtg is None:
                opp_drtg = self.matchup_adjuster.get_opponent_drtg(opp_team_id)
            # Note: Foul rate not available from NBA API, so keep manual/default
        
        # Step 2: Get matchup adjustments
        matchup = self.matchup_adjuster.get_all_adjustments(opp_pace, opp_drtg, opp_foul_rate)
        
        # Step 3: Project minutes
        mins_proj = self.minutes_model.project(
            player_id,
            is_b2b=is_b2b,
            blowout_prob=blowout_prob,
            custom_adjustments=custom_minutes_adj
        )
        if mins_proj is None:
            return None
            
        # Step 4: Project shot volume
        volume_proj = self.volume_model.project(
            player_id,
            projected_minutes=mins_proj['expected'],
            usage_adjustment=usage_adjustment * matchup['defense'],
            pace_adjustment=matchup['pace'],
            defense_adjustment=1.0  # Already factored into usage
        )
        if volume_proj is None:
            return None
            
        # Step 5: Get efficiency estimates
        efficiency = self.efficiency_model.get_efficiency(player_id)
        if efficiency is None:
            # Use league averages as fallback
            efficiency = self.efficiency_model.get_league_averages()
            efficiency['sample_size'] = 0
            
        # Step 6: Calculate expected points
        expected_points = self._calculate_expected_points(
            fga=volume_proj['fga']['expected'],
            fg3a=volume_proj['fg3a']['expected'],
            fta=volume_proj['fta']['expected'],
            fg_pct=efficiency['fg_pct'],
            fg3_pct=efficiency['fg3_pct'],
            ft_pct=efficiency['ft_pct']
        )
        
        # Step 7: Calculate uncertainty
        points_std = self._calculate_uncertainty(
            expected_points=expected_points,
            mins_volatility=mins_proj['volatility'],
            volume_proj=volume_proj
        )
        
        # Calculate confidence interval
        lower_90 = expected_points - 1.645 * points_std
        upper_90 = expected_points + 1.645 * points_std
        
        return {
            'player_id': player_id,
            'expected_points': expected_points,
            'std': points_std,
            'lower_90': max(0, lower_90),
            'upper_90': upper_90,
            'components': {
                'minutes': mins_proj,
                'volume': {
                    'fga': volume_proj['fga']['expected'],
                    'fg3a': volume_proj['fg3a']['expected'],
                    'fta': volume_proj['fta']['expected'],
                },
                'efficiency': {
                    'fg_pct': efficiency['fg_pct'],
                    'fg3_pct': efficiency['fg3_pct'],
                    'ft_pct': efficiency['ft_pct'],
                },
            },
            'adjustments': {
                'matchup': matchup,
                'usage': usage_adjustment,
                'is_b2b': is_b2b,
                'blowout_prob': blowout_prob,
            },
            'breakdown': self._get_points_breakdown(
                fga=volume_proj['fga']['expected'],
                fg3a=volume_proj['fg3a']['expected'],
                fta=volume_proj['fta']['expected'],
                fg_pct=efficiency['fg_pct'],
                fg3_pct=efficiency['fg3_pct'],
                ft_pct=efficiency['ft_pct']
            )
        }
    
    def _calculate_expected_points(
        self,
        fga: float,
        fg3a: float,
        fta: float,
        fg_pct: float,
        fg3_pct: float,
        ft_pct: float
    ) -> float:
        """
        Calculate expected points from components.
        
        PTS = 2 × (FGA - FG3A) × FG% + 3 × FG3A × FG3% + FTA × FT%
        
        Note: Using overall FG% for 2-pointers is a simplification.
        More accurate would be to use 2PT% specifically.
        """
        fg2a = fga - fg3a  # 2-point attempts
        
        # Expected points from each source
        expected_2pt = fg2a * fg_pct * 2
        expected_3pt = fg3a * fg3_pct * 3
        expected_ft = fta * ft_pct
        
        return expected_2pt + expected_3pt + expected_ft
    
    def _calculate_uncertainty(
        self,
        expected_points: float,
        mins_volatility: float,
        volume_proj: Dict
    ) -> float:
        """
        Calculate uncertainty in points projection.
        
        Uses a simplified variance propagation approach.
        Full Monte Carlo simulation would be more accurate.
        """
        # Base uncertainty from coefficient of variation (~18% for points)
        base_std = expected_points * 0.18
        
        # Additional uncertainty from minutes volatility
        mins_contribution = mins_volatility * 0.5
        
        # Combine (assuming independence)
        return np.sqrt(base_std**2 + mins_contribution**2)
    
    def _get_points_breakdown(
        self,
        fga: float,
        fg3a: float,
        fta: float,
        fg_pct: float,
        fg3_pct: float,
        ft_pct: float
    ) -> Dict:
        """Get detailed breakdown of expected points by source."""
        fg2a = fga - fg3a
        
        pts_2pt = fg2a * fg_pct * 2
        pts_3pt = fg3a * fg3_pct * 3
        pts_ft = fta * ft_pct
        total = pts_2pt + pts_3pt + pts_ft
        
        return {
            'from_2pt': pts_2pt,
            'from_3pt': pts_3pt,
            'from_ft': pts_ft,
            'pct_from_2pt': pts_2pt / total if total > 0 else 0,
            'pct_from_3pt': pts_3pt / total if total > 0 else 0,
            'pct_from_ft': pts_ft / total if total > 0 else 0,
        }
