"""
Matchup Adjuster
================
Calculates adjustments based on opponent characteristics.

Key factors:
1. Opponent pace (possessions per game) - affects volume
2. Opponent defensive rating (DRTG) - affects efficiency and volume
3. Position-specific defense adjustments
4. Foul rate differentials
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from nba_api.stats.endpoints import leaguedashteamstats


class MatchupAdjuster:
    """
    Calculates multipliers for projections based on opponent matchup.
    """
    
    LEAGUE_AVG_PACE = 100.0
    LEAGUE_AVG_DRTG = 114.0 
    LEAGUE_AVG_FOUL_RATE = 21.8  
    
    def __init__(
        self,
        league_avg_pace: float = None,
        league_avg_drtg: float = None,
        league_avg_foul_rate: float = None,
        fetch_team_stats: bool = True
    ):
        """
        Args:
            league_avg_pace: League average pace (possessions per game)
            league_avg_drtg: League average defensive rating
            league_avg_foul_rate: League average foul rate
            fetch_team_stats: If True, fetch current team stats from NBA API
        """
        self.team_stats = None
        self.fouls_per_game = {}  # Dictionary mapping team_id -> fouls per game
        if fetch_team_stats:
            self._fetch_team_stats()
            if self.team_stats is not None:
                # Calculate league averages from fetched data
                self.league_avg_pace = self.team_stats['PACE'].mean() if 'PACE' in self.team_stats.columns else (league_avg_pace or self.LEAGUE_AVG_PACE)
                self.league_avg_drtg = self.team_stats['DEF_RATING'].mean() if 'DEF_RATING' in self.team_stats.columns else (league_avg_drtg or self.LEAGUE_AVG_DRTG)
                # Foul rate might not be in the API response, use default or provided
                self.league_avg_foul_rate = league_avg_foul_rate or self.LEAGUE_AVG_FOUL_RATE
            else:
                # Fallback to defaults if fetch fails
                self.league_avg_pace = league_avg_pace or self.LEAGUE_AVG_PACE
                self.league_avg_drtg = league_avg_drtg or self.LEAGUE_AVG_DRTG
                self.league_avg_foul_rate = league_avg_foul_rate or self.LEAGUE_AVG_FOUL_RATE
        else:
            self.league_avg_pace = league_avg_pace or self.LEAGUE_AVG_PACE
            self.league_avg_drtg = league_avg_drtg or self.LEAGUE_AVG_DRTG
            self.league_avg_foul_rate = league_avg_foul_rate or self.LEAGUE_AVG_FOUL_RATE
    
    def _fetch_team_stats(self):
        """
        Fetch current team stats from NBA API.
        Stores results in self.team_stats as a DataFrame indexed by TEAM_ID.
        """
        try:
            league_df = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable='00',
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            self.team_stats = league_df.set_index('TEAM_ID')
        except Exception as e:
            # If API call fails, set to None and use defaults
            print(f"Warning: Could not fetch team stats from NBA API: {e}")
            self.team_stats = None
    
    def get_team_stats(self, team_id: int = None) -> Optional[pd.Series]:
        """
        Get stats for a specific team by TEAM_ID.
        
        Args:
            team_id: NBA team ID
            
        Returns:
            Series with team stats, or None if not found
        """
        if self.team_stats is None or team_id is None:
            return None
        return self.team_stats.loc[team_id] if team_id in self.team_stats.index else None
    
    def get_opponent_pace(self, team_id: int) -> Optional[float]:
        """
        Get opponent pace by team ID.
        
        Args:
            team_id: NBA team ID
            
        Returns:
            Pace (possessions per game) or None if not found
        """
        team_data = self.get_team_stats(team_id)
        if team_data is not None and 'PACE' in team_data:
            return float(team_data['PACE'])
        return None
    
    def get_opponent_drtg(self, team_id: int) -> Optional[float]:
        """
        Get opponent defensive rating by team ID.
        
        Args:
            team_id: NBA team ID
            
        Returns:
            Defensive rating or None if not found
        """
        team_data = self.get_team_stats(team_id)
        if team_data is not None and 'DEF_RATING' in team_data:
            return float(team_data['DEF_RATING'])
        return None
    
    def set_fouls_per_game(self, fouls_dict: Dict[int, float], league_avg: float = None):
        """
        Set fouls per game data from league_base_df.
        
        Args:
            fouls_dict: Dictionary mapping team_id -> fouls per game
            league_avg: League average fouls per game (auto-calculated if not provided)
        """
        self.fouls_per_game = fouls_dict
        if league_avg is not None:
            self.league_avg_foul_rate = league_avg
        elif fouls_dict:
            # Calculate league average from provided data
            self.league_avg_foul_rate = sum(fouls_dict.values()) / len(fouls_dict)
    
    def get_opponent_foul_rate(self, team_id: int) -> Optional[float]:
        """
        Get opponent foul rate (fouls per game) by team ID.
        
        Args:
            team_id: NBA team ID
            
        Returns:
            Fouls per game (float), or league average if not found
        """
        if not self.fouls_per_game:
            return self.league_avg_foul_rate
        
        team_id_int = int(team_id)
        return self.fouls_per_game.get(team_id_int, self.league_avg_foul_rate)
        
    def pace_adjustment(self, opp_pace: float) -> float:
        """
        Calculate pace multiplier.
        
        More possessions = more shot opportunities.
        Linear scaling relative to league average.
        
        Args:
            opp_pace: Opponent's pace (possessions per game)
            
        Returns:
            Multiplier for shot volume (e.g., 1.05 for 5% faster pace)
        """
        return opp_pace / self.league_avg_pace
    
    def defensive_adjustment(self, opp_drtg: float, impact_factor: float = 0.01) -> float:
        """
        Calculate defensive adjustment multiplier.
        
        Higher DRTG = worse defense = more points expected.
        
        Args:
            opp_drtg: Opponent's defensive rating
            impact_factor: Points adjustment per DRTG point (default: 1% per point)
            
        Returns:
            Multiplier for scoring projection
        """
        drtg_diff = opp_drtg - self.league_avg_drtg
        return 1 + (drtg_diff * impact_factor)
    
    def foul_rate_adjustment(self, opp_foul_rate: float) -> float:
        """
        Calculate free throw opportunity adjustment.
        
        Higher foul rate = more FTA expected.
        
        Args:
            opp_foul_rate: Opponent's foul rate
            
        Returns:
            Multiplier for FTA projection
        """
        return opp_foul_rate / self.league_avg_foul_rate
    
    def get_all_adjustments(
        self,
        opp_pace: float = None,
        opp_drtg: float = None,
        opp_foul_rate: float = None
    ) -> Dict:
        """
        Get all matchup adjustments at once.
        
        Args:
            opp_pace: Opponent's pace (defaults to league average)
            opp_drtg: Opponent's defensive rating (defaults to league average)
            opp_foul_rate: Opponent's foul rate (defaults to league average)
            
        Returns:
            Dictionary with all adjustment multipliers
        """
        pace_adj = self.pace_adjustment(opp_pace) if opp_pace else 1.0
        def_adj = self.defensive_adjustment(opp_drtg) if opp_drtg else 1.0
        foul_adj = self.foul_rate_adjustment(opp_foul_rate) if opp_foul_rate else 1.0
        
        return {
            'pace': pace_adj,
            'defense': def_adj,
            'foul_rate': foul_adj,
            'inputs': {
                'opp_pace': opp_pace or self.league_avg_pace,
                'opp_drtg': opp_drtg or self.league_avg_drtg,
                'opp_foul_rate': opp_foul_rate or self.league_avg_foul_rate,
            }
        }
    
    def estimate_unpriced_pace_effect(
        self,
        player_avg_opp_pace: float,
        tonight_opp_pace: float,
        book_adjustment_pct: float = 0.5
    ) -> Dict:
        """
        Estimate how much pace differential the market hasn't priced in.
        
        Books typically price in ~50% of pace differential. This identifies
        potential edge from pace mismatches.
        
        Args:
            player_avg_opp_pace: Player's typical opponent pace
            tonight_opp_pace: Tonight's opponent pace
            book_adjustment_pct: How much books typically adjust (0.5 = 50%)
            
        Returns:
            Dictionary with unpriced pace effect analysis
        """
        pace_diff_from_usual = (tonight_opp_pace - player_avg_opp_pace) / player_avg_opp_pace
        unpriced_effect = pace_diff_from_usual * (1 - book_adjustment_pct)
        
        edge_direction = None
        if abs(unpriced_effect) > 0.02:
            edge_direction = "OVER" if unpriced_effect > 0 else "UNDER"
            
        return {
            'tonight_pace': tonight_opp_pace,
            'usual_opp_pace': player_avg_opp_pace,
            'pace_differential': pace_diff_from_usual,
            'unpriced_effect': unpriced_effect,
            'edge_direction': edge_direction,
            'edge_magnitude': abs(unpriced_effect),
        }
