"""
Volatility Analyzer
===================
Analyzes player performance volatility and distribution characteristics.

Key insight: Points props are priced assuming normal distributions,
but actual distributions often have fatter tails (more extreme outcomes).

This creates systematic edge opportunities when the true distribution
differs significantly from what books assume.

Edge Sources Identified:
1. Minutes volatility underpriced by books
2. Fat tails (more extreme outcomes than normal)
3. Recency bias / mean reversion
4. Skewness in distributions
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


class VolatilityAnalyzer:
    """
    Analyzes and exploits volatility in player performance.
    """
    
    def __init__(self):
        self.player_distributions: Dict[int, Dict] = {}
        
    def fit_distribution(self, game_logs: pd.DataFrame, stat_col: str = 'pts') -> Dict:
        """
        Fit various distributions to historical performance and find best fit.
        
        Args:
            game_logs: DataFrame with player game log data
            stat_col: Column to analyze (default: 'pts')
            
        Returns:
            Distribution fit analysis dictionary
        """
        values = game_logs[stat_col].values
        
        if len(values) < 15:
            return {'error': 'insufficient_data', 'sample_size': len(values)}
            
        # Fit normal distribution
        norm_params = stats.norm.fit(values)
        norm_ks = stats.kstest(values, 'norm', norm_params)
        
        # Fit Student's t (fatter tails)
        t_params = stats.t.fit(values)
        t_ks = stats.kstest(values, 't', t_params)
        
        # Fit skew normal
        skewnorm_params = stats.skewnorm.fit(values)
        skewnorm_ks = stats.kstest(values, 'skewnorm', skewnorm_params)
        
        # Analyze tails vs normal assumption
        mean, std = norm_params
        tail_analysis = self._analyze_tails(values, mean, std)
        
        # Determine best fit based on KS test p-value
        fits = [
            ('normal', norm_ks.pvalue),
            ('student_t', t_ks.pvalue),
            ('skew_normal', skewnorm_ks.pvalue)
        ]
        best_fit = max(fits, key=lambda x: x[1])
        
        return {
            'mean': mean,
            'std': std,
            'sample_size': len(values),
            'normal_fit': {
                'params': norm_params,
                'ks_pvalue': norm_ks.pvalue,
            },
            't_fit': {
                'params': t_params,
                'ks_pvalue': t_ks.pvalue,
            },
            'skewnorm_fit': {
                'params': skewnorm_params,
                'ks_pvalue': skewnorm_ks.pvalue,
            },
            'tail_analysis': tail_analysis,
            'kurtosis': stats.kurtosis(values),
            'skewness': stats.skew(values),
            'best_fit': best_fit[0],
        }
    
    def _analyze_tails(self, values: np.ndarray, mean: float, std: float) -> Dict:
        """Analyze actual tail probabilities vs normal assumption."""
        # Upper tail: P(X > mean + 1.5*std)
        upper_threshold = mean + 1.5 * std
        normal_upper_tail = 1 - stats.norm.cdf(upper_threshold, mean, std)
        actual_upper_tail = (values > upper_threshold).mean()
        
        # Lower tail: P(X < mean - 1.5*std)
        lower_threshold = mean - 1.5 * std
        normal_lower_tail = stats.norm.cdf(lower_threshold, mean, std)
        actual_lower_tail = (values < lower_threshold).mean()
        
        return {
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'upper_tail_normal': normal_upper_tail,
            'upper_tail_actual': actual_upper_tail,
            'upper_tail_ratio': actual_upper_tail / normal_upper_tail if normal_upper_tail > 0 else None,
            'lower_tail_normal': normal_lower_tail,
            'lower_tail_actual': actual_lower_tail,
            'lower_tail_ratio': actual_lower_tail / normal_lower_tail if normal_lower_tail > 0 else None,
            'fat_upper_tail': actual_upper_tail > normal_upper_tail * 1.5,
            'fat_lower_tail': actual_lower_tail > normal_lower_tail * 1.5,
        }
    
    def calculate_true_probability(
        self,
        game_logs: pd.DataFrame,
        line: float,
        stat_col: str = 'pts',
        use_empirical: bool = True
    ) -> Dict:
        """
        Calculate true probability of going over a line using better
        distribution assumptions than books typically use.
        
        Args:
            game_logs: DataFrame with game log data
            line: Prop line to evaluate
            stat_col: Statistic column
            use_empirical: Whether to include empirical probability
            
        Returns:
            Probability analysis dictionary
        """
        values = game_logs[stat_col].values
        
        # Empirical probability (most accurate with enough data)
        empirical_over = (values > line).mean()
        empirical_under = (values <= line).mean()
        
        # Normal assumption (what books likely use)
        mean, std = np.mean(values), np.std(values)
        normal_over = 1 - stats.norm.cdf(line, mean, std)
        normal_under = stats.norm.cdf(line, mean, std)
        
        # Student's t assumption (fatter tails)
        t_params = stats.t.fit(values)
        t_over = 1 - stats.t.cdf(line, *t_params)
        t_under = stats.t.cdf(line, *t_params)
        
        # Edge: difference between true probability and normal assumption
        over_edge_vs_normal = empirical_over - normal_over
        under_edge_vs_normal = empirical_under - normal_under
        
        return {
            'line': line,
            'sample_size': len(values),
            'empirical': {
                'over': empirical_over,
                'under': empirical_under,
            },
            'normal': {
                'over': normal_over,
                'under': normal_under,
            },
            't_dist': {
                'over': t_over,
                'under': t_under,
            },
            'edge_vs_normal': {
                'over': over_edge_vs_normal,
                'under': under_edge_vs_normal,
            },
            'mean': mean,
            'std': std,
        }
    
    def detect_minutes_volatility_edge(
        self,
        game_logs: pd.DataFrame,
        market_assumes_std: float = 3.0
    ) -> Dict:
        """
        Detect edge from minutes volatility being different than market assumes.
        
        Books often assume low minutes variance (~3 min std).
        Reality: Most players have 4-6 min std, especially role players.
        
        Args:
            game_logs: DataFrame with game log data
            market_assumes_std: What market assumes for minutes std
            
        Returns:
            Minutes volatility edge analysis
        """
        actual_std = game_logs['minutes'].std()
        variance_ratio = actual_std / market_assumes_std
        
        edge_direction = None
        edge_magnitude = 0.0
        
        if variance_ratio > 1.3:
            edge_direction = "MINUTES_VOLATILITY_HIGH"
            edge_magnitude = (variance_ratio - 1) * 0.02  # 2% edge per 100% excess variance
        elif variance_ratio < 0.7:
            edge_direction = "MINUTES_VOLATILITY_LOW"
            edge_magnitude = (1 - variance_ratio) * 0.015
            
        return {
            'actual_std': actual_std,
            'assumed_std': market_assumes_std,
            'variance_ratio': variance_ratio,
            'edge_direction': edge_direction,
            'edge_magnitude': edge_magnitude,
        }
    
    def detect_recency_bias_edge(
        self,
        game_logs: pd.DataFrame,
        stat_col: str = 'pts',
        lookback_short: int = 5,
        lookback_long: int = 20
    ) -> Dict:
        """
        Detect mean reversion opportunity from recency bias.
        
        Public overweights recent performance (last 5 games).
        When recent diverges significantly from season baseline, expect reversion.
        
        Args:
            game_logs: DataFrame sorted by date (most recent first)
            stat_col: Statistic to analyze
            lookback_short: Recent games window
            lookback_long: Season baseline window
            
        Returns:
            Recency bias edge analysis
        """
        if len(game_logs) < lookback_long:
            return {'edge_direction': None, 'reason': 'insufficient_data'}
            
        recent_avg = game_logs[stat_col].head(lookback_short).mean()
        season_avg = game_logs[stat_col].head(lookback_long).mean()
        season_std = game_logs[stat_col].head(lookback_long).std()
        
        # Z-score of recent vs season
        z_score = (recent_avg - season_avg) / (season_std / np.sqrt(lookback_short))
        
        edge_direction = None
        edge_magnitude = 0.0
        
        if z_score > 1.5:
            edge_direction = "UNDER"  # Hot streak likely to cool
            edge_magnitude = min(0.05, (z_score - 1.5) * 0.02)
        elif z_score < -1.5:
            edge_direction = "OVER"  # Cold streak likely to warm
            edge_magnitude = min(0.05, (-z_score - 1.5) * 0.02)
            
        return {
            'recent_avg': recent_avg,
            'season_avg': season_avg,
            'z_score': z_score,
            'edge_direction': edge_direction,
            'edge_magnitude': edge_magnitude,
            'explanation': f"Recent {lookback_short}G avg ({recent_avg:.1f}) vs Season ({season_avg:.1f})",
        }
    
    def aggregate_edges(self, edges: list) -> Dict:
        """
        Combine multiple edge sources into unified recommendation.
        
        Args:
            edges: List of edge analysis dictionaries
            
        Returns:
            Aggregated edge recommendation
        """
        over_edges = []
        under_edges = []
        
        for edge in edges:
            direction = edge.get('edge_direction') or edge.get('recommended')
            magnitude = edge.get('edge_magnitude', 0.02)
            
            if direction == 'OVER':
                over_edges.append(magnitude)
            elif direction == 'UNDER':
                under_edges.append(magnitude)
                
        total_over = sum(over_edges)
        total_under = sum(under_edges)
        net_edge = total_over - total_under
        
        if abs(net_edge) < 0.02:
            recommendation = "NO_EDGE"
            confidence = "LOW"
        elif net_edge > 0:
            recommendation = "OVER"
            confidence = "HIGH" if net_edge > 0.05 else "MEDIUM"
        else:
            recommendation = "UNDER"
            confidence = "HIGH" if abs(net_edge) > 0.05 else "MEDIUM"
            
        return {
            'total_over_edge': total_over,
            'total_under_edge': total_under,
            'net_edge': net_edge,
            'recommendation': recommendation,
            'confidence': confidence,
            'edge_sources': len(over_edges) + len(under_edges),
        }
