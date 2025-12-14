"""
Utility Functions
=================
Helper functions for data generation, testing, and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def generate_synthetic_game_logs(
    n_players: int = 50,
    games_per_player: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic NBA game log data for testing.
    
    Creates players with three archetypes: stars, starters, and role players,
    each with appropriate baseline stats and variance.
    
    Args:
        n_players: Number of players to generate
        games_per_player: Number of games per player
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic game logs
    """
    np.random.seed(seed)
    
    data = []
    
    for player_id in range(1, n_players + 1):
        # Player archetype
        archetype = np.random.choice(
            ['star', 'starter', 'role'],
            p=[0.15, 0.35, 0.5]
        )
        
        # Set baselines by archetype
        if archetype == 'star':
            base_mins = np.random.uniform(32, 36)
            base_fga = np.random.uniform(18, 24)
            base_fg3a = np.random.uniform(5, 10)
            base_fta = np.random.uniform(6, 10)
            fg_pct = np.random.uniform(0.45, 0.55)
            fg3_pct = np.random.uniform(0.35, 0.42)
            ft_pct = np.random.uniform(0.80, 0.92)
        elif archetype == 'starter':
            base_mins = np.random.uniform(26, 32)
            base_fga = np.random.uniform(10, 16)
            base_fg3a = np.random.uniform(3, 7)
            base_fta = np.random.uniform(3, 6)
            fg_pct = np.random.uniform(0.43, 0.50)
            fg3_pct = np.random.uniform(0.33, 0.40)
            ft_pct = np.random.uniform(0.75, 0.85)
        else:  # role player
            base_mins = np.random.uniform(18, 26)
            base_fga = np.random.uniform(5, 10)
            base_fg3a = np.random.uniform(1, 4)
            base_fta = np.random.uniform(1, 3)
            fg_pct = np.random.uniform(0.42, 0.48)
            fg3_pct = np.random.uniform(0.32, 0.38)
            ft_pct = np.random.uniform(0.70, 0.82)
            
        for game in range(games_per_player):
            # Situational factors
            is_b2b = np.random.random() < 0.15
            is_blowout = np.random.random() < 0.12
            
            # Minutes with situational adjustments
            mins_adj = -3 if is_b2b else 0
            mins_adj += -8 if is_blowout else 0
            minutes = max(10, base_mins + mins_adj + np.random.normal(0, 4))
            
            # Shot attempts (scale with minutes)
            mins_ratio = minutes / base_mins
            fga = max(1, base_fga * mins_ratio + np.random.normal(0, 3))
            fg3a = max(0, min(fga, base_fg3a * mins_ratio + np.random.normal(0, 2)))
            fta = max(0, base_fta * mins_ratio + np.random.normal(0, 2))
            
            # Made shots with game-level efficiency variance
            game_fg_adj = np.random.normal(0, 0.08)
            game_3pt_adj = np.random.normal(0, 0.12)
            game_ft_adj = np.random.normal(0, 0.06)
            
            fgm = np.random.binomial(
                int(fga),
                np.clip(fg_pct + game_fg_adj, 0.2, 0.7)
            )
            fg3m = np.random.binomial(
                int(fg3a),
                np.clip(fg3_pct + game_3pt_adj, 0.15, 0.6)
            )
            ftm = np.random.binomial(
                int(fta),
                np.clip(ft_pct + game_ft_adj, 0.5, 0.98)
            )
            
            # Calculate points
            pts = 2 * (fgm - fg3m) + 3 * fg3m + ftm
            
            # Random game margin for blowout detection
            if is_blowout:
                margin = np.random.choice([-1, 1]) * np.random.uniform(20, 35)
            else:
                margin = np.random.normal(0, 12)
            
            data.append({
                'player_id': player_id,
                'game_id': game,
                'archetype': archetype,
                'minutes': round(minutes, 1),
                'fga': int(fga),
                'fg3a': int(fg3a),
                'fta': int(fta),
                'fgm': int(fgm),
                'fg3m': int(fg3m),
                'ftm': int(ftm),
                'pts': int(pts),
                'is_b2b': int(is_b2b),
                'margin': round(margin, 1),
            })
            
    return pd.DataFrame(data)


def calculate_metrics(
    predictions: List[float],
    actuals: List[float]
) -> Dict:
    """
    Calculate prediction accuracy metrics.
    
    Args:
        predictions: List of predicted values
        actuals: List of actual values
        
    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    errors = predictions - actuals
    
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    bias = errors.mean()
    
    # Correlation
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'correlation': correlation,
        'n_predictions': len(predictions),
    }


def format_odds(american_odds: int) -> str:
    """Format American odds with proper sign."""
    if american_odds > 0:
        return f"+{american_odds}"
    return str(american_odds)


def calculate_vig(over_odds: int, under_odds: int) -> float:
    """
    Calculate the vigorish (vig) from over/under odds.
    
    Args:
        over_odds: American odds for over
        under_odds: American odds for under
        
    Returns:
        Vig as decimal (e.g., 0.0476 for 4.76%)
    """
    def american_to_implied(odds):
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        return 100 / (odds + 100)
    
    over_implied = american_to_implied(over_odds)
    under_implied = american_to_implied(under_odds)
    
    # Vig is the excess over 100%
    return over_implied + under_implied - 1
