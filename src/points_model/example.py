"""
Example Usage
=============
Demonstrates how to use the Points Prop Model.

Run this script to see the full pipeline in action:
    python -m src.points_model.example
"""

import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.points_model import PointsPropModel
from src.points_model.main import print_projection_report
from src.points_model.utils import generate_synthetic_game_logs


def main():
    """Run complete example demonstration."""
    print("=" * 60)
    print("NBA Points Prop Quantitative Model")
    print("=" * 60)
    
    # 1. Generate synthetic data for demonstration
    print("\n1. Generating synthetic game log data...")
    game_logs = generate_synthetic_game_logs(n_players=50, games_per_player=30)
    print(f"   Created {len(game_logs)} game logs for {game_logs['player_id'].nunique()} players")
    
    # 2. Initialize and fit model
    print("\n2. Fitting model components...")
    model = PointsPropModel(
        min_edge=0.03,      # 3% minimum edge
        min_confidence=0.55  # 55% minimum confidence
    )
    model.fit(game_logs)
    print("   ✓ Minutes model fitted")
    print("   ✓ Volume model fitted")
    print("   ✓ Efficiency model fitted")
    
    # 3. Get player summary
    print("\n3. Player Summary (Player ID: 1)")
    summary = model.get_player_summary(player_id=1)
    if summary:
        print(f"   Minutes baseline: {summary['minutes']['baseline']:.1f}")
        print(f"   Minutes volatility: {summary['minutes']['volatility']:.1f}")
        print(f"   FGA per minute: {summary['volume']['fga_per_min']:.2f}")
        print(f"   FG3A per minute: {summary['volume']['fg3a_per_min']:.2f}")
        print(f"   FG%: {summary['efficiency']['fg_pct']:.1%}")
        print(f"   FG3%: {summary['efficiency']['fg3_pct']:.1%}")
    
    # 4. Generate projection
    print("\n4. Generating Projection (Player ID: 1)")
    projection = model.project_points(
        player_id=1,
        is_b2b=False,
        blowout_prob=0.15,
        opp_pace=102,
        opp_drtg=112
    )
    
    if projection:
        print(f"   Expected Points: {projection['expected_points']:.1f}")
        print(f"   90% CI: [{projection['lower_90']:.1f}, {projection['upper_90']:.1f}]")
        print(f"   Minutes Projection: {projection['components']['minutes']['expected']:.1f}")
        print(f"\n   Points Breakdown:")
        print(f"     From 2PT: {projection['breakdown']['from_2pt']:.1f} ({projection['breakdown']['pct_from_2pt']:.1%})")
        print(f"     From 3PT: {projection['breakdown']['from_3pt']:.1f} ({projection['breakdown']['pct_from_3pt']:.1%})")
        print(f"     From FT: {projection['breakdown']['from_ft']:.1f} ({projection['breakdown']['pct_from_ft']:.1%})")
    
    # 5. Evaluate a prop bet
    print("\n5. Evaluating Prop: Over/Under 22.5 points @ -110")
    evaluation = model.evaluate_prop(
        player_id=1,
        market_line=22.5,
        market_juice=-110,
        opp_pace=102,
        opp_drtg=112
    )
    
    if evaluation:
        print_projection_report(evaluation)
    
    # 6. Volatility analysis
    print("\n6. Volatility Analysis")
    vol_analysis = model.analyze_volatility(player_id=1)
    if vol_analysis:
        dist = vol_analysis['distribution']
        print(f"   Kurtosis: {dist['kurtosis']:.2f} (>0 = fatter tails than normal)")
        print(f"   Skewness: {dist['skewness']:.2f}")
        print(f"   Best distribution fit: {dist['best_fit']}")
        
        mins_vol = vol_analysis['minutes_volatility']
        print(f"\n   Minutes Volatility:")
        print(f"     Actual std: {mins_vol['actual_std']:.1f}")
        print(f"     Market assumes: {mins_vol['assumed_std']:.1f}")
        if mins_vol['edge_direction']:
            print(f"     Edge: {mins_vol['edge_direction']} ({mins_vol['edge_magnitude']:.1%})")
        
        recency = vol_analysis['recency_bias']
        print(f"\n   Recency Bias:")
        print(f"     Recent avg: {recency['recent_avg']:.1f}")
        print(f"     Season avg: {recency['season_avg']:.1f}")
        print(f"     Z-score: {recency['z_score']:.2f}")
        if recency['edge_direction']:
            print(f"     Edge: {recency['edge_direction']} ({recency['edge_magnitude']:.1%})")
    
    # 7. Find best props
    print("\n7. Finding Best Props (Top 5)")
    
    # Create hypothetical market lines
    market_lines = {}
    for player_id in range(1, 11):
        proj = model.project_points(player_id)
        if proj:
            # Set line slightly off from projection to create some edge
            offset = (-2 if player_id % 2 == 0 else 2)
            market_lines[player_id] = proj['expected_points'] + offset
    
    best_props = model.find_best_props(
        player_ids=list(range(1, 11)),
        market_lines=market_lines,
        min_edge=0.02,
        opp_pace=100,
        opp_drtg=114
    )
    
    print(f"   Found {len(best_props)} props with edge >= 2%")
    for i, prop in enumerate(best_props[:5], 1):
        edge = prop['edge_analysis']
        proj = prop['projection']
        print(f"\n   {i}. Player {prop['player_id']}")
        print(f"      Line: {edge['line']:.1f} | Proj: {proj['expected_points']:.1f}")
        print(f"      Rec: {edge['recommendation']} | Edge: {edge['edge']:.1%}")
        print(f"      Bet Size: {edge['unit_recommendation']}")
    
    print("\n" + "=" * 60)
    print("Model demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
