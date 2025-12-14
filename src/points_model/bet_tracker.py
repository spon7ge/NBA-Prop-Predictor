"""
NBA Points Prop Bet Tracker & Performance Analysis
===================================================
Track your bets, measure edge, evaluate model calibration.

Key Metrics:
1. CLV (Closing Line Value) - Are you beating the market?
2. Calibration - Are your probability estimates accurate?
3. ROI by Edge Bucket - Does higher predicted edge = more wins?
4. Component Attribution - Which edge sources are working?

Uses timestamps from player_lines to track opening vs closing lines.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, Optional, List, Tuple
import os


class LineTracker:
    """
    Track line movement using player_lines timestamps.
    
    Uses DATA_PULLED_AT to identify earliest (opening) and latest (closing) lines.
    """
    
    def __init__(self, lines_directory: str = 'data/raw/player_lines'):
        self.lines_directory = lines_directory
        self.line_history = {}
        
    def load_all_lines(self, date_filter: str = None) -> pd.DataFrame:
        """
        Load all player lines files and combine with timestamps.
        
        Args:
            date_filter: Optional date string (YYYYMMDD) to filter files
            
        Returns:
            Combined DataFrame with all lines and timestamps
        """
        all_lines = []
        
        for filename in os.listdir(self.lines_directory):
            if not filename.endswith('.csv'):
                continue
            if date_filter and date_filter not in filename:
                continue
                
            filepath = os.path.join(self.lines_directory, filename)
            df = pd.read_csv(filepath)
            
            # Parse timestamps
            if 'DATA_PULLED_AT' in df.columns:
                df['pulled_at'] = pd.to_datetime(df['DATA_PULLED_AT'])
            if 'LAST_UPDATE' in df.columns:
                df['last_update'] = pd.to_datetime(df['LAST_UPDATE'])
                
            all_lines.append(df)
            
        if not all_lines:
            return pd.DataFrame()
            
        combined = pd.concat(all_lines, ignore_index=True)
        return combined.sort_values('pulled_at')
    
    def get_opening_closing_lines(
        self, 
        player_name: str, 
        prop_type: str = 'player_points',
        game_date: str = None
    ) -> Dict:
        """
        Get opening and closing lines for a player prop.
        
        Args:
            player_name: Player name
            prop_type: Prop category (default: player_points)
            game_date: Game date to filter (YYYY-MM-DD format)
            
        Returns:
            Dictionary with opening/closing line info by bookmaker
        """
        lines = self.load_all_lines()
        
        if lines.empty:
            return {}
            
        # Filter to this player and prop type
        mask = (lines['NAME'] == player_name) & (lines['CATEGORY'] == prop_type)
        if game_date:
            mask &= (lines['COMMENCE_TIME'] == game_date)
            
        player_lines = lines[mask].copy()
        
        if player_lines.empty:
            return {}
            
        results = {}
        
        for book in player_lines['BOOKMAKER'].unique():
            book_lines = player_lines[player_lines['BOOKMAKER'] == book].sort_values('pulled_at')
            
            # Get over lines
            over_lines = book_lines[book_lines['OVER/UNDER'] == 'Over']
            under_lines = book_lines[book_lines['OVER/UNDER'] == 'Under']
            
            if len(over_lines) > 0:
                opening_over = over_lines.iloc[0]
                closing_over = over_lines.iloc[-1]
                
                results[book] = {
                    'opening_line': opening_over['LINE'],
                    'opening_over_odds': opening_over['ODDS'],
                    'opening_time': opening_over['pulled_at'],
                    'closing_line': closing_over['LINE'],
                    'closing_over_odds': closing_over['ODDS'],
                    'closing_time': closing_over['pulled_at'],
                    'line_moved': closing_over['LINE'] != opening_over['LINE'],
                    'line_movement': closing_over['LINE'] - opening_over['LINE'],
                }
                
                if len(under_lines) > 0:
                    results[book]['opening_under_odds'] = under_lines.iloc[0]['ODDS']
                    results[book]['closing_under_odds'] = under_lines.iloc[-1]['ODDS']
                    
        return results
    
    def calculate_clv(
        self,
        player_name: str,
        side: str,
        line_at_bet: float,
        game_date: str = None
    ) -> Dict:
        """
        Calculate Closing Line Value.
        
        CLV = closing_line - line_at_bet (adjusted for side)
        Positive CLV means you got a better number than the market settled on.
        
        Args:
            player_name: Player name
            side: 'OVER' or 'UNDER'
            line_at_bet: The line when you placed your bet
            game_date: Game date
            
        Returns:
            CLV analysis dictionary
        """
        line_info = self.get_opening_closing_lines(player_name, game_date=game_date)
        
        if not line_info:
            return {'clv': None, 'error': 'No line data found'}
            
        # Use consensus closing line (average across books)
        closing_lines = [info['closing_line'] for info in line_info.values()]
        avg_closing = np.mean(closing_lines)
        
        # CLV direction depends on side
        if side == 'OVER':
            # For overs, lower closing line = positive CLV (you got better number)
            clv = line_at_bet - avg_closing
        else:
            # For unders, higher closing line = positive CLV
            clv = avg_closing - line_at_bet
            
        return {
            'clv': clv,
            'line_at_bet': line_at_bet,
            'closing_line': avg_closing,
            'side': side,
            'books_tracked': len(line_info),
            'line_movement_by_book': {
                book: info['line_movement'] 
                for book, info in line_info.items()
            }
        }


class PropBetTracker:
    """
    Track and analyze prop bet performance.
    """
    
    def __init__(self, filepath: str = 'data/prop_bets.csv', lines_dir: str = 'data/raw/player_lines'):
        self.filepath = filepath
        self.bets = self._load_bets()
        self.line_tracker = LineTracker(lines_dir)
        
    def _load_bets(self) -> pd.DataFrame:
        """Load existing bets or create empty DataFrame."""
        if os.path.exists(self.filepath):
            return pd.read_csv(self.filepath, parse_dates=['date', 'bet_time'])
        else:
            return pd.DataFrame(columns=[
                'date', 'bet_time', 'player', 'player_id', 'team', 'opponent',
                'prop_type', 'side', 'bookmaker',
                'line_at_bet', 'odds_at_bet', 'opening_line', 'closing_line', 'closing_odds',
                'projection', 'projection_std', 'predicted_prob', 'predicted_edge',
                'actual_result', 'won', 'units_bet', 'units_won',
                'mins_proj', 'mins_actual', 'fga_proj', 'fga_actual',
                'opp_pace', 'opp_drtg',
                'edge_sources', 'notes'
            ])
    
    def add_bet(self, bet_data: dict) -> int:
        """
        Add a new bet to the tracker.
        
        Required fields:
        - date, player, prop_type, side, line_at_bet, odds_at_bet
        - projection, projection_std, predicted_edge
        
        Optional (fill in after game):
        - closing_line, closing_odds, actual_result, won
        """
        # Add bet timestamp
        if 'bet_time' not in bet_data:
            bet_data['bet_time'] = datetime.now()
            
        # Calculate predicted probability if not provided
        if 'predicted_prob' not in bet_data:
            if all(k in bet_data for k in ['projection', 'projection_std', 'line_at_bet']):
                z = (bet_data['line_at_bet'] - bet_data['projection']) / bet_data['projection_std']
                if bet_data['side'] == 'OVER':
                    bet_data['predicted_prob'] = 1 - stats.norm.cdf(z)
                else:
                    bet_data['predicted_prob'] = stats.norm.cdf(z)
        
        # Default units
        if 'units_bet' not in bet_data:
            bet_data['units_bet'] = 1.0
        
        # Try to get opening line from line tracker
        if 'opening_line' not in bet_data:
            line_info = self.line_tracker.get_opening_closing_lines(
                bet_data.get('player'),
                game_date=str(bet_data.get('date'))[:10] if bet_data.get('date') else None
            )
            if line_info:
                # Use first bookmaker's opening line as reference
                first_book = list(line_info.keys())[0]
                bet_data['opening_line'] = line_info[first_book]['opening_line']
            
        new_bet = pd.DataFrame([bet_data])
        self.bets = pd.concat([self.bets, new_bet], ignore_index=True)
        self._save_bets()
        
        print(f"✓ Added bet: {bet_data['player']} {bet_data['side']} {bet_data['line_at_bet']} pts")
        return len(self.bets) - 1  # Return bet index
    
    def update_result(
        self, 
        bet_index: int, 
        actual_result: float,
        closing_line: float = None, 
        closing_odds: int = None,
        mins_actual: float = None, 
        fga_actual: float = None
    ):
        """Update a bet with actual results after the game."""
        
        bet = self.bets.loc[bet_index]
        
        self.bets.loc[bet_index, 'actual_result'] = actual_result
        
        # Try to auto-fetch closing line if not provided
        if closing_line is None:
            clv_info = self.line_tracker.calculate_clv(
                bet['player'],
                bet['side'],
                bet['line_at_bet'],
                str(bet['date'])[:10] if pd.notna(bet['date']) else None
            )
            if clv_info.get('closing_line'):
                closing_line = clv_info['closing_line']
        
        if closing_line is not None:
            self.bets.loc[bet_index, 'closing_line'] = closing_line
        if closing_odds is not None:
            self.bets.loc[bet_index, 'closing_odds'] = closing_odds
        if mins_actual is not None:
            self.bets.loc[bet_index, 'mins_actual'] = mins_actual
        if fga_actual is not None:
            self.bets.loc[bet_index, 'fga_actual'] = fga_actual
            
        # Determine if bet won
        line = bet['line_at_bet']
        side = bet['side']
        
        if side == 'OVER':
            won = actual_result > line
        else:
            won = actual_result < line
            
        self.bets.loc[bet_index, 'won'] = won
        
        # Calculate units won/lost
        odds = bet['odds_at_bet']
        units_bet = bet['units_bet']
        
        if won:
            if odds < 0:
                units_won = units_bet * (100 / abs(odds))
            else:
                units_won = units_bet * (odds / 100)
        else:
            units_won = -units_bet
            
        self.bets.loc[bet_index, 'units_won'] = units_won
        
        self._save_bets()
        
        result_str = "✓ WON" if won else "✗ LOST"
        print(f"{result_str}: {bet['player']} scored {actual_result} (line: {line})")
        
    def _save_bets(self):
        """Save bets to CSV."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.bets.to_csv(self.filepath, index=False)
    
    def get_pending_bets(self) -> pd.DataFrame:
        """Get bets that haven't been settled yet."""
        return self.bets[self.bets['won'].isna()]
    
    def get_settled_bets(self) -> pd.DataFrame:
        """Get settled bets."""
        return self.bets[self.bets['won'].notna()]
    
    def add_bet_from_evaluation(self, evaluation: dict, bookmaker: str, odds: int, units: float = 1.0):
        """
        Add a bet directly from a model evaluation result.
        
        Args:
            evaluation: Result from PointsPropModel.evaluate_prop()
            bookmaker: Which bookmaker you're betting at
            odds: The odds you're getting
            units: Units to bet
        """
        proj = evaluation['projection']
        edge = evaluation['edge_analysis']
        
        bet_data = {
            'date': datetime.now().date(),
            'player': evaluation.get('player_name', f"Player {evaluation['player_id']}"),
            'player_id': evaluation['player_id'],
            'prop_type': 'points',
            'side': edge['recommendation'],
            'bookmaker': bookmaker,
            'line_at_bet': edge['line'],
            'odds_at_bet': odds,
            'projection': proj['expected_points'],
            'projection_std': proj['std'],
            'predicted_prob': edge['prob_over'] if edge['recommendation'] == 'OVER' else edge['prob_under'],
            'predicted_edge': edge['edge'],
            'units_bet': units,
            'mins_proj': proj['components']['minutes']['expected'],
            'opp_pace': proj['adjustments']['matchup']['inputs']['opp_pace'],
            'opp_drtg': proj['adjustments']['matchup']['inputs']['opp_drtg'],
        }
        
        return self.add_bet(bet_data)


class PerformanceAnalyzer:
    """
    Analyze betting performance with focus on process over results.
    """
    
    def __init__(self, tracker: PropBetTracker):
        self.tracker = tracker
        self.bets = tracker.bets
        
    def summary(self, last_n_days: int = None):
        """Overall performance summary."""
        
        df = self.bets.copy()
        
        if last_n_days:
            cutoff = datetime.now() - timedelta(days=last_n_days)
            df = df[pd.to_datetime(df['date']) >= cutoff]
            
        if len(df) == 0:
            print("No bets to analyze.")
            return
            
        settled = df[df['won'].notna()]
        
        if len(settled) == 0:
            print(f"Total bets: {len(df)} (none settled yet)")
            return
            
        print("=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Basic stats
        total_bets = len(settled)
        wins = settled['won'].sum()
        win_rate = wins / total_bets
        
        print(f"\nRecord: {int(wins)}-{int(total_bets - wins)} ({win_rate:.1%})")
        
        # Units
        total_units_bet = settled['units_bet'].sum()
        total_units_won = settled['units_won'].sum()
        roi = total_units_won / total_units_bet if total_units_bet > 0 else 0
        
        print(f"Units: {total_units_won:+.2f} / {total_units_bet:.2f} wagered")
        print(f"ROI: {roi:.1%}")
        
        # CLV Analysis
        clv_data = settled[settled['closing_line'].notna()]
        if len(clv_data) > 0:
            print(f"\n--- Closing Line Value ---")
            
            clv_values = []
            for _, bet in clv_data.iterrows():
                if bet['side'] == 'OVER':
                    clv = bet['line_at_bet'] - bet['closing_line']
                else:
                    clv = bet['closing_line'] - bet['line_at_bet']
                clv_values.append(clv)
                
            avg_clv = np.mean(clv_values)
            clv_positive_pct = np.mean([c > 0 for c in clv_values])
            
            print(f"Average CLV: {avg_clv:+.2f} points")
            print(f"Positive CLV rate: {clv_positive_pct:.1%}")
            
            if avg_clv > 0.3:
                print("→ Strong edge indicator! Keep betting.")
            elif avg_clv > 0:
                print("→ Slight edge. Monitor closely.")
            else:
                print("→ Negative CLV suggests no edge. Review model.")
        
        # Prediction accuracy
        pred_data = settled[settled['projection'].notna() & settled['actual_result'].notna()]
        if len(pred_data) > 0:
            print(f"\n--- Prediction Accuracy ---")
            
            errors = pred_data['actual_result'] - pred_data['projection']
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors ** 2).mean())
            bias = errors.mean()
            
            print(f"MAE: {mae:.2f} points")
            print(f"RMSE: {rmse:.2f} points")
            print(f"Bias: {bias:+.2f} points")
            
            if bias > 1:
                print("→ Model underestimates. Consider adjusting up.")
            elif bias < -1:
                print("→ Model overestimates. Consider adjusting down.")
        
        # By bookmaker
        if 'bookmaker' in settled.columns:
            print(f"\n--- By Bookmaker ---")
            by_book = settled.groupby('bookmaker').agg({
                'won': ['sum', 'count'],
                'units_won': 'sum',
                'units_bet': 'sum'
            })
            by_book.columns = ['wins', 'total', 'units_won', 'units_bet']
            by_book['win_rate'] = by_book['wins'] / by_book['total']
            by_book['roi'] = by_book['units_won'] / by_book['units_bet']
            
            for book, row in by_book.iterrows():
                print(f"  {book}: {int(row['wins'])}-{int(row['total']-row['wins'])} "
                      f"({row['win_rate']:.1%}) | ROI: {row['roi']:.1%}")
                
        print("=" * 60)
        
    def calibration_analysis(self):
        """Check if probability estimates are well-calibrated."""
        
        df = self.bets[self.bets['won'].notna() & self.bets['predicted_prob'].notna()].copy()
        
        if len(df) < 20:
            print("Need at least 20 settled bets for calibration analysis.")
            return
            
        print("\n" + "=" * 60)
        print("CALIBRATION ANALYSIS")
        print("=" * 60)
        print("\nAre your probability estimates accurate?")
        print("-" * 60)
        print(f"{'Predicted Prob':<18} {'Actual Win%':<15} {'Count':<10} {'Status'}")
        print("-" * 60)
        
        # Bucket by predicted probability
        buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]
        
        for low, high in buckets:
            bucket_df = df[(df['predicted_prob'] >= low) & (df['predicted_prob'] < high)]
            
            if len(bucket_df) >= 3:
                actual_win = bucket_df['won'].mean()
                expected = (low + high) / 2
                diff = actual_win - expected
                
                status = "✓" if abs(diff) < 0.08 else "⚠" if abs(diff) < 0.15 else "✗"
                
                print(f"{low:.0%}-{high:.0%}            {actual_win:>6.1%}          {len(bucket_df):<10} {status}")
            else:
                print(f"{low:.0%}-{high:.0%}            {'--':<6}          {len(bucket_df):<10} (insufficient)")
                
        print("-" * 60)
        print("✓ = Well calibrated  ⚠ = Slight miscalibration  ✗ = Poor calibration")
        
    def edge_bucket_analysis(self):
        """Analyze performance by predicted edge size."""
        
        df = self.bets[self.bets['won'].notna() & self.bets['predicted_edge'].notna()].copy()
        
        if len(df) < 20:
            print("Need at least 20 settled bets for edge analysis.")
            return
            
        print("\n" + "=" * 60)
        print("EDGE BUCKET ANALYSIS")
        print("=" * 60)
        print("\nDoes higher predicted edge = better results?")
        print("-" * 60)
        print(f"{'Edge Bucket':<15} {'Win Rate':<12} {'ROI':<12} {'Count':<10}")
        print("-" * 60)
        
        buckets = [(0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.12), (0.12, 1.0)]
        
        results = []
        for low, high in buckets:
            bucket_df = df[(df['predicted_edge'] >= low) & (df['predicted_edge'] < high)]
            
            if len(bucket_df) >= 3:
                win_rate = bucket_df['won'].mean()
                roi = bucket_df['units_won'].sum() / bucket_df['units_bet'].sum()
                
                print(f"{low:.0%}-{high:.0%}          {win_rate:>6.1%}       {roi:>+6.1%}       {len(bucket_df)}")
                results.append({'bucket': f"{low:.0%}-{high:.0%}", 'win_rate': win_rate, 'count': len(bucket_df)})
            else:
                print(f"{low:.0%}-{high:.0%}          {'--':<6}       {'--':<6}       {len(bucket_df)}")
                
        # Check if higher edge = better results
        if len(results) >= 3:
            win_rates = [r['win_rate'] for r in results]
            is_monotonic = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))
            
            print("-" * 60)
            if is_monotonic:
                print("✓ Higher edge correlates with better results. Model has signal!")
            else:
                print("⚠ Edge buckets not monotonic. Review model or need more sample.")
                
    def component_attribution(self):
        """Analyze which model components are contributing to edge."""
        
        df = self.bets[
            self.bets['won'].notna() & 
            self.bets['mins_proj'].notna() & 
            self.bets['mins_actual'].notna()
        ].copy()
        
        if len(df) < 10:
            print("Need at least 10 bets with component data for attribution.")
            return
            
        print("\n" + "=" * 60)
        print("COMPONENT ATTRIBUTION")
        print("=" * 60)
        print("\nWhere did projection errors come from?")
        print("-" * 60)
        
        # Minutes attribution
        mins_error = df['mins_actual'] - df['mins_proj']
        print(f"Minutes: Avg Error = {mins_error.mean():+.1f}, Std = {mins_error.std():.1f}")
        
        # FGA attribution
        if 'fga_proj' in df.columns and 'fga_actual' in df.columns:
            fga_df = df[df['fga_proj'].notna() & df['fga_actual'].notna()]
            if len(fga_df) > 5:
                fga_error = fga_df['fga_actual'] - fga_df['fga_proj']
                print(f"FGA: Avg Error = {fga_error.mean():+.1f}, Std = {fga_error.std():.1f}")
        
        # Points attribution
        pts_error = df['actual_result'] - df['projection']
        print(f"Points: Avg Error = {pts_error.mean():+.1f}, Std = {pts_error.std():.1f}")
        
        # Correlation between minutes error and points error
        if len(df) >= 10:
            corr = np.corrcoef(mins_error, pts_error)[0, 1]
            print(f"\nMins Error ↔ Points Error Correlation: {corr:.2f}")
            
            if corr > 0.5:
                print("→ Minutes projection is key driver of points error. Focus there.")
            else:
                print("→ Volume/efficiency variance also significant.")
                
    def clv_analysis(self):
        """Detailed CLV analysis using line movement data."""
        
        df = self.bets[
            self.bets['won'].notna() & 
            self.bets['closing_line'].notna() &
            self.bets['line_at_bet'].notna()
        ].copy()
        
        if len(df) < 10:
            print("Need at least 10 bets with closing line data for CLV analysis.")
            return
            
        print("\n" + "=" * 60)
        print("CLOSING LINE VALUE (CLV) ANALYSIS")
        print("=" * 60)
        
        # Calculate CLV for each bet
        clv_values = []
        for _, bet in df.iterrows():
            if bet['side'] == 'OVER':
                clv = bet['line_at_bet'] - bet['closing_line']
            else:
                clv = bet['closing_line'] - bet['line_at_bet']
            clv_values.append(clv)
        
        df['clv'] = clv_values
        
        print(f"\nOverall CLV Stats:")
        print(f"  Average CLV: {df['clv'].mean():+.2f} points")
        print(f"  Median CLV: {df['clv'].median():+.2f} points")
        print(f"  CLV > 0: {(df['clv'] > 0).mean():.1%} of bets")
        
        # CLV by result
        print(f"\nCLV by Result:")
        print(f"  Winners avg CLV: {df[df['won']==True]['clv'].mean():+.2f}")
        print(f"  Losers avg CLV: {df[df['won']==False]['clv'].mean():+.2f}")
        
        # CLV correlation with winning
        if len(df) >= 20:
            corr = df['clv'].corr(df['won'].astype(float))
            print(f"\nCLV ↔ Win Correlation: {corr:.3f}")
            
            if corr > 0.1:
                print("→ Positive CLV correlates with wins. Good sign!")
            
    def cold_streak_analysis(self):
        """Analyze if recent losses are variance or model failure."""
        
        df = self.bets[self.bets['won'].notna()].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) < 20:
            print("Need at least 20 bets for streak analysis.")
            return
            
        # Find recent streak
        recent_10 = df.tail(10)
        recent_win_rate = recent_10['won'].mean()
        
        # Overall expected win rate
        overall_expected = df['predicted_prob'].mean()
        
        print("\n" + "=" * 60)
        print("VARIANCE CHECK")
        print("=" * 60)
        
        print(f"\nLast 10 bets: {int(recent_10['won'].sum())}-{int(10 - recent_10['won'].sum())} ({recent_win_rate:.0%})")
        print(f"Expected win rate: {overall_expected:.0%}")
        
        # Binomial test
        from scipy.stats import binomtest
        result = binomtest(int(recent_10['won'].sum()), n=10, p=overall_expected, alternative='two-sided')
        p_value = result.pvalue
        
        print(f"P-value (is this just variance?): {p_value:.3f}")
        
        if p_value < 0.05:
            print("\n⚠ Statistically unusual streak. Review model for potential issues.")
        else:
            print("\n✓ Within normal variance. Trust the process, keep betting.")
            
        # Calculate probability of this streak under true edge
        print(f"\nProbability of {int(recent_10['won'].sum())}/10 or worse given {overall_expected:.0%} expected:")
        prob_this_bad = stats.binom.cdf(int(recent_10['won'].sum()), 10, overall_expected)
        print(f"  {prob_this_bad:.1%} - {'Normal variance' if prob_this_bad > 0.05 else 'Concerning'}")
        
    def full_report(self):
        """Run all analyses."""
        self.summary()
        self.calibration_analysis()
        self.edge_bucket_analysis()
        self.clv_analysis()
        self.component_attribution()
        self.cold_streak_analysis()


# Quick reference guide
TRACKING_GUIDE = """
================================================================================
                        PROP BET TRACKING QUICK GUIDE
================================================================================

BEFORE PLACING BET:
------------------
1. Run your projection model
2. Record: player, line, odds, projection, projected_edge
3. Only bet if edge > 3% (after accounting for juice)

AFTER GAME:
-----------
1. Record: actual_result, closing_line, closing_odds
2. Note any factors that affected outcome (injury, blowout, etc.)

WEEKLY REVIEW:
--------------
1. Run summary() - check overall ROI and CLV
2. Run calibration_analysis() - are probabilities accurate?
3. Run edge_bucket_analysis() - does more edge = more wins?

MONTHLY REVIEW:
---------------
1. Run component_attribution() - where are errors coming from?
2. Adjust model based on systematic biases found
3. Track if adjustments improve future performance

RED FLAGS:
----------
- Negative CLV consistently → you're not beating the market
- Poor calibration → probability estimates are off
- No edge bucket correlation → model has no signal
- Minutes error > 3 → need better minutes projection

GREEN FLAGS:
------------
- Positive CLV > 0.5 points → strong market edge
- Calibration within 5% → good probability estimates
- Monotonic edge buckets → model signal is real
- Consistent ROI > 5% → sustainable edge

================================================================================
"""


def print_tracking_guide():
    """Print the tracking guide."""
    print(TRACKING_GUIDE)
