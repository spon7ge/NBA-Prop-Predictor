import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class BetResult:
    date: str
    player1: str
    category1: str
    line1: float
    actual1: float
    player2: str
    category2: str
    line2: float
    actual2: float
    bet_type: str
    ev: float
    probability: float
    kelly: float
    won: bool
    profit: float

class PrizePicksBacktest:
    def __init__(self, 
                 props_ev_dir: str = "../CSV_FILES/HISTORICAL_PROP_PAIRS",
                 regular_data_dir: str = "../CSV_FILES/REGULAR_DATA",
                 min_ev: float = 60.0,
                 stake: float = 100,
                 max_bets_per_day: int = 3,
                 kelly_fraction: float = 0.25):
        """
        Initialize backtester for PrizePicks pairs using actual results
        
        Args:
            props_ev_dir: Directory containing PrizePicks EV CSV files
            regular_data_dir: Directory containing actual game results
            min_ev: Minimum EV threshold for taking a bet
            stake: Stake size for each bet
        """
        self.props_ev_dir = Path(props_ev_dir)
        self.regular_data_dir = Path(regular_data_dir)
        self.min_ev = min_ev
        self.max_bets_per_day = max_bets_per_day
        self.stake = stake
        self.kelly_fraction = kelly_fraction
        self.results: List[BetResult] = []
        self.bet_selection_log = []  # Track bet selection process
        
        # Load actual results data
        self.actual_results = self._load_actual_results()
        
    def _load_actual_results(self) -> Dict[str, pd.DataFrame]:
        """Load actual results for each stat category"""
        results = {}
        stat_types = {
            'player_points': 'PTS',
            'player_rebounds': 'REB',
            'player_assists': 'AST'
        }
        
        for category, stat in stat_types.items():
            file_path = self.regular_data_dir / f'season_25_{stat}_features.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Convert date to YYYYMMDD format
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y%m%d')
                results[category] = df
                
        return results
    
    def _get_actual_stat(self, date: str, player: str, category: str) -> float:
        """Get actual stat value for a player on a given date"""
        if category not in self.actual_results:
            return None
            
        df = self.actual_results[category]
        result = df[(df['GAME_DATE'] == date) & (df['PLAYER_NAME'] == player)]
        
        if result.empty:
            return None
            
        stat_map = {
            'player_points': 'PTS',
            'player_rebounds': 'REB',
            'player_assists': 'AST'
        }
        
        return result[stat_map[category]].iloc[0]

    def _check_bet_result(self, bet_type: str, actual1: float, line1: float, 
                     actual2: float, line2: float) -> bool:
        """Check if a bet won based on actual results"""
        if actual1 is None or actual2 is None:
            return None
            
        bet_parts = bet_type.split('/')
        result1 = actual1 > line1 if bet_parts[0] == 'OVER' else actual1 < line1
        result2 = actual2 > line2 if bet_parts[1] == 'OVER' else actual2 < line2
        
        return result1 and result2

    def load_daily_ev_data(self, date_str: str) -> pd.DataFrame:
        """Load PrizePicks pairs EV data for a specific date"""
        file_path = self.props_ev_dir / f"{date_str}_PAIRS.csv"
        if not file_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        # First filter by minimum EV
        qualified_bets = df[df['EV'] >= self.min_ev].sort_values('EV', ascending=False)
        # Then take only top N bets
        return qualified_bets.head(self.max_bets_per_day)

    def simulate_bets(self) -> None:
        """Run backtest simulation using actual results"""
        ev_files = list(self.props_ev_dir.glob("*_PAIRS.csv"))
        
        for file in sorted(ev_files):
            date_str = file.stem.split('_')[0]  # Get YYYYMMDD from filename
            
            # Load all bets for the day
            all_bets = pd.read_csv(file)
            daily_bets = self.load_daily_ev_data(date_str)
            
            # Log bet selection process
            self.bet_selection_log.append({
                'date': date_str,
                'total_available_bets': len(all_bets),
                'bets_above_min_ev': len(all_bets[all_bets['EV'] >= self.min_ev]),
                'bets_selected': len(daily_bets),
                'min_ev_selected': daily_bets['EV'].min() if not daily_bets.empty else None,
                'max_ev_selected': daily_bets['EV'].max() if not daily_bets.empty else None
            })
            
            if daily_bets.empty:
                continue
                
            for _, bet in daily_bets.iterrows():
                # Get actual results
                actual1 = self._get_actual_stat(date_str, bet['PLAYER 1'], bet['CATEGORY 1'])
                actual2 = self._get_actual_stat(date_str, bet['PLAYER 2'], bet['CATEGORY 2'])
                
                # Skip if we don't have actual results
                if actual1 is None or actual2 is None:
                    continue
                
                # Check if bet won
                won = self._check_bet_result(
                    bet['TYPE'], 
                    actual1, bet['PLAYER 1 LINE'],
                    actual2, bet['PLAYER 2 LINE']
                )
                
                if won is None:
                    continue
                    
                kelly_stake = self.stake * bet['KELLY CRITERION'] * self.kelly_fraction
                profit = kelly_stake if won else -kelly_stake
                
                result = BetResult(
                    date=date_str,
                    player1=bet['PLAYER 1'],
                    category1=bet['CATEGORY 1'],
                    line1=bet['PLAYER 1 LINE'],
                    actual1=actual1,
                    player2=bet['PLAYER 2'],
                    category2=bet['CATEGORY 2'],
                    line2=bet['PLAYER 2 LINE'],
                    actual2=actual2,
                    bet_type=bet['TYPE'],
                    ev=bet['EV'],
                    probability=bet['PROBABILITY'],
                    kelly=bet['KELLY CRITERION'],
                    won=won,
                    profit=profit
                )
                self.results.append(result)

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.results:
            return {}
            
        profits = [r.profit for r in self.results]
        cumulative_profits = np.cumsum(profits)
        
        total_bets = len(self.results)
        winning_bets = sum(1 for r in self.results if r.won)
        total_risked = sum(abs(r.profit) for r in self.results)  # Sum of absolute profits since each profit represents the Kelly stake

        metrics = {
            'total_bets': total_bets,
            'hit_rate': winning_bets / total_bets if total_bets > 0 else 0,
            'total_profit': sum(profits),
            'roi': (sum(profits) / total_risked) if total_bets > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(cumulative_profits),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, cumulative_profits: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        rolling_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = rolling_max - cumulative_profits
        return np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    def _calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if not profits:
            return 0.0
        
        returns = np.array(profits) / self.stake
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0.0
            
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) != 0 else 0.0

    def print_summary(self) -> None:
        """Print backtest summary with actual results"""
        metrics = self.calculate_metrics()
        if not metrics:
            print("No results to display")
            return
            
        print("\n=== PrizePicks Pairs Backtest Summary ===")
        print(f"Total Bets: {metrics['total_bets']}")
        print(f"Total Days: {len(self.bet_selection_log)}")
        print(f"Average Bets Per Day: {metrics['total_bets']/len(self.bet_selection_log):.1f}")
        print(f"Hit Rate: {metrics['hit_rate']:.2%}")
        
        # Calculate and display Kelly sizing statistics
        total_risked = sum(abs(r.profit) for r in self.results)
        avg_stake = total_risked / metrics['total_bets'] if metrics['total_bets'] > 0 else 0
        print(f"Total Amount Risked: ${total_risked:.2f}")
        print(f"Average Stake Size: ${avg_stake:.2f}")
        
        print(f"Total Profit: ${metrics['total_profit']:.2f}")
        print(f"ROI: {metrics['roi']:.2%}")
        print(f"Max Drawdown: ${metrics['max_drawdown']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        # Print bet selection statistics
        print("\nBet Selection Statistics:")
        total_available = sum(log['total_available_bets'] for log in self.bet_selection_log)
        total_above_min_ev = sum(log['bets_above_min_ev'] for log in self.bet_selection_log)
        print(f"Average Available Bets Per Day: {total_available/len(self.bet_selection_log):.1f}")
        print(f"Average Bets Above Min EV Per Day: {total_above_min_ev/len(self.bet_selection_log):.1f}")
        
        # Print top 5 highest EV bets and their actual results
        print("\nTop 5 Highest EV Bets:")
        top_ev_bets = sorted(self.results, key=lambda x: x.ev, reverse=True)[:5]
        for bet in top_ev_bets:
            print(f"{bet.date}: {bet.player1} {bet.category1} {bet.line1} (Actual: {bet.actual1:.1f}) & "
                  f"{bet.player2} {bet.category2} {bet.line2} (Actual: {bet.actual2:.1f})")
            print(f"Type: {bet.bet_type}, EV: {bet.ev:.2f}, Won: {bet.won}")

    def plot_performance(self) -> None:
        """Plot cumulative performance over time"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                print("No results to plot")
                return
                
            profits = [r.profit for r in self.results]
            cumulative_profits = np.cumsum(profits)
            dates = [datetime.strptime(r.date, '%Y%m%d') for r in self.results]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, cumulative_profits, label='Cumulative Profit')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.title('PrizePicks Pairs Performance')
            plt.xlabel('Date')
            plt.ylabel('Profit ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib is required for plotting")