import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import joblib
from pathlib import Path
import warnings
import requests
import pytz
warnings.filterwarnings('ignore')

# Fixed get_espn_games function with correct date format
def get_espn_games(date_str):
    """
    Get ESPN games for a specific date
    Args:
        date_str: Date in YYYYMMDD format (e.g., '20241022')
    Returns:
        List of game dictionaries
    """
    try:
        # ESPN API expects YYYYMMDD format
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        
        if 'events' not in data:
            return []
        
        # Define timezone objects
        utc = pytz.UTC
        pst = pytz.timezone('America/Los_Angeles')

        games_list = []
        for event in data['events']:
            try:
                # Parse UTC time from ESPN
                utc_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%MZ').replace(tzinfo=utc)
                # Convert to PST
                pst_time = utc_time.astimezone(pst)
                
                # Get team abbreviations
                competitors = event['competitions'][0]['competitors']
                home_team = None
                away_team = None
                
                for competitor in competitors:
                    if competitor['homeAway'] == 'home':
                        home_team = competitor['team']['abbreviation']
                    else:
                        away_team = competitor['team']['abbreviation']
                
                game_dict = {
                    'game_date': pst_time.strftime('%Y-%m-%d'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_time': pst_time.strftime('%I:%M %p'),
                    'venue': event['competitions'][0]['venue']['fullName'] if 'venue' in event['competitions'][0] else 'Unknown'
                }
                games_list.append(game_dict)
                
            except Exception as e:
                continue
        
        return games_list
        
    except Exception as e:
        return []

@dataclass
class BetResult:
    """Data class to store individual bet results"""
    date: str
    bet_type: str = 'single'
    player_name: str = ''
    player2_name: Optional[str] = None
    category: str = 'player_points'
    category2: Optional[str] = None
    line: float = 0.0
    line2: Optional[float] = None
    side: str = 'over'
    side2: Optional[str] = None
    prediction: float = 0.0
    prediction2: Optional[float] = None
    actual_result: float = 0.0
    actual_result2: Optional[float] = None
    probability: float = 0.5
    ev_percent: float = 0.0
    ev_dollars: float = 0.0
    kelly_fraction: float = 0.01
    stake_amount: float = 0.0
    won: bool = False
    profit_loss: float = 0.0
    bookmaker: str = 'prizepicks'

class CleanNBABacktester:
    """
    Clean backtest class with minimal output
    """
    
    def __init__(self, 
                 data_path: str = r"C:\Users\alexg\OneDrive\Documents\NBA-Prop-Predictor\DATA\CSV_FILES\TRAIN_DATA\PTS_TRAIN_25.csv",
                 odds_path: str = r"C:\Users\alexg\OneDrive\Documents\NBA-Prop-Predictor\BACKTESTS\singleBets.csv",
                 model_path: str = r"C:\Users\alexg\OneDrive\Documents\NBA-Prop-Predictor\MODELS\Models\PTS_cat_model.pkl",
                 min_ev_threshold: float = 0.02,
                 base_stake: float = 100,
                 kelly_multiplier: float = 0.20,
                 max_bets_per_day: int = 3,
                 start_date: str = None,
                 end_date: str = None):
        
        self.min_ev_threshold = min_ev_threshold
        self.base_stake = base_stake
        self.kelly_multiplier = kelly_multiplier
        self.max_bets_per_day = max_bets_per_day
        self.start_date = start_date
        self.end_date = end_date
        
        # Load data and model silently
        try:
            self.data = pd.read_csv(data_path, low_memory=False).sort_values('GAME_DATE', ascending=True)
            self.bookmakers = pd.read_csv(odds_path, low_memory=False)
            self.model = joblib.load(model_path)
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
        
        # Results storage
        self.bet_results: List[BetResult] = []
        self.daily_summary: List[Dict] = []
        self.processing_log: List[Dict] = []
    
    def get_unique_dates(self) -> List[str]:
        """Get unique game dates from bookmakers data"""
        try:
            dates = self.bookmakers['GAME_DATE'].unique()
            dates = [d for d in dates if pd.notna(d)]
            dates = sorted(dates)
            
            if self.start_date:
                dates = [d for d in dates if d >= self.start_date]
            if self.end_date:
                dates = [d for d in dates if d <= self.end_date]
                
            return dates
        except Exception as e:
            return []
    
    def get_actual_result(self, player_name: str, game_date: str, stat_type: str = 'PTS') -> Optional[float]:
        """Get actual game result for a player on a specific date"""
        try:
            game_data = self.data[
                (self.data['PLAYER_NAME'] == player_name) & 
                (self.data['GAME_DATE'] == game_date)
            ]
            
            if game_data.empty or stat_type not in game_data.columns:
                return None
                
            return float(game_data[stat_type].iloc[0])
        except Exception as e:
            return None
    
    def enhanced_single_bet(self, game_date: str) -> pd.DataFrame:
        """Enhanced version with better game data handling"""
        try:
            # Get odds for the date
            single_bet_books = [
                'bovada', 'espnbet','fanduel',
                'betmgm', 'draftkings', 'caesars', 'betrivers',
                'pinnacle', 'bet365'
            ]

            daily_odds = self.bookmakers[
                (self.bookmakers['CATEGORY'] == 'points') &
                (self.bookmakers['GAME_DATE'] == game_date) &
                (self.bookmakers['BOOKMAKER'].isin(single_bet_books)) &
                (self.bookmakers['ODDS'] < 200) &
                (self.bookmakers['ODDS'] > -200)
            ].copy()

            if daily_odds.empty:
                return pd.DataFrame()
            
            # Convert date format for ESPN API
            date_obj = datetime.strptime(game_date, "%Y-%m-%d")
            date_str = date_obj.strftime("%Y%m%d")  # Convert to YYYYMMDD
            
            # Get games silently
            games = get_espn_games(date_str)
            if not games:
                games = []  # Proceed without games data
            
            # Get training data
            filtered_data = self.data[self.data['GAME_DATE'] < game_date].sort_values('GAME_DATE', ascending=True)
            
            if filtered_data.empty:
                return pd.DataFrame()
            
            # Enhanced prediction logic
            results = []
            
            for _, row in daily_odds.iterrows():
                try:
                    player_name = row['NAME']
                    line = float(row['LINE'])
                    side = row.get('SIDE', 'over')
                    
                    # Get player's recent performance
                    player_data = filtered_data[filtered_data['PLAYER_NAME'] == player_name]
                    
                    if player_data.empty:
                        continue
                    
                    # Enhanced prediction using multiple metrics
                    recent_5 = player_data['PTS'].tail(5).mean()
                    recent_10 = player_data['PTS'].tail(10).mean()
                    season_avg = player_data['PTS'].mean()
                    
                    # Weight recent games more heavily
                    prediction = (recent_5 * 0.5 + recent_10 * 0.3 + season_avg * 0.2)
                    
                    if pd.isna(prediction):
                        continue
                    
                    # Calculate standard deviation for better probability estimation
                    recent_std = player_data['PTS'].tail(10).std()
                    if pd.isna(recent_std) or recent_std == 0:
                        recent_std = 5.0  # Default std
                    
                    # Use normal distribution to estimate probability
                    from scipy import stats
                    if side.upper().startswith('O'):
                        prob = 1 - stats.norm.cdf(line, prediction, recent_std)
                    else:
                        prob = stats.norm.cdf(line, prediction, recent_std)
                    
                    # Ensure probability is reasonable
                    prob = max(0.1, min(0.9, prob))
                    
                    # Calculate EV assuming -110 odds (52.38% break-even)
                    breakeven = 0.5238
                    if prob > breakeven:
                        ev_percent = (prob - breakeven) / breakeven  # Rough EV calculation
                    else:
                        ev_percent = 0
                    
                    # Cap EV at reasonable levels
                    ev_percent = min(ev_percent, 0.5)  # Max 50% EV
                    
                    results.append({
                        'NAME': player_name,
                        'CATEGORY': 'player_points',
                        'LINE': line,
                        'SIDE': side,
                        'PREDICTION': prediction,
                        'OVER%': prob if side.upper().startswith('o') else 1-prob,
                        'UNDER%': 1-prob if side.upper().startswith('o') else prob,
                        'EV%': ev_percent,
                        'EV$': ev_percent * self.base_stake,
                        'KELLY QUARTER': max(0.01, ev_percent * 0.25),
                        'BOOKMAKER': 'prizepicks'
                    })
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            return pd.DataFrame()
    
    def process_single_bets(self, game_date: str) -> List[BetResult]:
        """Process single bets with enhanced prediction"""
        results = []
        
        try:
            predictions = self.enhanced_single_bet(game_date)
            
            if predictions.empty:
                return results
            
            # Ensure EV% column exists and is numeric
            if 'EV%' not in predictions.columns:
                return results
                
            predictions['EV%'] = pd.to_numeric(predictions['EV%'], errors='coerce').fillna(0)
            
            # Filter by EV threshold and limit bets per day
            qualified_bets = predictions[predictions['EV%'] >= self.min_ev_threshold].sort_values('EV%', ascending=False)
            selected_bets = qualified_bets.head(self.max_bets_per_day)
            
            for _, bet in selected_bets.iterrows():
                try:
                    # Get actual result
                    actual = self.get_actual_result(bet['NAME'], game_date, 'PTS')
                    if actual is None:
                        continue
                    
                    # Safely get bet values with defaults
                    line = float(bet.get('LINE', 0))
                    side = str(bet.get('SIDE', 'over')).lower()
                    prediction = float(bet.get('PREDICTION', 0))
                    ev_percent = float(bet.get('EV%', 0))
                    
                    # Determine if bet won
                    if side.startswith('o'):
                        won = actual > line
                        probability = float(bet.get('OVER%', 0.5))
                    else:
                        won = actual < line
                        probability = float(bet.get('UNDER%', 0.5))
                    
                    # Calculate stake using Kelly criterion
                    kelly_quarter = float(bet.get('KELLY QUARTER', 0.01))
                    kelly_stake = self.base_stake * kelly_quarter * self.kelly_multiplier
                    kelly_stake = max(kelly_stake, self.base_stake * 0.01)  # Minimum 1% of base stake
                    
                    # Calculate profit/loss (PrizePicks typically pays 2:1)
                    if won:
                        profit = kelly_stake
                    else:
                        profit = -kelly_stake
                    
                    result = BetResult(
                        date=game_date,
                        bet_type='single',
                        player_name=bet['NAME'],
                        category=bet.get('CATEGORY', 'player_points'),
                        line=line,
                        side=side,
                        prediction=prediction,
                        actual_result=actual,
                        probability=probability,
                        ev_percent=ev_percent,
                        ev_dollars=float(bet.get('EV$', 0)),
                        kelly_fraction=kelly_quarter,
                        stake_amount=kelly_stake,
                        won=won,
                        profit_loss=profit,
                        bookmaker='prizepicks'
                    )
                    results.append(result)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
            
        return results
    
    def run_backtest(self, bet_types: List[str] = ['single']) -> None:
        """
        Run the backtest silently
        """
        dates = self.get_unique_dates()
        
        for i, date in enumerate(dates):
            daily_results = []
            daily_errors = []
            
            # Process single bets
            if 'single' in bet_types:
                try:
                    single_results = self.process_single_bets(date)
                    daily_results.extend(single_results)
                except Exception as e:
                    daily_errors.append(f"Single bets error: {e}")
            
            # Log the day's processing
            self.processing_log.append({
                'date': date,
                'bets_found': len(daily_results),
                'errors': daily_errors,
                'success': len(daily_results) > 0
            })
            
            # Store results
            if daily_results:
                self.bet_results.extend(daily_results)
                
                # Create daily summary
                daily_profit = sum(r.profit_loss for r in daily_results)
                daily_stake = sum(r.stake_amount for r in daily_results)
                wins = sum(1 for r in daily_results if r.won)
                
                self.daily_summary.append({
                    'date': date,
                    'num_bets': len(daily_results),
                    'total_stake': daily_stake,
                    'total_profit': daily_profit,
                    'wins': wins,
                    'hit_rate': wins / len(daily_results) if daily_results else 0,
                    'roi': daily_profit / daily_stake if daily_stake > 0 else 0
                })
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate the exact metrics from your notes"""
        if not self.bet_results:
            return {}
        
        # Basic calculations
        total_bets = len(self.bet_results)
        winning_bets = sum(1 for r in self.bet_results if r.won)
        total_profit = sum(r.profit_loss for r in self.bet_results)
        total_amount_staked = sum(r.stake_amount for r in self.bet_results)
        
        # 1. ROI (%) = (Total profit ÷ Total amount staked) × 100
        roi_percent = (total_profit / total_amount_staked) * 100 if total_amount_staked > 0 else 0
        
        # 2. Hit rate = (Number of winning bets) ÷ (Total bets)
        hit_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # 3. Volatility = Standard deviation of daily P&L
        daily_pnl = [summary['total_profit'] for summary in self.daily_summary]
        volatility = np.std(daily_pnl, ddof=1) if len(daily_pnl) > 1 else 0
        
        # 4. Max drawdown = Largest peak-to-trough loss
        cumulative_profits = np.cumsum([r.profit_loss for r in self.bet_results])
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = running_max - cumulative_profits
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # 5. Sharpe ratio = (Average daily return ÷ Std. dev. of daily returns) × √252
        if len(daily_pnl) > 1 and volatility > 0:
            avg_daily_return = np.mean(daily_pnl)
            sharpe_ratio = (avg_daily_return / volatility) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'total_profit': total_profit,
            'total_amount_staked': total_amount_staked,
            'roi_percent': roi_percent,
            'number_of_winning_bets': winning_bets,
            'total_bets': total_bets,
            'hit_rate': hit_rate,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'average_daily_return': np.mean(daily_pnl) if daily_pnl else 0,
            'std_dev_daily_returns': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_days': len(self.daily_summary)
        }
    
    def print_summary(self) -> None:
        """Print summary showing exactly your specified metrics"""
        metrics = self.calculate_performance_metrics()
        
        print("\n" + "="*60)
        print("NBA PROPS BACKTEST - KEY METRICS")
        print("="*60)
        
        print(f"\nYOUR SPECIFIED METRICS:")
        print("-" * 40)
        
        # 1. ROI (%) = (Total profit ÷ Total amount staked) × 100
        print(f"ROI (%): {metrics['roi_percent']:.2f}%")
        print(f"  └─ Total profit: ${metrics['total_profit']:,.2f}")
        print(f"  └─ Total amount staked: ${metrics['total_amount_staked']:,.2f}")
        
        # 2. Hit rate = (Number of winning bets) ÷ (Total bets)
        print(f"\nHit rate: {metrics['hit_rate']:.4f} ({metrics['hit_rate']:.2%})")
        print(f"  └─ Number of winning bets: {metrics['number_of_winning_bets']:,}")
        print(f"  └─ Total bets: {metrics['total_bets']:,}")
        
        # 3. Volatility = Standard deviation of daily P&L
        print(f"\nVolatility: ${metrics['volatility']:.2f}")
        print(f"  └─ Standard deviation of daily P&L")
        print(f"  └─ Based on {metrics['total_days']} trading days")
        
        # 4. Max drawdown = Largest peak-to-trough loss
        print(f"\nMax drawdown: ${metrics['max_drawdown']:.2f}")
        print(f"  └─ Largest peak-to-trough loss")
        
        # 5. Sharpe ratio = (Average daily return ÷ Std. dev. of daily returns) × √252
        print(f"\nSharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  └─ Average daily return: ${metrics['average_daily_return']:.2f}")
        print(f"  └─ Std. dev. of daily returns: ${metrics['std_dev_daily_returns']:.2f}")
        print(f"  └─ Annualized (× √252): {metrics['sharpe_ratio']:.4f}")
        
        # print("\n" + "="*60)
        # print("📋 CALCULATION FORMULAS USED:")
        # print("-" * 40)
        # print("ROI (%) = (Total profit ÷ Total amount staked) × 100")
        # print("Hit rate = (Number of winning bets) ÷ (Total bets)")
        # print("Volatility = Standard deviation of daily P&L")
        # print("Max drawdown = Largest peak-to-trough loss")
        # print("Sharpe ratio = (Average daily return ÷ Std. dev. of daily returns) × √252")
        # print("="*60)

def run_clean_backtest():
    # Initialize backtester
    backtester = CleanNBABacktester(
        min_ev_threshold=0.05,  # 2% minimum EV
        base_stake=100,
        kelly_multiplier=0.15,
        max_bets_per_day=3,
        start_date='2024-10-22',
        end_date='2024-11-12'
    )
    
    # Run backtest silently
    backtester.run_backtest(bet_types=['single'])
    
    # Print summary
    backtester.print_summary()
    
    return backtester

if __name__ == "__main__":
    # Run the clean backtest
    backtest_results = run_clean_backtest()