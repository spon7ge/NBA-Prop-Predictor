import pandas as pd
import numpy as np
import time
import joblib
import sys
import os

# Add parent directory to path to access MODELS module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from MODELS.pipeline import *
from BACKTEST.backtestCalculateEVS import backtest2legs as calculatePairs
from BACKTEST.backtestCalculateEVS import backtestSingleBet as calculateSingleBet
from BACKTEST.backtestCalculateEVS import backtest3Legs as calculate3Legs

def backtestSingle(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                  variance_inflation=1.1, use_monte_carlo=True, 
                  n_simulations=10000, max_kelly=0.25, stake=100):
    
    data = data[data['GAME_DATE'] == gameDate]
    category = 'points'
    backtestData = backtestData[(backtestData['CATEGORY'] == category) & (backtestData['GAME_DATE'] == gameDate)]
    if backtestData.empty:
        print(f"No bets found for {gameDate}")
        return pd.DataFrame()

    results = calculateSingleBet(
        data=data,
        bookmakers=backtestData,
        models=models,
        features=features,
        edge_threshold=edge_threshold,
        stake=stake,
        variance_inflation=variance_inflation,
        use_monte_carlo=use_monte_carlo,
        n_simulations=n_simulations,
        max_kelly=max_kelly
    )
    
    # Sort by edge and take top bets
    evData = results.sort_values(by='EV%', ascending=False).head(top_n)
    results = []

    for idx, row in evData.iterrows():
        player = row['NAME']
        bookmaker = row['BOOKMAKER']
        side = row['SIDE']
        line = row['LINE']
        odds = row['ODDS']
        pred = row['PREDICTION']  # This is the mean prediction
        edge = row['EDGE']  # This is the probability edge
        
        playerData = data[data['PLAYER_NAME'] == player]

        if playerData.empty:
            print(f"Player {player} not found in data for {gameDate}")
            continue

        actual = playerData['PTS'].iloc[-1]
        
        # Use the recommendation from the updated function (includes Kelly constraint)
        recommendation = row['RECOMMENDATION']

        # Determine if bet won
        if side == 'over':
            won = 1 if actual > line else 0
        elif side == 'under':
            won = 1 if actual < line else 0
        else:
            won = 0

        results.append({
            'player': player, 
            'bookmaker': bookmaker, 
            'side': side, 
            'line': line, 
            'odds': odds, 
            'pred': pred,  # Mean prediction
            'actual': actual,
            'edge': edge,  # Probability edge
            'model_prob': row['MODEL PROB'],  # Model probability for the side
            'market_prob': row['IMPLIED PROB'],  # Market implied probability
            'ev_percent': row['EV%'],  # Expected value percentage
            'kelly_fraction': row['KELLY_FRACTION'],  # Kelly as fraction
            'kelly_dollars': row['KELLY_DOLLARS'],  # Kelly in dollars
            'recommendation': recommendation,
            'won': won,
            'simulation_method': row['SIMULATION_METHOD'],  # Monte Carlo or Analytical
            'date': gameDate
        })

    return pd.DataFrame(results)


def backtestPairs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, stat_col='PTS', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100):
    print(f"Starting backtest for pairs on {gameDate}")
    total_start = time.time()

    data = data[data['GAME_DATE'] == gameDate]
    backtestData = backtestData[(backtestData['CATEGORY'] == 'player_points') & (backtestData['GAME_DATE'] == gameDate)]
    
    if backtestData.empty:
        print(f"No bets found for {gameDate}")
        return pd.DataFrame()

    # Get PrizePicks pairs EV calculations using updated function
    results = calculatePairs(
        data=data,
        backtestData=backtestData,
        gameDate=gameDate,
        models=models,
        features=features,
        edge_threshold=edge_threshold,
        top_n=top_n,
        variance_inflation=variance_inflation,
        stat_col=stat_col,
        use_monte_carlo=use_monte_carlo,
        n_simulations=n_simulations,
        max_kelly=max_kelly,
        stake=stake
    )
    print(f"Time taken for pairs: {time.time() - total_start} seconds")

    top_ev = results.sort_values(by='EV$', ascending=False).head(top_n)
    top_kelly = results.sort_values(by='KELLY FULL', ascending=False).head(top_n)

    print(f"Time taken for sorting: {time.time() - total_start} seconds")
    top_ev['selection'] = 'top_ev'
    top_kelly['selection'] = 'top_kelly'
    
    all_results = pd.concat([top_ev, top_kelly])
    # Use NAME columns for deduplication
    name1_col = 'NAME 1' if 'NAME 1' in all_results.columns else 'player1'
    name2_col = 'NAME 2' if 'NAME 2' in all_results.columns else 'player2'
    evData = all_results.drop_duplicates(subset=[name1_col, name2_col], keep='first').sort_values(by='EV$', ascending=False)

    print(f"Time taken for dropping duplicates: {time.time() - total_start} seconds")
    if evData.empty:
        print(f"No valid pairs found for {gameDate}")
        return pd.DataFrame()
    
    backtest_results = []
    print(f"Time taken for iterating over evData: {time.time() - total_start} seconds")
    for idx, row in evData.iterrows():
        # Use new column names (matching calculateEVS.py format) with fallback to old names
        player1 = row.get('NAME 1', row.get('player1'))
        player2 = row.get('NAME 2', row.get('player2'))
        
        line1 = row.get('LINE 1', row.get('line1'))
        line2 = row.get('LINE 2', row.get('line2'))
        
        pred1 = row.get('PREDICTION 1', row.get('pred1'))
        pred2 = row.get('PREDICTION 2', row.get('pred2'))
        
        # Get model's recommended sides
        model_side1 = row.get('MODEL SIDE 1', row.get('model_side1'))
        model_side2 = row.get('MODEL SIDE 2', row.get('model_side2'))
        
        # Get actual results for both players
        player1Data = data[data['PLAYER_NAME'] == player1]
        player2Data = data[data['PLAYER_NAME'] == player2]

        if player1Data.empty or player2Data.empty:
            print(f"Player data not found for {player1} or {player2} on {gameDate}")
            continue
        
        if stat_col == 'PTS':
            actual1 = player1Data['PTS'].iloc[-1]
            actual2 = player2Data['PTS'].iloc[-1]
        elif stat_col == 'REB':
            actual1 = player1Data['REB'].iloc[-1]
            actual2 = player2Data['REB'].iloc[-1]
        else:
            print(f"Invalid stat column: {stat_col}")
            continue
        
        # Calculate edges for each leg (use new column names with fallback)
        edge1 = row.get('EDGE 1', row.get('edge1'))
        edge2 = row.get('EDGE 2', row.get('edge2'))
        combined_edge = row.get('COMBINED EDGE', row.get('combined_edge'))
        
        # Determine if each leg wins
        if model_side1 == 'over':
            won1 = 1 if actual1 > line1 else 0
        else:  # model_side1 == 'under'
            won1 = 1 if actual1 < line1 else 0

        if model_side2 == 'over':
            won2 = 1 if actual2 > line2 else 0
        else:  # model_side2 == 'under'
            won2 = 1 if actual2 < line2 else 0
        
        
        # Pair wins only if both legs win
        pair_won = 1 if (won1 == 1 and won2 == 1) else 0

        backtest_results.append({
            # New format columns (matching calculateEVS.py)
            'NAME 1': player1,
            'NAME 2': player2,
            'LINE 1': line1,
            'LINE 2': line2,
            'PREDICTION 1': pred1,
            'PREDICTION 2': pred2,
            'MODEL SIDE 1': model_side1,
            'MODEL SIDE 2': model_side2,
            'PROB 1': row.get('PROB 1', row.get('prob1')),
            'PROB 2': row.get('PROB 2', row.get('prob2')),
            'PROB BOTH': row.get('PROB BOTH', row.get('prob_both')),
            'EDGE 1': edge1,
            'EDGE 2': edge2,
            'COMBINED EDGE': combined_edge,
            'EV$': row.get('EV$', row.get('ev_percent', 0)),
            'KELLY FULL': row.get('KELLY FULL', row.get('kelly_full', 0)),
            'RECOMMENDATION': row.get('RECOMMENDATION', row.get('recommendation')),
            'INTERVAL WIDTH 1': row.get('INTERVAL WIDTH 1', None),
            'INTERVAL WIDTH 2': row.get('INTERVAL WIDTH 2', None),
            'SIGMA 1': row.get('SIGMA 1', None),
            'SIGMA 2': row.get('SIGMA 2', None),
            'SIGMA FLAG 1': row.get('SIGMA FLAG 1', None),
            'SIGMA FLAG 2': row.get('SIGMA FLAG 2', None),
            'EXPECTED ROI': row.get('EXPECTED ROI', None),
            'SIMULATION METHOD': row.get('SIMULATION METHOD', row.get('simulation_method')),
            # Backtest-specific columns
            'actual1': actual1,
            'actual2': actual2,
            'leg1_won': won1,
            'leg2_won': won2,
            'pair_won': pair_won,
            'pair_recommendation': row.get('RECOMMENDATION', row.get('recommendation')),
            'pair_profit': row.get('profit', 0),
            'selection': row['selection'],
            'date': gameDate,
        })
    print(f"Time taken for backtest_results: {time.time() - total_start} seconds")
    return pd.DataFrame(backtest_results)

def backtestTrios(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, stat_col='PTS', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100):
    data = data[data['GAME_DATE'] == gameDate]
    if stat_col == 'PTS':
        category = 'player_points'
    elif stat_col == 'REB':
        category = 'player_rebounds'
    else:
        print(f"Invalid stat column: {stat_col}")
        return pd.DataFrame()
        
    backtestData = backtestData[(backtestData['CATEGORY'] == category) & (backtestData['GAME_DATE'] == gameDate)]
    
    if backtestData.empty:
        print(f"No bets found for {gameDate}")
        return pd.DataFrame()

    # Get PrizePicks 3-leg EV calculations using updated function
    results = calculate3Legs(
        data=data,
        backtestData=backtestData,
        gameDate=gameDate,
        models=models,
        features=features,
        edge_threshold=edge_threshold,
        top_n=top_n,
        variance_inflation=variance_inflation,
        stat_col=stat_col,
        use_monte_carlo=use_monte_carlo,
        n_simulations=n_simulations,
        max_kelly=max_kelly,
        stake=stake
    )
    
    # Sort by combined edge and take top bets
    # Use EV$ column name (matching calculateEVS.py format)
    ev_col = 'EV$' if 'EV$' in results.columns else 'ev_percent'
    evData = results.sort_values(by=ev_col, ascending=False).head(top_n)
    
    if evData.empty:
        print(f"No valid 3-leg parlays found for {gameDate}")
        return pd.DataFrame()
    
    backtest_results = []

    for idx, row in evData.iterrows():
        # Use new column names (matching calculateEVS.py format) with fallback to old names
        player1 = row.get('NAME 1', row.get('player1'))
        player2 = row.get('NAME 2', row.get('player2'))
        player3 = row.get('NAME 3', row.get('player3'))
        
        line1 = row.get('LINE 1', row.get('line1'))
        line2 = row.get('LINE 2', row.get('line2'))
        line3 = row.get('LINE 3', row.get('line3'))
        
        pred1 = row.get('PREDICTION 1', row.get('pred1'))
        pred2 = row.get('PREDICTION 2', row.get('pred2'))
        pred3 = row.get('PREDICTION 3', row.get('pred3'))
        
        # Get model's recommended sides
        model_side1 = row.get('MODEL SIDE 1', row.get('model_side1'))
        model_side2 = row.get('MODEL SIDE 2', row.get('model_side2'))
        model_side3 = row.get('MODEL SIDE 3', row.get('model_side3'))
        
        # Get actual results for all three players
        player1Data = data[data['PLAYER_NAME'] == player1]
        player2Data = data[data['PLAYER_NAME'] == player2]
        player3Data = data[data['PLAYER_NAME'] == player3]

        if player1Data.empty or player2Data.empty or player3Data.empty:
            print(f"Player data not found for {player1}, {player2}, or {player3} on {gameDate}")
            continue

        if stat_col == 'PTS':
            actual1 = player1Data['PTS'].iloc[-1]
            actual2 = player2Data['PTS'].iloc[-1]
            actual3 = player3Data['PTS'].iloc[-1]
        elif stat_col == 'REB':
            actual1 = player1Data['REB'].iloc[-1]
            actual2 = player2Data['REB'].iloc[-1]
            actual3 = player3Data['REB'].iloc[-1]
        else:
            print(f"Invalid stat column: {stat_col}")
            continue
        
        # Calculate edges for each leg (use new column names with fallback)
        edge1 = row.get('EDGE 1', row.get('edge1'))
        edge2 = row.get('EDGE 2', row.get('edge2'))
        edge3 = row.get('EDGE 3', row.get('edge3'))
        combined_edge = row.get('COMBINED EDGE', row.get('combined_edge'))
        
        # Determine if each leg wins
        # Leg 1
        if model_side1 == 'over':
            won1 = 1 if actual1 > line1 else 0
        else:  # UNDER
            won1 = 1 if actual1 < line1 else 0

        # Leg 2
        if model_side2 == 'over':
            won2 = 1 if actual2 > line2 else 0
        else:  # UNDER
            won2 = 1 if actual2 < line2 else 0
        
        # Leg 3
        if model_side3 == 'over':
            won3 = 1 if actual3 > line3 else 0
        else:  # UNDER
            won3 = 1 if actual3 < line3 else 0
        
        
        # Parlay wins only if all three legs win
        parlay_won = 1 if (won1 == 1 and won2 == 1 and won3 == 1) else 0

        backtest_results.append({
            # New format columns (matching calculateEVS.py)
            'NAME 1': player1,
            'NAME 2': player2,
            'NAME 3': player3,
            'LINE 1': line1,
            'LINE 2': line2,
            'LINE 3': line3,
            'PREDICTION 1': pred1,
            'PREDICTION 2': pred2,
            'PREDICTION 3': pred3,
            'MODEL SIDE 1': model_side1,
            'MODEL SIDE 2': model_side2,
            'MODEL SIDE 3': model_side3,
            'PROB 1': row.get('PROB 1', row.get('prob1')),
            'PROB 2': row.get('PROB 2', row.get('prob2')),
            'PROB 3': row.get('PROB 3', row.get('prob3')),
            'PROB ALL THREE': row.get('PROB ALL THREE', row.get('prob_all_three')),
            'EDGE 1': edge1,
            'EDGE 2': edge2,
            'EDGE 3': edge3,
            'COMBINED EDGE': combined_edge,
            'EV$': row.get('EV$', row.get('ev_percent', 0)),
            'KELLY FULL': row.get('KELLY FULL', row.get('kelly_full', 0)),
            'RECOMMENDATION': row.get('RECOMMENDATION', row.get('recommendation')),
            'CONFIDENCE INTERVAL 1': row.get('CONFIDENCE INTERVAL 1', None),
            'CONFIDENCE INTERVAL 2': row.get('CONFIDENCE INTERVAL 2', None),
            'CONFIDENCE INTERVAL 3': row.get('CONFIDENCE INTERVAL 3', None),
            'INTERVAL WIDTH 1': row.get('INTERVAL WIDTH 1', None),
            'INTERVAL WIDTH 2': row.get('INTERVAL WIDTH 2', None),
            'INTERVAL WIDTH 3': row.get('INTERVAL WIDTH 3', None),
            'SIGMA 1': row.get('SIGMA 1', None),
            'SIGMA 2': row.get('SIGMA 2', None),
            'SIGMA 3': row.get('SIGMA 3', None),
            'SIGMA FLAG 1': row.get('SIGMA FLAG 1', None),
            'SIGMA FLAG 2': row.get('SIGMA FLAG 2', None),
            'SIGMA FLAG 3': row.get('SIGMA FLAG 3', None),
            'EXPECTED ROI': row.get('EXPECTED ROI', None),
            'SIMULATION METHOD': row.get('SIMULATION METHOD', row.get('simulation_method')),
            # Backtest-specific columns
            'actual1': actual1,
            'actual2': actual2,
            'actual3': actual3,
            'leg1_won': won1,
            'leg2_won': won2,
            'leg3_won': won3,
            'parlay_won': parlay_won,
            'parlay_recommendation': row.get('RECOMMENDATION', row.get('recommendation')),
            'parlay_profit': row.get('profit', 0),
            'date': gameDate,
        })

    return pd.DataFrame(backtest_results)

def calculate_advanced_metrics(results_df, stake=5):
    """
    Calculate advanced betting metrics including ROI, volatility, max drawdown, and Sharpe ratio
    """
    if results_df.empty:
        return {}
    
    total_bets = len(results_df)
    total_wins = results_df['won'].sum()
    win_rate = results_df['won'].mean()
    
    results_df = results_df.copy()
    def calculate_profit(row):
        if row['won'] == 1:
            odds = row['odds']
            if odds > 0:  # Positive odds (e.g., +150)
                return (odds / 100) * stake
            else:  # Negative odds (e.g., -200)
                return (stake / (abs(odds) / 100))
        else:
            return -stake
    
    results_df['profit'] = results_df.apply(calculate_profit, axis=1)
    
    # Total profit and ROI
    total_profit = results_df['profit'].sum()
    total_staked = total_bets * stake
    roi_percent = (total_profit / total_staked) * 100
    
    # Daily P&L (group by date)
    daily_pnl = results_df.groupby('date')['profit'].sum().reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl = daily_pnl.sort_values('date')
    
    # Volatility (standard deviation of daily P&L)
    volatility = daily_pnl['profit'].std()
    
    # Max drawdown (largest peak-to-trough loss)
    daily_pnl['cumulative_profit'] = daily_pnl['profit'].cumsum()
    daily_pnl['running_max'] = daily_pnl['cumulative_profit'].expanding().max()
    daily_pnl['drawdown'] = daily_pnl['cumulative_profit'] - daily_pnl['running_max']
    max_drawdown = daily_pnl['drawdown'].min()
    
    # Sharpe ratio (assuming 252 trading days per year)
    if volatility > 0:
        daily_return = daily_pnl['profit'].mean()
        sharpe_ratio = (daily_return / volatility) * np.sqrt(len(daily_pnl))
    else:
        sharpe_ratio = 0
    
    return {
        'total_bets': total_bets,
        'total_wins': total_wins,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi_percent': roi_percent,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'daily_pnl': daily_pnl
    }
    
def calculatePairMetrics(results_df, stake=5):
    """
    Calculate advanced betting metrics including ROI, volatility, max drawdown, Sharpe ratio,
    and probability metrics (AUC-ROC, log loss, Brier score)
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    
    if results_df.empty:
        return {}
    
    total_bets = len(results_df)
    total_wins = results_df['pair_won'].sum()
    win_rate = results_df['pair_won'].mean()
    
    # Calculate probability metrics (AUC-ROC, log loss, Brier score)
    # These metrics evaluate how well the predicted probabilities match actual outcomes
    prob_metrics = {}
    # Check for both old and new column names
    prob_col = 'PROB BOTH' if 'PROB BOTH' in results_df.columns else 'prob_both'
    if prob_col in results_df.columns and 'pair_won' in results_df.columns:
        prob_pred = results_df[prob_col].values
        y_actual = results_df['pair_won'].values
        
        # Filter out NaN/infinite values
        valid_mask = np.isfinite(prob_pred) & np.isfinite(y_actual)
        prob_pred_clean = prob_pred[valid_mask]
        y_actual_clean = y_actual[valid_mask]
        
        if len(prob_pred_clean) > 0:
            n_positives = np.sum(y_actual_clean)
            n_negatives = len(y_actual_clean) - n_positives
            
            # Brier score (lower is better, range: 0 to 1)
            prob_metrics['brier_score'] = float(brier_score_loss(y_actual_clean, prob_pred_clean))
            
            # Log loss (lower is better)
            # Clip probabilities to avoid log(0) or log(1)
            prob_clipped = np.clip(prob_pred_clean, 1e-15, 1 - 1e-15)
            if n_positives > 0 and n_negatives > 0:
                prob_metrics['log_loss'] = float(log_loss(y_actual_clean, prob_clipped))
            else:
                prob_metrics['log_loss'] = float('nan')
                prob_metrics['log_loss_note'] = (
                    "Log loss undefined with a single class present "
                    f"(positives: {n_positives}, negatives: {n_negatives})"
                )
            
            # AUC-ROC (higher is better, range: 0 to 1)
            # Only calculate if we have both positive and negative cases
            if n_positives > 0 and n_negatives > 0:
                prob_metrics['auc_roc'] = float(roc_auc_score(y_actual_clean, prob_pred_clean))
            else:
                prob_metrics['auc_roc'] = float('nan')
                prob_metrics['auc_roc_note'] = f"Only one class present (positives: {n_positives}, negatives: {n_negatives})"
            
            prob_metrics['n_samples'] = len(prob_pred_clean)
            prob_metrics['n_positives'] = int(n_positives)
            prob_metrics['n_negatives'] = int(n_negatives)
        else:
            prob_metrics['error'] = "No valid predictions found"
    else:
        prob_metrics['error'] = "Missing required columns: 'prob_both' or 'pair_won'"
    
    results_df = results_df.copy()
    def calculate_profit(row):
        if row['pair_won'] == 1:
            # PrizePicks 2-leg pays 3x stake (profit = 2x stake)
            return 2 * stake
        else:
            return -stake
    
    results_df['profit'] = results_df.apply(calculate_profit, axis=1)
    
    # Total profit and ROI
    total_profit = results_df['profit'].sum()
    total_staked = total_bets * stake
    roi_percent = (total_profit / total_staked) * 100
    
    # Daily P&L (group by date)
    daily_pnl = results_df.groupby('date')['profit'].sum().reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl = daily_pnl.sort_values('date')
    
    # Volatility (standard deviation of daily P&L)
    volatility = daily_pnl['profit'].std()
    
    # Max drawdown (largest peak-to-trough loss)
    daily_pnl['cumulative_profit'] = daily_pnl['profit'].cumsum()
    daily_pnl['running_max'] = daily_pnl['cumulative_profit'].expanding().max()
    daily_pnl['drawdown'] = daily_pnl['cumulative_profit'] - daily_pnl['running_max']
    max_drawdown = daily_pnl['drawdown'].min()
    
    # Sharpe ratio (annualized, assuming ~365 betting days per year for NBA)
    # Sharpe = (mean_return / std_return) * sqrt(periods_per_year)
    # Using sqrt(365) to annualize daily P&L data
    # Note: This assumes daily betting opportunities. For NBA, there are ~365 days/year
    if volatility > 0:
        daily_return = daily_pnl['profit'].mean()
        sharpe_ratio = (daily_return / volatility) * np.sqrt(365)
    else:
        sharpe_ratio = 0
    
    return {
        'total_bets': total_bets,
        'total_wins': total_wins,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi_percent': roi_percent,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'daily_pnl': daily_pnl,
        'probability_metrics': prob_metrics  # AUC-ROC, log loss, Brier score
    }


def calculate3LegMetrics(results_df, stake=5):
    """
    Calculate advanced betting metrics for 3-leg parlays with 6x payout,
    including probability metrics (AUC-ROC, log loss, Brier score)
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    
    if results_df.empty:
        return {}
    
    total_bets = len(results_df)
    total_wins = results_df['parlay_won'].sum()
    win_rate = results_df['parlay_won'].mean()
    
    # Calculate probability metrics (AUC-ROC, log loss, Brier score)
    # These metrics evaluate how well the predicted probabilities match actual outcomes
    prob_metrics = {}
    # Check for both old and new column names
    prob_col = 'PROB ALL THREE' if 'PROB ALL THREE' in results_df.columns else 'prob_all_three'
    if prob_col in results_df.columns and 'parlay_won' in results_df.columns:
        prob_pred = results_df[prob_col].values
        y_actual = results_df['parlay_won'].values
        
        # Filter out NaN/infinite values
        valid_mask = np.isfinite(prob_pred) & np.isfinite(y_actual)
        prob_pred_clean = prob_pred[valid_mask]
        y_actual_clean = y_actual[valid_mask]
        
        if len(prob_pred_clean) > 0:
            n_positives = np.sum(y_actual_clean)
            n_negatives = len(y_actual_clean) - n_positives
            
            # Brier score (lower is better, range: 0 to 1)
            prob_metrics['brier_score'] = float(brier_score_loss(y_actual_clean, prob_pred_clean))
            
            # Log loss (lower is better)
            # Clip probabilities to avoid log(0) or log(1)
            prob_clipped = np.clip(prob_pred_clean, 1e-15, 1 - 1e-15)
            prob_metrics['log_loss'] = float(log_loss(y_actual_clean, prob_clipped))
            
            # AUC-ROC (higher is better, range: 0 to 1)
            # Only calculate if we have both positive and negative cases
            if n_positives > 0 and n_negatives > 0:
                prob_metrics['auc_roc'] = float(roc_auc_score(y_actual_clean, prob_pred_clean))
            else:
                prob_metrics['auc_roc'] = float('nan')
                prob_metrics['auc_roc_note'] = f"Only one class present (positives: {n_positives}, negatives: {n_negatives})"
            
            prob_metrics['n_samples'] = len(prob_pred_clean)
            prob_metrics['n_positives'] = int(n_positives)
            prob_metrics['n_negatives'] = int(n_negatives)
        else:
            prob_metrics['error'] = "No valid predictions found"
    else:
        prob_metrics['error'] = "Missing required columns: 'prob_all_three' or 'parlay_won'"
    
    results_df = results_df.copy()
    def calculate_profit(row):
        if row['parlay_won'] == 1:
            # 3-leg pays 6x stake (profit = 5x stake)
            return 5 * stake
        else:
            return -stake
    
    results_df['profit'] = results_df.apply(calculate_profit, axis=1)
    
    # Total profit and ROI
    total_profit = results_df['profit'].sum()
    total_staked = total_bets * stake
    roi_percent = (total_profit / total_staked) * 100
    
    # Daily P&L (group by date)
    daily_pnl = results_df.groupby('date')['profit'].sum().reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl = daily_pnl.sort_values('date')
    
    # Volatility (standard deviation of daily P&L)
    volatility = daily_pnl['profit'].std()
    
    # Max drawdown (largest peak-to-trough loss)
    daily_pnl['cumulative_profit'] = daily_pnl['profit'].cumsum()
    daily_pnl['running_max'] = daily_pnl['cumulative_profit'].expanding().max()
    daily_pnl['drawdown'] = daily_pnl['cumulative_profit'] - daily_pnl['running_max']
    max_drawdown = daily_pnl['drawdown'].min()
    
    # Sharpe ratio
    if volatility > 0:
        daily_return = daily_pnl['profit'].mean()
        sharpe_ratio = (daily_return / volatility) * np.sqrt(len(daily_pnl))
    else:
        sharpe_ratio = 0
    
    return {
        'total_bets': total_bets,
        'total_wins': total_wins,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi_percent': roi_percent,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'daily_pnl': daily_pnl,
        'probability_metrics': prob_metrics
    }