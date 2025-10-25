import pandas as pd
import numpy as np
import joblib
from MODELS.pipeline import *
from BACKTEST.backtestCalculateEVS import backtest2legs as calculatePairs
from BACKTEST.backtestCalculateEVS import backtestSingleBet as calculateSingleBet
from BACKTEST.backtestCalculateEVS import backtest3Legs as calculate3Legs

def backtestSingle(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                  variance_inflation=1.1, distribution_type='normal', use_monte_carlo=True, 
                  n_simulations=10000, max_kelly=0.25, stake=100):
    
    data = data[data['GAME_DATE'] <= gameDate]
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
        distribution_type=distribution_type,
        use_monte_carlo=use_monte_carlo,
        n_simulations=n_simulations,
        max_kelly=max_kelly
    )
    
    # Sort by edge and take top bets
    evData = results.sort_values(by='ev_percent', ascending=False).head(top_n)
    results = []

    for idx, row in evData.iterrows():
        player = row['NAME']
        bookmaker = row['BOOKMAKER']
        side = row['SIDE']
        line = row['LINE']
        odds = row['ODDS']
        pred = row['PREDICTION']  # This is q50 (median)
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
            'pred': pred,  # q50 prediction
            'q10': row['Q10'],  # Lower bound
            'q90': row['Q90'],  # Upper bound
            'actual': actual,
            'edge': edge,  # Probability edge
            'model_prob': row['MODEL PROB'],  # Model probability for the side
            'market_prob': row['IMPLIED PROB'],  # Market implied probability
            'ev_percent': row['EV%'],  # Expected value percentage
            'ev_total': row['EV_TOTAL'],  # Total EV in dollars
            'kelly_fraction': row['KELLY_FRACTION'],  # Kelly as fraction
            'kelly_dollars': row['KELLY_DOLLARS'],  # Kelly in dollars
            'kelly_capped_fraction': row['KELLY_CAPPED_FRACTION'],  # Kelly fraction (capped)
            'kelly_capped_dollars': row['KELLY_CAPPED_DOLLARS'],  # Kelly dollars (capped)
            'stake': stake,  # Show the stake amount
            'recommendation': recommendation,
            'won': won,
            'simulation_method': row['SIMULATION_METHOD'],  # Monte Carlo or Analytical
            'date': gameDate
        })

    return pd.DataFrame(results)


def backtestPairs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100):
                 
    data = data[data['GAME_DATE'] <= gameDate]
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
        distribution_type=distribution_type,
        stat_col=stat_col,
        use_monte_carlo=use_monte_carlo,
        n_simulations=n_simulations,
        max_kelly=max_kelly,
        stake=stake
    )
    
    # Sort by combined edge and take top bets
    evData = results.sort_values(by='ev_percent', ascending=False).head(top_n)
    
    if evData.empty:
        print(f"No valid pairs found for {gameDate}")
        return pd.DataFrame()
    
    backtest_results = []

    for idx, row in evData.iterrows():
        player1 = row['player1']
        player2 = row['player2']
        
        line1 = row['line1']
        line2 = row['line2']
        
        pred1 = row['pred1']
        pred2 = row['pred2']
        
        # Get model's recommended sides
        model_side1 = row['model_side1']
        model_side2 = row['model_side2']
        
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
        
        # Calculate edges for each leg
        edge1 = row['edge1']
        edge2 = row['edge2']
        combined_edge = row['combined_edge']
        
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
            'player1': player1,
            'player2': player2,
            'model_side1': model_side1,
            'model_side2': model_side2,
            'line1': line1,
            'line2': line2,
            'pred1': pred1,
            'pred2': pred2,
            'q10_1': row['q10_1'],
            'q90_1': row['q90_1'],
            'q10_2': row['q10_2'],
            'q90_2': row['q90_2'],
            'actual1': actual1,
            'actual2': actual2,
            'edge1': edge1,
            'edge2': edge2,
            'leg1_won': won1,
            'leg2_won': won2,
            'pair_won': pair_won,
            'pair_recommendation': row['recommendation'],
            'pair_profit': row['profit'],  
            'ev_percent': row['ev_percent'],
            'kelly_full': row['kelly_full'],
            'kelly_capped': row['kelly_capped'],
            'kelly_bet_size': row['kelly_bet_size'],
            'kelly_bet_size_full': row['kelly_bet_size_full'],
            'stake': row['stake'],
            'prob_both': row['prob_both'],
            'combined_edge': combined_edge,
            'simulation_method': row['simulation_method'],
            'date': gameDate
        })

    return pd.DataFrame(backtest_results)

def backtestTrios(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100):
    data = data[data['GAME_DATE'] <= gameDate]
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
        distribution_type=distribution_type,
        stat_col=stat_col,
        use_monte_carlo=use_monte_carlo,
        n_simulations=n_simulations,
        max_kelly=max_kelly,
        stake=stake
    )
    
    # Sort by combined edge and take top bets
    evData = results.sort_values(by='ev_percent', ascending=False).head(top_n)
    
    if evData.empty:
        print(f"No valid 3-leg parlays found for {gameDate}")
        return pd.DataFrame()
    
    backtest_results = []

    for idx, row in evData.iterrows():
        player1 = row['player1']
        player2 = row['player2']
        player3 = row['player3']
        
        line1 = row['line1']
        line2 = row['line2']
        line3 = row['line3']
        
        pred1 = row['pred1']
        pred2 = row['pred2']
        pred3 = row['pred3']
        
        # Get model's recommended sides
        model_side1 = row['model_side1']
        model_side2 = row['model_side2']
        model_side3 = row['model_side3']
        
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
        
        # Calculate edges for each leg
        edge1 = row['edge1']
        edge2 = row['edge2']
        edge3 = row['edge3']
        combined_edge = row['combined_edge']
        
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
            'player1': player1,
            'player2': player2,
            'player3': player3,
            'model_side1': model_side1,
            'model_side2': model_side2,
            'model_side3': model_side3,
            'line1': line1,
            'line2': line2,
            'line3': line3,
            'pred1': pred1,
            'pred2': pred2,
            'pred3': pred3,
            'q10_1': row['q10_1'],
            'q90_1': row['q90_1'],
            'q10_2': row['q10_2'],
            'q90_2': row['q90_2'],
            'q10_3': row['q10_3'],
            'q90_3': row['q90_3'],
            'actual1': actual1,
            'actual2': actual2,
            'actual3': actual3,
            'edge1': edge1,
            'edge2': edge2,
            'edge3': edge3,
            'leg1_won': won1,
            'leg2_won': won2,
            'leg3_won': won3,
            'parlay_recommendation': row['recommendation'],
            'parlay_won': parlay_won,
            'parlay_profit': row['profit'],  # Use profit from updated function
            'ev_percent': row['ev_percent'],
            'kelly_full': row['kelly_full'],
            'kelly_capped': row['kelly_capped'],
            'kelly_bet_size': row['kelly_bet_size'],
            'kelly_bet_size_full': row['kelly_bet_size_full'],
            'stake': row['stake'],
            'prob_all_three': row['prob_all_three'],
            'combined_edge': combined_edge,
            'simulation_method': row['simulation_method'],
            'date': gameDate
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
    Calculate advanced betting metrics including ROI, volatility, max drawdown, and Sharpe ratio
    """
    if results_df.empty:
        return {}
    
    total_bets = len(results_df)
    total_wins = results_df['pair_won'].sum()
    win_rate = results_df['pair_won'].mean()
    
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


def calculate3LegMetrics(results_df, stake=5):
    """
    Calculate advanced betting metrics for 3-leg parlays with 6x payout
    """
    if results_df.empty:
        return {}
    
    total_bets = len(results_df)
    total_wins = results_df['parlay_won'].sum()
    win_rate = results_df['parlay_won'].mean()
    
    results_df = results_df.copy()
    def calculate_profit(row):
        if row['parlay_won'] == 1:
            # PrizePicks 3-leg pays 6x stake (profit = 5x stake)
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
        'daily_pnl': daily_pnl
    }