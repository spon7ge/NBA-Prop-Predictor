import pandas as pd
import numpy as np
import joblib
from MODELS.pipeline import *
from BACKTEST.backtestCalculateEVS import backtest2legs as calculatePairs
from BACKTEST.backtestCalculateEVS import backtestSingleBet as calculateSingleBet
from BACKTEST.backtestCalculateEVS import backtest3Legs as calculate3Legs

def backtestSingle(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                  variance_inflation=1.1, distribution_type='normal'):
    
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
        stake=100,
        variance_inflation=variance_inflation,
        distribution_type=distribution_type
    )
    
    # Sort by edge and take top bets
    evData = results.sort_values(by='EDGE', ascending=False).head(top_n)
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
        
        # Recommendation based on edge threshold (probability edge, not point edge)
        recommendation = 1 if edge > edge_threshold else 0

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
            'kelly_full': row['KELLY FULL'],  # Kelly criterion
            'recommendation': recommendation,
            'won': won,
            'date': gameDate
        })

    return pd.DataFrame(results)


def backtestPairs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal'):
    data = data[data['GAME_DATE'] <= gameDate]
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
        distribution_type=distribution_type
    )
    
    # Sort by combined edge and take top bets
    evData = results.sort_values(by='combined_edge', ascending=False).head(top_n)
    
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
        
        actual1 = player1Data['PTS'].iloc[-1]
        actual2 = player2Data['PTS'].iloc[-1]

        
        # Calculate edges for each leg
        edge1 = row['edge1']
        edge2 = row['edge2']
        
        # Determine if each leg should be recommended based on MODEL'S recommendation
        # Use model's side instead of bookmaker's side
        if model_side1 == 'over':
            rec1 = 1 if (pred1 > line1 and edge1 > edge_threshold) else 0
            won1 = 1 if actual1 > line1 else 0
        else:  # model_side1 == 'under'
            rec1 = 1 if (pred1 < line1 and edge1 > edge_threshold) else 0
            won1 = 1 if actual1 < line1 else 0

        if model_side2 == 'over':
            rec2 = 1 if (pred2 > line2 and edge2 > edge_threshold) else 0
            won2 = 1 if actual2 > line2 else 0
        else:  # model_side2 == 'under'
            rec2 = 1 if (pred2 < line2 and edge2 > edge_threshold) else 0
            won2 = 1 if actual2 < line2 else 0
        
        # Overall pair recommendation (both legs must be recommended)
        pair_recommendation = 1 if (rec1 == 1 and rec2 == 1) else 0
        
        # Pair wins only if both legs win
        pair_won = 1 if (won1 == 1 and won2 == 1) else 0
        
        # Calculate combined edge
        combined_edge = row['combined_edge']

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
            'pair_recommendation': pair_recommendation,
            'pair_won': pair_won,
            'ev_percent': row['ev_percent'],
            'kelly_full': row['kelly_full'],
            'prob_both': row['prob_both'],
            'combined_edge': combined_edge,
            'date': gameDate
        })

    return pd.DataFrame(backtest_results)

def backtestTrios(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal'):
    data = data[data['GAME_DATE'] <= gameDate]
    category = 'player_points'
        
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
        stat_col=stat_col
    )
    
    # Sort by combined edge and take top bets
    evData = results.sort_values(by='combined_edge', ascending=False).head(top_n)
    
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
        
        # Determine if each leg should be recommended based on MODEL'S recommendation
        # Leg 1
        if model_side1 == 'over':
            rec1 = 1 if (pred1 > line1 and edge1 > edge_threshold) else 0
            won1 = 1 if actual1 > line1 else 0
        else:  # UNDER
            rec1 = 1 if (pred1 < line1 and edge1 > edge_threshold) else 0
            won1 = 1 if actual1 < line1 else 0

        # Leg 2
        if model_side2 == 'over':
            rec2 = 1 if (pred2 > line2 and edge2 > edge_threshold) else 0
            won2 = 1 if actual2 > line2 else 0
        else:  # UNDER
            rec2 = 1 if (pred2 < line2 and edge2 > edge_threshold) else 0
            won2 = 1 if actual2 < line2 else 0
        
        # Leg 3
        if model_side3 == 'over':
            rec3 = 1 if (pred3 > line3 and edge3 > edge_threshold) else 0
            won3 = 1 if actual3 > line3 else 0
        else:  # UNDER
            rec3 = 1 if (pred3 < line3 and edge3 > edge_threshold) else 0
            won3 = 1 if actual3 < line3 else 0
        
        # Overall parlay recommendation (all three legs must be recommended)
        parlay_recommendation = 1 if (rec1 == 1 and rec2 == 1 and rec3 == 1) else 0
        
        # Parlay wins only if all three legs win
        parlay_won = 1 if (won1 == 1 and won2 == 1 and won3 == 1) else 0
        
        # Calculate combined edge
        combined_edge = row['combined_edge']

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
            'parlay_recommendation': parlay_recommendation,
            'parlay_won': parlay_won,
            'ev_percent': row['ev_percent'],
            'kelly_full': row['kelly_full'],
            'prob_all_three': row['prob_all_three'],
            'combined_edge': combined_edge,
            'date': gameDate
        })

    return pd.DataFrame(backtest_results)