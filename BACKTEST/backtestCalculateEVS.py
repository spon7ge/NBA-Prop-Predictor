import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import truncnorm
from nba_api.stats.endpoints import scoreboardv2
from MODELS.teamInfo import *


# Convert UTC to ET and create game_date column
def convert_to_et(utc_time):
    utc_dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
    et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
    return et_dt.strftime('%Y-%m-%d')  

def impliedProb(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds):
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def kelly_criterion(probability, payout, stake, kelly_fraction=1.0):
    netProfit = payout - stake
    probabilityOfLoss = 1 - probability
    kelly = (netProfit * probability - probabilityOfLoss) / netProfit
    return max(0, round(kelly * kelly_fraction, 4))


def fairProb(bookmakersData, name, line, category, over_under, fixed_buffer=0.035):
    df = bookmakersData[
        (bookmakersData['NAME'] == name) &
        (bookmakersData['CATEGORY'] == category)
    ]
    
    grouped_df = df.groupby('LINE').agg({
        'BOOKMAKER': list,  # Collect all bookmakers offering the same line
        'ODDS': list,
        'OVER/UNDER': list
    }).reset_index() 

    res = []
    for idx, row in grouped_df.iterrows():
        if line == row['LINE']:
            for odds, OU in zip(row['ODDS'], row['OVER/UNDER']):
                if OU == over_under:
                    res.append(round(impliedProb(odds), 2))
    
    # Apply a fixed buffer for one-sided props
    adjusted_probs = [prob - fixed_buffer for prob in res]

    # Calculate the fair odds
    if len(adjusted_probs) == 0:
        raise ValueError("No valid probabilities found for the given line and over/under condition.")
    
    fair_odds = sum(adjusted_probs) / len(adjusted_probs)
    
    if fair_odds == 0:
        raise ValueError("Calculated fair probability is zero, cannot convert to odds.")
    
    odds_to_decimal = 1 / fair_odds
    
    # Convert to American odds
    if odds_to_decimal == 2.0:
        return +100
    elif odds_to_decimal > 2.0:
        return round((odds_to_decimal - 1) * 100)
    else:
        return round(-100 / (odds_to_decimal - 1))

def predictStats(playerName, data, models, features):
    playerData = data[data['PLAYER_NAME'] == playerName]
    sorted_data = playerData.sort_values(by='GAME_DATE')
    
    # Get latest row for game context and opponent context
    latestRow = sorted_data.iloc[-1]
    
    # Get second-to-latest row for all other features
    secondLatestRow = sorted_data.iloc[-2] if len(sorted_data) > 1 else latestRow
    
    # Define feature categories
    game_context_features = [
        'HOME_GAME', 'STARTING', 'PLAYER_IS_TEAM_STAR', 'STAR_SAT_OUT', 
        'IS_BACK_TO_BACK', 'PLAYER_DAYS_REST', 'spread', 'TEAM_IMPLIED_PTS'
    ]
    
    opponent_context_features = [
        'OPP_DEF_RATING_AVG_TO_DATE', 'OPP_PACE_AVG_TO_DATE', 'OPP_BLK_AVG_TO_DATE', 
        'OPP_TOV_AVG_TO_DATE', 'OPP_GUARD_DEF_RATING', 'OPP_GUARD_DEF_FG_PCT_ALLOWED',
        'OPP_GUARD_DEF_3PT_PCT_ALLOWED', 'OPP_GUARD_PTS_ALLOWED_PER_MIN',
        'OPP_FORWARD_DEF_RATING', 'OPP_FORWARD_DEF_FG_PCT_ALLOWED',
        'OPP_FORWARD_DEF_3PT_PCT_ALLOWED', 'OPP_FORWARD_PTS_ALLOWED_PER_MIN',
        'OPP_CENTER_DEF_RATING', 'OPP_CENTER_DEF_FG_PCT_ALLOWED',
        'OPP_CENTER_DEF_3PT_PCT_ALLOWED', 'OPP_CENTER_PTS_ALLOWED_PER_MIN',
        'PTS_PER_MIN_X_OPP_DEF_RATING'
    ]
    
    available_features = [f for f in features if f in data.columns]
    playerInput = {}
    
    # Use latest row for game context and opponent context
    for feature in available_features:
        if feature in game_context_features or feature in opponent_context_features:
            playerInput[feature] = latestRow[feature]
        else:
            # Use second-to-latest row for all other features
            playerInput[feature] = secondLatestRow[feature]
    
    playerInput_df = pd.DataFrame([list(playerInput.values())], columns=list(playerInput.keys()))
    
    for col in playerInput_df.columns:
        if playerInput_df[col].dtype == 'object':
            playerInput_df[col] = pd.to_numeric(playerInput_df[col], errors='coerce').fillna(0)
        elif playerInput_df[col].dtype == 'bool':
            playerInput_df[col] = playerInput_df[col].astype(int)
    
    # Get predictions from all quantile models
    predictions = {}
    for quantile_name, model in models.items():
        pred = model.predict(playerInput_df)[0]
        predictions[quantile_name] = float(pred)
    
    return predictions


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def backtestSingleBet(data, bookmakers, models, features, edge_threshold=0.05, stake=100, 
                     variance_inflation=1.1, distribution_type='normal', stat_col='PTS'):
    print("Processing single bets with quantile models...")
    
    results = []
    
    for _, row in bookmakers.iterrows():
        name = row['NAME']
        bookmaker = row['BOOKMAKER']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')
        odds = int(row['ODDS'])
        
        # Handle name variations
        if name == 'Nikola Jokic':
            name = 'Nikola Jokić'
        elif name == 'Luka Doncic':
            name = 'Luka Dončić'
        elif name == 'Kristaps Porzingis':
            name = 'Kristaps Porziņģis'
        elif name == 'Alperen Sengun':
            name = 'Alperen Şengün'
        elif name == 'Nikola Vucevic':
            name = 'Nikola Vučević'
        
        # Get player data
        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue
        
        # Get quantile predictions using updated function
        try:
            predictions = predictStats(name, data, models, features)
            q10_pred = predictions['q10']
            q50_pred = predictions['q50']
            q90_pred = predictions['q90']
            
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue
        
        # Convert quantiles to distribution parameters
        mu = q50_pred  # Median as mean
        sigma_raw = (q90_pred - q10_pred) / 2.56  # 80% interval to std dev
        sigma = sigma_raw * variance_inflation  # Apply variance inflation
        
        # Calculate probabilities using the distribution
        if distribution_type == 'normal':
            from scipy.stats import norm
            p_over = 1 - norm.cdf(line, mu, sigma)
        elif distribution_type == 't':
            from scipy.stats import t
            df = max(3, 2 * sigma**2 / (sigma**2 - 1))  # Rough estimate
            p_over = 1 - t.cdf(line, df, mu, sigma)
        else:
            raise ValueError("distribution_type must be 'normal' or 't'")
        
        p_under = 1.0 - p_over
        
        # Choose probability based on the offered side
        if str(side).upper().startswith('O'):
            p = p_over
        else:
            p = p_under
        
        # Convert odds to decimal and calculate EV
        dec_odds = american_to_decimal(odds)
        b = dec_odds - 1.0
        
        # EV calculations
        ev_per_unit = p * b - (1 - p)
        ev_percent = ev_per_unit * 100
        
        # Kelly fraction
        kelly_full = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
        
        # Edge calculation (difference between model and market probabilities)
        market_prob = impliedProb(odds)
        model_prob = p_over if str(side).upper().startswith('O') else p_under
        edge = model_prob - market_prob
        
        # Recommendation based on edge threshold
        if edge > edge_threshold:
            recommendation = 1
        else:
            recommendation = 0
        
        # Confidence interval (using quantiles)
        confidence_interval = (q10_pred, q90_pred)
        
        results.append({
            'NAME': name,
            'BOOKMAKER': bookmaker,
            'CATEGORY': category,
            'LINE': line,
            'ODDS': odds,
            'SIDE': side,
            'PREDICTION': round(q50_pred, 2),
            'Q10': round(q10_pred, 2),
            'Q90': round(q90_pred, 2),
            'RECOMMENDATION': recommendation,
            'OVER%': round(p_over, 3),
            'UNDER%': round(p_under, 3),
            'IMPLIED PROB': round(market_prob, 3),
            'MODEL PROB': round(model_prob, 3),
            'EDGE': round(edge, 3),
            'EV%': round(ev_percent, 2),
            'KELLY FULL': round(kelly_full, 2),
            'KELLY HALF': round(0.5 * kelly_full, 2),
            'KELLY QUARTER': round(0.25 * kelly_full, 2),
            'CONFIDENCE INTERVAL': f"({confidence_interval[0]:.1f}, {confidence_interval[1]:.1f})",
            'SIGMA': round(sigma, 2)
        })
    
    return pd.DataFrame(results)
    
def backtest2legs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', stat_col='PTS'):
    data = data[data['GAME_DATE'] <= gameDate]
    category = 'player_points'
    backtestData = backtestData[(backtestData['CATEGORY'] == category) & (backtestData['GAME_DATE'] == gameDate)]
    if backtestData.empty:
        print(f"No bets found for {gameDate}")
        return pd.DataFrame()

    # Get all available players for 2-leg combinations
    available_players = backtestData['NAME'].unique()
    if len(available_players) < 2:
        print("Not enough players for 2-leg bets")
        return pd.DataFrame()

    results = []
    
    # Generate all 2-leg combinations
    for i in range(len(available_players)):
        for j in range(i + 1, len(available_players)):
            player1 = available_players[i]
            player2 = available_players[j]
            
            # Get player data
            player1_data = data[data['PLAYER_NAME'] == player1]
            player2_data = data[data['PLAYER_NAME'] == player2]
            
            if player1_data.empty or player2_data.empty:
                continue
                
            # Get betting lines for both players
            player1_bets = backtestData[backtestData['NAME'] == player1]
            player2_bets = backtestData[backtestData['NAME'] == player2]
            
            if player1_bets.empty or player2_bets.empty:
                continue
            
            # Use the first available line for each player
            player1_line = player1_bets.iloc[0]
            player2_line = player2_bets.iloc[0]
            
            # Get quantile predictions for both players
            try:
                pred1 = predictStats(player1, data, models, features)
                pred2 = predictStats(player2, data, models, features)
                
                q10_1, q50_1, q90_1 = pred1['q10'], pred1['q50'], pred1['q90']
                q10_2, q50_2, q90_2 = pred2['q10'], pred2['q50'], pred2['q90']
                
            except Exception as e:
                print(f"Error getting predictions for {player1} or {player2}: {e}")
                continue
            
            # Convert quantiles to distribution parameters for both players
            mu1 = q50_1
            sigma1_raw = (q90_1 - q10_1) / 2.56
            sigma1 = sigma1_raw * variance_inflation
            
            mu2 = q50_2
            sigma2_raw = (q90_2 - q10_2) / 2.56
            sigma2 = sigma2_raw * variance_inflation
            
            # Calculate probabilities for both players
            if distribution_type == 'normal':
                from scipy.stats import norm
                p1_over = 1 - norm.cdf(player1_line['LINE'], mu1, sigma1)
                p2_over = 1 - norm.cdf(player2_line['LINE'], mu2, sigma2)
            elif distribution_type == 't':
                from scipy.stats import t
                df1 = max(3, 2 * sigma1**2 / (sigma1**2 - 1))
                df2 = max(3, 2 * sigma2**2 / (sigma2**2 - 1))
                p1_over = 1 - t.cdf(player1_line['LINE'], df1, mu1, sigma1)
                p2_over = 1 - t.cdf(player2_line['LINE'], df2, mu2, sigma2)
            else:
                raise ValueError("distribution_type must be 'normal' or 't'")
            
            # Determine model sides based on predictions vs lines
            if q50_1 > player1_line['LINE']:
                model_side1 = 'over'
                p1 = p1_over
            else:
                model_side1 = 'under'
                p1 = 1 - p1_over
                
            if q50_2 > player2_line['LINE']:
                model_side2 = 'over'
                p2 = p2_over
            else:
                model_side2 = 'under'
                p2 = 1 - p2_over
            
            # Calculate combined probability and EV
            p_both = p1 * p2
            payout_multiple = 3.0  # 3x payout
            ev = payout_multiple * p_both - 1
            
            # Kelly criterion
            b = payout_multiple - 1.0  # b = 2.0
            kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b) if b > 0 else 0.0
            
            # Edge calculation (probability edge for both players)
            market_prob1 = impliedProb(-137)  # Fixed odds
            market_prob2 = impliedProb(-137)  # Fixed odds
            edge1 = p1 - market_prob1
            edge2 = p2 - market_prob2
            combined_edge = (edge1 + edge2) / 2  # Average edge
            
            # Recommendation based on edge threshold
            recommendation = 1 if combined_edge > edge_threshold else 0
            
            # Get actual results
            actual1 = player1_data[player1_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player1_data[player1_data['GAME_DATE'] == gameDate]) > 0 else None
            actual2 = player2_data[player2_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player2_data[player2_data['GAME_DATE'] == gameDate]) > 0 else None
            
            if actual1 is None or actual2 is None:
                continue
            
            # Determine if bet won
            won1 = (actual1 > player1_line['LINE']) if model_side1 == 'over' else (actual1 < player1_line['LINE'])
            won2 = (actual2 > player2_line['LINE']) if model_side2 == 'over' else (actual2 < player2_line['LINE'])
            won_both = won1 and won2
            
            results.append({
                'player1': player1,
                'player2': player2,
                'line1': player1_line['LINE'],
                'line2': player2_line['LINE'],
                'pred1': round(q50_1, 2),
                'pred2': round(q50_2, 2),
                'q10_1': round(q10_1, 2),
                'q90_1': round(q90_1, 2),
                'q10_2': round(q10_2, 2),
                'q90_2': round(q90_2, 2),
                'model_side1': model_side1,
                'model_side2': model_side2,
                'prob1': round(p1, 3),
                'prob2': round(p2, 3),
                'prob_both': round(p_both, 4),
                'edge1': round(edge1, 3),
                'edge2': round(edge2, 3),
                'combined_edge': round(combined_edge, 3),
                'ev_percent': round(ev * 100, 2),
                'kelly_full': round(kelly_full, 3),
                'recommendation': recommendation,
                'actual1': actual1,
                'actual2': actual2,
                'won1': won1,
                'won2': won2,
                'won_both': won_both,
                'date': gameDate
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def backtest3Legs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', stat_col='PTS'):
    data = data[data['GAME_DATE'] <= gameDate]
    category = 'player_points'
    backtestData = backtestData[(backtestData['CATEGORY'] == category) & (backtestData['GAME_DATE'] == gameDate)]
    if backtestData.empty:
        print(f"No bets found for {gameDate}")
        return pd.DataFrame()

    # Get all available players for 3-leg combinations
    available_players = backtestData['NAME'].unique()
    if len(available_players) < 3:
        print("Not enough players for 3-leg bets")
        return pd.DataFrame()

    results = []
    
    # Generate all 3-leg combinations
    for i in range(len(available_players)):
        for j in range(i + 1, len(available_players)):
            for k in range(j + 1, len(available_players)):
                player1 = available_players[i]
                player2 = available_players[j]
                player3 = available_players[k]
                
                # Get player data
                player1_data = data[data['PLAYER_NAME'] == player1]
                player2_data = data[data['PLAYER_NAME'] == player2]
                player3_data = data[data['PLAYER_NAME'] == player3]
                
                if player1_data.empty or player2_data.empty or player3_data.empty:
                    continue
                    
                # Get betting lines for all three players
                player1_bets = backtestData[backtestData['NAME'] == player1]
                player2_bets = backtestData[backtestData['NAME'] == player2]
                player3_bets = backtestData[backtestData['NAME'] == player3]
                
                if player1_bets.empty or player2_bets.empty or player3_bets.empty:
                    continue
                
                # Use the first available line for each player
                player1_line = player1_bets.iloc[0]
                player2_line = player2_bets.iloc[0]
                player3_line = player3_bets.iloc[0]
                
                # Get quantile predictions for all three players
                try:
                    pred1 = predictStats(player1, data, models, features)
                    pred2 = predictStats(player2, data, models, features)
                    pred3 = predictStats(player3, data, models, features)
                    
                    q10_1, q50_1, q90_1 = pred1['q10'], pred1['q50'], pred1['q90']
                    q10_2, q50_2, q90_2 = pred2['q10'], pred2['q50'], pred2['q90']
                    q10_3, q50_3, q90_3 = pred3['q10'], pred3['q50'], pred3['q90']
                    
                except Exception as e:
                    print(f"Error getting predictions for {player1}, {player2}, or {player3}: {e}")
                    continue
                
                # Convert quantiles to distribution parameters for all three players
                mu1 = q50_1
                sigma1_raw = (q90_1 - q10_1) / 2.56
                sigma1 = sigma1_raw * variance_inflation
                
                mu2 = q50_2
                sigma2_raw = (q90_2 - q10_2) / 2.56
                sigma2 = sigma2_raw * variance_inflation
                
                mu3 = q50_3
                sigma3_raw = (q90_3 - q10_3) / 2.56
                sigma3 = sigma3_raw * variance_inflation
                
                # Calculate probabilities for all three players
                if distribution_type == 'normal':
                    from scipy.stats import norm
                    p1_over = 1 - norm.cdf(player1_line['LINE'], mu1, sigma1)
                    p2_over = 1 - norm.cdf(player2_line['LINE'], mu2, sigma2)
                    p3_over = 1 - norm.cdf(player3_line['LINE'], mu3, sigma3)
                elif distribution_type == 't':
                    from scipy.stats import t
                    df1 = max(3, 2 * sigma1**2 / (sigma1**2 - 1))
                    df2 = max(3, 2 * sigma2**2 / (sigma2**2 - 1))
                    df3 = max(3, 2 * sigma3**2 / (sigma3**2 - 1))
                    p1_over = 1 - t.cdf(player1_line['LINE'], df1, mu1, sigma1)
                    p2_over = 1 - t.cdf(player2_line['LINE'], df2, mu2, sigma2)
                    p3_over = 1 - t.cdf(player3_line['LINE'], df3, mu3, sigma3)
                else:
                    raise ValueError("distribution_type must be 'normal' or 't'")
                
                # Determine model sides based on predictions vs lines
                if q50_1 > player1_line['LINE']:
                    model_side1 = 'over'
                    p1 = p1_over
                else:
                    model_side1 = 'under'
                    p1 = 1 - p1_over
                    
                if q50_2 > player2_line['LINE']:
                    model_side2 = 'over'
                    p2 = p2_over
                else:
                    model_side2 = 'under'
                    p2 = 1 - p2_over
                    
                if q50_3 > player3_line['LINE']:
                    model_side3 = 'over'
                    p3 = p3_over
                else:
                    model_side3 = 'under'
                    p3 = 1 - p3_over
                
                # Calculate combined probability and EV
                p_all_three = p1 * p2 * p3
                payout_multiple = 6.0  # 6x payout for 3-leg parlay
                ev = payout_multiple * p_all_three - 1
                
                # Kelly criterion
                b = payout_multiple - 1.0  # b = 5.0
                kelly_full = max(0.0, (b * p_all_three - (1 - p_all_three)) / b) if b > 0 else 0.0
                
                # Edge calculation (probability edge for all three players)
                market_prob1 = impliedProb(-137)  # Fixed odds
                market_prob2 = impliedProb(-137)  # Fixed odds
                market_prob3 = impliedProb(-137)  # Fixed odds
                edge1 = p1 - market_prob1
                edge2 = p2 - market_prob2
                edge3 = p3 - market_prob3
                combined_edge = (edge1 + edge2 + edge3) / 3  # Average edge
                
                # Recommendation based on edge threshold
                recommendation = 1 if combined_edge > edge_threshold else 0
                
                # Get actual results
                actual1 = player1_data[player1_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player1_data[player1_data['GAME_DATE'] == gameDate]) > 0 else None
                actual2 = player2_data[player2_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player2_data[player2_data['GAME_DATE'] == gameDate]) > 0 else None
                actual3 = player3_data[player3_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player3_data[player3_data['GAME_DATE'] == gameDate]) > 0 else None
                
                if actual1 is None or actual2 is None or actual3 is None:
                    continue
                
                # Determine if bet won
                won1 = (actual1 > player1_line['LINE']) if model_side1 == 'over' else (actual1 < player1_line['LINE'])
                won2 = (actual2 > player2_line['LINE']) if model_side2 == 'over' else (actual2 < player2_line['LINE'])
                won3 = (actual3 > player3_line['LINE']) if model_side3 == 'over' else (actual3 < player3_line['LINE'])
                won_all_three = won1 and won2 and won3
                
                results.append({
                    'player1': player1,
                    'player2': player2,
                    'player3': player3,
                    'line1': player1_line['LINE'],
                    'line2': player2_line['LINE'],
                    'line3': player3_line['LINE'],
                    'pred1': round(q50_1, 2),
                    'pred2': round(q50_2, 2),
                    'pred3': round(q50_3, 2),
                    'q10_1': round(q10_1, 2),
                    'q90_1': round(q90_1, 2),
                    'q10_2': round(q10_2, 2),
                    'q90_2': round(q90_2, 2),
                    'q10_3': round(q10_3, 2),
                    'q90_3': round(q90_3, 2),
                    'model_side1': model_side1,
                    'model_side2': model_side2,
                    'model_side3': model_side3,
                    'prob1': round(p1, 3),
                    'prob2': round(p2, 3),
                    'prob3': round(p3, 3),
                    'prob_all_three': round(p_all_three, 4),
                    'edge1': round(edge1, 3),
                    'edge2': round(edge2, 3),
                    'edge3': round(edge3, 3),
                    'combined_edge': round(combined_edge, 3),
                    'ev_percent': round(ev * 100, 2),
                    'kelly_full': round(kelly_full, 3),
                    'recommendation': recommendation,
                    'actual1': actual1,
                    'actual2': actual2,
                    'actual3': actual3,
                    'won1': won1,
                    'won2': won2,
                    'won3': won3,
                    'won_all_three': won_all_three,
                    'date': gameDate
                })
    
    results_df = pd.DataFrame(results)
    return results_df