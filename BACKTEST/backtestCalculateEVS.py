import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm
import scipy.stats as stats
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import truncnorm
from MODELS.teamInfo import *
from itertools import combinations
from MODELS.ngboostModel import predict_mean_variance_split


nameDict = {
    'Nikola Jokic': 'Nikola Jokić',
    'Luka Doncic': 'Luka Dončić',
    'Nikola Vucevic': 'Nikola Vučević',
    'Jonas Valanciunas': 'Jonas Valančiūnas',
    'Dennis Schroder': 'Dennis Schröder',
    'Kristaps Porzingis': 'Kristaps Porziņģis',
    'Bogdan Bogdanovic': 'Bogdan Bogdanović',
    'Dario Saric': 'Dario Šarić',
    'Nikola Jovic': 'Nikola Jović',
    'Vlatko Cancar': 'Vlatko Čančar',
}

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

# Global prediction cache
_prediction_cache = {}

def get_cached_prediction(player_name, data, models, features, stat_col='PTS', game_date=None):
    """Get cached prediction for a player, computing if not already cached.
    
    Returns:
        Dictionary with 'prediction' (mu/mean), 'sigma' keys (matching calculateEVS.py format)
    """
    cache_key = f"{player_name}_{stat_col}_{game_date}" if game_date else f"{player_name}_{stat_col}"
    
    if cache_key not in _prediction_cache:
        try:
            _prediction_cache[cache_key] = predictStats(player_name, data, models, features)
        except Exception as e:
            print(f"Error getting prediction for {player_name}: {e}")
            return None
    return _prediction_cache[cache_key]

def predictStats(playerName, data, models, features):
    playerData = data[data['PLAYER_NAME'] == playerName]
    sorted_data = playerData.sort_values(by='GAME_DATE')
    
    if len(sorted_data) == 0:
        raise ValueError(f"No data found for player {playerName}")
    
    # Use most recent game row (.iloc[-1]) for all features
    latestRow = sorted_data.iloc[-1]
    
    available_features = [f for f in features if f in data.columns]
    playerInput = {}
    
    # Use latest row for all features
    for feature in available_features:
        playerInput[feature] = latestRow[feature]
    
    playerInput_df = pd.DataFrame([list(playerInput.values())], columns=list(playerInput.keys()))
    
    # Clean the input data
    for col in playerInput_df.columns:
        if playerInput_df[col].dtype == 'object':
            playerInput_df[col] = pd.to_numeric(playerInput_df[col], errors='coerce').fillna(0)
        elif playerInput_df[col].dtype == 'bool':
            playerInput_df[col] = playerInput_df[col].astype(int)
    
    # Handle NaN and inf values
    playerInput_df = playerInput_df.replace([np.inf, -np.inf], np.nan)
    playerInput_df = playerInput_df.fillna(playerInput_df.median())
    
    # Get predictions from NGBoost models
    if isinstance(models, dict):
        # Check if it's the new format with 'mean' and 'variance' keys
        if 'mean' in models and 'variance' in models:
            mean_model = models['mean']
            variance_model = models['variance']
            calibration_factor = models.get('calibration_factor', 1.25)
        else:
            # Old quantile format - convert if needed
            raise ValueError("Models dict must contain 'mean' and 'variance' keys for NGBoost models")
    elif isinstance(models, tuple):
        # Tuple format: (mean_model, variance_model, calibration_factor)
        mean_model = models[0]
        variance_model = models[1]
        calibration_factor = models[2] if len(models) > 2 else 1.25
    else:
        raise ValueError("Models must be a dict with 'mean' and 'variance' keys or a tuple of (mean_model, variance_model, calibration_factor)")
    
    # Get mean and variance predictions
    mu, variance = predict_mean_variance_split(
        mean_model, variance_model, playerInput_df, features, calibration_factor
    )
    
    # Convert to scalars if arrays
    mu = float(mu[0] if isinstance(mu, (np.ndarray, pd.Series)) else mu)
    variance = float(variance[0] if isinstance(variance, (np.ndarray, pd.Series)) else variance)
    sigma = np.sqrt(variance)
    
    # Clip predictions to be non-negative (points/rebounds/assists can't be negative)
    # If prediction is negative, it likely means the player has very low historical stats
    # or the model encountered unusual feature values. Floor at 0.5 to avoid extreme edge cases.
    if mu < 0:
        warnings.warn(f"Negative prediction ({mu:.2f}) for {playerName}. Clipping to 0.5. This may indicate insufficient data or unusual feature values.")
        mu = 0.5
    elif mu < 0.5:
        # Very low but positive predictions should also be floored to avoid unrealistic values
        mu = 0.5
    
    predictions = {
        'prediction': mu,  # Match calculateEVS.py format
        'sigma': sigma
    }
    
    return predictions


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def backtestSingleBet(data, bookmakers, models, features, edge_threshold=0.05, stake=100, 
                     variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                     use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, df_t=5, skew_a=-2.0):
    print("Processing single bets with NGBoost models...")
    
    results = []
    
    for _, row in bookmakers.iterrows():
        name = row['NAME']
        bookmaker = row['BOOKMAKER']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')
        odds = int(row['ODDS'])
        
        # Handle name variations
        if name in nameDict:
            name = nameDict[name]
        
        # Get player data
        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue
        
        # Get predictions using cached function (returns prediction and sigma)
        try:
            predictions = get_cached_prediction(name, data, models, features, stat_col)
            if predictions is None:
                continue
            mu = predictions['prediction']
            sigma_raw = predictions['sigma']
            # Apply variance inflation if needed
            sigma = sigma_raw * variance_inflation
            
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue
        
        # Set random seed outside conditional for reproducibility across runs
        np.random.seed(42)  # For reproducibility
        
        # Calculate probabilities using Monte Carlo simulation or analytical method
        if use_monte_carlo:
            # Monte Carlo simulation (10k draws)
            if distribution_type == 'normal':
                simulations = np.random.normal(mu, sigma, n_simulations)
            elif distribution_type == 't':
                from scipy.stats import t
                df = df_t  # treat df as hyperparameter controlling tail thickness
                scale = sigma * np.sqrt((df - 2) / df)
                simulations = t.rvs(df, loc=mu, scale=scale, size=n_simulations, random_state=42)
            elif distribution_type == 'skew_t':
                from scipy.stats import skewnorm
                # Approximate skew-t with skew-normal using configurable skew parameter
                simulations = skewnorm.rvs(skew_a, loc=mu, scale=sigma, size=n_simulations, random_state=42)
            else:
                raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
            
            # Clip simulations at zero since points cannot be negative
            simulations = np.maximum(simulations, 0)
            p_over = np.mean(simulations > line)
        else:
            # Analytical method (original)
            if distribution_type == 'normal':
                from scipy.stats import norm
                p_over = 1 - norm.cdf(line, mu, sigma)
            elif distribution_type == 't':
                from scipy.stats import t
                df = df_t
                scale = sigma * np.sqrt((df - 2) / df)
                p_over = 1 - t.cdf(line, df, loc=mu, scale=scale)
            elif distribution_type == 'skew_t':
                from scipy.stats import skewnorm
                p_over = 1 - skewnorm.cdf(line, skew_a, loc=mu, scale=sigma)
            else:
                raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
        
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
        ev_per_dollar = p * b - (1 - p)
        ev_total = ev_per_dollar * stake  # Total EV in dollars
        
        # Kelly criterion with variance-adjusted constraint
        kelly_fraction = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
        kelly_capped_fraction = min(kelly_fraction, max_kelly)  
        kelly_dollars = kelly_capped_fraction * stake * b  
        
        # Edge calculation (difference between model and market probabilities)
        market_prob = impliedProb(odds)
        model_prob = p_over if str(side).upper().startswith('O') else p_under
        edge = model_prob - market_prob
        
        # Recommendation based on edge threshold
        if (edge > edge_threshold and 
            kelly_capped_fraction > -0.02 and
            p > 0.40 and 
            ev_total > 80.00):
            recommendation = 1
        else:
            recommendation = 0
        
        results.append({
        'NAME': name,
        'BOOKMAKER': bookmaker,
        'CATEGORY': category,
        'LINE': line,
        'ODDS': odds,
        'SIDE': side,
        'PREDICTION': round(mu, 2),
        'RECOMMENDATION': recommendation,
        'OVER%': round(p_over, 3),
        'UNDER%': round(p_under, 3),
        'IMPLIED PROB': round(market_prob, 3),
        'MODEL PROB': round(model_prob, 3),
        'EDGE': round(edge, 3),
        'EV%': round(ev_total, 2),
        'KELLY_FRACTION': round(kelly_fraction, 3),  # Kelly as fraction
        'KELLY_DOLLARS': round(kelly_dollars, 2),  # Kelly in dollars
        'SIGMA': round(sigma, 2),
        'SIMULATION_METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical'
    })
    
    return pd.DataFrame(results)    

def backtest2legs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100, df_t=5, skew_a=-2.0):
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
    player_combinations = list(combinations(available_players, 2))
    # Generate all 2-leg combinations
    for player1, player2 in player_combinations:
        
            # Handle name variations
            if player1 in nameDict:
                player1 = nameDict[player1]
            if player2 in nameDict:
                player2 = nameDict[player2]
                
            # Get player data
            player1_data = data[data['PLAYER_NAME'] == player1]
            player2_data = data[data['PLAYER_NAME'] == player2]
            
            if player1_data.empty or player2_data.empty:
                continue
            
            # Check if players are from the same team (prevent same-team combinations)
            player1_team = player1_data['TEAM_ABBREVIATION'].iloc[-1]
            player2_team = player2_data['TEAM_ABBREVIATION'].iloc[-1]
            
            if player1_team == player2_team:
                continue
                
            # Get betting lines for both players
            player1_bets = backtestData[backtestData['NAME'] == player1]
            player2_bets = backtestData[backtestData['NAME'] == player2]
            
            if player1_bets.empty or player2_bets.empty:
                continue
            
            # Use the first available line for each player
            player1_line = player1_bets.iloc[0]
            player2_line = player2_bets.iloc[0]
            
            # Get predictions for both players using cached function
            try:
                pred1 = get_cached_prediction(player1, data, models, features, stat_col, gameDate)
                pred2 = get_cached_prediction(player2, data, models, features, stat_col, gameDate)
                
                if pred1 is None or pred2 is None:
                    continue
                
                # Extract prediction (mu) and sigma directly
                mu1 = pred1['prediction']
                sigma1_raw = pred1['sigma']
                sigma1 = sigma1_raw * variance_inflation
                
                mu2 = pred2['prediction']
                sigma2_raw = pred2['sigma']
                sigma2 = sigma2_raw * variance_inflation
                
            except Exception as e:
                print(f"Error getting predictions for {player1} or {player2}: {e}")
                continue
            
            # Set random seed outside conditional for reproducibility across runs
            np.random.seed(42)  # For reproducibility
            
            # Calculate probabilities for both players using Monte Carlo or analytical method
            if use_monte_carlo:
                # Monte Carlo simulation for both players
                if distribution_type == 'normal':
                    sim1 = np.random.normal(mu1, sigma1, n_simulations)
                    sim2 = np.random.normal(mu2, sigma2, n_simulations)
                elif distribution_type == 't':
                    from scipy.stats import t
                    df1 = df_t
                    df2 = df_t
                    scale1 = sigma1 * np.sqrt((df1 - 2) / df1)
                    scale2 = sigma2 * np.sqrt((df2 - 2) / df2)
                    sim1 = t.rvs(df1, loc=mu1, scale=scale1, size=n_simulations, random_state=42)
                    sim2 = t.rvs(df2, loc=mu2, scale=scale2, size=n_simulations, random_state=42)
                elif distribution_type == 'skew_t':
                    from scipy.stats import skewnorm
                    sim1 = skewnorm.rvs(skew_a, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                    sim2 = skewnorm.rvs(skew_a, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                else:
                    raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
                
                # Clip simulations at zero since points cannot be negative
                sim1 = np.maximum(sim1, 0)
                sim2 = np.maximum(sim2, 0)
                p1_over = np.mean(sim1 > player1_line['LINE'])
                p2_over = np.mean(sim2 > player2_line['LINE'])
            else:
                # Analytical method (original)
                if distribution_type == 'normal':
                    from scipy.stats import norm
                    p1_over = 1 - norm.cdf(player1_line['LINE'], mu1, sigma1)
                    p2_over = 1 - norm.cdf(player2_line['LINE'], mu2, sigma2)
                elif distribution_type == 't':
                    from scipy.stats import t
                    df1 = df_t
                    df2 = df_t
                    scale1 = sigma1 * np.sqrt((df1 - 2) / df1)
                    scale2 = sigma2 * np.sqrt((df2 - 2) / df2)
                    p1_over = 1 - t.cdf(player1_line['LINE'], df1, loc=mu1, scale=scale1)
                    p2_over = 1 - t.cdf(player2_line['LINE'], df2, loc=mu2, scale=scale2)
                elif distribution_type == 'skew_t':
                    from scipy.stats import skewnorm
                    p1_over = 1 - skewnorm.cdf(player1_line['LINE'], skew_a, loc=mu1, scale=sigma1)
                    p2_over = 1 - skewnorm.cdf(player2_line['LINE'], skew_a, loc=mu2, scale=sigma2)
                else:
                    raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
            
            # Determine model sides based on predictions vs lines
            if mu1 > player1_line['LINE']:
                model_side1 = 'over'
                p1 = p1_over
            else:
                model_side1 = 'under'
                p1 = 1 - p1_over
                
            if mu2 > player2_line['LINE']:
                model_side2 = 'over'
                p2 = p2_over
            else:
                model_side2 = 'under'
                p2 = 1 - p2_over
            
            # Calculate confidence intervals and sigma flags for both players
            ci1 = (max(0, mu1 - 1.96 * sigma1), mu1 + 1.96 * sigma1)
            ci2 = (max(0, mu2 - 1.96 * sigma2), mu2 + 1.96 * sigma2)
            width1 = round(ci1[1] - ci1[0], 2)
            width2 = round(ci2[1] - ci2[0], 2)
            
            # Sigma flags for readability
            if sigma1 <= 5.0:
                sigma_flag1 = 'Low'
            elif sigma1 <= 6.0:
                sigma_flag1 = 'Med'
            else:
                sigma_flag1 = 'High'
            if sigma2 <= 5.0:
                sigma_flag2 = 'Low'
            elif sigma2 <= 6.0:
                sigma_flag2 = 'Med'
            else:
                sigma_flag2 = 'High'
            
            # Calculate combined probability and EV
            p_both = p1 * p2
            payout_multiple = 3.0  # 3x payout for 2-leg parlay
            ev = payout_multiple * p_both - 1
            ev_dollars = ev * stake
            
            # Kelly criterion with variance-adjusted constraint
            b = payout_multiple - 1.0  # b = 2.0
            kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b) if b > 0 else 0.0
            
            # Edge calculation (probability edge for both players)
            market_prob1 = impliedProb(-137)  # Fixed odds
            market_prob2 = impliedProb(-137)  # Fixed odds
            edge1 = p1 - market_prob1
            edge2 = p2 - market_prob2
            
            # Calculate combined probabilities and edge
            combined_model_prob = p1 * p2
            combined_market_prob = market_prob1 * market_prob2
            combined_edge = combined_model_prob - combined_market_prob
            
            # Recommendation based on multiple criteria (matching calculateEVS.py)
            if (combined_edge > 0 and 
            combined_model_prob > 0.335 and 
            ev_dollars > 0 and 
            sigma1 <= 5.5 and 
            sigma2 <= 5.5):
                recommendation = 1
            else:
                recommendation = 0
            
            # Get actual results for backtesting
            actual1 = player1_data[player1_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player1_data[player1_data['GAME_DATE'] == gameDate]) > 0 else None
            actual2 = player2_data[player2_data['GAME_DATE'] == gameDate]['PTS'].iloc[0] if len(player2_data[player2_data['GAME_DATE'] == gameDate]) > 0 else None
            
            if actual1 is None or actual2 is None:
                continue
            
            # Determine if bet won
            won1 = (actual1 > player1_line['LINE']) if model_side1 == 'over' else (actual1 < player1_line['LINE'])
            won2 = (actual2 > player2_line['LINE']) if model_side2 == 'over' else (actual2 < player2_line['LINE'])
            won_both = won1 and won2
            
            # Calculate profit/loss based on stake
            if won_both:
                profit = (payout_multiple - 1) * stake  # Win: (3-1) * stake = 2 * stake
            else:
                profit = -stake  # Loss: lose the stake
            
            results.append({
                'NAME 1': player1,
                'NAME 2': player2,
                'LINE 1': player1_line['LINE'],
                'LINE 2': player2_line['LINE'],
                'PREDICTION 1': round(mu1, 2),
                'PREDICTION 2': round(mu2, 2),
                'MODEL SIDE 1': model_side1,
                'MODEL SIDE 2': model_side2,
                'PROB 1': round(p1, 3),
                'PROB 2': round(p2, 3),
                'PROB BOTH': round(p_both, 4),
                'EDGE 1': round(edge1, 3),
                'EDGE 2': round(edge2, 3),
                'COMBINED EDGE': round(combined_edge, 3),
                'EV$': round(ev_dollars, 2),
                'KELLY FULL': round(kelly_full, 3),
                'RECOMMENDATION': recommendation,
                'INTERVAL WIDTH 1': width1,
                'INTERVAL WIDTH 2': width2,
                'SIGMA 1': round(sigma1, 2),
                'SIGMA 2': round(sigma2, 2),
                'SIGMA FLAG 1': sigma_flag1,
                'SIGMA FLAG 2': sigma_flag2,
                'EXPECTED ROI': round((ev_dollars / stake) * 100, 1),
                'SIMULATION METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical',
                # Backtest-specific columns (for backtest.py compatibility)
                'actual1': actual1,
                'actual2': actual2,
                'won1': won1,
                'won2': won2,
                'won_both': won_both,
                'profit': round(profit, 2),
                'date': gameDate
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def backtest3Legs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100, df_t=5, skew_a=-2.0):
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
    player_combinations = list(combinations(available_players, 3))
    # Generate all 3-leg combinations
    for player1, player2, player3 in player_combinations:
        
                # Handle name variations
                if player1 in nameDict:
                    player1 = nameDict[player1]
                if player2 in nameDict:
                    player2 = nameDict[player2]
                if player3 in nameDict:
                    player3 = nameDict[player3]
                
                # Get player data
                player1_data = data[data['PLAYER_NAME'] == player1]
                player2_data = data[data['PLAYER_NAME'] == player2]
                player3_data = data[data['PLAYER_NAME'] == player3]
                
                if player1_data.empty or player2_data.empty or player3_data.empty:
                    continue
                
                # Check if any players are from the same team (prevent same-team combinations)
                player1_team = player1_data['TEAM_ABBREVIATION'].iloc[-1]
                player2_team = player2_data['TEAM_ABBREVIATION'].iloc[-1]
                player3_team = player3_data['TEAM_ABBREVIATION'].iloc[-1]
                
                # Skip if any two players are from the same team
                if (player1_team == player2_team and
                    player1_team == player3_team and
                    player2_team == player3_team):
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
                
                # Get predictions for all three players using cached function
                try:
                    pred1 = get_cached_prediction(player1, data, models, features, stat_col, gameDate)
                    pred2 = get_cached_prediction(player2, data, models, features, stat_col, gameDate)
                    pred3 = get_cached_prediction(player3, data, models, features, stat_col, gameDate)
                    
                    if pred1 is None or pred2 is None or pred3 is None:
                        continue
                    
                    # Extract prediction (mu) and sigma directly
                    mu1 = pred1['prediction']
                    sigma1_raw = pred1['sigma']
                    sigma1 = sigma1_raw * variance_inflation
                    
                    mu2 = pred2['prediction']
                    sigma2_raw = pred2['sigma']
                    sigma2 = sigma2_raw * variance_inflation
                    
                    mu3 = pred3['prediction']
                    sigma3_raw = pred3['sigma']
                    sigma3 = sigma3_raw * variance_inflation
                    
                except Exception as e:
                    print(f"Error getting predictions for {player1}, {player2}, or {player3}: {e}")
                    continue
                
                # Set random seed outside conditional for reproducibility across runs
                np.random.seed(42)  # For reproducibility
                
                # Calculate probabilities for all three players using Monte Carlo or analytical method
                if use_monte_carlo:
                    # Monte Carlo simulation for all three players
                    if distribution_type == 'normal':
                        sim1 = np.random.normal(mu1, sigma1, n_simulations)
                        sim2 = np.random.normal(mu2, sigma2, n_simulations)
                        sim3 = np.random.normal(mu3, sigma3, n_simulations)
                    elif distribution_type == 't':
                        from scipy.stats import t
                        df1 = df_t
                        df2 = df_t
                        df3 = df_t
                        scale1 = sigma1 * np.sqrt((df1 - 2) / df1)
                        scale2 = sigma2 * np.sqrt((df2 - 2) / df2)
                        scale3 = sigma3 * np.sqrt((df3 - 2) / df3)
                        sim1 = t.rvs(df1, loc=mu1, scale=scale1, size=n_simulations, random_state=42)
                        sim2 = t.rvs(df2, loc=mu2, scale=scale2, size=n_simulations, random_state=42)
                        sim3 = t.rvs(df3, loc=mu3, scale=scale3, size=n_simulations, random_state=42)
                    elif distribution_type == 'skew_t':
                        from scipy.stats import skewnorm
                        sim1 = skewnorm.rvs(skew_a, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                        sim2 = skewnorm.rvs(skew_a, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                        sim3 = skewnorm.rvs(skew_a, loc=mu3, scale=sigma3, size=n_simulations, random_state=42)
                    else:
                        raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
                    
                    # Clip simulations at zero since points cannot be negative
                    sim1 = np.maximum(sim1, 0)
                    sim2 = np.maximum(sim2, 0)
                    sim3 = np.maximum(sim3, 0)
                    p1_over = np.mean(sim1 > player1_line['LINE'])
                    p2_over = np.mean(sim2 > player2_line['LINE'])
                    p3_over = np.mean(sim3 > player3_line['LINE'])
                else:
                    # Analytical method (original)
                    if distribution_type == 'normal':
                        from scipy.stats import norm
                        p1_over = 1 - norm.cdf(player1_line['LINE'], mu1, sigma1)
                        p2_over = 1 - norm.cdf(player2_line['LINE'], mu2, sigma2)
                        p3_over = 1 - norm.cdf(player3_line['LINE'], mu3, sigma3)
                    elif distribution_type == 't':
                        from scipy.stats import t
                        df1 = df_t
                        df2 = df_t
                        df3 = df_t
                        scale1 = sigma1 * np.sqrt((df1 - 2) / df1)
                        scale2 = sigma2 * np.sqrt((df2 - 2) / df2)
                        scale3 = sigma3 * np.sqrt((df3 - 2) / df3)
                        p1_over = 1 - t.cdf(player1_line['LINE'], df1, loc=mu1, scale=scale1)
                        p2_over = 1 - t.cdf(player2_line['LINE'], df2, loc=mu2, scale=scale2)
                        p3_over = 1 - t.cdf(player3_line['LINE'], df3, loc=mu3, scale=scale3)
                    elif distribution_type == 'skew_t':
                        from scipy.stats import skewnorm
                        p1_over = 1 - skewnorm.cdf(player1_line['LINE'], skew_a, loc=mu1, scale=sigma1)
                        p2_over = 1 - skewnorm.cdf(player2_line['LINE'], skew_a, loc=mu2, scale=sigma2)
                        p3_over = 1 - skewnorm.cdf(player3_line['LINE'], skew_a, loc=mu3, scale=sigma3)
                    else:
                        raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
                
                # Determine model sides based on predictions vs lines
                if mu1 > player1_line['LINE']:
                    model_side1 = 'over'
                    p1 = p1_over
                else:
                    model_side1 = 'under'
                    p1 = 1 - p1_over
                    
                if mu2 > player2_line['LINE']:
                    model_side2 = 'over'
                    p2 = p2_over
                else:
                    model_side2 = 'under'
                    p2 = 1 - p2_over
                    
                if mu3 > player3_line['LINE']:
                    model_side3 = 'over'
                    p3 = p3_over
                else:
                    model_side3 = 'under'
                    p3 = 1 - p3_over
                
                # Confidence intervals and sigma flags for all three players
                ci1 = (max(0, mu1 - 1.96 * sigma1), mu1 + 1.96 * sigma1)
                ci2 = (max(0, mu2 - 1.96 * sigma2), mu2 + 1.96 * sigma2)
                ci3 = (max(0, mu3 - 1.96 * sigma3), mu3 + 1.96 * sigma3)
                width1 = round(ci1[1] - ci1[0], 2)
                width2 = round(ci2[1] - ci2[0], 2)
                width3 = round(ci3[1] - ci3[0], 2)
                
                # Sigma flags helper function
                def flag_sigma(s):
                    if s <= 5.0:
                        return 'Low'
                    elif s <= 6.0:
                        return 'Med'
                    else:
                        return 'High'
                sigma_flag1 = flag_sigma(sigma1)
                sigma_flag2 = flag_sigma(sigma2)
                sigma_flag3 = flag_sigma(sigma3)
                
                # Calculate combined probability and EV
                p_all_three = p1 * p2 * p3
                payout_multiple = 6.0  # 6x payout for 3-leg parlay
                ev = payout_multiple * p_all_three - 1
                ev_dollars = ev * stake
                
                # Kelly criterion with variance-adjusted constraint
                b = payout_multiple - 1.0  # b = 5.0
                kelly_full = max(0.0, (b * p_all_three - (1 - p_all_three)) / b) if b > 0 else 0.0
                
                # Edge calculation (probability edge for all three players)
                market_prob1 = impliedProb(-137)  # Fixed odds
                market_prob2 = impliedProb(-137)  # Fixed odds
                market_prob3 = impliedProb(-137)  # Fixed odds
                edge1 = p1 - market_prob1
                edge2 = p2 - market_prob2
                edge3 = p3 - market_prob3
                
                # Calculate combined probabilities and edge
                combined_model_prob = p1 * p2 * p3
                combined_market_prob = market_prob1 * market_prob2 * market_prob3
                combined_edge = combined_model_prob - combined_market_prob
                
                # Recommendation based on multiple criteria (matching calculateEVS.py)
                if combined_edge > 0 and combined_model_prob > 0.185 and ev_dollars > 0:
                    recommendation = 1
                else:
                    recommendation = 0
                
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
                
                # Calculate profit/loss based on stake
                if won_all_three:
                    profit = (payout_multiple - 1) * stake  # Win: (6-1) * stake = 5 * stake
                else:
                    profit = -stake  # Loss: lose the stake
                
                results.append({
                    'NAME 1': player1,
                    'NAME 2': player2,
                    'NAME 3': player3,
                    'LINE 1': player1_line['LINE'],
                    'LINE 2': player2_line['LINE'],
                    'LINE 3': player3_line['LINE'],
                    'PREDICTION 1': round(mu1, 2),
                    'PREDICTION 2': round(mu2, 2),
                    'PREDICTION 3': round(mu3, 2),
                    'MODEL SIDE 1': model_side1,
                    'MODEL SIDE 2': model_side2,
                    'MODEL SIDE 3': model_side3,
                    'PROB 1': round(p1, 3),
                    'PROB 2': round(p2, 3),
                    'PROB 3': round(p3, 3),
                    'PROB ALL THREE': round(p_all_three, 4),
                    'EDGE 1': round(edge1, 3),
                    'EDGE 2': round(edge2, 3),
                    'EDGE 3': round(edge3, 3),
                    'COMBINED EDGE': round(combined_edge, 3),
                    'EV$': round(ev_dollars, 2),
                    'KELLY FULL': round(kelly_full, 3),
                    'RECOMMENDATION': recommendation,
                    'CONFIDENCE INTERVAL 1': f"({ci1[0]:.1f}, {ci1[1]:.1f})",
                    'CONFIDENCE INTERVAL 2': f"({ci2[0]:.1f}, {ci2[1]:.1f})",
                    'CONFIDENCE INTERVAL 3': f"({ci3[0]:.1f}, {ci3[1]:.1f})",
                    'INTERVAL WIDTH 1': width1,
                    'INTERVAL WIDTH 2': width2,
                    'INTERVAL WIDTH 3': width3,
                    'SIGMA 1': round(sigma1, 2),
                    'SIGMA 2': round(sigma2, 2),
                    'SIGMA 3': round(sigma3, 2),
                    'SIGMA FLAG 1': sigma_flag1,
                    'SIGMA FLAG 2': sigma_flag2,
                    'SIGMA FLAG 3': sigma_flag3,
                    'EXPECTED ROI': round((ev_dollars / stake) * 100, 1),
                    'SIMULATION METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical',
                    # Backtest-specific columns (for backtest.py compatibility)
                    'actual1': actual1,
                    'actual2': actual2,
                    'actual3': actual3,
                    'won1': won1,
                    'won2': won2,
                    'won3': won3,
                    'won_all_three': won_all_three,
                    'profit': round(profit, 2),
                    'date': gameDate
                })
    
    results_df = pd.DataFrame(results)
    return results_df    