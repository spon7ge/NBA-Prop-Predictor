import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import truncnorm
from nba_api.stats.endpoints import scoreboardv2
from MODELS.teamInfo import *
from MODELS.pipeline import *
from itertools import combinations
from collections import defaultdict

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

_prediction_cache = {}

def get_cached_prediction(player_name, data, models, features, current_date, projectedStartingFive, teamStarPlayer):
    cache_key = f"{player_name}_{current_date}"
    
    if cache_key not in _prediction_cache:
        try:
            # Filter once
            player_df = data[data['PLAYER_NAME'] == player_name].sort_values(by='GAME_DATE')
            if player_df.empty:
                return None
            
            # Build vector once
            vector = buildVector(player_name, data, current_date, projectedStartingFive, teamStarPlayer)
            vector = [item for sublist in vector for item in sublist]
            vector = pd.DataFrame([vector], columns=features)
            
            for col in vector.columns:
                vector[col] = pd.to_numeric(vector[col], errors='coerce')
            
            vector = vector.fillna(0)
            
            # Predict with all three models on same vector
            q10_pred = round(float(models['q10'].predict(vector)[0]), 3)
            q50_pred = round(float(models['q50'].predict(vector)[0]), 3)
            q90_pred = round(float(models['q90'].predict(vector)[0]), 3)
            
            _prediction_cache[cache_key] = {
                'q10': q10_pred,
                'q50': q50_pred,
                'q90': q90_pred
            }
        except Exception as e:
            print(f"Error getting prediction for {player_name}: {e}")
            return None
    return _prediction_cache[cache_key]

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculateSingleBets(data, bookmakers, models, features, current_date, edge_threshold=0.05, stake=100, 
                     variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                     use_monte_carlo=True, n_simulations=10000, max_kelly=0.25):
    print("Processing single bets with quantile models...")
    
    results = []
    
    # Process each bet individually to capture all opportunities
    for _, row in bookmakers.iterrows():
        name = row['NAME']
        bookmaker = row['BOOKMAKER']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('OVER/UNDER', 'over')
        odds = int(row['ODDS'])
        
        # Handle name variations
        original_name = name
        if name in nameDict:
            name = nameDict[name]
        
        # Get quantile predictions for this player
        try:
            predictions = get_cached_prediction(name, data, models, features, current_date, projectedStartingFive, teamStarPlayer)
            if predictions is None:
                continue
            q10_pred = predictions['q10']
            q50_pred = predictions['q50']
            q90_pred = predictions['q90']
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue
        
        # Pre-calculate distribution parameters
        mu = q50_pred
        sigma_raw = (q90_pred - q10_pred) / 2.56
        sigma = sigma_raw * variance_inflation
        
        # Calculate probabilities using Monte Carlo simulation or analytical method
        if use_monte_carlo and distribution_type == 'normal':
            # Use faster analytical method for normal distributions
            from scipy.stats import norm
            p_over = 1 - norm.cdf(line, mu, sigma)
        elif use_monte_carlo:
            # Monte Carlo simulation for non-normal distributions
            np.random.seed(hash(f"{name}_{line}") % 2**32)
            if distribution_type == 't':
                from scipy.stats import t
                df = max(3, 2 * sigma**2 / (sigma**2 - 1))
                simulations = t.rvs(df, loc=mu, scale=sigma, size=n_simulations, random_state=42)
            elif distribution_type == 'skew_t':
                from scipy.stats import skewnorm
                simulations = skewnorm.rvs(0, loc=mu, scale=sigma, size=n_simulations, random_state=42)
            else:
                raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
            
            p_over = np.mean(simulations > line)
        else:
            # Analytical method
            if distribution_type == 'normal':
                from scipy.stats import norm
                p_over = 1 - norm.cdf(line, mu, sigma)
            elif distribution_type == 't':
                from scipy.stats import t
                df = max(3, 2 * sigma**2 / (sigma**2 - 1))
                p_over = 1 - t.cdf(line, df, mu, sigma)
            elif distribution_type == 'skew_t':
                from scipy.stats import skewnorm
                p_over = 1 - skewnorm.cdf(line, 0, loc=mu, scale=sigma)
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
        ev_total = ev_per_dollar * stake
        
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
            kelly_capped_fraction > 0 and
            p > 0.40 and 
            ev_total > 1.00):
            recommendation = 1
        else:
            recommendation = 0
        
        # Confidence interval (using quantiles)
        confidence_interval = (q10_pred, q90_pred)
        
        results.append({
            'NAME': original_name,
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
            'EV%': round(ev_total, 2),
            'KELLY_FRACTION': round(kelly_fraction, 3),
            'KELLY_DOLLARS': round(kelly_dollars, 2),
            'CONFIDENCE INTERVAL': f"({confidence_interval[0]:.1f}, {confidence_interval[1]:.1f})",
            'SIGMA': round(sigma, 2),
            'SIMULATION_METHOD': 'Analytical' if not use_monte_carlo or distribution_type == 'normal' else 'Monte Carlo'
        })
    
    return pd.DataFrame(results)

def calculate2LegBets(data, bookmakers, models, features, current_date, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100):

    category = 'player_points'
    bookmakers = bookmakers[(bookmakers['CATEGORY'] == category)]
    if bookmakers.empty:
        print(f"No bets found for {category}")
        return pd.DataFrame()

    # Get all available players for 2-leg combinations
    available_players = bookmakers['NAME'].unique()
    if len(available_players) < 2:
        print("Not enough players for 2-leg bets")
        return pd.DataFrame()

    results = []
    player_combinations = list(combinations(available_players, 2))

    for player1, player2 in player_combinations:
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
            player1_bets = bookmakers[bookmakers['NAME'] == player1]
            player2_bets = bookmakers[bookmakers['NAME'] == player2]
            
            if player1_bets.empty or player2_bets.empty:
                continue
            
            # Use the first available line for each player
            player1_line = player1_bets.iloc[0]
            player2_line = player2_bets.iloc[0]
            
            # Get quantile predictions for both players using cached function
            try:
                pred1 = get_cached_prediction(player1, data, models, features, current_date, projectedStartingFive, teamStarPlayer)
                pred2 = get_cached_prediction(player2, data, models, features, current_date, projectedStartingFive, teamStarPlayer)
                
                if pred1 is None or pred2 is None:
                    continue
                
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
            
            # Calculate probabilities for both players using Monte Carlo or analytical method
            if use_monte_carlo:
                # Monte Carlo simulation for both players
                np.random.seed(42)  # For reproducibility
                if distribution_type == 'normal':
                    sim1 = np.random.normal(mu1, sigma1, n_simulations)
                    sim2 = np.random.normal(mu2, sigma2, n_simulations)
                elif distribution_type == 't':
                    from scipy.stats import t
                    df1 = max(3, 2 * sigma1**2 / (sigma1**2 - 1))
                    df2 = max(3, 2 * sigma2**2 / (sigma2**2 - 1))
                    sim1 = t.rvs(df1, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                    sim2 = t.rvs(df2, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                elif distribution_type == 'skew_t':
                    from scipy.stats import skewnorm
                    sim1 = skewnorm.rvs(0, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                    sim2 = skewnorm.rvs(0, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                else:
                    raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
                
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
                    df1 = max(3, 2 * sigma1**2 / (sigma1**2 - 1))
                    df2 = max(3, 2 * sigma2**2 / (sigma2**2 - 1))
                    p1_over = 1 - t.cdf(player1_line['LINE'], df1, mu1, sigma1)
                    p2_over = 1 - t.cdf(player2_line['LINE'], df2, mu2, sigma2)
                elif distribution_type == 'skew_t':
                    from scipy.stats import skewnorm
                    p1_over = 1 - skewnorm.cdf(player1_line['LINE'], 0, loc=mu1, scale=sigma1)
                    p2_over = 1 - skewnorm.cdf(player2_line['LINE'], 0, loc=mu2, scale=sigma2)
                else:
                    raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
            
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
            payout_multiple = 3.0  # 3x payout for 2-leg parlay
            ev = payout_multiple * p_both - 1
            
            # Kelly criterion with variance-adjusted constraint
            b = payout_multiple - 1.0  # b = 2.0
            kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b) if b > 0 else 0.0
            
            # Edge calculation (probability edge for both players)
            market_prob1 = impliedProb(-137)  # Fixed odds
            market_prob2 = impliedProb(-137)  # Fixed odds
            edge1 = p1 - market_prob1
            edge2 = p2 - market_prob2
            combined_edge = (edge1 + edge2) / 2  # Average edge
            
            # Recommendation based on multiple criteria
            if (combined_edge > edge_threshold and 
                kelly_full > -0.02 and
                p_both > 0.40 and 
                ev > 0.4):
                recommendation = 1
            else:
                recommendation = 0
            
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
                'ev_percent': round(ev, 2),
                'kelly_full': round(kelly_full, 3),
                'recommendation': recommendation,
                'simulation_method': 'Monte Carlo' if use_monte_carlo else 'Analytical'
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def calculate3LegBets(data, bookmakers, models, features, current_date, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100):
    category = 'player_points'
    bookmakers = bookmakers[(bookmakers['CATEGORY'] == category)]
    if bookmakers.empty:
        print(f"No bets found for {category}")
        return pd.DataFrame()

    # Get all available players for 3-leg combinations
    available_players = bookmakers['NAME'].unique()
    if len(available_players) < 3:
        print("Not enough players for 3-leg bets")
        return pd.DataFrame()

    results = []
    player_combinations = list(combinations(available_players, 3))
    # Generate all 3-leg combinations
    for player1, player2, player3 in player_combinations:
        
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
                
                # Prevent all 3 players from being on the same team
                if player1_team == player2_team == player3_team:
                    continue
                    
                # Get betting lines for all three players
                player1_bets = bookmakers[bookmakers['NAME'] == player1]
                player2_bets = bookmakers[bookmakers['NAME'] == player2]
                player3_bets = bookmakers[bookmakers['NAME'] == player3]
                
                if player1_bets.empty or player2_bets.empty or player3_bets.empty:
                    continue
                
                # Use the first available line for each player
                player1_line = player1_bets.iloc[0]
                player2_line = player2_bets.iloc[0]
                player3_line = player3_bets.iloc[0]
                
                # Get quantile predictions for all three players using cached function
                try:
                    pred1 = get_cached_prediction(player1, data, models, features, current_date, projectedStartingFive, teamStarPlayer)
                    pred2 = get_cached_prediction(player2, data, models, features, current_date, projectedStartingFive, teamStarPlayer)
                    pred3 = get_cached_prediction(player3, data, models, features, current_date, projectedStartingFive, teamStarPlayer)
                    
                    if pred1 is None or pred2 is None or pred3 is None:
                        continue
                    
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
                
                # Calculate probabilities for all three players using Monte Carlo or analytical method
                if use_monte_carlo:
                    # Monte Carlo simulation for all three players
                    np.random.seed(42)  # For reproducibility
                    if distribution_type == 'normal':
                        sim1 = np.random.normal(mu1, sigma1, n_simulations)
                        sim2 = np.random.normal(mu2, sigma2, n_simulations)
                        sim3 = np.random.normal(mu3, sigma3, n_simulations)
                    elif distribution_type == 't':
                        from scipy.stats import t
                        df1 = max(3, 2 * sigma1**2 / (sigma1**2 - 1))
                        df2 = max(3, 2 * sigma2**2 / (sigma2**2 - 1))
                        df3 = max(3, 2 * sigma3**2 / (sigma3**2 - 1))
                        sim1 = t.rvs(df1, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                        sim2 = t.rvs(df2, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                        sim3 = t.rvs(df3, loc=mu3, scale=sigma3, size=n_simulations, random_state=42)
                    elif distribution_type == 'skew_t':
                        from scipy.stats import skewnorm
                        sim1 = skewnorm.rvs(0, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                        sim2 = skewnorm.rvs(0, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                        sim3 = skewnorm.rvs(0, loc=mu3, scale=sigma3, size=n_simulations, random_state=42)
                    else:
                        raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
                    
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
                        df1 = max(3, 2 * sigma1**2 / (sigma1**2 - 1))
                        df2 = max(3, 2 * sigma2**2 / (sigma2**2 - 1))
                        df3 = max(3, 2 * sigma3**2 / (sigma3**2 - 1))
                        p1_over = 1 - t.cdf(player1_line['LINE'], df1, mu1, sigma1)
                        p2_over = 1 - t.cdf(player2_line['LINE'], df2, mu2, sigma2)
                        p3_over = 1 - t.cdf(player3_line['LINE'], df3, mu3, sigma3)
                    elif distribution_type == 'skew_t':
                        from scipy.stats import skewnorm
                        p1_over = 1 - skewnorm.cdf(player1_line['LINE'], 0, loc=mu1, scale=sigma1)
                        p2_over = 1 - skewnorm.cdf(player2_line['LINE'], 0, loc=mu2, scale=sigma2)
                        p3_over = 1 - skewnorm.cdf(player3_line['LINE'], 0, loc=mu3, scale=sigma3)
                    else:
                        raise ValueError("distribution_type must be 'normal', 't', or 'skew_t'")
                
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
                combined_edge = (edge1 + edge2 + edge3) / 3  # Average edge
                
                # Recommendation based on multiple criteria
                if (combined_edge > edge_threshold and 
                    kelly_full > -0.02 and
                    p_all_three > 0.40 and
                    ev > 0.40):
                    recommendation = 1
                else:
                    recommendation = 0
                
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
                    'ev_percent': round(ev, 2),
                    'kelly_full': round(kelly_full, 3),
                    'recommendation': recommendation,
                    'simulation_method': 'Monte Carlo' if use_monte_carlo else 'Analytical'
                })

    results_df = pd.DataFrame(results)
    return results_df    