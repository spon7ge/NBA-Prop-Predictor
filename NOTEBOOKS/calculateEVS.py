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

def estimate_skew_from_residuals(residuals: np.ndarray) -> float:
    """Estimate skew-normal shape parameter from residuals.
    Clips to a reasonable range for stability.
    """
    try:
        from scipy.stats import skew as _skew
        est = float(_skew(residuals, bias=False))
        return float(np.clip(est, -4.0, 4.0))
    except Exception:
        return -2.0

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

def get_cached_prediction(player_name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    cache_key = f"{player_name}_{current_date}"
    
    if cache_key not in _prediction_cache:
        try:
            player_df = data[data['PLAYER_NAME'] == player_name].sort_values(by='GAME_DATE')
            if player_df.empty:
                return None
            
            vector = buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
            vector = [item for sublist in vector for item in sublist]
            vector = pd.DataFrame([vector], columns=features)
            
            for col in vector.columns:
                vector[col] = pd.to_numeric(vector[col], errors='coerce')
            
            vector = vector.fillna(0)
            
            # Check if model is a tuple (split model: mean, variance, calibration_factor)
            if isinstance(model, tuple):
                mean_model = model[0]
                variance_model = model[1]
                calibration_factor = model[2] if len(model) > 2 else 1.25  # Use calibration factor if provided
                from MODELS.ngboostModel import predict_mean_variance_split
                mu, variance = predict_mean_variance_split(mean_model, variance_model, vector, features, calibration_factor)
                pred = round(float(mu[0] if isinstance(mu, (np.ndarray, pd.Series)) else mu), 3)
                sigma = float(np.sqrt(variance[0] if isinstance(variance, (np.ndarray, pd.Series)) else variance))
                skew = 0.0
            elif hasattr(model, "pred_dist"):
                # Single NGBoost model with pred_dist
                dist = model.pred_dist(vector)
                pred = round(float(dist.loc[0]), 3)
                sigma = float(dist.scale[0])
                skew = 0.0
            else:
                # Point model (XGBoost or similar)
                pred = round(float(model.predict(vector)[0]), 3)
                sigma = calculate_player_sigma(player_df.iloc[-1], pred)
                skew = calculate_player_skew(player_df.iloc[-1])
            
            _prediction_cache[cache_key] = {
                'prediction': pred,
                'sigma': sigma,
                'skew': skew
            }
        except Exception as e:
            print(f"Error getting prediction for {player_name}: {e}")
            return None
    return _prediction_cache[cache_key]

def calculate_player_sigma(player_row, prediction):
    try:
        # Base sigma from points volatility (25-game rolling standard deviation)
        pts_std_25 = player_row.get('PTS_STD_LAST_25', 0)
        
        # If no volatility data available, use prediction-based estimate
        if pd.isna(pts_std_25) or pts_std_25 == 0:
            # Estimate sigma based on prediction level (higher scorers tend to be more volatile)
            # INCREASED minimum to prevent overly narrow distributions
            base_sigma = max(3.5, min(8.5, prediction * 0.18))  # Was 2.5 and 0.15
        else:
            # Use actual volatility data with scaling factor
            # INCREASED minimum to prevent overly narrow distributions
            base_sigma = max(3.5, min(8.5, 1.3 * pts_std_25))  # Was 2.5 and 1.2
        
        # Adjustments based on game context
        adjustments = []
        
        # Back-to-back adjustment (10-25% increase)
        if player_row.get('IS_BACK_TO_BACK', 0) == 1:
            adjustments.append(0.15)  # 15% increase
        
        # High pace adjustment (10-20% increase)
        game_pace = player_row.get('TEAM_PACE_AVG_TO_DATE', 100)
        if game_pace > 105:  # High pace threshold
            pace_adjustment = min(0.20, (game_pace - 105) * 0.01)  # Up to 20% increase
            adjustments.append(pace_adjustment)
        
        # Star player with extreme usage adjustment (10-25% increase)
        usage_pct = player_row.get('USG_PCT_AVG_TO_DATE', 20)
        if usage_pct > 30:  # High usage threshold
            usage_adjustment = min(0.25, (usage_pct - 30) * 0.01)  # Up to 25% increase
            adjustments.append(usage_adjustment)
        
        # Minutes volatility low adjustment (10-20% decrease)
        min_std = player_row.get('MIN_STD_LAST_25', 0)
        if not pd.isna(min_std) and min_std < 3:  # Low minutes volatility
            min_adjustment = -0.15  # 15% decrease
            adjustments.append(min_adjustment)
        
        # Opponent slow pace adjustment (10-20% decrease)
        # opp_pace = player_row.get('OPP_PACE_AVG_TO_DATE', 100)
        # if opp_pace < 95:  # Slow pace threshold
        #     opp_adjustment = -min(0.20, (95 - opp_pace) * 0.01)  # Up to 20% decrease
        #     adjustments.append(opp_adjustment)
        
        # Apply adjustments
        total_adjustment = sum(adjustments)
        adjusted_sigma = base_sigma * (1 + total_adjustment)
        
        # Final clipping to sane range
        # INCREASED minimum from 1.5 to 3.0 to prevent extreme probabilities
        final_sigma = max(3.0, min(12.0, adjusted_sigma))  # Was max(1.5, ...)
        
        return round(final_sigma, 2)
        
    except Exception as e:
        print(f"Error calculating sigma for player: {e}")
        # Fallback to prediction-based sigma with higher minimum
        return max(3.5, min(8.5, prediction * 0.18))  # Was 2.5 and 0.15

def calculate_player_skew(player_row):
    try:
        # Try to get empirical skew from recent points volatility features
        # Look for skew-related features in the data
        recent_skew = None
        
        # Check for explicit skew features first
        if 'PTS_SKEW_LAST_25' in player_row.index:
            recent_skew = player_row.get('PTS_SKEW_LAST_25', None)
        elif 'PTS_SKEW_LAST_15' in player_row.index:
            recent_skew = player_row.get('PTS_SKEW_LAST_15', None)
        elif 'PTS_SKEW_LAST_10' in player_row.index:
            recent_skew = player_row.get('PTS_SKEW_LAST_10', None)
        
        # If no explicit skew data, estimate from volatility patterns
        if pd.isna(recent_skew) or recent_skew == 0:
            # Use volatility ratios as proxy for skew
            pts_std_25 = player_row.get('PTS_STD_LAST_25', 0)
            pts_std_10 = player_row.get('PTS_STD_LAST_10', 0)
            
            if not pd.isna(pts_std_25) and not pd.isna(pts_std_10) and pts_std_25 > 0:
                # Higher short-term volatility relative to long-term suggests positive skew
                volatility_ratio = pts_std_10 / pts_std_25
                recent_skew = (volatility_ratio - 1.0) * 2.0  # Scale to reasonable skew range
            else:
                # Default to mild right tail skew
                recent_skew = 0.5
        
        # Convert empirical skew to skewnorm shape parameter
        # Clip to reasonable range and scale
        alpha = np.clip(3 * recent_skew, -4, 4)
        
        return round(alpha, 2)
        
    except Exception as e:
        print(f"Error calculating skew for player: {e}")
        # Fallback to mild right tail skew
        return 1.0

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculateSingleBets(data, bookmakers, model, features, current_date, edge_threshold=0.05, stake=100, 
                     variance_inflation=1.1, distribution_type='normal', stat_col='PTS', 
                     use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, df_t=5,
                     enforce_downside_skew: bool = False, skew_override: float | None = None):
    print("Processing single bets with single model...")
    
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
        
        # Get prediction, sigma, and skew for this player
        try:
            prediction_data = get_cached_prediction(name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
            if prediction_data is None:
                continue
            pred = prediction_data['prediction']
            sigma = prediction_data['sigma']
            skew = prediction_data['skew']
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue
        
        # Pre-calculate distribution parameters using player-specific sigma
        mu = pred
        
        # Apply variance inflation if specified
        sigma = sigma * variance_inflation
        
        # Set random seed outside conditional for reproducibility across runs
        np.random.seed(hash(f"{name}_{line}") % 2**32)
        
        # Calculate probabilities using Monte Carlo simulation or analytical method
        if use_monte_carlo and distribution_type == 'normal':
            # Use faster analytical method for normal distributions
            from scipy.stats import norm
            p_over = 1 - norm.cdf(line, mu, sigma)
        elif use_monte_carlo:
            # Monte Carlo simulation for non-normal distributions
            if distribution_type == 't':
                from scipy.stats import t
                df = df_t
                scale = sigma * np.sqrt((df - 2) / df)
                simulations = t.rvs(df, loc=mu, scale=scale, size=n_simulations, random_state=42)
            elif distribution_type == 'skewnorm':
                from scipy.stats import skewnorm
                # Optionally enforce downside risk or override skew
                a = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew)
                simulations = skewnorm.rvs(a, loc=mu, scale=sigma, size=n_simulations, random_state=42)
            else:
                raise ValueError("distribution_type must be 'normal', 't', or 'skewnorm'")
            
            # Clip simulations at zero since points cannot be negative
            simulations = np.maximum(simulations, 0)
            p_over = np.mean(simulations > line)
        else:
            # Analytical method
            if distribution_type == 'normal':
                from scipy.stats import norm
                p_over = 1 - norm.cdf(line, mu, sigma)
            elif distribution_type == 't':
                from scipy.stats import t
                df = df_t
                scale = sigma * np.sqrt((df - 2) / df)
                p_over = 1 - t.cdf(line, df, loc=mu, scale=scale)
            elif distribution_type == 'skewnorm':
                from scipy.stats import skewnorm
                a = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew)
                p_over = 1 - skewnorm.cdf(line, a, loc=mu, scale=sigma)
            else:
                raise ValueError("distribution_type must be 'normal', 't', or 'skewnorm'")
        
        p_under = 1.0 - p_over
        
        # Choose probability based on the offered side (not model prediction)
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
        
        # Confidence interval (using proper statistical formula with player-specific sigma)
        confidence_interval = (
            max(0, pred - 1.96 * sigma),  # Lower bound, clipped at zero
            pred + 1.96 * sigma           # Upper bound
        )
        interval_width = round(confidence_interval[1] - confidence_interval[0], 2)
        # Sigma flag for readability
        if sigma <= 5.0:
            sigma_flag = 'Low'
        elif sigma <= 6.0:
            sigma_flag = 'Med'
        else:
            sigma_flag = 'High'
        
        results.append({
            'NAME': original_name,
            'BOOKMAKER': bookmaker,
            'CATEGORY': category,
            'LINE': line,
            'ODDS': odds,
            'SIDE': side,
            'PREDICTION': round(pred, 2),
            'RECOMMENDATION': recommendation,
            'OVER%': round(p_over, 3),
            'UNDER%': round(p_under, 3),
            'IMPLIED PROB': round(market_prob, 3),
            'MODEL PROB': round(model_prob, 3),
            'EDGE': round(edge, 3),
            'EV$': round(ev_total, 2),
            'KELLY_FRACTION': round(kelly_fraction, 3),
            'KELLY_DOLLARS': round(kelly_dollars, 2),
            'CONFIDENCE INTERVAL': f"({confidence_interval[0]:.1f}, {confidence_interval[1]:.1f})",
            'INTERVAL WIDTH': interval_width,
            'SIGMA': round(sigma, 2),
            'SIGMA FLAG': sigma_flag,
            'EXPECTED ROI': round((ev_total / stake )* 100, 1),
            'SIMULATION_METHOD': 'Analytical' if not use_monte_carlo or distribution_type == 'normal' else 'Monte Carlo'
        })
    
    return pd.DataFrame(results)

def calculate2LegBets(data, bookmakers, model, features, current_date, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100, df_t=5,
                 enforce_downside_skew: bool = False, skew_override: float | None = None):

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
            
            # Get predictions, sigmas, and skews for both players using cached function
            try:
                pred1_data = get_cached_prediction(player1, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
                pred2_data = get_cached_prediction(player2, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
                
                if pred1_data is None or pred2_data is None:
                    continue
                
                pred1_val = pred1_data['prediction']
                pred2_val = pred2_data['prediction']
                sigma1 = pred1_data['sigma']
                sigma2 = pred2_data['sigma']
                skew1 = pred1_data['skew']
                skew2 = pred2_data['skew']

            except Exception as e:
                print(f"Error getting predictions for {player1} or {player2}: {e}")
                continue
            
            # Convert predictions to distribution parameters for both players
            mu1 = pred1_val
            mu2 = pred2_val
            
            # Apply variance inflation if specified
            sigma1 = sigma1 * variance_inflation
            sigma2 = sigma2 * variance_inflation
            
            # Calculate confidence intervals and sigma flags for both players
            ci1 = (max(0, mu1 - 1.96 * sigma1), mu1 + 1.96 * sigma1)
            ci2 = (max(0, mu2 - 1.96 * sigma2), mu2 + 1.96 * sigma2)
            width1 = round(ci1[1] - ci1[0], 2)
            width2 = round(ci2[1] - ci2[0], 2)
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
                elif distribution_type == 'skewnorm':
                    from scipy.stats import skewnorm
                    a1 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew1)
                    a2 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew2)
                    sim1 = skewnorm.rvs(a1, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                    sim2 = skewnorm.rvs(a2, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                else:
                    raise ValueError("distribution_type must be 'normal', 't', or 'skewnorm'")
                
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
                elif distribution_type == 'skewnorm':
                    from scipy.stats import skewnorm
                    a1 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew1)
                    a2 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew2)
                    p1_over = 1 - skewnorm.cdf(player1_line['LINE'], a1, loc=mu1, scale=sigma1)
                    p2_over = 1 - skewnorm.cdf(player2_line['LINE'], a2, loc=mu2, scale=sigma2)
                else:
                    raise ValueError("distribution_type must be 'normal', 't', or 'skewnorm'")
            
            # Determine model sides based on predictions vs lines
            if pred1_val > player1_line['LINE']:
                model_side1 = 'over'
                p1 = p1_over
            else:
                model_side1 = 'under'
                p1 = 1 - p1_over
                
            if pred2_val > player2_line['LINE']:
                model_side2 = 'over'
                p2 = p2_over
            else:
                model_side2 = 'under'
                p2 = 1 - p2_over
            
            # Calculate combined probability and EV with correlation adjustment
            p_both = p1 * p2
            corr_factor = 0.9  # 10% dependence adjustment
            p_both *= corr_factor
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
            
            # Recommendation based on multiple criteria
            if combined_edge > 0 and combined_model_prob > 0.335 and ev_dollars > 0:
                recommendation = 1
            else:
                recommendation = 0
            
            results.append({
                'NAME 1': player1,
                'NAME 2': player2,
                'LINE 1': player1_line['LINE'],
                'LINE 2': player2_line['LINE'],
                'PREDICTION 1': round(pred1_val, 2),
                'PREDICTION 2': round(pred2_val, 2),
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
                'EXPECTED ROI': round((ev_dollars / stake )* 100, 1),
                'SIMULATION METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical'
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def calculate3LegBets(data, bookmakers, model, features, current_date, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, distribution_type='normal', 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100, df_t=5,
                 enforce_downside_skew: bool = False, skew_override: float | None = None):
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
                
                # Get predictions, sigmas, and skews for all three players using cached function
                try:
                    pred1_data = get_cached_prediction(player1, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
                    pred2_data = get_cached_prediction(player2, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
                    pred3_data = get_cached_prediction(player3, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
                    
                    if pred1_data is None or pred2_data is None or pred3_data is None:
                        continue
                    
                    pred1_val = pred1_data['prediction']
                    pred2_val = pred2_data['prediction']
                    pred3_val = pred3_data['prediction']
                    sigma1 = pred1_data['sigma']
                    sigma2 = pred2_data['sigma']
                    sigma3 = pred3_data['sigma']
                    skew1 = pred1_data['skew']
                    skew2 = pred2_data['skew']
                    skew3 = pred3_data['skew']
                    
                except Exception as e:
                    print(f"Error getting predictions for {player1}, {player2}, or {player3}: {e}")
                    continue
                
                # Convert predictions to distribution parameters for all three players
                mu1 = pred1_val
                mu2 = pred2_val
                mu3 = pred3_val
                
                # Apply variance inflation if specified
                sigma1 = sigma1 * variance_inflation
                sigma2 = sigma2 * variance_inflation
                sigma3 = sigma3 * variance_inflation
                
                # Confidence intervals and sigma flags for all three players
                ci1 = (max(0, mu1 - 1.96 * sigma1), mu1 + 1.96 * sigma1)
                ci2 = (max(0, mu2 - 1.96 * sigma2), mu2 + 1.96 * sigma2)
                ci3 = (max(0, mu3 - 1.96 * sigma3), mu3 + 1.96 * sigma3)
                width1 = round(ci1[1] - ci1[0], 2)
                width2 = round(ci2[1] - ci2[0], 2)
                width3 = round(ci3[1] - ci3[0], 2)
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
                    elif distribution_type == 'skewnorm':
                        from scipy.stats import skewnorm
                        a1 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew1)
                        a2 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew2)
                        a3 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew3)
                        sim1 = skewnorm.rvs(a1, loc=mu1, scale=sigma1, size=n_simulations, random_state=42)
                        sim2 = skewnorm.rvs(a2, loc=mu2, scale=sigma2, size=n_simulations, random_state=42)
                        sim3 = skewnorm.rvs(a3, loc=mu3, scale=sigma3, size=n_simulations, random_state=42)
                    else:
                        raise ValueError("distribution_type must be 'normal', 't', or 'skewnorm'")
                    
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
                    elif distribution_type == 'skewnorm':
                        from scipy.stats import skewnorm
                        a1 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew1)
                        a2 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew2)
                        a3 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew3)
                        p1_over = 1 - skewnorm.cdf(player1_line['LINE'], a1, loc=mu1, scale=sigma1)
                        p2_over = 1 - skewnorm.cdf(player2_line['LINE'], a2, loc=mu2, scale=sigma2)
                        p3_over = 1 - skewnorm.cdf(player3_line['LINE'], a3, loc=mu3, scale=sigma3)
                    else:
                        raise ValueError("distribution_type must be 'normal', 't', or 'skewnorm'")
                
                # Determine model sides based on predictions vs lines
                if pred1_val > player1_line['LINE']:
                    model_side1 = 'over'
                    p1 = p1_over
                else:
                    model_side1 = 'under'
                    p1 = 1 - p1_over
                    
                if pred2_val > player2_line['LINE']:
                    model_side2 = 'over'
                    p2 = p2_over
                else:
                    model_side2 = 'under'
                    p2 = 1 - p2_over
                    
                if pred3_val > player3_line['LINE']:
                    model_side3 = 'over'
                    p3 = p3_over
                else:
                    model_side3 = 'under'
                    p3 = 1 - p3_over
                
                # Calculate combined probability and EV with correlation adjustment
                p_all_three = p1 * p2 * p3
                corr_factor = 0.9  # 10% dependence adjustment
                p_all_three *= corr_factor ** 2  # Squared for 3-leg parlay
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
                
                # Recommendation based on multiple criteria
                if combined_edge > 0 and combined_model_prob > 0.185 and ev_dollars > 0:
                    recommendation = 1
                else:
                    recommendation = 0
                
                results.append({
                    'NAME 1': player1,
                    'NAME 2': player2,
                    'NAME 3': player3,
                    'LINE 1': player1_line['LINE'],
                    'LINE 2': player2_line['LINE'],
                    'LINE 3': player3_line['LINE'],
                    'PREDICTION 1': round(pred1_val, 2),
                    'PREDICTION 2': round(pred2_val, 2),
                    'PREDICTION 3': round(pred3_val, 2),
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
                    'EXPECTED ROI': round((ev_dollars / stake )* 100, 1),
                    'SIMULATION METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical'
                })

    results_df = pd.DataFrame(results)
    return results_df    