import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import truncnorm
from nba_api.stats.endpoints import scoreboardv2
from PRODUCTION.teamInfo import *
from PRODUCTION.pipeline import *
from itertools import combinations
from collections import defaultdict
from PRODUCTION.pipeline import calculate_volatility
from PRODUCTION.teamInfo import nameDict


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

def get_cached_prediction(player_name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer, isotonic_calibrator=None):
    """
    Get cached prediction for a player.
    
    Args:
        player_name: Name of the player
        data: Historical data
        model: Model or tuple of (mean_model, variance_model, calibration_factor)
        features: List of feature names
        current_date: Current date for prediction
        projectedStartingFive: Projected starting five
        mainStartingFive: Main starting five
        teamStarPlayer: Team star player info
        isotonic_calibrator: Optional isotonic regression calibrator for mean predictions
    
    Returns:
        Dictionary with 'prediction', 'sigma', and 'skew'
    """
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
                mu, variance = predict_mean_variance_split(
                    mean_model, variance_model, vector, features, 
                    calibration_factor, isotonic_calibrator
                )
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
                # Use simple default sigma and neutral skew when model doesn't provide distribution
                sigma = max(1.5, min(12.0, max(2.5, min(8.5, pred * 0.15))))
                skew = 0.0
            
            _prediction_cache[cache_key] = {
                'prediction': pred,
                'sigma': sigma,
                'skew': skew
            }
        except Exception as e:
            print(f"Error getting prediction for {player_name}: {e}")
            return None
    return _prediction_cache[cache_key]



#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculateSingleBets(data, bookmakers, model, features, current_date, edge_threshold=0.05, stake=100, 
                     variance_inflation=1.1, stat_col='PTS', 
                     use_monte_carlo=True, n_simulations=10000, max_kelly=0.25,
                     enforce_downside_skew: bool = False, skew_override: float | None = None,
                     isotonic_calibrator=None):
    """
    Calculate single bet opportunities with optional isotonic calibration.
    
    Args:
        isotonic_calibrator: Optional isotonic regression calibrator for mean predictions
    """
    print("Processing single bets with single model...")
    if isotonic_calibrator is not None:
        print("Using isotonic regression calibration")
    
    # Set random seed once before the loop for reproducibility
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Pre-compute predictions for all unique players
    unique_players = bookmakers['NAME'].unique()
    player_predictions = {}
    
    print(f"Pre-computing predictions for {len(unique_players)} unique players...")
    for player in unique_players:
        mapped_player = nameDict.get(player, player)
        pred_data = get_cached_prediction(mapped_player, data, model, features, current_date, 
                                         projectedStartingFive, mainStartingFive, teamStarPlayer,
                                         isotonic_calibrator)
        if pred_data is not None:
            player_predictions[player] = pred_data
    
    # Import skewnorm once
    from scipy.stats import skewnorm
    
    # Helper function for sigma flag
    def flag_sigma(s):
        if s <= 5.0:
            return 'Low'
        elif s <= 6.0:
            return 'Med'
        else:
            return 'High'
    
    results = []
    batch_size = n_simulations * 2
    
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
        mapped_name = nameDict.get(name, name)
        
        # Get pre-computed prediction data
        if name not in player_predictions:
            continue
        
        prediction_data = player_predictions[name]
        pred = prediction_data['prediction']
        sigma = prediction_data['sigma'] * variance_inflation
        skew = prediction_data['skew']
        
        # Pre-calculate distribution parameters using player-specific sigma
        mu = pred
        
        # Determine skew parameter: use override if provided, otherwise use enforce_downside_skew or player-specific skew
        a = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew)
        
        if use_monte_carlo:
            # Monte Carlo simulation using rejection sampling to properly handle zero-bound
            simulations = []
            while len(simulations) < n_simulations:
                batch = skewnorm.rvs(a, loc=mu, scale=sigma, size=batch_size, random_state=rng)
                positive_batch = batch[batch >= 0]
                simulations.extend(positive_batch)
                if len(simulations) >= n_simulations:
                    break
            simulations = np.array(simulations[:n_simulations])
            p_over_raw = np.mean(simulations > line)
        else:
            # Analytical method using conditional probability: P(X > line | X >= 0)
            p_above_line = 1 - skewnorm.cdf(line, a, loc=mu, scale=sigma)
            p_non_negative = 1 - skewnorm.cdf(0, a, loc=mu, scale=sigma)
            if p_non_negative > 1e-10:
                p_over_raw = p_above_line / p_non_negative
            else:
                p_over_raw = max(0.0, min(1.0, p_above_line))
        
        # Use raw probabilities for all decision-making (edge, EV, Kelly, recommendations)
        p_under_raw = 1.0 - p_over_raw
        
        # Choose probability based on the offered side (not model prediction)
        is_over = str(side).upper().startswith('O')
        p = p_over_raw if is_over else p_under_raw
        
        # Convert odds to decimal and calculate EV
        dec_odds = american_to_decimal(odds)
        b = dec_odds - 1.0
        
        # EV calculations using raw probabilities
        ev_per_dollar = p * b - (1 - p)
        ev_total = ev_per_dollar * stake
        
        # Kelly criterion with variance-adjusted constraint (using raw probabilities)
        kelly_fraction = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
        kelly_capped_fraction = min(kelly_fraction, max_kelly)
        kelly_dollars = kelly_capped_fraction * stake * b
        
        # Edge calculation (difference between model and market probabilities) - using raw probabilities
        market_prob = impliedProb(odds)
        model_prob = p_over_raw if is_over else p_under_raw
        edge = model_prob - market_prob
        
        # Recommendation based on edge threshold (using raw probabilities)
        recommendation = 1 if (abs(line - pred) > 4.5) else 0
        
        # Confidence interval (using proper statistical formula with player-specific sigma)
        confidence_interval = (
            max(0, pred - 1.96 * sigma),  # Lower bound, clipped at zero
            pred + 1.96 * sigma           # Upper bound
        )
        interval_width = round(confidence_interval[1] - confidence_interval[0], 2)
        sigma_flag = flag_sigma(sigma)
        
        results.append({
            'NAME': original_name,
            'BOOKMAKER': bookmaker,
            'CATEGORY': category,
            'LINE': line,
            'ODDS': odds,
            'SIDE': side,
            'PREDICTION': round(pred, 2),
            'RECOMMENDATION': recommendation,
            'OVER%': round(p_over_raw, 3),
            'UNDER%': round(p_under_raw, 3),
            'IMPLIED PROB': round(market_prob, 3),
            'MODEL PROB': round(model_prob, 3),  # Raw probability for edge calculation (already stored)
            'EDGE': round(edge, 3),  # Calculated using raw probabilities
            'EV$': round(ev_total, 2),
            'KELLY_FRACTION': round(kelly_fraction, 3),
            'KELLY_DOLLARS': round(kelly_dollars, 2),
            'CONFIDENCE INTERVAL': f"({confidence_interval[0]:.1f}, {confidence_interval[1]:.1f})",
            'INTERVAL WIDTH': interval_width,
            'SIGMA': round(sigma, 2),
            'SIGMA FLAG': sigma_flag,
            'EXPECTED ROI': round((ev_total / stake )* 100, 1),
            'SIMULATION_METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical'
        })
    
    return pd.DataFrame(results)

def calculate2LegBets(data, bookmakers, model, features, current_date, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100,
                 enforce_downside_skew: bool = False, skew_override: float | None = None,
                 max_player_appearances: int = 3, isotonic_calibrator=None):
    """
    Calculate 2-leg parlay opportunities with optional isotonic calibration.
    
    Args:
        isotonic_calibrator: Optional isotonic regression calibrator for mean predictions
    """
    if isotonic_calibrator is not None:
        print("Using isotonic regression calibration for 2-leg bets")
    
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

    # Set random seed once before the loop for reproducibility
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Pre-compute all predictions once
    player_predictions = {}
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Pre-computing predictions for {len(available_players)} players...")
    current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
    
    for player in available_players:
        # Map player name
        mapped_player = nameDict.get(player, player)
        
        # Get prediction data
        pred_data = get_cached_prediction(mapped_player, data, model, features, current_date, 
                                         projectedStartingFive, mainStartingFive, teamStarPlayer,
                                         isotonic_calibrator)
        if pred_data is None:
            continue
        
        # Get player data for team lookup
        player_data = data[data['PLAYER_NAME'] == mapped_player]
        if player_data.empty:
            continue
        
        # Get team
        player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
        
        # Get opponent using findOpp
        opp_team, _ = findOpp(mapped_player, player_data, current_date_str)
        if opp_team is None:
            continue
        
        # Get betting line (first available)
        player_bets = bookmakers[bookmakers['NAME'] == player]
        if player_bets.empty:
            continue
        
        # Store all data
        player_predictions[player] = pred_data
        player_teams[player] = player_team
        player_opponents[player] = opp_team
        player_lines[player] = player_bets.iloc[0]
    
    # Filter to only players with valid predictions, teams, opponents, and lines
    available_players = [p for p in available_players if p in player_predictions and p in player_teams and p in player_opponents and p in player_lines]
    
    if len(available_players) < 2:
        print("Not enough players with valid predictions for 2-leg bets")
        return pd.DataFrame()
    
    print(f"Processing {len(available_players)} players with valid predictions...")
    
    # Generate only valid combinations (different teams)
    valid_combinations = []
    for p1, p2 in combinations(available_players, 2):
        team1 = player_teams[p1]
        team2 = player_teams[p2]
        opp1 = player_opponents[p1]
        opp2 = player_opponents[p2]
        
        # Prevent same-team combinations
        if team1 == team2:
            continue
        
        valid_combinations.append((p1, p2))
    
    print(f"Generated {len(valid_combinations)} valid 2-leg combinations")
    
    # Pre-compute market probability (constant for all)
    market_prob = impliedProb(-137)
    market_prob_combined = market_prob ** 2
    
    # Pre-compute constants
    corr_factor = 0.90
    payout_multiple = 3.0
    b = payout_multiple - 1.0
    
    # Helper function for sigma flag
    def flag_sigma(s):
        if s <= 5.0:
            return 'Low'
        elif s <= 6.0:
            return 'Med'
        else:
            return 'High'
    
    # Import skewnorm once
    from scipy.stats import skewnorm
    
    results = []
    batch_size = n_simulations * 2

    for player1, player2 in valid_combinations:
        # Map player names
        mapped_p1 = nameDict.get(player1, player1)
        mapped_p2 = nameDict.get(player2, player2)
        
        # Get pre-computed data
        pred1_data = player_predictions[player1]
        pred2_data = player_predictions[player2]
        
        pred1_val = pred1_data['prediction']
        pred2_val = pred2_data['prediction']
        sigma1 = pred1_data['sigma'] * variance_inflation
        sigma2 = pred2_data['sigma'] * variance_inflation
        skew1 = pred1_data['skew']
        skew2 = pred2_data['skew']
        
        # Get lines
        line1 = float(player_lines[player1]['LINE'])
        line2 = float(player_lines[player2]['LINE'])
        
        # Distribution parameters
        mu1 = pred1_val
        mu2 = pred2_val
        
        # Determine skew parameters
        a1 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew1)
        a2 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew2)
        
        # Confidence intervals and sigma flags
        ci1 = (max(0, mu1 - 1.96 * sigma1), mu1 + 1.96 * sigma1)
        ci2 = (max(0, mu2 - 1.96 * sigma2), mu2 + 1.96 * sigma2)
        width1 = round(ci1[1] - ci1[0], 2)
        width2 = round(ci2[1] - ci2[0], 2)
        sigma_flag1 = flag_sigma(sigma1)
        sigma_flag2 = flag_sigma(sigma2)
        
        # Calculate probabilities using Monte Carlo or analytical method
        if use_monte_carlo:
            # Player 1
            sim1_list = []
            while len(sim1_list) < n_simulations:
                batch = skewnorm.rvs(a1, loc=mu1, scale=sigma1, size=batch_size, random_state=rng)
                positive_batch = batch[batch >= 0]
                sim1_list.extend(positive_batch)
                if len(sim1_list) >= n_simulations:
                    break
            sim1 = np.array(sim1_list[:n_simulations])
            
            # Player 2
            sim2_list = []
            while len(sim2_list) < n_simulations:
                batch = skewnorm.rvs(a2, loc=mu2, scale=sigma2, size=batch_size, random_state=rng)
                positive_batch = batch[batch >= 0]
                sim2_list.extend(positive_batch)
                if len(sim2_list) >= n_simulations:
                    break
            sim2 = np.array(sim2_list[:n_simulations])
            
            p1_over_raw = np.mean(sim1 > line1)
            p2_over_raw = np.mean(sim2 > line2)
        else:
            # Analytical method using conditional probability
            # Player 1
            p1_above_line = 1 - skewnorm.cdf(line1, a1, loc=mu1, scale=sigma1)
            p1_non_negative = 1 - skewnorm.cdf(0, a1, loc=mu1, scale=sigma1)
            p1_over_raw = (p1_above_line / p1_non_negative) if p1_non_negative > 1e-10 else max(0.0, min(1.0, p1_above_line))
            
            # Player 2
            p2_above_line = 1 - skewnorm.cdf(line2, a2, loc=mu2, scale=sigma2)
            p2_non_negative = 1 - skewnorm.cdf(0, a2, loc=mu2, scale=sigma2)
            p2_over_raw = (p2_above_line / p2_non_negative) if p2_non_negative > 1e-10 else max(0.0, min(1.0, p2_above_line))
        
        # Determine model sides based on predictions vs lines
        if pred1_val > line1:
            model_side1 = 'over'
            p1 = p1_over_raw
            p1_raw = p1_over_raw
        else:
            model_side1 = 'under'
            p1 = 1 - p1_over_raw
            p1_raw = 1 - p1_over_raw
            
        if pred2_val > line2:
            model_side2 = 'over'
            p2 = p2_over_raw
            p2_raw = p2_over_raw
        else:
            model_side2 = 'under'
            p2 = 1 - p2_over_raw
            p2_raw = 1 - p2_over_raw
        
        # Calculate combined probability and EV with correlation adjustment (using raw probabilities)
        p_both_raw = p1_raw * p2_raw
        p_both = p_both_raw * corr_factor
        ev = payout_multiple * p_both - 1
        ev_dollars = ev * stake
        
        # Kelly criterion with variance-adjusted constraint
        kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b) if b > 0 else 0.0
        
        # Edge calculation (using raw probabilities for accuracy)
        edge1 = p1_raw - market_prob
        edge2 = p2_raw - market_prob
        
        # Calculate combined probabilities and edge (using raw probabilities)
        combined_model_prob = p1_raw * p2_raw
        combined_edge = combined_model_prob - market_prob_combined
        
        # Recommendation based on multiple criteria
        recommendation = 1 if (abs(line1 - pred1_val) > 4.5 and abs(line2 - pred2_val) > 4.5) else 0
        
        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'LINE 1': line1,
            'LINE 2': line2,
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
            'CONFIDENCE INTERVAL 1': f"({ci1[0]:.1f}, {ci1[1]:.1f})",
            'CONFIDENCE INTERVAL 2': f"({ci2[0]:.1f}, {ci2[1]:.1f})",
            'INTERVAL WIDTH 1': width1,
            'INTERVAL WIDTH 2': width2,
            'SIGMA 1': round(sigma1, 2),
            'SIGMA 2': round(sigma2, 2),
            'SIGMA FLAG 1': sigma_flag1,
            'SIGMA FLAG 2': sigma_flag2,
            'EXPECTED ROI': round((ev_dollars / stake) * 100, 1),
            'SIMULATION METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical'
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply player frequency limit for diversification
    if max_player_appearances is not None and len(results_df) > 0:
        # Sort by EV descending to prioritize best bets
        results_df = results_df.sort_values('EV$', ascending=False).reset_index(drop=True)
        
        # Track how many times each player appears
        player_count = defaultdict(int)
        selected_rows = []
        
        for idx, row in results_df.iterrows():
            p1 = row['NAME 1']
            p2 = row['NAME 2']
            
            # Check if adding this combination would exceed the limit for any player
            if (player_count[p1] < max_player_appearances and 
                player_count[p2] < max_player_appearances):
                selected_rows.append(idx)
                player_count[p1] += 1
                player_count[p2] += 1
        
        results_df = results_df.loc[selected_rows].reset_index(drop=True)
        print(f"Applied player frequency limit ({max_player_appearances} max appearances per player)")
        print(f"Selected {len(selected_rows)} combinations from {len(results)} candidates")
    
    return results_df

def calculate3LegBets(data, bookmakers, model, features, current_date, edge_threshold=0.05, top_n=10, 
                 variance_inflation=1.1, 
                 use_monte_carlo=True, n_simulations=10000, max_kelly=0.25, stake=100,
                 enforce_downside_skew: bool = False, skew_override: float | None = None,
                 max_player_appearances: int = 2, isotonic_calibrator=None):
    """
    Calculate 3-leg parlay opportunities with optional isotonic calibration.
    
    Args:
        isotonic_calibrator: Optional isotonic regression calibrator for mean predictions
    """
    if isotonic_calibrator is not None:
        print("Using isotonic regression calibration for 3-leg bets")
    
    category = 'player_points'
    bookmakers = bookmakers[(bookmakers['CATEGORY'] == category)]
    if bookmakers.empty:
        print(f"No bets found for {category}")
        return pd.DataFrame()

    # Get all available players for 3-leg combinations
    available_players = bookmakers['NAME'].unique()

    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Pre-compute all predictions once
    player_predictions = {}
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Pre-computing predictions for {len(available_players)} players...")
    current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
    
    for player in available_players:
        # Map player name
        mapped_player = nameDict.get(player, player)
        
        # Get prediction data
        pred_data = get_cached_prediction(mapped_player, data, model, features, current_date, 
                                         projectedStartingFive, mainStartingFive, teamStarPlayer,
                                         isotonic_calibrator)
        if pred_data is None:
            continue
        
        # Get player data for team lookup
        player_data = data[data['PLAYER_NAME'] == mapped_player]
        if player_data.empty:
            continue
        
        # Get team
        player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
        
        # Get opponent using findOpp
        opp_team, _ = findOpp(mapped_player, player_data, current_date_str)
        if opp_team is None:
            continue
        
        # Get betting line (first available)
        player_bets = bookmakers[bookmakers['NAME'] == player]
        if player_bets.empty:
            continue
        
        # Store all data
        player_predictions[player] = pred_data
        player_teams[player] = player_team
        player_opponents[player] = opp_team
        player_lines[player] = player_bets.iloc[0]
    
    # Filter to only players with valid predictions, teams, opponents, and lines
    available_players = [p for p in available_players if p in player_predictions and p in player_teams and p in player_opponents and p in player_lines]
    
    if len(available_players) < 3:
        print("Not enough players with valid predictions for 3-leg bets")
        return pd.DataFrame()
    
    print(f"Processing {len(available_players)} players with valid predictions...")
    
    # Generate only valid combinations (different teams and not all in same game)
    valid_combinations = []
    for p1, p2, p3 in combinations(available_players, 3):
        team1 = player_teams[p1]
        team2 = player_teams[p2]
        team3 = player_teams[p3]
        opp1 = player_opponents[p1]
        opp2 = player_opponents[p2]
        opp3 = player_opponents[p3]
        
        # Prevent all 3 players from being on the same team
        if team1 == team2 == team3:
            continue
        
        # Prevent all 3 players from being in the same game
        # For 3 players to be in the same game, they must be from exactly 2 teams
        unique_teams = set([team1, team2, team3])
        if len(unique_teams) == 2:
            # Check if the two teams are playing each other (same game)
            # Get the two teams
            teams_list = list(unique_teams)
            team_a = teams_list[0]
            team_b = teams_list[1]
            
            # Check if team_a's opponent is team_b and team_b's opponent is team_a
            # Find which players are on which team
            p1_team = team1
            p2_team = team2
            p3_team = team3
            
            # Get opponents for each team
            if p1_team == team_a:
                team_a_opp = opp1
            elif p2_team == team_a:
                team_a_opp = opp2
            else:  # p3_team == team_a
                team_a_opp = opp3
            
            if p1_team == team_b:
                team_b_opp = opp1
            elif p2_team == team_b:
                team_b_opp = opp2
            else:  # p3_team == team_b
                team_b_opp = opp3
            
            # If team_a's opponent is team_b and team_b's opponent is team_a, they're in the same game
            if team_a_opp == team_b and team_b_opp == team_a:
                continue  # Skip - all 3 players are in the same game
        
        valid_combinations.append((p1, p2, p3))
    
    print(f"Generated {len(valid_combinations)} valid 3-leg combinations")
    
    # Pre-compute market probability (constant for all)
    market_prob = impliedProb(-137)
    market_prob_combined = market_prob ** 3
    
    # Pre-compute constants
    corr_factor = 0.90
    payout_multiple = 6.0
    b = payout_multiple - 1.0
    
    # Helper function for sigma flag
    def flag_sigma(s):
        if s <= 5.0:
            return 'Low'
        elif s <= 6.0:
            return 'Med'
        else:
            return 'High'
    
    # Import skewnorm once
    from scipy.stats import skewnorm
    
    results = []
    batch_size = n_simulations * 2
    
    # Process each valid combination
    for player1, player2, player3 in valid_combinations:
        # Map player names
        mapped_p1 = nameDict.get(player1, player1)
        mapped_p2 = nameDict.get(player2, player2)
        mapped_p3 = nameDict.get(player3, player3)
        
        # Get pre-computed data
        pred1_data = player_predictions[player1]
        pred2_data = player_predictions[player2]
        pred3_data = player_predictions[player3]
        
        pred1_val = pred1_data['prediction']
        pred2_val = pred2_data['prediction']
        pred3_val = pred3_data['prediction']
        sigma1 = pred1_data['sigma'] * variance_inflation
        sigma2 = pred2_data['sigma'] * variance_inflation
        sigma3 = pred3_data['sigma'] * variance_inflation
        skew1 = pred1_data['skew']
        skew2 = pred2_data['skew']
        skew3 = pred3_data['skew']
        
        # Get lines
        line1 = float(player_lines[player1]['LINE'])
        line2 = float(player_lines[player2]['LINE'])
        line3 = float(player_lines[player3]['LINE'])
        
        # Distribution parameters
        mu1 = pred1_val
        mu2 = pred2_val
        mu3 = pred3_val
        
        # Determine skew parameters
        a1 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew1)
        a2 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew2)
        a3 = skew_override if skew_override is not None else (-2.0 if enforce_downside_skew else skew3)
        
        # Confidence intervals and sigma flags
        ci1 = (max(0, mu1 - 1.96 * sigma1), mu1 + 1.96 * sigma1)
        ci2 = (max(0, mu2 - 1.96 * sigma2), mu2 + 1.96 * sigma2)
        ci3 = (max(0, mu3 - 1.96 * sigma3), mu3 + 1.96 * sigma3)
        width1 = round(ci1[1] - ci1[0], 2)
        width2 = round(ci2[1] - ci2[0], 2)
        width3 = round(ci3[1] - ci3[0], 2)
        sigma_flag1 = flag_sigma(sigma1)
        sigma_flag2 = flag_sigma(sigma2)
        sigma_flag3 = flag_sigma(sigma3)
        
        # Calculate probabilities using Monte Carlo or analytical method
        if use_monte_carlo:
            # Optimized Monte Carlo with vectorized operations
            # Player 1
            sim1_list = []
            while len(sim1_list) < n_simulations:
                batch = skewnorm.rvs(a1, loc=mu1, scale=sigma1, size=batch_size, random_state=rng)
                positive_batch = batch[batch >= 0]
                sim1_list.extend(positive_batch)
                if len(sim1_list) >= n_simulations:
                    break
            sim1 = np.array(sim1_list[:n_simulations])
            
            # Player 2
            sim2_list = []
            while len(sim2_list) < n_simulations:
                batch = skewnorm.rvs(a2, loc=mu2, scale=sigma2, size=batch_size, random_state=rng)
                positive_batch = batch[batch >= 0]
                sim2_list.extend(positive_batch)
                if len(sim2_list) >= n_simulations:
                    break
            sim2 = np.array(sim2_list[:n_simulations])
            
            # Player 3
            sim3_list = []
            while len(sim3_list) < n_simulations:
                batch = skewnorm.rvs(a3, loc=mu3, scale=sigma3, size=batch_size, random_state=rng)
                positive_batch = batch[batch >= 0]
                sim3_list.extend(positive_batch)
                if len(sim3_list) >= n_simulations:
                    break
            sim3 = np.array(sim3_list[:n_simulations])
            
            p1_over_raw = np.mean(sim1 > line1)
            p2_over_raw = np.mean(sim2 > line2)
            p3_over_raw = np.mean(sim3 > line3)
        else:
            # Analytical method using conditional probability
            # Player 1
            p1_above_line = 1 - skewnorm.cdf(line1, a1, loc=mu1, scale=sigma1)
            p1_non_negative = 1 - skewnorm.cdf(0, a1, loc=mu1, scale=sigma1)
            p1_over_raw = (p1_above_line / p1_non_negative) if p1_non_negative > 1e-10 else max(0.0, min(1.0, p1_above_line))
            
            # Player 2
            p2_above_line = 1 - skewnorm.cdf(line2, a2, loc=mu2, scale=sigma2)
            p2_non_negative = 1 - skewnorm.cdf(0, a2, loc=mu2, scale=sigma2)
            p2_over_raw = (p2_above_line / p2_non_negative) if p2_non_negative > 1e-10 else max(0.0, min(1.0, p2_above_line))
            
            # Player 3
            p3_above_line = 1 - skewnorm.cdf(line3, a3, loc=mu3, scale=sigma3)
            p3_non_negative = 1 - skewnorm.cdf(0, a3, loc=mu3, scale=sigma3)
            p3_over_raw = (p3_above_line / p3_non_negative) if p3_non_negative > 1e-10 else max(0.0, min(1.0, p3_above_line))
        
        # Determine model sides based on predictions vs lines
        if pred1_val > line1:
            model_side1 = 'over'
            p1 = p1_over_raw
            p1_raw = p1_over_raw
        else:
            model_side1 = 'under'
            p1 = 1 - p1_over_raw
            p1_raw = 1 - p1_over_raw
            
        if pred2_val > line2:
            model_side2 = 'over'
            p2 = p2_over_raw
            p2_raw = p2_over_raw
        else:
            model_side2 = 'under'
            p2 = 1 - p2_over_raw
            p2_raw = 1 - p2_over_raw
            
        if pred3_val > line3:
            model_side3 = 'over'
            p3 = p3_over_raw
            p3_raw = p3_over_raw
        else:
            model_side3 = 'under'
            p3 = 1 - p3_over_raw
            p3_raw = 1 - p3_over_raw
        
        # Calculate combined probability and EV with correlation adjustment (using raw probabilities)
        p_all_three_raw = p1_raw * p2_raw * p3_raw
        p_all_three = p_all_three_raw * (corr_factor ** 2)  # Squared for 3-leg parlay
        ev = payout_multiple * p_all_three - 1
        ev_dollars = ev * stake

        # Kelly criterion with variance-adjusted constraint
        kelly_full = max(0.0, (b * p_all_three - (1 - p_all_three)) / b) if b > 0 else 0.0
        
        # Edge calculation (using raw probabilities for accuracy)
        edge1 = p1_raw - market_prob
        edge2 = p2_raw - market_prob
        edge3 = p3_raw - market_prob
        
        # Calculate combined probabilities and edge (using raw probabilities)
        combined_model_prob = p1_raw * p2_raw * p3_raw
        combined_edge = combined_model_prob - market_prob_combined
        
        # Recommendation based on multiple criteria
        recommendation = 1 if (abs(line1 - pred1_val) > 4.5 and abs(line2 - pred2_val) > 4.5 and abs(line3 - pred3_val) > 4.5) else 0
        
        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'NAME 3': mapped_p3,
            'LINE 1': line1,
            'LINE 2': line2,
            'LINE 3': line3,
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
            'EXPECTED ROI': round((ev_dollars / stake) * 100, 1),
            'SIMULATION METHOD': 'Monte Carlo' if use_monte_carlo else 'Analytical'
        })

    results_df = pd.DataFrame(results)
    
    # Apply player frequency limit for diversification
    if max_player_appearances is not None and len(results_df) > 0:
        # Sort by EV descending to prioritize best bets
        results_df = results_df.sort_values('EV$', ascending=False).reset_index(drop=True)
        
        # Track how many times each player appears
        player_count = defaultdict(int)
        selected_rows = []
        
        for idx, row in results_df.iterrows():
            p1 = row['NAME 1']
            p2 = row['NAME 2']
            p3 = row['NAME 3']
            
            # Check if adding this combination would exceed the limit for any player
            if (player_count[p1] < max_player_appearances and 
                player_count[p2] < max_player_appearances and
                player_count[p3] < max_player_appearances):
                selected_rows.append(idx)
                player_count[p1] += 1
                player_count[p2] += 1
                player_count[p3] += 1
        
        results_df = results_df.loc[selected_rows].reset_index(drop=True)
        print(f"Applied player frequency limit ({max_player_appearances} max appearances per player)")
        print(f"Selected {len(selected_rows)} combinations from {len(results)} candidates")
    
    return results_df