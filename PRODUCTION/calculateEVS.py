import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from zoneinfo import ZoneInfo
from PRODUCTION.teamInfo import *
from PRODUCTION.pipelineV2 import *
from itertools import combinations
from PRODUCTION.teamInfo import nameDict
import os


# Convert UTC to ET and create game_date column
def convert_to_et(utc_time):
    utc_dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
    et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
    return et_dt.strftime('%Y-%m-%d')  

def impliedProb(odds, vig=0.05):
    if odds > 0:
        prob = 100 / (odds + 100)
    else:
        prob = abs(odds) / (abs(odds) + 100)

    # For 5% total vig, each side has ~2.5% vig, so divide by (1 + vig/2)
    fair_prob = prob / (1.0 + vig / 2)
    return fair_prob

def american_to_decimal(odds):
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds"""
    if decimal_odds == 2.0:
        return +100
    elif decimal_odds > 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))

def calculate_parlay_odds(odds_list, correlation_adjustment=1.0):
    if len(odds_list) == 0:
        return -137
    
    # Convert all to decimal odds
    decimal_odds = [american_to_decimal(odds) for odds in odds_list]
    
    # Calculate combined decimal odds (multiply them)
    combined_decimal = 1.0
    for dec in decimal_odds:
        combined_decimal *= dec
    
    # Apply correlation adjustment
    # If correlation_adjustment < 1.0, the true probability is lower
    # Lower probability = higher odds, so we multiply decimal odds by (1 / corr_adjustment)
    if correlation_adjustment > 0:
        adjusted_decimal = combined_decimal / correlation_adjustment
    else:
        adjusted_decimal = combined_decimal
    
    return decimal_to_american(adjusted_decimal)

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

# Cache for player odds - stores hashmap per date
_odds_cache = {}  # {date_str: {(player_name, line, side): best_odds}}

def _load_odds_cache(date_str):
    """Load and cache odds data for a specific date. Returns hashmap for O(1) lookup."""
    if date_str in _odds_cache:
        return _odds_cache[date_str]
    
    try:
        # Construct file path
        csv_path = f'DATA/CSV_FILES/PROP_DATA/PLAYER_LINES/NBA_US_{date_str}.csv'
        
        # Try absolute path if relative doesn't work
        if not os.path.exists(csv_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, '..', 'DATA', 'CSV_FILES', 'PROP_DATA', 'PLAYER_LINES', f'NBA_US_{date_str}.csv')
            csv_path = os.path.normpath(csv_path)
        
        # Check if file exists
        if not os.path.exists(csv_path):
            _odds_cache[date_str] = {}
            return {}
        
        # Read CSV and filter to only player_points
        df = pd.read_csv(csv_path)
        
        # Filter to only player_points category
        if 'CATEGORY' in df.columns:
            df = df[df['CATEGORY'] == 'player_points']
        
        if df.empty:
            _odds_cache[date_str] = {}
            return {}
        
        # Create hashmap: (player_name, line, side) -> best_odds
        odds_map = {}
        
        # Normalize OVER/UNDER column first to handle case variations
        df['SIDE_NORMALIZED'] = df['OVER/UNDER'].str.upper().str.strip().apply(
            lambda x: 'over' if str(x).startswith('O') else 'under'
        )
        
        # Group by player, line, and normalized side, then get best (maximum) odds
        for (name, line, side_norm), group in df.groupby(['NAME', 'LINE', 'SIDE_NORMALIZED']):
            line_float = float(line)
            
            # Get best (maximum) odds for this combination
            best_odds = int(group['ODDS'].max())
            
            # Store in hashmap
            key = (str(name), line_float, side_norm)
            # If key already exists, keep the best (maximum) odds
            if key in odds_map:
                odds_map[key] = max(odds_map[key], best_odds)
            else:
                odds_map[key] = best_odds
        
        # Cache the hashmap
        _odds_cache[date_str] = odds_map
        return odds_map
    
    except Exception as e:
        # On error, cache empty dict to avoid repeated file reads
        _odds_cache[date_str] = {}
        return {}

def get_cached_prediction(player_name, data, model, features, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer):
    cache_key = f"{player_name}_{current_date}"
    
    if cache_key not in _prediction_cache:
        try:
            player_df = data[data['PLAYER_NAME'] == player_name].sort_values(by='GAME_DATE')
            if player_df.empty:
                return None
            
            vector = buildVector(player_name, data, current_date, projectedStartingFive, mainStartingFive, teamStarPlayer)
            if vector is None:
                raise ValueError(f"buildVector returned None for {player_name}")
            
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
                # Check if score-dependent params are provided (4th element of tuple)
                score_dependent_params = model[3] if len(model) > 3 else None
                use_score_dependent = score_dependent_params is not None
                
                from MODELS.ngboostModel import predict_mean_variance_split
                
                # Get raw mean prediction in ORIGINAL space
                raw_mean = mean_model.predict(vector[features])
                bias_correction = getattr(mean_model, 'bias_correction_', 0.0)
                if bias_correction is None:
                    bias_correction = 0.0
                
                # Extract scalar
                if isinstance(raw_mean, np.ndarray):
                    raw_mean_val = raw_mean.flat[0]
                elif isinstance(raw_mean, pd.Series):
                    raw_mean_val = raw_mean.iloc[0]
                else:
                    raw_mean_val = raw_mean
                
                # Convert to log-space
                mu_log = np.log1p(raw_mean_val + bias_correction)
                
                # Get variance in log space (UNCALIBRATED)
                # variance_model predicts log(variance) where variance is Var[log(Y+1)]
                X = vector[features].replace([np.inf, -np.inf], np.nan)
                if X.isnull().any().any():
                    X = X.fillna(X.median())
                log_var_pred = variance_model.predict(X)  # This is log(variance of log(Y+1))
                
                # Extract log_var_pred scalar
                if isinstance(log_var_pred, np.ndarray):
                    log_var_val = log_var_pred.flat[0] if log_var_pred.size > 0 else None
                elif isinstance(log_var_pred, pd.Series):
                    log_var_val = log_var_pred.iloc[0]
                else:
                    log_var_val = log_var_pred
                
                if log_var_val is None or np.isnan(log_var_val) or np.isinf(log_var_val):
                    raise ValueError(f"Invalid log_var_pred value for {player_name}: {log_var_val}")
                
                # Convert to variance and then to sigma (UNCALIBRATED)
                # variance_log = exp(log_var_val) is the variance of log(Y+1) (in log space)
                variance_log_uncalibrated = np.exp(log_var_val)
                
                # Extract uncalibrated variance
                if isinstance(variance_log_uncalibrated, np.ndarray):
                    var_log_val_uncalib = variance_log_uncalibrated.flat[0] if variance_log_uncalibrated.size > 0 else None
                elif isinstance(variance_log_uncalibrated, pd.Series):
                    var_log_val_uncalib = variance_log_uncalibrated.iloc[0]
                else:
                    var_log_val_uncalib = variance_log_uncalibrated
                
                if var_log_val_uncalib is None or np.isnan(var_log_val_uncalib) or var_log_val_uncalib <= 0:
                    raise ValueError(f"Invalid variance_log_uncalibrated value for {player_name}: {var_log_val_uncalib}")
                
                # sigma_log_uncalibrated = sqrt(variance_log) is the standard deviation of log(Y+1) (in log space, uncalibrated)
                sigma_log_uncalibrated = float(np.sqrt(var_log_val_uncalib))
                
                # mu_log is already a scalar (converted above)
                # Validate before converting to float
                if mu_log is None or np.isnan(mu_log) or np.isinf(mu_log):
                    raise ValueError(f"Invalid mu_log value for {player_name}: {mu_log}")
                
                mu_log_val = float(mu_log)
                
                # Get transformed predictions for display (original space)
                mu, variance = predict_mean_variance_split(
                    mean_model, variance_model, vector, features, calibration_factor,
                    prediction_type='mean',
                    use_score_dependent_calibration=use_score_dependent,
                    score_dependent_params=score_dependent_params
                )
                
                # Extract transformed mean for display
                if isinstance(mu, np.ndarray):
                    pred = float(mu.flat[0])
                elif isinstance(mu, pd.Series):
                    pred = float(mu.iloc[0])
                else:
                    pred = float(mu)
                pred = round(pred, 3)
                
                # Extract transformed sigma for display (but we'll use log-space for calculations)
                if isinstance(variance, np.ndarray):
                    sigma = float(np.sqrt(variance.flat[0]))
                elif isinstance(variance, pd.Series):
                    sigma = float(np.sqrt(variance.iloc[0]))
                else:
                    sigma = float(np.sqrt(variance))
                
                skew = 0.0
            elif hasattr(model, "pred_dist"):
                # Single NGBoost model with pred_dist
                dist = model.pred_dist(vector)
                pred = round(float(dist.loc[0]), 3)
                sigma = float(dist.scale[0])
                # For single model, we need to compute log-space params
                mu_log_val = dist.loc[0]  # Already in log space if trained on log-transformed target
                sigma_log_uncalibrated = dist.scale[0]
                calibration_factor = 1.0
                use_score_dependent = False
                score_dependent_params = None
                skew = 0.0
            else:
                # Point model (XGBoost or similar)
                pred = round(float(model.predict(vector)[0]), 3)
                # Use simple default sigma and neutral skew when model doesn't provide distribution
                sigma = max(1.5, min(12.0, max(2.5, min(8.5, pred * 0.15))))
                # For point model, estimate log-space params
                mu_log_val = np.log1p(pred)
                sigma_log_uncalibrated = 0.15  # Default uncertainty
                calibration_factor = 1.0
                use_score_dependent = False
                score_dependent_params = None
                skew = 0.0
            
            _prediction_cache[cache_key] = {
                'prediction': pred,  # Original space (for display)
                'sigma': sigma,  # Original space (for display, calibrated)
                'mu_log': mu_log_val,  # Log space (for probability calculations)
                'sigma_log': sigma_log_uncalibrated,  # Log space (UNCALIBRATED, for probability calculations)
                'calibration_factor': calibration_factor,  # Store for later use
                'use_score_dependent': use_score_dependent,  # Store flag
                'score_dependent_params': score_dependent_params.copy() if score_dependent_params else None,  # COPY the dict!
                'skew': skew
            }
        except Exception as e:
            print(f"Error getting prediction for {player_name}: {e}")
            return None
    return _prediction_cache[cache_key]



#----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def calculate2LegBets(data, bookmakers, model, features, current_date, 
                     edge_threshold=0.05, top_n=10, 
                     stake=100, max_player_appearances: int = 2):
    """
    Optimized 2-leg bet calculator using calibrated Normal distribution
    """

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

    # Pre-compute all predictions
    player_predictions = {}
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Pre-computing predictions for {len(available_players)} players...")
    current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
    
    for player in available_players:
        mapped_player = nameDict.get(player, player)
        
        pred_data = get_cached_prediction(
            mapped_player, data, model, features, current_date, 
            projectedStartingFive, mainStartingFive, teamStarPlayer
        )
        if pred_data is None:
            continue
        
        player_data = data[data['PLAYER_NAME'] == mapped_player]
        if player_data.empty:
            continue
        
        player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
        opp_team, _ = findOpp(mapped_player, player_data, current_date_str)
        if opp_team is None:
            continue
        
        player_bets = bookmakers[bookmakers['NAME'] == player]
        if player_bets.empty:
            continue
        
        player_predictions[player] = pred_data
        player_teams[player] = player_team
        player_opponents[player] = opp_team
        player_lines[player] = player_bets.iloc[0]
    
    available_players = [p for p in available_players 
                        if p in player_predictions and p in player_teams 
                        and p in player_opponents and p in player_lines]
    
    if len(available_players) < 2:
        print("Not enough players with valid predictions for 2-leg bets")
        return pd.DataFrame()
    
    print(f"Processing {len(available_players)} players...")
    
    # Generate valid combinations
    valid_combinations = []
    for p1, p2 in combinations(available_players, 2):
        if player_teams[p1] != player_teams[p2]:  # Different teams
            valid_combinations.append((p1, p2))
    
    print(f"Generated {len(valid_combinations)} valid 2-leg combinations")
    
    # Constants
    from scipy.stats import norm
    payout_multiple = 3.0
    
    results = []

    for player1, player2 in valid_combinations:
        # Map player names
        mapped_p1 = nameDict.get(player1, player1)
        mapped_p2 = nameDict.get(player2, player2)
        
        # Get pre-computed data
        pred1_data = player_predictions[player1]
        pred2_data = player_predictions[player2]
        
        mu1_raw = pred1_data['prediction']
        mu2_raw = pred2_data['prediction']
        sigma1 = pred1_data['sigma']  # Already calibrated!
        sigma2 = pred2_data['sigma']
        
        line1 = float(player_lines[player1]['LINE'])
        line2 = float(player_lines[player2]['LINE'])
        
        # Use raw predictions directly
        mu1 = mu1_raw
        mu2 = mu2_raw
        
        # Get uncalibrated log-space parameters from cache
        mu1_log = pred1_data.get('mu_log', np.log1p(mu1_raw))
        sigma1_log_uncalib = pred1_data.get('sigma_log', sigma1 / (mu1_raw + 1))
        mu2_log = pred2_data.get('mu_log', np.log1p(mu2_raw))
        sigma2_log_uncalib = pred2_data.get('sigma_log', sigma2 / (mu2_raw + 1))
        
        # Get calibration factors
        cal_factor1 = pred1_data.get('calibration_factor', 1.25)
        cal_factor2 = pred2_data.get('calibration_factor', 1.25)
        use_score_dep1 = pred1_data.get('use_score_dependent', False)
        use_score_dep2 = pred2_data.get('use_score_dependent', False)
        score_dep_params1 = pred1_data.get('score_dependent_params', None)
        score_dep_params2 = pred2_data.get('score_dependent_params', None)
        
        # Apply calibration consistently: multiply SIGMA by calibration factor
        # (Score-dependent calibration returns sigma multipliers directly)
        if use_score_dep1 and score_dep_params1:
            from MODELS.ngboostModel import get_score_dependent_calibration_factor
            temp_mean_orig1 = np.expm1(mu1_log)
            temp_mean_orig1 = np.asarray(temp_mean_orig1)
            temp_mean_orig1 = np.nan_to_num(temp_mean_orig1, nan=0.0, posinf=0.0, neginf=0.0)
            
            params1 = score_dep_params1.copy()
            very_high_threshold1 = params1.pop('very_high_threshold', None)
            very_high_factor1 = params1.pop('very_high_factor', None)
            
            cal_factors1 = get_score_dependent_calibration_factor(
                np.array([temp_mean_orig1]),
                base_factor=1.0,  # Use 1.0 as base since function returns absolute factors
                very_high_threshold=very_high_threshold1,
                very_high_factor=very_high_factor1,
                **params1
            )
            sigma_multiplier1 = float(cal_factors1[0]) if isinstance(cal_factors1, np.ndarray) else float(cal_factors1)
            sigma1_log = sigma1_log_uncalib * sigma_multiplier1
        else:
            # Base calibration: cal_factor applies to variance, so use sqrt for sigma
            sigma1_log = sigma1_log_uncalib * np.sqrt(cal_factor1)
        
        if use_score_dep2 and score_dep_params2:
            from MODELS.ngboostModel import get_score_dependent_calibration_factor
            temp_mean_orig2 = np.expm1(mu2_log)
            temp_mean_orig2 = np.asarray(temp_mean_orig2)
            temp_mean_orig2 = np.nan_to_num(temp_mean_orig2, nan=0.0, posinf=0.0, neginf=0.0)
            
            params2 = score_dep_params2.copy()
            very_high_threshold2 = params2.pop('very_high_threshold', None)
            very_high_factor2 = params2.pop('very_high_factor', None)
            
            cal_factors2 = get_score_dependent_calibration_factor(
                np.array([temp_mean_orig2]),
                base_factor=1.0,
                very_high_threshold=very_high_threshold2,
                very_high_factor=very_high_factor2,
                **params2
            )
            sigma_multiplier2 = float(cal_factors2[0]) if isinstance(cal_factors2, np.ndarray) else float(cal_factors2)
            sigma2_log = sigma2_log_uncalib * sigma_multiplier2
        else:
            sigma2_log = sigma2_log_uncalib * np.sqrt(cal_factor2)
        
        # Calculate probabilities in log space (CORRECT for log-normal distribution)
        p1_over = float(1 - norm.cdf(np.log1p(line1), loc=mu1_log, scale=sigma1_log))
        p1_under = 1.0 - p1_over
        p2_over = float(1 - norm.cdf(np.log1p(line2), loc=mu2_log, scale=sigma2_log))
        p2_under = 1.0 - p2_over
        
        # Look up worst-case odds for BOTH sides
        odds1_over = get_player_odds_from_csv(player1, line1, current_date_str, 'over')
        odds1_under = get_player_odds_from_csv(player1, line1, current_date_str, 'under')
        odds2_over = get_player_odds_from_csv(player2, line2, current_date_str, 'over')
        odds2_under = get_player_odds_from_csv(player2, line2, current_date_str, 'under')
        
        # Dynamic correlation adjustment (same for all combinations)
        team1 = player_teams[player1]
        team2 = player_teams[player2]
        opp1 = player_opponents[player1]
        opp2 = player_opponents[player2]
        
        # Check if same game
        if team1 == opp2 or team2 == opp1:
            corr_adjustment = 0.85  # Same game
            correlation = 0.30
        else:
            corr_adjustment = 0.98  # Different games
            correlation = 0.05
        
        # Try all 4 side combinations and find the best EV
        best_ev = float('-inf')
        best_config = None
        
        for side1, side2 in [('over', 'over'), ('over', 'under'), ('under', 'over'), ('under', 'under')]:
            # Get probabilities and odds for this combination
            p1 = p1_over if side1 == 'over' else p1_under
            p2 = p2_over if side2 == 'over' else p2_under
            odds1 = odds1_over if side1 == 'over' else odds1_under
            odds2 = odds2_over if side2 == 'over' else odds2_under
            
            # Calculate market probabilities from individual odds
            market_prob1 = impliedProb(odds1)
            market_prob2 = impliedProb(odds2)
            
            # Calculate fair parlay odds from individual odds (treating as independent)
            fair_parlay_odds = calculate_parlay_odds([odds1, odds2], correlation_adjustment=corr_adjustment)
            fair_parlay_prob = impliedProb(fair_parlay_odds)
            
            # Calculate model's combined probability
            p_both_raw = p1 * p2
            p_both = p_both_raw * corr_adjustment
            
            # Calculate EV using fixed payout (PrizePicks/Underdog pays 3x regardless)
            ev_percent = payout_multiple * p_both - 1  # EV per unit as decimal
            
            # Also calculate what the fair market probability would be
            market_prob = fair_parlay_prob
            
            # Store this configuration if it's the best so far
            if ev_percent > best_ev:
                best_ev = ev_percent
                best_config = {
                    'side1': side1,
                    'side2': side2,
                    'p1': p1,
                    'p2': p2,
                    'odds1': odds1,
                    'odds2': odds2,
                    'market_prob1': market_prob1,
                    'market_prob2': market_prob2,
                    'fair_parlay_prob': fair_parlay_prob,
                    'market_prob': market_prob,
                    'p_both_raw': p_both_raw,
                    'p_both': p_both,
                    'ev_percent': ev_percent
                }
        
        # Use the best configuration found
        if best_config is None:
            continue
        
        # Determine model sides based on prediction vs line (not best EV)
        model_side1 = 'over' if mu1 > line1 else 'under'
        model_side2 = 'over' if mu2 > line2 else 'under'
        
        # Use best EV configuration for actual bet
        side1 = best_config['side1']
        side2 = best_config['side2']
        p1 = best_config['p1']
        p2 = best_config['p2']
        odds1 = best_config['odds1']
        odds2 = best_config['odds2']
        market_prob1 = best_config['market_prob1']
        market_prob2 = best_config['market_prob2']
        fair_parlay_prob = best_config['fair_parlay_prob']
        p_both_raw = best_config['p_both_raw']
        p_both = best_config['p_both']
        ev_percent = best_config['ev_percent']
        
        # Edge calculations
        edge1 = p1 - market_prob1  # Individual leg edge
        edge2 = p2 - market_prob2  # Individual leg edge
        combined_edge = p_both - fair_parlay_prob  # Combined edge using fair parlay probability
        
        # Kelly criterion
        b = payout_multiple - 1.0
        kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b)
        
        # Recommendation based on edge threshold
        recommendation = 1 if (abs(line1 - mu1) > edge_threshold and abs(line2 - mu2) > edge_threshold) and (ev_percent > 0) else 0
        
        # Confidence intervals
        ci1_lower = max(0, mu1 - 1.96 * sigma1)
        ci1_upper = mu1 + 1.96 * sigma1
        ci2_lower = max(0, mu2 - 1.96 * sigma2)
        ci2_upper = mu2 + 1.96 * sigma2
        
        # Sigma flags
        sigma_flag1 = flag_sigma(sigma1)
        sigma_flag2 = flag_sigma(sigma2)
        
        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'LINE 1': line1,
            'LINE 2': line2,
            'ODDS 1': odds1,
            'ODDS 2': odds2,
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
            'EV%': round(ev_percent * 100, 2),
            'KELLY FULL': round(kelly_full, 3),
            'RECOMMENDATION': recommendation,
            'SIGMA 1': round(sigma1, 2),
            'SIGMA 2': round(sigma2, 2),
            'SIGMA FLAG 1': sigma_flag1,
            'SIGMA FLAG 2': sigma_flag2,
            'CI 1': f"({ci1_lower:.1f}, {ci1_upper:.1f})",
            'CI 2': f"({ci2_lower:.1f}, {ci2_upper:.1f})",
            'CORRELATION': round(correlation, 3),
            'SAME_GAME': 1 if (team1 == opp2 or team2 == opp1) else 0,
            'EXPECTED ROI': round(ev_percent * 100, 1)
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by EV
    results_df = results_df.sort_values('EV%', ascending=False)
    
    # Limit player appearances
    results_df = limit_player_appearances(results_df, max_appearances=max_player_appearances)
    
    # Return top N
    return results_df.head(top_n)

def clear_odds_cache():
    """Clear the odds cache. Useful when switching dates or reloading data."""
    global _odds_cache
    _odds_cache = {}

def get_player_odds_from_csv(player_name, line, current_date, side='over'):
    try:
        # Convert current_date to string if needed
        if isinstance(current_date, datetime):
            date_str = current_date.strftime('%Y%m%d')
        else:
            # Assume format is 'YYYY-MM-DD' or already 'YYYYMMDD'
            date_str = str(current_date).replace('-', '')
            if len(date_str) == 10:  # YYYY-MM-DD format
                date_str = date_str.replace('-', '')
        
        # Normalize side to 'over' or 'under'
        side_normalized = 'over' if str(side).upper().startswith('O') else 'under'
        line_float = float(line)
        
        # Load cache for this date (will load once and cache)
        odds_map = _load_odds_cache(date_str)
        
        # O(1) lookup using hashmap
        key = (str(player_name), line_float, side_normalized)
        best_odds = odds_map.get(key, -137)
        
        return int(best_odds)
    
    except Exception as e:
        # Silently return default on error
        return -137

def flag_sigma(s):
    """Categorize sigma values for volatility assessment"""
    if s <= 5.0:
        return 'Low'
    elif s <= 6.0:
        return 'Med'
    else:
        return 'High'

def limit_player_appearances(results_df, max_appearances=3):
    """Limit how many times each player appears"""
    player_counts = {}
    filtered_results = []
    
    for _, row in results_df.iterrows():
        p1 = row['NAME 1']
        p2 = row['NAME 2']
        
        count1 = player_counts.get(p1, 0)
        count2 = player_counts.get(p2, 0)
        
        if count1 < max_appearances and count2 < max_appearances:
            filtered_results.append(row)
            player_counts[p1] = count1 + 1
            player_counts[p2] = count2 + 1
    
    return pd.DataFrame(filtered_results)

def calculate3LegBets(data, bookmakers, model, features, current_date, 
                     edge_threshold=0.05, top_n=10, 
                     stake=100, max_player_appearances: int = 2):
    """
    Optimized 3-leg bet calculator using calibrated Normal distribution
    """
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
    
    # Pre-compute all predictions
    player_predictions = {}
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Pre-computing predictions for {len(available_players)} players...")
    current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
    
    for player in available_players:
        mapped_player = nameDict.get(player, player)
        
        pred_data = get_cached_prediction(
            mapped_player, data, model, features, current_date, 
            projectedStartingFive, mainStartingFive, teamStarPlayer
        )
        if pred_data is None:
            continue
        
        player_data = data[data['PLAYER_NAME'] == mapped_player]
        if player_data.empty:
            continue
        
        player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
        opp_team, _ = findOpp(mapped_player, player_data, current_date_str)
        if opp_team is None:
            continue
        
        player_bets = bookmakers[bookmakers['NAME'] == player]
        if player_bets.empty:
            continue
        
        player_predictions[player] = pred_data
        player_teams[player] = player_team
        player_opponents[player] = opp_team
        player_lines[player] = player_bets.iloc[0]
    
    available_players = [p for p in available_players 
                        if p in player_predictions and p in player_teams 
                        and p in player_opponents and p in player_lines]
    
    if len(available_players) < 3:
        print("Not enough players with valid predictions for 3-leg bets")
        return pd.DataFrame()
    
    print(f"Processing {len(available_players)} players...")
    
    # Generate only valid combinations (different teams and not all in same game)
    valid_combinations = []
    for p1, p2, p3 in combinations(available_players, 3):
        team1 = player_teams[p1]
        team2 = player_teams[p2]
        team3 = player_teams[p3]
        
        # Prevent ANY two players from being on the same team (all 3 must be different teams)
        unique_teams = set([team1, team2, team3])
        if len(unique_teams) < 3:
            continue  # Skip if any players share a team
        
        valid_combinations.append((p1, p2, p3))
    
    print(f"Generated {len(valid_combinations)} valid 3-leg combinations")
    
    # Constants
    from scipy.stats import norm
    payout_multiple = 6.0
    
    results = []
    
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
        
        mu1_raw = pred1_data['prediction']
        mu2_raw = pred2_data['prediction']
        mu3_raw = pred3_data['prediction']
        sigma1 = pred1_data['sigma']
        sigma2 = pred2_data['sigma']
        sigma3 = pred3_data['sigma']
        
        line1 = float(player_lines[player1]['LINE'])
        line2 = float(player_lines[player2]['LINE'])
        line3 = float(player_lines[player3]['LINE'])
        
        mu1 = mu1_raw
        mu2 = mu2_raw
        mu3 = mu3_raw
        
        # Get uncalibrated log-space parameters from cache
        mu1_log = pred1_data.get('mu_log', np.log1p(mu1_raw))
        sigma1_log_uncalib = pred1_data.get('sigma_log', sigma1 / (mu1_raw + 1))
        mu2_log = pred2_data.get('mu_log', np.log1p(mu2_raw))
        sigma2_log_uncalib = pred2_data.get('sigma_log', sigma2 / (mu2_raw + 1))
        mu3_log = pred3_data.get('mu_log', np.log1p(mu3_raw))
        sigma3_log_uncalib = pred3_data.get('sigma_log', sigma3 / (mu3_raw + 1))
        
        # Get calibration factors
        cal_factor1 = pred1_data.get('calibration_factor', 1.25)
        cal_factor2 = pred2_data.get('calibration_factor', 1.25)
        cal_factor3 = pred3_data.get('calibration_factor', 1.25)
        use_score_dep1 = pred1_data.get('use_score_dependent', False)
        use_score_dep2 = pred2_data.get('use_score_dependent', False)
        use_score_dep3 = pred3_data.get('use_score_dependent', False)
        score_dep_params1 = pred1_data.get('score_dependent_params', None)
        score_dep_params2 = pred2_data.get('score_dependent_params', None)
        score_dep_params3 = pred3_data.get('score_dependent_params', None)
        
        # Apply calibration consistently for all 3 players
        if use_score_dep1 and score_dep_params1:
            from MODELS.ngboostModel import get_score_dependent_calibration_factor
            temp_mean_orig1 = np.expm1(mu1_log)
            temp_mean_orig1 = np.asarray(temp_mean_orig1)
            temp_mean_orig1 = np.nan_to_num(temp_mean_orig1, nan=0.0, posinf=0.0, neginf=0.0)
            
            params1 = score_dep_params1.copy()
            very_high_threshold1 = params1.pop('very_high_threshold', None)
            very_high_factor1 = params1.pop('very_high_factor', None)
            
            cal_factors1 = get_score_dependent_calibration_factor(
                np.array([temp_mean_orig1]),
                base_factor=1.0,
                very_high_threshold=very_high_threshold1,
                very_high_factor=very_high_factor1,
                **params1
            )
            sigma_multiplier1 = float(cal_factors1[0]) if isinstance(cal_factors1, np.ndarray) else float(cal_factors1)
            sigma1_log = sigma1_log_uncalib * sigma_multiplier1
        else:
            sigma1_log = sigma1_log_uncalib * np.sqrt(cal_factor1)
        
        if use_score_dep2 and score_dep_params2:
            from MODELS.ngboostModel import get_score_dependent_calibration_factor
            temp_mean_orig2 = np.expm1(mu2_log)
            temp_mean_orig2 = np.asarray(temp_mean_orig2)
            temp_mean_orig2 = np.nan_to_num(temp_mean_orig2, nan=0.0, posinf=0.0, neginf=0.0)
            
            params2 = score_dep_params2.copy()
            very_high_threshold2 = params2.pop('very_high_threshold', None)
            very_high_factor2 = params2.pop('very_high_factor', None)
            
            cal_factors2 = get_score_dependent_calibration_factor(
                np.array([temp_mean_orig2]),
                base_factor=1.0,
                very_high_threshold=very_high_threshold2,
                very_high_factor=very_high_factor2,
                **params2
            )
            sigma_multiplier2 = float(cal_factors2[0]) if isinstance(cal_factors2, np.ndarray) else float(cal_factors2)
            sigma2_log = sigma2_log_uncalib * sigma_multiplier2
        else:
            sigma2_log = sigma2_log_uncalib * np.sqrt(cal_factor2)
        
        if use_score_dep3 and score_dep_params3:
            from MODELS.ngboostModel import get_score_dependent_calibration_factor
            temp_mean_orig3 = np.expm1(mu3_log)
            temp_mean_orig3 = np.asarray(temp_mean_orig3)
            temp_mean_orig3 = np.nan_to_num(temp_mean_orig3, nan=0.0, posinf=0.0, neginf=0.0)
            
            params3 = score_dep_params3.copy()
            very_high_threshold3 = params3.pop('very_high_threshold', None)
            very_high_factor3 = params3.pop('very_high_factor', None)
            
            cal_factors3 = get_score_dependent_calibration_factor(
                np.array([temp_mean_orig3]),
                base_factor=1.0,
                very_high_threshold=very_high_threshold3,
                very_high_factor=very_high_factor3,
                **params3
            )
            sigma_multiplier3 = float(cal_factors3[0]) if isinstance(cal_factors3, np.ndarray) else float(cal_factors3)
            sigma3_log = sigma3_log_uncalib * sigma_multiplier3
        else:
            sigma3_log = sigma3_log_uncalib * np.sqrt(cal_factor3)
        
        # Calculate probabilities in log space
        p1_over = float(1 - norm.cdf(np.log1p(line1), loc=mu1_log, scale=sigma1_log))
        p1_under = 1.0 - p1_over
        p2_over = float(1 - norm.cdf(np.log1p(line2), loc=mu2_log, scale=sigma2_log))
        p2_under = 1.0 - p2_over
        p3_over = float(1 - norm.cdf(np.log1p(line3), loc=mu3_log, scale=sigma3_log))
        p3_under = 1.0 - p3_over
        
        # Look up worst-case odds for BOTH sides
        odds1_over = get_player_odds_from_csv(player1, line1, current_date_str, 'over')
        odds1_under = get_player_odds_from_csv(player1, line1, current_date_str, 'under')
        odds2_over = get_player_odds_from_csv(player2, line2, current_date_str, 'over')
        odds2_under = get_player_odds_from_csv(player2, line2, current_date_str, 'under')
        odds3_over = get_player_odds_from_csv(player3, line3, current_date_str, 'over')
        odds3_under = get_player_odds_from_csv(player3, line3, current_date_str, 'under')
        
        # Dynamic correlation adjustment
        team1 = player_teams[player1]
        team2 = player_teams[player2]
        team3 = player_teams[player3]
        opp1 = player_opponents[player1]
        opp2 = player_opponents[player2]
        opp3 = player_opponents[player3]
        
        # Count pairs in same game
        same_game_count = 0
        if team1 == opp2 or team2 == opp1:
            same_game_count += 1
        if team1 == opp3 or team3 == opp1:
            same_game_count += 1
        if team2 == opp3 or team3 == opp2:
            same_game_count += 1
        
        if same_game_count >= 2:
            corr_adjustment = 0.50
            correlation = 0.50
        elif same_game_count == 1:
            corr_adjustment = 0.70
            correlation = 0.30
        else:
            corr_adjustment = 0.90
            correlation = 0.10
        
        # Try all 8 side combinations and find the best EV
        best_ev = float('-inf')
        best_config = None
        
        for side1, side2, side3 in [
            ('over', 'over', 'over'), ('over', 'over', 'under'),
            ('over', 'under', 'over'), ('over', 'under', 'under'),
            ('under', 'over', 'over'), ('under', 'over', 'under'),
            ('under', 'under', 'over'), ('under', 'under', 'under')
        ]:
            p1 = p1_over if side1 == 'over' else p1_under
            p2 = p2_over if side2 == 'over' else p2_under
            p3 = p3_over if side3 == 'over' else p3_under
            odds1 = odds1_over if side1 == 'over' else odds1_under
            odds2 = odds2_over if side2 == 'over' else odds2_under
            odds3 = odds3_over if side3 == 'over' else odds3_under
            
            market_prob1 = impliedProb(odds1)
            market_prob2 = impliedProb(odds2)
            market_prob3 = impliedProb(odds3)
            
            fair_parlay_odds = calculate_parlay_odds([odds1, odds2, odds3], correlation_adjustment=corr_adjustment)
            fair_parlay_prob = impliedProb(fair_parlay_odds)
            
            p_all_three_raw = p1 * p2 * p3
            p_all_three = p_all_three_raw * corr_adjustment
            
            ev_percent = payout_multiple * p_all_three - 1
            
            if ev_percent > best_ev:
                best_ev = ev_percent
                best_config = {
                    'side1': side1,
                    'side2': side2,
                    'side3': side3,
                    'p1': p1,
                    'p2': p2,
                    'p3': p3,
                    'odds1': odds1,
                    'odds2': odds2,
                    'odds3': odds3,
                    'market_prob1': market_prob1,
                    'market_prob2': market_prob2,
                    'market_prob3': market_prob3,
                    'fair_parlay_prob': fair_parlay_prob,
                    'p_all_three_raw': p_all_three_raw,
                    'p_all_three': p_all_three,
                    'ev_percent': ev_percent
                }
        
        if best_config is None:
            continue
        
        model_side1 = 'over' if mu1 > line1 else 'under'
        model_side2 = 'over' if mu2 > line2 else 'under'
        model_side3 = 'over' if mu3 > line3 else 'under'
        
        side1 = best_config['side1']
        side2 = best_config['side2']
        side3 = best_config['side3']
        p1 = best_config['p1']
        p2 = best_config['p2']
        p3 = best_config['p3']
        odds1 = best_config['odds1']
        odds2 = best_config['odds2']
        odds3 = best_config['odds3']
        market_prob1 = best_config['market_prob1']
        market_prob2 = best_config['market_prob2']
        market_prob3 = best_config['market_prob3']
        fair_parlay_prob = best_config['fair_parlay_prob']
        p_all_three_raw = best_config['p_all_three_raw']
        p_all_three = best_config['p_all_three']
        ev_percent = best_config['ev_percent']
        
        edge1 = p1 - market_prob1
        edge2 = p2 - market_prob2
        edge3 = p3 - market_prob3
        combined_edge = p_all_three - fair_parlay_prob
        
        b = payout_multiple - 1.0
        kelly_full = max(0.0, (b * p_all_three - (1 - p_all_three)) / b)
        
        recommendation = 1 if (abs(line1 - mu1) > edge_threshold and abs(line2 - mu2) > edge_threshold and abs(line3 - mu3) > edge_threshold) and (ev_percent > 0) else 0
        
        ci1_lower = max(0, mu1 - 1.96 * sigma1)
        ci1_upper = mu1 + 1.96 * sigma1
        ci2_lower = max(0, mu2 - 1.96 * sigma2)
        ci2_upper = mu2 + 1.96 * sigma2
        ci3_lower = max(0, mu3 - 1.96 * sigma3)
        ci3_upper = mu3 + 1.96 * sigma3
        
        sigma_flag1 = flag_sigma(sigma1)
        sigma_flag2 = flag_sigma(sigma2)
        sigma_flag3 = flag_sigma(sigma3)
        
        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'NAME 3': mapped_p3,
            'LINE 1': line1,
            'LINE 2': line2,
            'LINE 3': line3,
            'ODDS 1': odds1,
            'ODDS 2': odds2,
            'ODDS 3': odds3,
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
            'EV%': round(ev_percent * 100, 2),
            'KELLY FULL': round(kelly_full, 3),
            'RECOMMENDATION': recommendation,
            'SIGMA 1': round(sigma1, 2),
            'SIGMA 2': round(sigma2, 2),
            'SIGMA 3': round(sigma3, 2),
            'SIGMA FLAG 1': sigma_flag1,
            'SIGMA FLAG 2': sigma_flag2,
            'SIGMA FLAG 3': sigma_flag3,
            'CI 1': f"({ci1_lower:.1f}, {ci1_upper:.1f})",
            'CI 2': f"({ci2_lower:.1f}, {ci2_upper:.1f})",
            'CI 3': f"({ci3_lower:.1f}, {ci3_upper:.1f})",
            'CORRELATION': round(correlation, 3),
            'SAME_GAME_PAIRS': same_game_count,
            'EXPECTED ROI': round(ev_percent * 100, 1)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('EV%', ascending=False)
    results_df = limit_player_appearances_3leg(results_df, max_appearances=max_player_appearances)
    
    return results_df.head(top_n)


def limit_player_appearances_3leg(results_df, max_appearances=3):
    """Limit how many times each player appears in 3-leg bets"""
    player_counts = {}
    filtered_results = []
    
    for _, row in results_df.iterrows():
        p1 = row['NAME 1']
        p2 = row['NAME 2']
        p3 = row['NAME 3']
        
        count1 = player_counts.get(p1, 0)
        count2 = player_counts.get(p2, 0)
        count3 = player_counts.get(p3, 0)
        
        if (count1 < max_appearances and 
            count2 < max_appearances and 
            count3 < max_appearances):
            filtered_results.append(row)
            player_counts[p1] = count1 + 1
            player_counts[p2] = count2 + 1
            player_counts[p3] = count3 + 1
    
    return pd.DataFrame(filtered_results)