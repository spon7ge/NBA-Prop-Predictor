import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from src.utils.team_info import *
from src.utils.helper_functions import *
from itertools import combinations
from src.utils.team_info import nameDict
from scipy.stats import lognorm
from src.pipeline.pipeline_pts import build_ngboost_points_features


def convert_to_et(utc_time):
    utc_dt = datetime.fromisoformat(utc_time.replace('Z', '+00:00'))
    et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
    return et_dt.strftime('%Y-%m-%d')  

def impliedProb(odds, vig=0.05):
    if odds > 0:
        prob = 100 / (odds + 100)
    else:
        prob = abs(odds) / (abs(odds) + 100)
    return prob / (1.0 + vig / 2)

def american_to_decimal(odds):
    return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

def decimal_to_american(decimal_odds):
    if decimal_odds == 2.0:
        return +100
    elif decimal_odds > 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))

def calculate_parlay_odds(odds_list, correlation_adjustment=1.0):
    if len(odds_list) == 0:
        return -137
    
    decimal_odds = [american_to_decimal(odds) for odds in odds_list]
    combined_decimal = 1.0
    for dec in decimal_odds:
        combined_decimal *= dec
    
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

def fairProb(bookmakersData, name, line, category, over_under, fixed_buffer=0.035):
    df = bookmakersData[
        (bookmakersData['NAME'] == name) &
        (bookmakersData['CATEGORY'] == category)
    ]
    
    grouped_df = df.groupby('LINE').agg({
        'BOOKMAKER': list,
        'ODDS': list,
        'OVER/UNDER': list
    }).reset_index() 

    res = []
    for idx, row in grouped_df.iterrows():
        if line == row['LINE']:
            for odds, OU in zip(row['ODDS'], row['OVER/UNDER']):
                if OU == over_under:
                    res.append(round(impliedProb(odds), 2))
    
    adjusted_probs = [prob - fixed_buffer for prob in res]

    if len(adjusted_probs) == 0:
        raise ValueError("No valid probabilities found for the given line and over/under condition.")
    
    fair_odds = sum(adjusted_probs) / len(adjusted_probs)
    
    if fair_odds == 0:
        raise ValueError("Calculated fair probability is zero, cannot convert to odds.")
    
    odds_to_decimal = 1 / fair_odds
    
    if odds_to_decimal == 2.0:
        return +100
    elif odds_to_decimal > 2.0:
        return round((odds_to_decimal - 1) * 100)
    else:
        return round(-100 / (odds_to_decimal - 1))

_odds_cache = {}

def _load_odds_cache(date_str):
    if date_str in _odds_cache:
        return _odds_cache[date_str]
    
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        player_lines_dir = project_root / 'data' / 'raw' / 'player_lines'
        
        pattern = f'NBA_US_{date_str}*.csv'
        matching_files = list(player_lines_dir.glob(pattern))
        
        if not matching_files:
            _odds_cache[date_str] = {}
            return {}
        
        csv_path = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        df = pd.read_csv(str(csv_path))
        
        if 'CATEGORY' in df.columns:
            df = df[df['CATEGORY'] == 'player_points']
        
        if df.empty:
            _odds_cache[date_str] = {}
            return {}
        
        odds_map = {}
        df['SIDE_NORMALIZED'] = df['OVER/UNDER'].str.upper().str.strip().apply(
            lambda x: 'over' if str(x).startswith('O') else 'under'
        )
        
        for (name, line, side_norm), group in df.groupby(['NAME', 'LINE', 'SIDE_NORMALIZED']):
            line_float = float(line)
            best_odds = int(group['ODDS'].max())
            key = (str(name), line_float, side_norm)
            if key in odds_map:
                odds_map[key] = max(odds_map[key], best_odds)
            else:
                odds_map[key] = best_odds  
        
        _odds_cache[date_str] = odds_map
        return odds_map
    
    except Exception:
        _odds_cache[date_str] = {}
        return {}

def clear_odds_cache():
    global _odds_cache
    _odds_cache = {}

def get_player_odds_from_csv(player_name, line, current_date, side='over'):
    try:
        if isinstance(current_date, datetime):
            date_str = current_date.strftime('%Y%m%d')
        else:
            date_str = str(current_date).replace('-', '')
            if len(date_str) == 10:
                date_str = date_str.replace('-', '')
        
        side_normalized = 'over' if str(side).upper().startswith('O') else 'under'
        line_float = float(line)
        
        odds_map = _load_odds_cache(date_str)
        key = (str(player_name), line_float, side_normalized)
        best_odds = odds_map.get(key, -137)
        
        return int(best_odds)
    
    except Exception:
        return -137

def flag_sigma(s):
    if s <= 5.0:
        return 'Low'
    elif s <= 6.0:
        return 'Med'
    else:
        return 'High'

def prob_lognorm(line, mu, sigma):
    """
    Calculate P(X > line) for log-normal distribution.
    mu and sigma are in original space (not log space).
    """
    if mu <= 0 or sigma <= 0:
        return 0.0
    
    mu_log = np.log(mu)
    cv = sigma / mu
    sigma_log = np.sqrt(np.log(1 + cv**2))
    
    scale = np.exp(mu_log)
    dist = lognorm(s=sigma_log, scale=scale)
    
    return float(1 - dist.cdf(line + 0.5))

def get_calibrated_distribution_params(engine, player_name, data, date, 
                                      predicted_minutes=None, predicted_usage=None,
                                      projectedStartingFive=None, mainStartingFive=None,
                                      teamStarPlayer=None, league_df=None, findOpp=None):
    """
    Get calibrated distribution parameters (mean_log, std_log) for a player.
    Returns (mean_log, std_log, median_pred) or None if failed.
    """
    if engine.ngboost_model_wrapper is None:
        return None
    
    mean_model = engine.ngboost_model_wrapper['mean_model']
    features_list = engine.ngboost_model_wrapper['features']
    variance_calibration = engine.ngboost_model_wrapper.get('variance_calibration')
    bins = engine.ngboost_model_wrapper.get('bins')
    
    # Build features
    feature_dict = build_ngboost_points_features(
        player_name=player_name,
        data=data,
        current_date=date,
        projectedStartingFive=projectedStartingFive,
        mainStartingFive=mainStartingFive,
        teamStarPlayer=teamStarPlayer,
        league_df=league_df,
        findOpp=findOpp,
        predicted_minutes=predicted_minutes,
        predicted_usage=predicted_usage
    )
    
    if feature_dict is None:
        return None
    
    feature_vector = [feature_dict.get(f, 0.0) for f in features_list]
    feature_df = pd.DataFrame([feature_vector], columns=features_list)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    
    try:
        # Get distribution parameters from model
        X = feature_df[features_list].fillna(feature_df[features_list].median())
        dist = mean_model.pred_dist(X)
        
        mean_log = float(dist.loc[0])
        scale_log = float(dist.scale[0])
        var_log = scale_log ** 2
        
        # Get median prediction to determine which bin
        # Use the median from the model (calibrations don't change median, only variance)
        median_pred = np.expm1(mean_log)
        median_pred = max(0.0, median_pred)
        
        # Apply calibrations based on bins
        if variance_calibration and bins and len(bins) >= 3:
            low_bin, medium_bin, high_bin = bins[0], bins[1], bins[2]
            if (median_pred >= low_bin[0]) and (median_pred < low_bin[1]):
                var_log *= variance_calibration.get('low', 1.0)
            elif (median_pred >= medium_bin[0]) and (median_pred < medium_bin[1]):
                var_log *= variance_calibration.get('medium', 1.0)
            elif (median_pred >= high_bin[0]) and (median_pred < high_bin[1]):
                var_log *= variance_calibration.get('high', 1.0)
        elif variance_calibration:
            # Fallback to hardcoded bins
            if median_pred < 7:
                var_log *= variance_calibration.get('low', 1.0)
            elif median_pred < 15:
                var_log *= variance_calibration.get('medium', 1.0)
            else:
                var_log *= variance_calibration.get('high', 1.0)
        
        std_log = np.sqrt(var_log)
        
        return mean_log, std_log, median_pred
        
    except Exception as e:
        print(f"Error getting distribution params for {player_name}: {e}")
        return None

def prob_lognorm_calibrated(line, mean_log, std_log, side='over'):
    if mean_log <= 0 or std_log <= 0:
        return 0.0
    
    # Create log-normal distribution
    # lognorm(s=std_log, scale=exp(mean_log))
    distribution = lognorm(s=std_log, scale=np.exp(mean_log))
    
    # Calculate probability
    # For over: P(X > line)
    # For under: P(X < line)
    if side.lower().startswith('o'):
        prob = float(1 - distribution.cdf(line + 0.5))
    else:
        prob = float(distribution.cdf(line - 0.5))
    
    return max(0.0, min(1.0, prob))

def limit_player_appearances(results_df, max_appearances=3):
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

def limit_player_appearances_3leg(results_df, max_appearances=3):
    player_counts = {}
    filtered_results = []
    
    for _, row in results_df.iterrows():
        p1 = row['NAME 1']
        p2 = row['NAME 2']
        p3 = row['NAME 3']
        count1 = player_counts.get(p1, 0)
        count2 = player_counts.get(p2, 0)
        count3 = player_counts.get(p3, 0)
        
        if count1 < max_appearances and count2 < max_appearances and count3 < max_appearances:
            filtered_results.append(row)
            player_counts[p1] = count1 + 1
            player_counts[p2] = count2 + 1
            player_counts[p3] = count3 + 1
    
    return pd.DataFrame(filtered_results)

def calculate2LegBets(data, bookmakers, engine, current_date, 
                             top_n=10, max_player_appearances=1,
                             projectedStartingFive=None, mainStartingFive=None, 
                             teamStarPlayer=None, league_df=None, findOpp=None):
    
    # Filter for player points category
    bookmakers = bookmakers[bookmakers['CATEGORY'] == 'player_points']
    if bookmakers.empty:
        print("No bets found for player_points")
        return pd.DataFrame()

    available_players = bookmakers['NAME'].unique()
    if len(available_players) < 2:
        print("Not enough players for 2-leg bets")
        return pd.DataFrame()

    # Store predictions, distributions, and metadata for each player
    player_predictions = {}
    player_distributions = {}  # Store (mean_log, std_log) for each player
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Computing predictions for {len(available_players)} players...")
    current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
    
    # Get predictions and distributions for all available players
    for player in available_players:
        mapped_player = nameDict.get(player, player)
        
        try:
            result = engine.project_player(
                player_name=mapped_player,
                data=data,
                date=current_date_str,
                projectedStartingFive=projectedStartingFive,
                mainStartingFive=mainStartingFive,
                teamStarPlayer=teamStarPlayer,
                league_df=league_df,
                findOpp=findOpp
            )
            
            if result is None:
                continue
                
            # Get distribution parameters with calibrations
            dist_params = get_calibrated_distribution_params(
                engine=engine,
                player_name=mapped_player,
                data=data,
                date=current_date_str,
                predicted_minutes=result['predicted_minutes'],
                predicted_usage=result['predicted_usage'],
                projectedStartingFive=projectedStartingFive,
                mainStartingFive=mainStartingFive,
                teamStarPlayer=teamStarPlayer,
                league_df=league_df,
                findOpp=findOpp
            )
            
            if dist_params is None:
                continue
            
            mean_log, std_log, _ = dist_params
            
            # Use the prediction from engine.project_player() which uses predict_mean
            # This ensures consistency with what predict_mean actually returns
            median_pred = result['predicted_points']
                
            # Get player's team and opponent
            player_data = data[data['PLAYER_NAME'] == mapped_player]
            if player_data.empty:
                continue
                
            player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
            opp_team, _ = findOpp(mapped_player, player_data, current_date_str)
            if opp_team is None:
                continue
                
            # Get line from bookmakers
            player_bets = bookmakers[bookmakers['NAME'] == player]
            if player_bets.empty:
                continue
                
            # Store everything - use result['predicted_points'] to match what predict_mean returns
            player_predictions[player] = result['predicted_points']
            player_distributions[player] = (mean_log, std_log)
            player_teams[player] = player_team
            player_opponents[player] = opp_team
            player_lines[player] = float(player_bets.iloc[0]['LINE'])
            
        except Exception as e:
            print(f"Error processing {player}: {e}")
            continue
    
    # Filter to players with valid data
    available_players = [p for p in available_players 
                        if p in player_predictions and p in player_distributions
                        and p in player_teams and p in player_opponents 
                        and p in player_lines]
    
    if len(available_players) < 2:
        print("Not enough players with valid predictions for 2-leg bets")
        return pd.DataFrame()
    
    print(f"Found {len(available_players)} valid players")
    
    # Generate valid player combinations
    valid_combinations = []
    for p1, p2 in combinations(available_players, 2):
        team1 = player_teams[p1]
        team2 = player_teams[p2]
        opp1 = player_opponents[p1]
        opp2 = player_opponents[p2]
        
        # Ensure players aren't on same team or opposing teams
        if team1 != team2 and team1 != opp2 and team2 != opp1:
            valid_combinations.append((p1, p2))
    
    print(f"Generated {len(valid_combinations)} valid 2-leg combinations")
    
    # Build results
    results = []
    for player1, player2 in valid_combinations:
        mapped_p1 = nameDict.get(player1, player1)
        mapped_p2 = nameDict.get(player2, player2)
        
        prediction1 = player_predictions[player1]
        prediction2 = player_predictions[player2]
        
        mean_log1, std_log1 = player_distributions[player1]
        mean_log2, std_log2 = player_distributions[player2]
        
        line1 = player_lines[player1]
        line2 = player_lines[player2]
        
        # Determine side based on prediction vs line
        side1 = 'over' if prediction1 > line1 else 'under'
        side2 = 'over' if prediction2 > line2 else 'under'
        
        # Calculate probabilities using log-normal distribution
        model_prob1 = prob_lognorm_calibrated(line1, mean_log1, std_log1, side1)
        model_prob2 = prob_lognorm_calibrated(line2, mean_log2, std_log2, side2)
        
        # Calculate edge
        if prediction1 > line1:
            edge1 = prediction1 - line1
        else:
            edge1 = line1 - prediction1

        if prediction2 > line2:
            edge2 = prediction2 - line2
        else:
            edge2 = line2 - prediction2

        # Get odds and calculate implied probability
        odds1 = get_player_odds_from_csv(player1, line1, current_date, side1)
        odds2 = get_player_odds_from_csv(player2, line2, current_date, side2)
        implied_prob1 = round(impliedProb(odds1), 2)
        implied_prob2 = round(impliedProb(odds2), 2)
        
        # Calculate parlay probability using model probabilities
        parlay_prob = model_prob1 * model_prob2
        
        # Calculate parlay odds from individual bookmaker odds
        parlay_odds = calculate_parlay_odds([odds1, odds2])
        
        # Convert parlay odds to decimal for EV calculation
        parlay_decimal = american_to_decimal(parlay_odds)
        
        # Calculate Expected Value using bookmaker odds
        # EV = (probability of winning * payout) - (probability of losing * stake)
        # For a $1 bet: EV = (parlay_prob * (parlay_decimal - 1)) - ((1 - parlay_prob) * 1)
        # Simplified: EV = parlay_prob * parlay_decimal - 1
        ev = (parlay_prob * parlay_decimal) - 1
        
        # Also calculate edge (model prob vs implied prob)
        parlay_implied_prob = implied_prob1 * implied_prob2
        parlay_edge = parlay_prob - parlay_implied_prob
        
        # Calculate EV percentage (return on investment)
        ev_percent = ev * 100
        
        # Calculate Kelly fraction (quarter Kelly = 0.25)
        # Kelly = (probability * payout - 1) / (payout - 1)
        # For parlay: kelly = (parlay_prob * parlay_decimal - 1) / (parlay_decimal - 1)
        if parlay_decimal > 1:
            kelly_full = (parlay_prob * parlay_decimal - 1) / (parlay_decimal - 1)
            kelly_quarter = max(0, kelly_full * 0.25)  # Quarter Kelly
        else:
            kelly_quarter = 0

        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'LINE 1': line1,
            'LINE 2': line2,
            'SIDE 1': side1,
            'SIDE 2': side2,
            'PREDICTION 1': round(prediction1, 2),
            'PREDICTION 2': round(prediction2, 2),
            'MODEL_PROB 1': round(model_prob1, 3),
            'MODEL_PROB 2': round(model_prob2, 3),
            'IMPLIED_PROB 1': implied_prob1,
            'IMPLIED_PROB 2': implied_prob2,
            'PARLAY_PROB': round(parlay_prob, 3),
            'PARLAY_IMPLIED_PROB': round(parlay_implied_prob, 3),
            'PARLAY_EDGE': round(parlay_edge, 3),
            'EDGE 1': round(edge1, 2),
            'EDGE 2': round(edge2, 2),
            'ODDS 1': odds1,
            'ODDS 2': odds2,
            'PARLAY_ODDS': parlay_odds,
            'PARLAY_DECIMAL': round(parlay_decimal, 3),
            'EV': round(ev, 4),
            'EV_PERCENT': round(ev_percent, 2),
            'KELLY_QUARTER': round(kelly_quarter, 4),
        })
    
    # Convert to DataFrame and sort by EV (descending)
    results_df = pd.DataFrame(results)
    results_df['TOTAL_EDGE'] = results_df['EDGE 1'] + results_df['EDGE 2']
    results_df = results_df.sort_values('EV', ascending=False)
    
    # Limit player appearances
    results_df = limit_player_appearances(results_df, max_appearances=max_player_appearances)
    
    return results_df.head(top_n)

def calculate3LegBets(data, bookmakers, engine, current_date, 
                             top_n=10, max_player_appearances=1,
                             projectedStartingFive=None, mainStartingFive=None, 
                             teamStarPlayer=None, league_df=None, findOpp=None):
    # Filter for player points category
    bookmakers = bookmakers[bookmakers['CATEGORY'] == 'player_points']
    if bookmakers.empty:
        print("No bets found for player_points")
        return pd.DataFrame()

    available_players = bookmakers['NAME'].unique()
    if len(available_players) < 3:
        print("Not enough players for 3-leg bets")
        return pd.DataFrame()

    # Store predictions, distributions, and metadata for each player
    player_predictions = {}
    player_distributions = {}  # Store (mean_log, std_log) for each player
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Computing predictions for {len(available_players)} players...")
    current_date_str = pd.to_datetime(current_date).strftime('%Y-%m-%d')
    
    # Get predictions and distributions for all available players
    for player in available_players:
        mapped_player = nameDict.get(player, player)
        
        try:
            result = engine.project_player(
                player_name=mapped_player,
                data=data,
                date=current_date_str,
                projectedStartingFive=projectedStartingFive,
                mainStartingFive=mainStartingFive,
                teamStarPlayer=teamStarPlayer,
                league_df=league_df,
                findOpp=findOpp
            )
            
            if result is None:
                continue
                
            # Get distribution parameters with calibrations
            dist_params = get_calibrated_distribution_params(
                engine=engine,
                player_name=mapped_player,
                data=data,
                date=current_date_str,
                predicted_minutes=result['predicted_minutes'],
                predicted_usage=result['predicted_usage'],
                projectedStartingFive=projectedStartingFive,
                mainStartingFive=mainStartingFive,
                teamStarPlayer=teamStarPlayer,
                league_df=league_df,
                findOpp=findOpp
            )
            
            if dist_params is None:
                continue
            
            mean_log, std_log, median_pred = dist_params
                
            # Get player's team and opponent
            player_data = data[data['PLAYER_NAME'] == mapped_player]
            if player_data.empty:
                continue
                
            player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
            opp_team, _ = findOpp(mapped_player, player_data, current_date_str)
            if opp_team is None:
                continue
                
            # Get line from bookmakers
            player_bets = bookmakers[bookmakers['NAME'] == player]
            if player_bets.empty:
                continue
                
            # Store everything
            player_predictions[player] = median_pred
            player_distributions[player] = (mean_log, std_log)
            player_teams[player] = player_team
            player_opponents[player] = opp_team
            player_lines[player] = float(player_bets.iloc[0]['LINE'])
            
        except Exception as e:
            print(f"Error processing {player}: {e}")
            continue
    
    # Filter to players with valid data
    available_players = [p for p in available_players 
                        if p in player_predictions and p in player_distributions
                        and p in player_teams and p in player_opponents 
                        and p in player_lines]
    
    if len(available_players) < 3:
        print("Not enough players with valid predictions for 3-leg bets")
        return pd.DataFrame()
    
    print(f"Found {len(available_players)} valid players")
    
    # Generate valid player combinations (3 players)
    valid_combinations = []
    for p1, p2, p3 in combinations(available_players, 3):
        team1 = player_teams[p1]
        team2 = player_teams[p2]
        team3 = player_teams[p3]
        opp1 = player_opponents[p1]
        opp2 = player_opponents[p2]
        opp3 = player_opponents[p3]
        
        # Ensure no two players are on same team or opposing teams
        teams = {team1, team2, team3}
        opps = {opp1, opp2, opp3}
        
        # Check all pairs
        if (team1 != team2 and team1 != team3 and team2 != team3 and
            team1 not in {opp2, opp3} and team2 not in {opp1, opp3} and 
            team3 not in {opp1, opp2}):
            valid_combinations.append((p1, p2, p3))
    
    print(f"Generated {len(valid_combinations)} valid 3-leg combinations")
    
    # Build results
    results = []
    for player1, player2, player3 in valid_combinations:
        mapped_p1 = nameDict.get(player1, player1)
        mapped_p2 = nameDict.get(player2, player2)
        mapped_p3 = nameDict.get(player3, player3)
        
        prediction1 = player_predictions[player1]
        prediction2 = player_predictions[player2]
        prediction3 = player_predictions[player3]
        
        mean_log1, std_log1 = player_distributions[player1]
        mean_log2, std_log2 = player_distributions[player2]
        mean_log3, std_log3 = player_distributions[player3]
        
        line1 = player_lines[player1]
        line2 = player_lines[player2]
        line3 = player_lines[player3]
        
        # Determine side based on prediction vs line
        side1 = 'over' if prediction1 > line1 else 'under'
        side2 = 'over' if prediction2 > line2 else 'under'
        side3 = 'over' if prediction3 > line3 else 'under'
        
        # Calculate probabilities using log-normal distribution
        model_prob1 = prob_lognorm_calibrated(line1, mean_log1, std_log1, side1)
        model_prob2 = prob_lognorm_calibrated(line2, mean_log2, std_log2, side2)
        model_prob3 = prob_lognorm_calibrated(line3, mean_log3, std_log3, side3)
        
        # Calculate edge
        if prediction1 > line1:
            edge1 = prediction1 - line1
        else:
            edge1 = line1 - prediction1

        if prediction2 > line2:
            edge2 = prediction2 - line2
        else:
            edge2 = line2 - prediction2

        if prediction3 > line3:
            edge3 = prediction3 - line3
        else:
            edge3 = line3 - prediction3

        # Get odds and calculate implied probability
        odds1 = get_player_odds_from_csv(player1, line1, current_date, side1)
        odds2 = get_player_odds_from_csv(player2, line2, current_date, side2)
        odds3 = get_player_odds_from_csv(player3, line3, current_date, side3)
        implied_prob1 = round(impliedProb(odds1), 2)
        implied_prob2 = round(impliedProb(odds2), 2)
        implied_prob3 = round(impliedProb(odds3), 2)
        
        # Calculate parlay probability using model probabilities
        parlay_prob = model_prob1 * model_prob2 * model_prob3
        
        # Calculate parlay odds from individual bookmaker odds
        parlay_odds = calculate_parlay_odds([odds1, odds2, odds3])
        
        # Convert parlay odds to decimal for EV calculation
        parlay_decimal = american_to_decimal(parlay_odds)
        
        # Calculate Expected Value using bookmaker odds
        # EV = (probability of winning * payout) - (probability of losing * stake)
        # For a $1 bet: EV = (parlay_prob * (parlay_decimal - 1)) - ((1 - parlay_prob) * 1)
        # Simplified: EV = parlay_prob * parlay_decimal - 1
        ev = (parlay_prob * parlay_decimal) - 1
        
        # Also calculate edge (model prob vs implied prob)
        parlay_implied_prob = implied_prob1 * implied_prob2 * implied_prob3
        parlay_edge = parlay_prob - parlay_implied_prob
        
        # Calculate EV percentage (return on investment)
        ev_percent = ev * 100
        
        # Calculate Kelly fraction (quarter Kelly = 0.25)
        # Kelly = (probability * payout - 1) / (payout - 1)
        # For parlay: kelly = (parlay_prob * parlay_decimal - 1) / (parlay_decimal - 1)
        if parlay_decimal > 1:
            kelly_full = (parlay_prob * parlay_decimal - 1) / (parlay_decimal - 1)
            kelly_quarter = max(0, kelly_full * 0.25)  # Quarter Kelly
        else:
            kelly_quarter = 0

        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'NAME 3': mapped_p3,
            'LINE 1': line1,
            'LINE 2': line2,
            'LINE 3': line3,
            'SIDE 1': side1,
            'SIDE 2': side2,
            'SIDE 3': side3,
            'PREDICTION 1': round(prediction1, 2),
            'PREDICTION 2': round(prediction2, 2),
            'PREDICTION 3': round(prediction3, 2),
            'MODEL_PROB 1': round(model_prob1, 3),
            'MODEL_PROB 2': round(model_prob2, 3),
            'MODEL_PROB 3': round(model_prob3, 3),
            'IMPLIED_PROB 1': implied_prob1,
            'IMPLIED_PROB 2': implied_prob2,
            'IMPLIED_PROB 3': implied_prob3,
            'PARLAY_PROB': round(parlay_prob, 3),
            'PARLAY_IMPLIED_PROB': round(parlay_implied_prob, 3),
            'PARLAY_EDGE': round(parlay_edge, 3),
            'EDGE 1': round(edge1, 2),
            'EDGE 2': round(edge2, 2),
            'EDGE 3': round(edge3, 2),
            'ODDS 1': odds1,
            'ODDS 2': odds2,
            'ODDS 3': odds3,
            'PARLAY_ODDS': parlay_odds,
            'PARLAY_DECIMAL': round(parlay_decimal, 3),
            'EV': round(ev, 4),
            'EV_PERCENT': round(ev_percent, 2),
            'KELLY_QUARTER': round(kelly_quarter, 4),
        })
    
    # Convert to DataFrame and sort by EV (descending)
    results_df = pd.DataFrame(results)
    results_df['TOTAL_EDGE'] = results_df['EDGE 1'] + results_df['EDGE 2'] + results_df['EDGE 3']
    results_df = results_df.sort_values('EV', ascending=False)
    
    # Limit player appearances
    results_df = limit_player_appearances_3leg(results_df, max_appearances=max_player_appearances)
    
    return results_df.head(top_n)
