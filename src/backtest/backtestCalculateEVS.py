import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from itertools import combinations
from src.models.ngboost_model import predict_mean_variance_split
from nba_api.stats.endpoints import scheduleleaguev2


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

def flag_sigma(s):
    """Categorize sigma values for volatility assessment"""
    if s <= 5.0:
        return 'Low'
    elif s <= 6.0:
        return 'Med'
    else:
        return 'High'

_gameCache = {}

def getUpcomingGamesCached(date):
    if date not in _gameCache:
        schedule = scheduleleaguev2.ScheduleLeagueV2(season=2024).get_data_frames()[0]
        schedule['gameDate'] = pd.to_datetime(schedule['gameDate']).dt.strftime('%Y-%m-%d')
        _gameCache[date] = schedule
    return _gameCache[date]


def findOpp(playerName, players_df, gameDate, max_days_ahead=3):
    player_team = players_df.loc[
        players_df['PLAYER_NAME'] == playerName, 'TEAM_ABBREVIATION'
    ].iloc[-1]
    
    base_date = datetime.strptime(gameDate, '%Y-%m-%d')
    dates_to_check = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(max_days_ahead + 1)]
    
    for check_date in dates_to_check:
        schedule = getUpcomingGamesCached(check_date)
        schedule_filtered = schedule[schedule['gameDate'] == check_date]
        homeTeams = schedule_filtered['homeTeam_teamTricode'].unique().tolist()
        awayTeams = schedule_filtered['awayTeam_teamTricode'].unique().tolist()
        
        home = 0
        if player_team in homeTeams:
            opp_team = awayTeams[homeTeams.index(player_team)]
            home = 1
            return opp_team, home
        elif player_team in awayTeams:
            opp_team = homeTeams[awayTeams.index(player_team)]
            home = 0
            return opp_team, home
    
    print(f"No game found for {player_team} within {max_days_ahead} days from {gameDate}")
    return None, None


# Global prediction cache
_prediction_cache = {}

def get_cached_prediction(player_name, data, models, features, stat_col='PTS', game_date=None):
    """Get cached prediction for a player, computing if not already cached.
    
    Returns:
        Dictionary with 'prediction' (mu/mean), 'sigma', 'skew' keys (matching calculateEVS.py format)
    """
    cache_key = f"{player_name}_{stat_col}_{game_date}" if game_date else f"{player_name}_{stat_col}"
    
    if cache_key not in _prediction_cache:
        try:
            # Get base prediction from model
            base_prediction = predictStats(player_name, data, models, features)
            mu = base_prediction['prediction']
            sigma_model = base_prediction['sigma']
            
            # Get player data for sigma and skew adjustments
            player_df = data[data['PLAYER_NAME'] == player_name].sort_values(by='GAME_DATE')
            
            # Determine current_date: use game_date if provided, otherwise use latest game date
            if game_date:
                current_date = game_date
            elif not player_df.empty:
                current_date = player_df['GAME_DATE'].iloc[-1]
            else:
                current_date = None
            
            # Use model-provided sigma and neutral skew for backtests
            sigma_adjusted = sigma_model
            skew = 0.0
            
            _prediction_cache[cache_key] = {
                'prediction': mu,
                'sigma': sigma_adjusted,
                'skew': skew
            }
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
    
    if isinstance(models, dict):
        if 'mean' in models and 'variance' in models:
            mean_model = models['mean']
            variance_model = models['variance']
            calibration_factor = models.get('calibration_factor', 1.25)
            isotonic_calibrator = models.get('isotonic_calibrator', None)
        else:
            raise ValueError("Models dict must contain 'mean' and 'variance' keys for NGBoost models")
    elif isinstance(models, tuple):
        mean_model = models[0]
        variance_model = models[1]
        calibration_factor = models[2] if len(models) > 2 else 1.25
        isotonic_calibrator = models[3] if len(models) > 3 else None
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
    
    if mu < 0:
        warnings.warn(f"Negative prediction ({mu:.2f}) for {playerName}. Clipping to 0.5. This may indicate insufficient data or unusual feature values.")
        mu = 0.5
    elif mu < 0.5:
        mu = 0.5
    
    predictions = {
        'prediction': mu,
        'sigma': sigma
    }
    
    return predictions

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
def backtestSingleBet(data, bookmakers, models, features, edge_threshold=0.05, stake=100, 
                     stat_col='PTS', top_n=50, max_player_appearances: int = 1):
    """
    Optimized single bet calculator using calibrated Normal distribution
    """
    print("Processing single bets...")
    
    # Pre-compute predictions for all unique players
    unique_players = bookmakers['NAME'].unique()
    player_predictions = {}
    
    print(f"Pre-computing predictions for {len(unique_players)} unique players...")
    for player in unique_players:
        mapped_player = nameDict.get(player, player)
        pred_data = get_cached_prediction(mapped_player, data, models, features, stat_col)
        if pred_data is not None:
            player_predictions[player] = pred_data
    
    results = []
    
    # Process each bet
    for _, row in bookmakers.iterrows():
        name = row['NAME']
        bookmaker = row['BOOKMAKER']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')
        odds = int(row['ODDS'])
        
        # Get pre-computed prediction data
        if name not in player_predictions:
            continue
        
        prediction_data = player_predictions[name]
        mu = prediction_data['prediction']
        sigma = prediction_data['sigma']  # Already calibrated!
        
        # Calculate probabilities analytically (fast & accurate)
        p_over = float(1 - norm.cdf(line, loc=mu, scale=sigma))
        p_under = 1.0 - p_over
        
        # Choose probability based on the offered side
        is_over = str(side).upper().startswith('O')
        p = p_over if is_over else p_under
        
        # Convert odds to decimal and calculate EV
        dec_odds = american_to_decimal(odds)
        b = dec_odds - 1.0
        
        # EV calculations
        ev_per_dollar = p * b - (1 - p)
        ev_total = ev_per_dollar * stake
        
        # Kelly criterion
        kelly_fraction = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
        
        # Edge calculation
        market_prob = impliedProb(odds)
        model_prob = p_over if is_over else p_under
        edge = model_prob - market_prob
        
        # Recommendation based on edge threshold
        recommendation = 1 if abs(line - mu) > edge_threshold and ev_total > 0 else 0
        
        # Confidence interval
        ci_lower = max(0, mu - 1.96 * sigma)
        ci_upper = mu + 1.96 * sigma
        
        # Sigma flag
        sigma_flag = flag_sigma(sigma)
        
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
            'EV$': round(ev_total, 2),
            'KELLY_FRACTION': round(kelly_fraction, 3),
            'SIGMA': round(sigma, 2),
            'SIGMA FLAG': sigma_flag,
            'CI': f"({ci_lower:.1f}, {ci_upper:.1f})",
            'EXPECTED ROI': round((ev_total / stake) * 100, 1)
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by EV
    results_df = results_df.sort_values('EV$', ascending=False)
    
    # Limit player appearances
    results_df = limit_player_appearances_single(results_df, max_appearances=max_player_appearances)
    
    # Return top N
    return results_df.head(top_n)

def limit_player_appearances_single(results_df, max_appearances=1):
    """Limit how many times each player appears in single bets"""
    player_counts = {}
    filtered_results = []
    
    for _, row in results_df.iterrows():
        player = row['NAME']
        count = player_counts.get(player, 0)
        
        if count < max_appearances:
            filtered_results.append(row)
            player_counts[player] = count + 1
    
    return pd.DataFrame(filtered_results)    

def backtest2legs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 stat_col='PTS', stake=100, max_player_appearances: int = 2):
    """
    Optimized 2-leg bet calculator using calibrated Normal distribution
    """
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

    # Pre-compute all predictions
    player_predictions = {}
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Pre-computing predictions for {len(available_players)} players...")
    game_date_str = pd.to_datetime(gameDate).strftime('%Y-%m-%d') if isinstance(gameDate, (pd.Timestamp, datetime)) else gameDate
    
    for player in available_players:
        mapped_player = nameDict.get(player, player)
        
        pred_data = get_cached_prediction(mapped_player, data, models, features, stat_col, gameDate)
        if pred_data is None:
            continue
        
        player_data = data[data['PLAYER_NAME'] == mapped_player]
        if player_data.empty:
            continue
        
        player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
        opp_team, _ = findOpp(mapped_player, player_data, game_date_str)
        if opp_team is None:
            continue
        
        player_bets = backtestData[backtestData['NAME'] == player]
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
    market_prob = impliedProb(-137)
    payout_multiple = 3.0
    
    results = []

    for player1, player2 in valid_combinations:
        # Map player names
        mapped_p1 = nameDict.get(player1, player1)
        mapped_p2 = nameDict.get(player2, player2)
        
        # Get pre-computed data
        pred1_data = player_predictions[player1]
        pred2_data = player_predictions[player2]
        
        mu1 = pred1_data['prediction']
        mu2 = pred2_data['prediction']
        sigma1 = pred1_data['sigma']  # Already calibrated!
        sigma2 = pred2_data['sigma']
        
        line1 = float(player_lines[player1]['LINE'])
        line2 = float(player_lines[player2]['LINE'])
        
        # Calculate probabilities analytically (fast & accurate)
        p1_over = float(1 - norm.cdf(line1, loc=mu1, scale=sigma1))
        p2_over = float(1 - norm.cdf(line2, loc=mu2, scale=sigma2))
        
        # Determine model sides
        if mu1 > line1:
            model_side1 = 'over'
            p1 = p1_over
        else:
            model_side1 = 'under'
            p1 = 1 - p1_over
            
        if mu2 > line2:
            model_side2 = 'over'
            p2 = p2_over
        else:
            model_side2 = 'under'
            p2 = 1 - p2_over
        
        # Dynamic correlation adjustment
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
        
        # Calculate combined probability and EV
        p_both_raw = p1 * p2
        p_both = p_both_raw * corr_adjustment
        ev = payout_multiple * p_both - 1
        ev_dollars = ev * stake
        
        # Edge calculations
        edge1 = p1 - market_prob
        edge2 = p2 - market_prob
        combined_edge = p_both_raw - (market_prob ** 2)
        
        # Kelly criterion
        b = payout_multiple - 1.0
        kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b)
        
        # Recommendation based on edge threshold
        recommendation = 1 if (abs(line1 - mu1) > edge_threshold and abs(line2 - mu2) > edge_threshold) and ev_dollars > 0 else 0
        
        # Confidence intervals
        ci1_lower = max(0, mu1 - 1.96 * sigma1)
        ci1_upper = mu1 + 1.96 * sigma1
        ci2_lower = max(0, mu2 - 1.96 * sigma2)
        ci2_upper = mu2 + 1.96 * sigma2
        
        # Sigma flags
        sigma_flag1 = flag_sigma(sigma1)
        sigma_flag2 = flag_sigma(sigma2)
        
        # Get actual results for backtesting
        player1_data = data[data['PLAYER_NAME'] == mapped_p1]
        player2_data = data[data['PLAYER_NAME'] == mapped_p2]
        actual1 = player1_data[player1_data['GAME_DATE'] == gameDate][stat_col].iloc[0] if len(player1_data[player1_data['GAME_DATE'] == gameDate]) > 0 else None
        actual2 = player2_data[player2_data['GAME_DATE'] == gameDate][stat_col].iloc[0] if len(player2_data[player2_data['GAME_DATE'] == gameDate]) > 0 else None
        
        if actual1 is None or actual2 is None:
            continue
        
        # Determine if bet won
        won1 = (actual1 > line1) if model_side1 == 'over' else (actual1 < line1)
        won2 = (actual2 > line2) if model_side2 == 'over' else (actual2 < line2)
        won_both = won1 and won2
        
        # Calculate profit/loss based on stake
        if won_both:
            profit = (payout_multiple - 1) * stake  # Win: (3-1) * stake = 2 * stake
        else:
            profit = -stake  # Loss: lose the stake
        
        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'LINE 1': line1,
            'LINE 2': line2,
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
            'SIGMA 1': round(sigma1, 2),
            'SIGMA 2': round(sigma2, 2),
            'SIGMA FLAG 1': sigma_flag1,
            'SIGMA FLAG 2': sigma_flag2,
            'CI 1': f"({ci1_lower:.1f}, {ci1_upper:.1f})",
            'CI 2': f"({ci2_lower:.1f}, {ci2_upper:.1f})",
            'CORRELATION': round(correlation, 3),
            'SAME_GAME': 1 if (team1 == opp2 or team2 == opp1) else 0,
            'EXPECTED ROI': round((ev / 1.0) * 100, 1),
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
    
    # Sort by EV
    results_df = results_df.sort_values('EV$', ascending=False)
    
    # Limit player appearances
    results_df = limit_player_appearances(results_df, max_appearances=max_player_appearances)
    
    # Return top N
    return results_df.head(top_n)

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

def backtest3Legs(data, backtestData, gameDate, models, features, edge_threshold=0.05, top_n=10, 
                 stat_col='PTS', stake=100, max_player_appearances: int = 2):
    """
    Optimized 3-leg bet calculator using calibrated Normal distribution
    """
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
    
    # Pre-compute all predictions
    player_predictions = {}
    player_teams = {}
    player_opponents = {}
    player_lines = {}
    
    print(f"Pre-computing predictions for {len(available_players)} players...")
    game_date_str = pd.to_datetime(gameDate).strftime('%Y-%m-%d') if isinstance(gameDate, (pd.Timestamp, datetime)) else gameDate
    
    for player in available_players:
        mapped_player = nameDict.get(player, player)
        
        pred_data = get_cached_prediction(mapped_player, data, models, features, stat_col, gameDate)
        if pred_data is None:
            continue
        
        player_data = data[data['PLAYER_NAME'] == mapped_player]
        if player_data.empty:
            continue
        
        player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
        opp_team, _ = findOpp(mapped_player, player_data, game_date_str)
        if opp_team is None:
            continue
        
        player_bets = backtestData[backtestData['NAME'] == player]
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
    
    # Constants
    market_prob = impliedProb(-137)
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
        
        mu1 = pred1_data['prediction']
        mu2 = pred2_data['prediction']
        mu3 = pred3_data['prediction']
        sigma1 = pred1_data['sigma']  # Already calibrated!
        sigma2 = pred2_data['sigma']
        sigma3 = pred3_data['sigma']
        
        line1 = float(player_lines[player1]['LINE'])
        line2 = float(player_lines[player2]['LINE'])
        line3 = float(player_lines[player3]['LINE'])
        
        # Calculate probabilities analytically (fast & accurate)
        p1_over = float(1 - norm.cdf(line1, loc=mu1, scale=sigma1))
        p2_over = float(1 - norm.cdf(line2, loc=mu2, scale=sigma2))
        p3_over = float(1 - norm.cdf(line3, loc=mu3, scale=sigma3))
        
        # Determine model sides
        if mu1 > line1:
            model_side1 = 'over'
            p1 = p1_over
        else:
            model_side1 = 'under'
            p1 = 1 - p1_over
            
        if mu2 > line2:
            model_side2 = 'over'
            p2 = p2_over
        else:
            model_side2 = 'under'
            p2 = 1 - p2_over
            
        if mu3 > line3:
            model_side3 = 'over'
            p3 = p3_over
        else:
            model_side3 = 'under'
            p3 = 1 - p3_over
        
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
            corr_adjustment = 0.50  # Very high correlation
            correlation = 0.50
        elif same_game_count == 1:
            corr_adjustment = 0.70  # Medium correlation
            correlation = 0.30
        else:
            corr_adjustment = 0.90  # Low correlation
            correlation = 0.10
        
        # Calculate combined probability and EV
        p_all_three_raw = p1 * p2 * p3
        p_all_three = p_all_three_raw * corr_adjustment
        ev = payout_multiple * p_all_three - 1
        ev_dollars = ev * stake
        
        # Edge calculations
        edge1 = p1 - market_prob
        edge2 = p2 - market_prob
        edge3 = p3 - market_prob
        combined_edge = p_all_three_raw - (market_prob ** 3)
        
        # Kelly criterion
        b = payout_multiple - 1.0
        kelly_full = max(0.0, (b * p_all_three - (1 - p_all_three)) / b)
        
        # Recommendation based on edge threshold
        recommendation = 1 if (abs(line1 - mu1) > edge_threshold and abs(line2 - mu2) > edge_threshold and abs(line3 - mu3) > edge_threshold) and ev_dollars > 0 else 0
        
        # Confidence intervals
        ci1_lower = max(0, mu1 - 1.96 * sigma1)
        ci1_upper = mu1 + 1.96 * sigma1
        ci2_lower = max(0, mu2 - 1.96 * sigma2)
        ci2_upper = mu2 + 1.96 * sigma2
        ci3_lower = max(0, mu3 - 1.96 * sigma3)
        ci3_upper = mu3 + 1.96 * sigma3
        
        # Sigma flags
        sigma_flag1 = flag_sigma(sigma1)
        sigma_flag2 = flag_sigma(sigma2)
        sigma_flag3 = flag_sigma(sigma3)
        
        # Get actual results
        player1_data = data[data['PLAYER_NAME'] == mapped_p1]
        player2_data = data[data['PLAYER_NAME'] == mapped_p2]
        player3_data = data[data['PLAYER_NAME'] == mapped_p3]
        actual1 = player1_data[player1_data['GAME_DATE'] == gameDate][stat_col].iloc[0] if len(player1_data[player1_data['GAME_DATE'] == gameDate]) > 0 else None
        actual2 = player2_data[player2_data['GAME_DATE'] == gameDate][stat_col].iloc[0] if len(player2_data[player2_data['GAME_DATE'] == gameDate]) > 0 else None
        actual3 = player3_data[player3_data['GAME_DATE'] == gameDate][stat_col].iloc[0] if len(player3_data[player3_data['GAME_DATE'] == gameDate]) > 0 else None
        
        if actual1 is None or actual2 is None or actual3 is None:
            continue
        
        # Determine if bet won
        won1 = (actual1 > line1) if model_side1 == 'over' else (actual1 < line1)
        won2 = (actual2 > line2) if model_side2 == 'over' else (actual2 < line2)
        won3 = (actual3 > line3) if model_side3 == 'over' else (actual3 < line3)
        won_all_three = won1 and won2 and won3
        
        # Calculate profit/loss based on stake
        if won_all_three:
            profit = (payout_multiple - 1) * stake  # Win: (6-1) * stake = 5 * stake
        else:
            profit = -stake  # Loss: lose the stake
        
        results.append({
            'NAME 1': mapped_p1,
            'NAME 2': mapped_p2,
            'NAME 3': mapped_p3,
            'LINE 1': line1,
            'LINE 2': line2,
            'LINE 3': line3,
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
            'EXPECTED ROI': round((ev / 1.0) * 100, 1),
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
    
    # Sort by EV
    results_df = results_df.sort_values('EV$', ascending=False)
    
    # Limit player appearances
    results_df = limit_player_appearances_3leg(results_df, max_appearances=max_player_appearances)
    
    # Return top N
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