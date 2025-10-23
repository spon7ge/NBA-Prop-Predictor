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

def get_player_std(player_df, stat_col, std_window=10, min_std=2.0, max_std=8.5):
    s = player_df[stat_col].dropna()
    if s.empty:
        return 5.0
    sd = s.tail(std_window).std(ddof=1) if len(s) >= std_window else s.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        sd = 5.0
    return float(np.clip(sd, min_std, max_std))

def predictStats(playerName, data, model, features):
    playerData = data[data['PLAYER_NAME'] == playerName]
    latestRow = playerData.sort_values(by='GAME_DATE').iloc[-1]
    available_features = [f for f in features if f in data.columns] 
    playerInput = latestRow[available_features]
    
    playerInput_df = pd.DataFrame([playerInput.values], columns=available_features)
    
    for col in playerInput_df.columns:
        if playerInput_df[col].dtype == 'object':
            playerInput_df[col] = pd.to_numeric(playerInput_df[col], errors='coerce').fillna(0)
        elif playerInput_df[col].dtype == 'bool':
            playerInput_df[col] = playerInput_df[col].astype(int)
    
    pred = model.predict(playerInput_df)[0]
    return float(pred)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def monteCarloSim(player_df, modelPred, prop_line, std_dev, num_simulations=10000, min_std=2.0, max_std=10.0, stat_col='PTS'):
    baseSTD = std_dev
    volAdj = 1.0
    vol_col = f'{stat_col}_EXPANDING_VOLATILITY_TO_DATE'
    stat_col_actual = stat_col

    # Volatility adjustment using recent vs season volatility
    if vol_col in player_df.columns and len(player_df) >= 10:
        recent_vol = player_df[stat_col_actual].tail(10).std()
        season_vol = player_df[vol_col].iloc[-1]

        if season_vol > 0 and not pd.isna(season_vol) and not pd.isna(recent_vol):
            volAdj = recent_vol / season_vol
            volAdj = 1 + (volAdj - 1) * 0.3
            volAdj = np.clip(volAdj, 0.7, 2.0)

        stdDev = baseSTD * volAdj
    else:
        stdDev = baseSTD

    # Scale variance relative to mean
    stdDev = max(stdDev, 0.15 * modelPred)
    stdDev = float(np.clip(stdDev, min_std, max_std * 1.5))

    # Simulate outcomes (non-negative)
    simulated_points = np.random.normal(modelPred, stdDev, num_simulations)
    simulated_points = np.clip(simulated_points, 0, None)

    # Compute probabilities and confidence interval
    prob_over = np.mean(simulated_points > prop_line)
    prob_over = float(np.clip(prob_over, 0.05, 0.95))
    prob_under = 1 - prob_over
    ci = np.percentile(simulated_points, [2.5, 97.5])

    return {
        'model_prediction': modelPred,
        'simulated_mean': simulated_points.mean(),
        'simulated_std': simulated_points.std(),
        'std_used': stdDev,
        'vol_adjustment': volAdj,
        'prob_over': prob_over,
        'prob_under': prob_under,
        'confidence_interval': (ci[0], ci[1])
    }


def backtestSingleBet(data, bookmakers, model, features, edge_threshold=4.5, stake=100, simulations=10000, 
               std_window=10, min_std=2.0, max_std=8.5, stat_col='PTS'):

    print("Processing single bets...")

    results = []

    for _, row in bookmakers.iterrows():
        name = row['NAME']
        bookmaker = row['BOOKMAKER']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')
        odds = int(row['ODDS'])

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

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        # Get prediction using original function
        try:
            prediction = predictStats(name, data, model, features)
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue

        std_dev = get_player_std(player_df, stat_col, std_window, min_std, max_std)

        sim_results = monteCarloSim(
            player_df=player_df,
            modelPred=float(prediction),
            prop_line=line,
            std_dev=std_dev,
            num_simulations=simulations
        )

        p_over = float(sim_results['prob_over'])
        p_under = 1.0 - p_over

        # choose probability based on the offered side
        if str(side).upper().startswith('O'):
            p = p_over
        else:
            p = p_under

        # odds and returns
        dec_odds = american_to_decimal(odds)          
        b = dec_odds - 1.0                             
        profit_if_win = stake * b
        loss_if_lose = stake

        # EV in dollars and percent
        ev_per_unit = p * b - (1 - p)
        ev_percent = ev_per_unit * 100

        # Kelly fraction
        kelly_full = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0

        edge = abs(prediction - line)
        if abs(edge) > edge_threshold:
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
            'PREDICTION': float(prediction),
            'RECOMMENDATION': recommendation,
            'OVER%': round(p_over, 3),
            'UNDER%': round(p_under, 3),
            'IMPLIED PROB': round(impliedProb(odds), 3),
            'EV%': round(ev_percent, 2),
            'KELLY FULL': round(kelly_full, 2),
            'KELLY HALF': round(0.5 * kelly_full, 2),
            'KELLY QUARTER': round(0.25 * kelly_full, 2),
            'CONFIDENCE INTERVAL': f"({sim_results['confidence_interval'][0]:.1f}, {sim_results['confidence_interval'][1]:.1f})"
        })

    return pd.DataFrame(results)

    
def backtest2legs(data, bookmakers, model, features, edge_threshold=4.5, stake=100, 
                      simulations=10000, std_window=10, min_std=2.0, max_std=8.5, stat_col='PTS'):
    
    print("Processing pairs...")

    grouped_bookmakers = bookmakers.groupby(['NAME', 'LINE']).agg({
        'CATEGORY': 'first',
        'BOOKMAKER': 'first',
        'ODDS': 'first',
        'SIDE': 'first'
    }).reset_index()
    
    prediction_cache = {}
    legs = []
    
    for _, row in grouped_bookmakers.iterrows():
        name = row['NAME']
        category = row['CATEGORY']
        bookmaker = row['BOOKMAKER']
        odds = int(row['ODDS'])
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')

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

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        player_team = player_df['TEAM_ABBREVIATION'].iloc[0]

        if name not in prediction_cache:
            try:
                prediction_cache[name] = predictStats(name, data, model, features)
            except Exception as e:
                print(f"Error getting prediction for {name}: {e}")
                continue
        prediction = prediction_cache[name]

        std_dev = get_player_std(player_df, stat_col, std_window, min_std, max_std)
        sim_results = monteCarloSim(
            player_df=player_df,
            modelPred=float(prediction),
            prop_line=line,
            std_dev=std_dev,
            num_simulations=simulations
        )

        if prediction > line:
            model_side = 'OVER'
            model_prob = float(sim_results['prob_over'])
            model_opposite_prob = float(sim_results['prob_under'])
        else:
            model_side = 'UNDER'
            model_prob = float(sim_results['prob_under'])
            model_opposite_prob = float(sim_results['prob_over'])

        legs.append({
            'NAME': name,
            'TEAM': player_team,
            'CATEGORY': category,
            'BOOKMAKER': bookmaker,
            'ODDS': odds,
            'LINE': line,
            'PREDICTION': float(prediction),
            'MODEL_SIDE': model_side,
            'MODEL_PROB': model_prob,
            'MODEL_OPPOSITE_PROB': model_opposite_prob,
            'OVER%': float(sim_results['prob_over']),
            'UNDER%': float(sim_results['prob_under']),
            'CI': sim_results['confidence_interval']
        })

    payout_multiple = 3.0
    b = payout_multiple - 1.0

    pair_results = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            leg1 = legs[i]
            leg2 = legs[j]
            
            # Skip if same player (no duplicates)
            if leg1['NAME'] == leg2['NAME']:
                continue
            
            # Skip if players from same team
            if leg1['TEAM'] == leg2['TEAM']:
                continue

            # Use the model's recommended side and probability
            p1 = leg1['MODEL_PROB']
            p2 = leg2['MODEL_PROB']
            p_both = p1 * p2
            ev = payout_multiple * p_both - 1


            kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b) if b > 0 else 0.0

            edge1 = abs(leg1['PREDICTION'] - leg1['LINE'])
            edge2 = abs(leg2['PREDICTION'] - leg2['LINE'])
            if abs(edge1) > edge_threshold and abs(edge2) > edge_threshold:
                recommendation = 1
            else:
                recommendation = 0

            pair_results.append({
                'PLAYER 1': leg1['NAME'],
                'CATEGORY 1': leg1['CATEGORY'],
                'BOOKMAKER 1': leg1['BOOKMAKER'],
                'ODDS 1': leg1['ODDS'],
                'LINE 1': leg1['LINE'],
                'PREDICTION 1': round(leg1['PREDICTION'], 2),
                'MODEL_SIDE 1': leg1['MODEL_SIDE'],
                'OVER% 1': round(leg1['OVER%'], 3),
                'UNDER% 1': round(leg1['UNDER%'], 3),
                'CONFIDENCE INTERVAL 1': f"({leg1['CI'][0]:.1f}, {leg1['CI'][1]:.1f})",
                'PLAYER 2': leg2['NAME'],
                'CATEGORY 2': leg2['CATEGORY'],
                'BOOKMAKER 2': leg2['BOOKMAKER'],
                'ODDS 2': leg2['ODDS'],
                'LINE 2': leg2['LINE'],
                'PREDICTION 2': round(leg2['PREDICTION'], 2),
                'MODEL_SIDE 2': leg2['MODEL_SIDE'],
                'OVER% 2': round(leg2['OVER%'], 3),
                'UNDER% 2': round(leg2['UNDER%'], 3),
                'CONFIDENCE INTERVAL 2': f"({leg2['CI'][0]:.1f}, {leg2['CI'][1]:.1f})",
                'RECOMMENDED_TYPE': f"{leg1['MODEL_SIDE']}/{leg2['MODEL_SIDE']}",
                'RECOMMENDATION': recommendation,
                'PROBABILITY': round(p_both, 4),
                'EV%': round(ev, 3),
                'KELLY': round(kelly_full, 3),
            })

    return pd.DataFrame(pair_results)

def backtest3Legs(data, bookmakers, model, features, edge_threshold=4.5, stake=100,
                     simulations=10000, std_window=10, min_std=2.0, max_std=8.5, stat_col='PTS'):
    
    print("Processing 3-leg parlays...")

    grouped_bookmakers = bookmakers.groupby(['NAME', 'LINE']).agg({
        'CATEGORY': 'first',
        'BOOKMAKER': 'first',
        'ODDS': 'first',
        'SIDE': 'first'
    }).reset_index()
    
    legs = []
    
    for _, row in grouped_bookmakers.iterrows():
        name = row['NAME']
        category = row['CATEGORY']
        bookmaker = row['BOOKMAKER']
        odds = int(row['ODDS'])
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')

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

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        player_team = player_df['TEAM_ABBREVIATION'].iloc[0]

        # Get prediction
        try:
            prediction = predictStats(name, data, model, features)
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue

        std_dev = get_player_std(player_df, stat_col, std_window, min_std, max_std)
        sim_results = monteCarloSim(
            player_df=player_df,
            modelPred=float(prediction),
            prop_line=line,
            std_dev=std_dev,
            num_simulations=simulations
        )

        # Determine over/under based on prediction vs line
        if prediction > line:
            model_side = 'OVER'
            model_prob = float(sim_results['prob_over'])
            model_opposite_prob = float(sim_results['prob_under'])
        else:
            model_side = 'UNDER'
            model_prob = float(sim_results['prob_under'])
            model_opposite_prob = float(sim_results['prob_over'])

        legs.append({
            'NAME': name,
            'TEAM': player_team,
            'CATEGORY': category,
            'BOOKMAKER': bookmaker,
            'ODDS': odds,
            'LINE': line,
            'PREDICTION': float(prediction),
            'MODEL_SIDE': model_side,
            'MODEL_PROB': model_prob,
            'MODEL_OPPOSITE_PROB': model_opposite_prob,
            'OVER%': float(sim_results['prob_over']),
            'UNDER%': float(sim_results['prob_under']),
            'CI': sim_results['confidence_interval']
        })

    # 3-leg parlay pays 6x (profit = 5x stake)
    payout_multiple = 6.0
    b = payout_multiple - 1.0  # b = 5.0

    parlay_results = []
    
    # Triple nested loop for 3-leg combinations
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            for k in range(j + 1, len(legs)):
                leg1 = legs[i]
                leg2 = legs[j]
                leg3 = legs[k]
                
                # No duplicate players (should be automatic with i < j < k, but double-check)
                players = [leg1['NAME'], leg2['NAME'], leg3['NAME']]
                if len(set(players)) != 3:
                    continue
                
                # Check team constraint: not all 3 from same team (but 2 is OK)
                teams = [leg1['TEAM'], leg2['TEAM'], leg3['TEAM']]
                if len(set(teams)) == 1:  # All 3 same team - skip
                    continue

                # Calculate combined probability
                p1 = leg1['MODEL_PROB']
                p2 = leg2['MODEL_PROB']
                p3 = leg3['MODEL_PROB']
                p_all_three = p1 * p2 * p3
                ev = payout_multiple * p_all_three - 1

                # Kelly criterion
                kelly_full = max(0.0, (b * p_all_three - (1 - p_all_three)) / b) if b > 0 else 0.0

                # Recommendation: all 3 legs must have edge > threshold
                edge1 = abs(leg1['PREDICTION'] - leg1['LINE'])
                edge2 = abs(leg2['PREDICTION'] - leg2['LINE'])
                edge3 = abs(leg3['PREDICTION'] - leg3['LINE'])
                
                if (abs(edge1) > edge_threshold and 
                    abs(edge2) > edge_threshold and 
                    abs(edge3) > edge_threshold):
                    recommendation = 1
                else:
                    recommendation = 0

                parlay_results.append({
                    'PLAYER 1': leg1['NAME'],
                    'CATEGORY 1': leg1['CATEGORY'],
                    'BOOKMAKER 1': leg1['BOOKMAKER'],
                    'ODDS 1': leg1['ODDS'],
                    'LINE 1': leg1['LINE'],
                    'PREDICTION 1': round(leg1['PREDICTION'], 2),
                    'MODEL_SIDE 1': leg1['MODEL_SIDE'],
                    'OVER% 1': round(leg1['OVER%'], 3),
                    'UNDER% 1': round(leg1['UNDER%'], 3),
                    'CONFIDENCE INTERVAL 1': f"({leg1['CI'][0]:.1f}, {leg1['CI'][1]:.1f})",
                    
                    'PLAYER 2': leg2['NAME'],
                    'CATEGORY 2': leg2['CATEGORY'],
                    'BOOKMAKER 2': leg2['BOOKMAKER'],
                    'ODDS 2': leg2['ODDS'],
                    'LINE 2': leg2['LINE'],
                    'PREDICTION 2': round(leg2['PREDICTION'], 2),
                    'MODEL_SIDE 2': leg2['MODEL_SIDE'],
                    'OVER% 2': round(leg2['OVER%'], 3),
                    'UNDER% 2': round(leg2['UNDER%'], 3),
                    'CONFIDENCE INTERVAL 2': f"({leg2['CI'][0]:.1f}, {leg2['CI'][1]:.1f})",
                    
                    'PLAYER 3': leg3['NAME'],
                    'CATEGORY 3': leg3['CATEGORY'],
                    'BOOKMAKER 3': leg3['BOOKMAKER'],
                    'ODDS 3': leg3['ODDS'],
                    'LINE 3': leg3['LINE'],
                    'PREDICTION 3': round(leg3['PREDICTION'], 2),
                    'MODEL_SIDE 3': leg3['MODEL_SIDE'],
                    'OVER% 3': round(leg3['OVER%'], 3),
                    'UNDER% 3': round(leg3['UNDER%'], 3),
                    'CONFIDENCE INTERVAL 3': f"({leg3['CI'][0]:.1f}, {leg3['CI'][1]:.1f})",
                    
                    'RECOMMENDED_TYPE': f"{leg1['MODEL_SIDE']}/{leg2['MODEL_SIDE']}/{leg3['MODEL_SIDE']}",
                    'RECOMMENDATION': recommendation,
                    'PROBABILITY': round(p_all_three, 4),
                    'EV%': round(ev, 3),
                    'KELLY': round(kelly_full, 3),
                })

    return pd.DataFrame(parlay_results)