import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from MODELS.pipeline import *
from MODELS.model import *
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import truncnorm


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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#monte carlo simulation using my model to calculate the probability of the prop
def monteCarloSim(player_df, modelPred, prop_line, std_dev, num_simulations=1000):
    """
    Simulates player performance using truncated normal distribution to prevent negative values.
    """
    # Create truncated normal distribution (lower bound = 0)
    a = -modelPred / std_dev  # Lower bound in standard deviations
    b = np.inf  # No upper bound
    
    simulated_points = truncnorm.rvs(a, b, loc=modelPred, scale=std_dev, size=num_simulations)
    prob_over = np.mean(simulated_points > prop_line)
    prob_under = 1 - prob_over
    ci = np.percentile(simulated_points, [2.5, 97.5])
    
    return {
        'model_prediction': modelPred,
        'simulated_mean': simulated_points.mean(),
        'simulated_std': simulated_points.std(),
        'std_used': std_dev,
        'prob_over': prob_over,
        'prob_under': prob_under,
        'confidence_interval': (ci[0], ci[1])
    }


def single_bet(data, bookmakers, model, gamesSchedule, features, todayDate, stake=100,
               simulations=10000, std_window=10, min_std=2.0, max_std=9.5, stat_col='PTS'):

    print("Processing single bets...")
    date_obj = datetime.strptime(todayDate, "%Y%m%d")
    game_date = date_obj.strftime("%Y-%m-%d")
    todayDate = str(todayDate)

    def get_player_std(player_df, stat_col):
        s = player_df[stat_col].dropna()
        if s.empty:
            return 5.0
        sd = s.tail(std_window).std(ddof=1) if len(s) >= std_window else s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            sd = 5.0
        return float(np.clip(sd, min_std, max_std))

    def american_to_decimal(odds):
        return 1 + (odds / 100.0) if odds > 0 else 1 + (100.0 / abs(odds))

    results = []
    # bookmakers = bookmakers.drop_duplicates(subset=['NAME'])

    for _, row in bookmakers.iterrows():
        name = row['NAME']
        bookmaker = row['BOOKMAKER']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')
        odds = int(row['ODDS'])

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        # use most recent game row
        player_team = player_df['TEAM_ABBREVIATION'].iloc[0]
        game_id = int(player_df['GAME_ID'].iloc[0])

        starters = getStarters(game_id, player_team, data)
        opponent, _ = findOppTeam(name, data, gamesSchedule)
        if opponent is None:
            continue

        temp_props = pd.DataFrame({
            'player': [name],
            'line': [line],
            'NAME': [name],
            'LINE': [line],
            'CATEGORY': [category]
        })

        features_list = features
        if features_list is None:
            fv = buildFeatureVector(name, data, gamesSchedule, todayDate, starters, game_id)
            features_list = [f'f{i}' for i in range(len(fv))]

        pred = makePredictionCatBoost(
            player_name=name,
            data=data,
            model=model,
            bookmakers=temp_props,
            games=gamesSchedule,
            todayDate=todayDate,
            starters=starters,
            game_id=game_id,
            features=features_list,
        )

        std_dev = get_player_std(player_df, stat_col)

        sim_results = monteCarloSim(
            player_df=player_df,
            modelPred=float(pred['predicted_stat']),
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
        ev_dollars = stake * ev_per_unit
        ev_percent = ev_per_unit * 100

        # Kelly fraction
        kelly_full = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0

        # Break-even probability and edge
        breakeven_prob = 1.0 / dec_odds
        edge = p - breakeven_prob

        # Debug: Print the actual values
        print(f"\nDEBUG - {name}:")
        print(f"  Odds: {odds} ({side})")
        print(f"  Line: {line}")
        print(f"  Prediction: {float(pred['predicted_stat'])}")
        print(f"  Std Dev: {std_dev}")
        print(f"  Prob Over: {p_over:.3f}")
        print(f"  Decimal Odds: {dec_odds:.2f}")
        print(f"  Breakeven: {breakeven_prob:.3f}")
        print(f"  Simulated Mean: {sim_results['simulated_mean']:.2f}")
        print(f"  Model Prediction: {sim_results['model_prediction']:.2f}")

        results.append({
            'NAME': name,
            'BOOKMAKER': bookmaker,
            'CATEGORY': category,
            'LINE': line,
            'ODDS': odds,
            'SIDE': side,
            'PREDICTION': float(pred['predicted_stat']),
            'OVER%': round(p_over, 3),
            'UNDER%': round(p_under, 3),
            'IMPLIED PROB': round(impliedProb(odds), 3),
            'EV%': round(ev_percent,2),
            'KELLY FULL': round(kelly_full, 2),
            'KELLY HALF': round(0.5 * kelly_full, 2),
            'KELLY QUARTER': round(0.25 * kelly_full, 2),
            'CONFIDENCE INTERVAL': f"({sim_results['confidence_interval'][0]:.1f}, {sim_results['confidence_interval'][1]:.1f})"
        })

    return pd.DataFrame(results)

    
def prizepickspairsEV(data, bookmakers, model, gamesSchedule, features, todayDate, stake=100,
                      simulations=10000, std_window=10, min_std=2.0, max_std=9.5, stat_col='PTS', prevData=None):
    
    print("Processing PrizePicks pairs...")
    date_obj = datetime.strptime(todayDate, "%Y%m%d")
    _game_date = date_obj.strftime("%Y-%m-%d")
    todayDate = str(todayDate)

    def get_player_std(player_df, stat_col):
        s = player_df[stat_col].dropna()
        if s.empty:
            return 5.0
        sd = s.tail(std_window).std(ddof=1) if len(s) >= std_window else s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            sd = 5.0
        return float(np.clip(sd, min_std, max_std))

    legs = []
    processed_players = set()  # Track processed players to avoid duplicates
    
    for _, row in bookmakers.iterrows():
        name = row['NAME']
        category = row['CATEGORY']
        line = float(row['LINE'])
        side = row.get('SIDE', 'over')  # Updated to use 'SIDE' column

        # Skip if we've already processed this player
        if name in processed_players:
            continue
        
        processed_players.add(name)

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        player_team = player_df['TEAM_ABBREVIATION'].iloc[0]
        game_id = int(player_df['GAME_ID'].iloc[0])

        starters = getStarters(game_id, player_team, data)
        opponent, _ = findOppTeam(name, data, gamesSchedule)
        if opponent is None:
            continue

        temp_props = pd.DataFrame({
            'player': [name],
            'line': [line],
            'NAME': [name],
            'LINE': [line],
            'CATEGORY': [category]
        })

        features_list = features
        if features_list is None:
            fv = buildFeatureVector(name, data, gamesSchedule, todayDate, starters, game_id)
            features_list = [f'f{i}' for i in range(len(fv))]

        pred = makePredictionCatBoost(
            player_name=name,
            data=data,
            model=model,
            bookmakers=temp_props,
            games=gamesSchedule,
            todayDate=todayDate,
            starters=starters,
            game_id=game_id,
            features=features_list,
        )

        std_dev = get_player_std(player_df, stat_col)

        sim_results = monteCarloSim(
            player_df=player_df,
            modelPred=float(pred['predicted_stat']),
            prop_line=line,
            std_dev=std_dev,
            num_simulations=simulations
        )

        legs.append({
            'NAME': name,
            'TEAM': player_team,  # Store team for easier comparison
            'CATEGORY': category,
            'LINE': line,
            'SIDE': side,
            'PREDICTION': float(pred['predicted_stat']),
            'OVER%': float(sim_results['prob_over']),
            'UNDER%': float(sim_results['prob_under']),
            'CI': sim_results['confidence_interval']
        })

    payout_multiple = 3.0
    b = payout_multiple - 1.0

    pair_results = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):  # i + 1 prevents duplicate combos (a,b vs b,a)
            leg1 = legs[i]
            leg2 = legs[j]
            
            # Skip if same player (no duplicates)
            if leg1['NAME'] == leg2['NAME']:
                continue
            
            # Skip if players from same team
            if leg1['TEAM'] == leg2['TEAM']:
                continue

            p1 = leg1['OVER%'] if str(leg1['SIDE']).upper().startswith('O') else leg1['UNDER%']
            p2 = leg2['OVER%'] if str(leg2['SIDE']).upper().startswith('O') else leg2['UNDER%']
            p_both = p1 * p2

            ev_per_unit = p_both * b - (1 - p_both)
            ev_percent = ev_per_unit

            kelly_full = max(0.0, (b * p_both - (1 - p_both)) / b) if b > 0 else 0.0

            pair_results.append({
                'PLAYER 1': leg1['NAME'],
                'CATEGORY 1': leg1['CATEGORY'],
                'LINE 1': leg1['LINE'],
                'SIDE 1': leg1['SIDE'],
                'PREDICTION 1': round(leg1['PREDICTION'], 2),
                'OVER% 1': round(leg1['OVER%'], 3),
                'UNDER% 1': round(leg1['UNDER%'], 3),
                'CONFIDENCE INTERVAL 1': f"({leg1['CI'][0]:.1f}, {leg1['CI'][1]:.1f})",
                'PLAYER 2': leg2['NAME'],
                'CATEGORY 2': leg2['CATEGORY'],
                'LINE 2': leg2['LINE'],
                'SIDE 2': leg2['SIDE'],
                'PREDICTION 2': round(leg2['PREDICTION'], 2),
                'OVER% 2': round(leg2['OVER%'], 3),
                'UNDER% 2': round(leg2['UNDER%'], 3),
                'CONFIDENCE INTERVAL 2': f"({leg2['CI'][0]:.1f}, {leg2['CI'][1]:.1f})",
                'TYPE': f"{'OVER' if str(leg1['SIDE']).upper().startswith('O') else 'UNDER'}/"
                        f"{'OVER' if str(leg2['SIDE']).upper().startswith('O') else 'UNDER'}",
                'PROBABILITY': round(p_both, 4),
                'EV%': round(ev_percent, 3),
                'KELLY FULL': round(kelly_full, 2),
                'KELLY HALF': round(0.5 * kelly_full, 2),
                'KELLY QUARTER': round(0.25 * kelly_full, 2)
            })

    return pd.DataFrame(pair_results)