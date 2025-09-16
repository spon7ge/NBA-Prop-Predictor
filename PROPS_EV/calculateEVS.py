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
        'mean_prediction': modelPred,
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
        side = row.get('OVER/UNDER', 'over')
        odds = int(row['PRICE'])

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
            n_games=3
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
        dec_odds = american_to_decimal(odds)            # includes stake
        b = dec_odds - 1.0                              # net profit per 1 staked
        profit_if_win = stake * b
        loss_if_lose = stake

        # EV in dollars and percent
        ev_per_unit = p * b - (1 - p)
        ev_dollars = stake * ev_per_unit
        ev_percent = ev_per_unit

        # Kelly fraction
        kelly_full = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0

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
            'EV$': round(ev_dollars, 2),
            'EV%': round(ev_percent,2),
            'KELLY_FULL': round(kelly_full, 2),
            'KELLY_HALF': round(0.5 * kelly_full, 2),
            'KELLY_QUARTER': round(0.25 * kelly_full, 2),
            'CONFIDENCE_INTERVAL': f"({sim_results['confidence_interval'][0]:.1f}, {sim_results['confidence_interval'][1]:.1f})"
        })

    return pd.DataFrame(results)

    
def prizePicksPairsEV(prizePicks, propDict, models, games, current_datasets=None, simulations=10000, stake=100, payout=300):
    print("Loading datasets and generating valid combinations...")
    valid_combinations = []

    # Load datasets
    datasets = {}
    stat_types = list(propDict.values())
    for stat_type in stat_types:
        try:
            if current_datasets is not None and stat_type in current_datasets:
                datasets[stat_type] = current_datasets[stat_type]
                print(f"Using provided current dataset for {stat_type}")
            else:
                datasets[stat_type] = pd.read_csv(f'CSV_FILES/REGULAR_DATA/season_25_{stat_type}_FEATURES.csv')
                print(f"Loaded dataset for {stat_type}")
        except Exception as e:
            print(f"Error loading dataset for {stat_type}: {e}")
            return pd.DataFrame()

    # Get unique players in PrizePicks lines
    unique_players = prizePicks['NAME'].unique()

    print(f"Precomputing residual stds for {len(unique_players)} players...")
    residual_stds = precompute_player_residual_stds(unique_players, datasets, models, games, stat_types)

    available_players = []
    for category, stat_line in propDict.items():
        category_data = prizePicks[prizePicks['CATEGORY'] == category]

        for _, row in category_data.iterrows():
            player = row['NAME']
            line = row['LINE']
            data = datasets[stat_line]

            player_data = data[data['PLAYER_NAME'] == player]
            if player_data.empty:
                continue

            player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]

            opponent = None
            for game in games:
                if game['home_team'] == player_team:
                    opponent = game['away_team']
                    break
                elif game['away_team'] == player_team:
                    opponent = game['home_team']
                    break

            if opponent is None:
                continue

            try:
                temp_props = pd.DataFrame({
                    'NAME': [player],
                    'LINE': [line],
                    'CATEGORY': [category]
                })

                pred = make_prediction(
                    player_name=player,
                    bookmakers=temp_props,
                    opponent=opponent,
                    model=models[stat_line],
                    data=data,
                    games=games,
                    is_playoff=0,
                    stat_line=stat_line
                )

                available_players.append({
                    'player': player,
                    'category': category,
                    'prediction': pred,
                    'line': line,
                    'stat_line': stat_line,
                    'team': player_team,
                    'opponent': opponent,
                    'std_dev': residual_stds.get(player, {}).get(stat_line, 5.0)  # use fallback if missing
                })

            except Exception as e:
                print(f"Error getting prediction for {player} ({category}): {e}")
                continue

    def get_combination_key(player1_data, player2_data):
        players = sorted([
            (player1_data['player'], player1_data['category'], player1_data['line']),
            (player2_data['player'], player2_data['category'], player2_data['line'])
        ])
        return tuple(players)

    seen_combinations = set()
    for i in range(len(available_players)):
        for j in range(i + 1, len(available_players)):
            player1_data = available_players[i]
            player2_data = available_players[j]

            combo_key = get_combination_key(player1_data, player2_data)

            if combo_key in seen_combinations:
                continue

            if player1_data['player'] == player2_data['player'] or player1_data['team'] == player2_data['team']:
                continue

            seen_combinations.add(combo_key)
            valid_combinations.append({
                'players': [player1_data['player'], player2_data['player']],
                'categories': [player1_data['category'], player2_data['category']],
                'stat_lines': [player1_data['stat_line'], player2_data['stat_line']],
                'lines': [player1_data['line'], player2_data['line']],
                'predictions': [player1_data['prediction'], player2_data['prediction']],
                'opponents': [player1_data['opponent'], player2_data['opponent']],
                'std_devs': [player1_data['std_dev'], player2_data['std_dev']]
            })

    def process_combination(combo):
        players = combo['players']
        categories = combo['categories']
        stat_lines = combo['stat_lines']
        predictions = combo['predictions']
        lines = combo['lines']
        opponents = combo['opponents']
        std_devs = combo['std_devs']

        try:
            sims = []
            for i in range(2):
                data = datasets[stat_lines[i]]
                player_df = data[data['PLAYER_NAME'] == players[i]].sort_values('GAME_DATE')

                sim = monte_carlo_prop_simulation(
                    player_df=player_df,
                    modelPred=predictions[i]['predicted_stat'],
                    prop_line=lines[i],
                    std_dev=std_devs[i],
                    num_simulations=simulations
                )
                sims.append(sim)

            sim1_over, sim1_under = sims[0]['prob_over'], sims[0]['prob_under']
            sim2_over, sim2_under = sims[1]['prob_over'], sims[1]['prob_under']

            combo_probs = {
                'OVER/OVER': sim1_over * sim2_over,
                'UNDER/UNDER': sim1_under * sim2_under,
                'OVER/UNDER': sim1_over * sim2_under,
                'UNDER/OVER': sim1_under * sim2_over
            }

            evs = {k: round((combo_probs[k] * payout) - stake, 2) for k in combo_probs}

            best_type = max(evs, key=evs.get)
            best_ev = evs[best_type]
            best_prob = combo_probs[best_type]

            return {
                'players': players,
                'categories': categories,
                'stat_lines': stat_lines,
                'lines': lines,
                'predictions': [pred['predicted_stat'] for pred in predictions],
                'best_type': best_type,
                'best_ev': best_ev,
                'best_prob': best_prob
            }

        except Exception as e:
            print(f"Error processing combination {players}: {e}")
            return None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp

    results = []
    max_workers = min(mp.cpu_count(), len(valid_combinations))

    print(f"Processing {len(valid_combinations)} combinations with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {
            executor.submit(process_combination, combo): i
            for i, combo in enumerate(valid_combinations)
        }

        completed = 0
        for future in as_completed(future_to_combo):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                completed += 1
                if completed % 100 == 0:
                    print(f"Completed {completed}/{len(valid_combinations)} combinations")
            except Exception as e:
                print(f"Error in future: {e}")

    print(f"Successfully processed {len(results)} combinations")
    print("Building final results...")
    all_pairs = []
    for result in results:
        all_pairs.append({
            'PLAYER 1': result['players'][0],
            'CATEGORY 1': result['categories'][0],
            'STAT TYPE 1': result['stat_lines'][0],
            'PLAYER 1 LINE': result['lines'][0],
            'PLAYER 1 PREDICTION': result['predictions'][0],
            'PLAYER 2': result['players'][1],
            'CATEGORY 2': result['categories'][1],
            'STAT TYPE 2': result['stat_lines'][1],
            'PLAYER 2 LINE': result['lines'][1],
            'PLAYER 2 PREDICTION': result['predictions'][1],
            'TYPE': result['best_type'],
            'EV': round(result['best_ev'], 2),
            'PROBABILITY': round(result['best_prob'], 4),
            'KELLY CRITERION': kelly_criterion(result['best_prob'], payout, stake)
        })

    return pd.DataFrame(all_pairs)

def prizePicksTriosEV(prizePicks, propDict, models, games, current_datasets=None, simulations=10000, stake=100, payout=600):
    """
    Calculate EV for PrizePicks trios using model predictions and Monte Carlo simulations
    Uses separate feature datasets for different stat types
    """
    print("Loading datasets and generating valid combinations...")
    valid_combinations = []
    
    # Load datasets
    datasets = {}
    stat_types = list(propDict.values())
    for stat_type in stat_types:
        try:
            if current_datasets is not None and stat_type in current_datasets:
                datasets[stat_type] = current_datasets[stat_type]
                print(f"Using provided current dataset for {stat_type}")
            else:
                datasets[stat_type] = pd.read_csv(f'CSV_FILES/REGULAR_DATA/season_25_{stat_type}_FEATURES.csv')
                print(f"Loaded dataset for {stat_type}")
        except Exception as e:
            print(f"Error loading dataset for {stat_type}: {e}")
            return pd.DataFrame()
    
    # Get unique players and precompute residual stds
    unique_players = prizePicks['NAME'].unique()
    print(f"Precomputing residual stds for {len(unique_players)} players...")
    residual_stds = precompute_player_residual_stds(unique_players, datasets, models, games, stat_types)
    
    # Process each category
    available_players = []
    for category, stat_line in propDict.items():
        category_data = prizePicks[prizePicks['CATEGORY'] == category]
        
        for _, row in category_data.iterrows():
            player = row['NAME']
            line = row['LINE']
            data = datasets[stat_line]
            
            # Get player data
            player_data = data[data['PLAYER_NAME'] == player]
            if player_data.empty:
                continue
                
            # Get player's team
            player_team = player_data['TEAM_ABBREVIATION'].iloc[-1]
            
            # Find opponent
            opponent = None
            for game in games:
                if game['home_team'] == player_team:
                    opponent = game['away_team']
                    break
                elif game['away_team'] == player_team:
                    opponent = game['home_team']
                    break
                    
            if opponent is None:
                continue
                
            try:
                temp_props = pd.DataFrame({
                    'NAME': [player],
                    'LINE': [line],
                    'CATEGORY': [category]
                })
                
                pred = make_prediction(
                    player_name=player,
                    bookmakers=temp_props,
                    opponent=opponent,
                    model=models[stat_line],
                    data=data,
                    games=games,
                    is_playoff=0,
                    stat_line=stat_line
                )
                
                available_players.append({
                    'player': player,
                    'category': category,
                    'prediction': pred,
                    'line': line,
                    'stat_line': stat_line,
                    'team': player_team,
                    'opponent': opponent,
                    'std_dev': residual_stds.get(player, {}).get(stat_line, 5.0)  # use fallback if missing
                })
                
            except Exception as e:
                print(f"Error getting prediction for {player} ({category}): {e}")
                continue
    
    # For trios:
    def get_trio_combination_key(player1_data, player2_data, player3_data):
        """Create a unique key for a trio combination that is order-independent"""
        players = sorted([
            (player1_data['player'], player1_data['category'], player1_data['line']),
            (player2_data['player'], player2_data['category'], player2_data['line']),
            (player3_data['player'], player3_data['category'], player3_data['line'])
        ])
        return tuple(players)

    # Keep track of seen combinations
    seen_combinations = set()

    # Generate all valid trios
    for i in range(len(available_players)):
        for j in range(i + 1, len(available_players)):
            for k in range(j + 1, len(available_players)):
                player1_data = available_players[i]
                player2_data = available_players[j]
                player3_data = available_players[k]
                
                # Create unique key for this combination
                combo_key = get_trio_combination_key(player1_data, player2_data, player3_data)
                
                # Skip if we've seen this combination before
                if combo_key in seen_combinations:
                    continue
                
                # Count how many players are from each team
                team_counts = {}
                for player_data in [player1_data, player2_data, player3_data]:
                    team = player_data['team']
                    team_counts[team] = team_counts.get(team, 0) + 1
                
                # Skip if any team has more than 2 players
                if any(count > 2 for count in team_counts.values()):
                    continue
                
                # Skip if same player
                if (player1_data['player'] == player2_data['player'] or 
                    player1_data['player'] == player3_data['player'] or 
                    player2_data['player'] == player3_data['player']):
                    continue
                
                seen_combinations.add(combo_key)
                valid_combinations.append({
                    'players': [player1_data['player'], player2_data['player'], player3_data['player']],
                    'categories': [player1_data['category'], player2_data['category'], player3_data['category']],
                    'stat_lines': [player1_data['stat_line'], player2_data['stat_line'], player3_data['stat_line']],
                    'lines': [player1_data['line'], player2_data['line'], player3_data['line']],
                    'predictions': [player1_data['prediction'], player2_data['prediction'], player3_data['prediction']],
                    'opponents': [player1_data['opponent'], player2_data['opponent'], player3_data['opponent']]
                })
    
    def process_combination(combo):
        """Process a single combination"""
        players = combo['players']
        categories = combo['categories']
        stat_lines = combo['stat_lines']
        predictions = combo['predictions']
        lines = combo['lines']
        opponents = combo['opponents']
        std_devs = [residual_stds.get(player, {}).get(stat_line, 5.0) 
                   for player, stat_line in zip(players, stat_lines)]
        
        try:
            # Run Monte Carlo simulations for each player
            sims = []
            for i in range(3):
                data = datasets[stat_lines[i]]
                player_df = data[data['PLAYER_NAME'] == players[i]].sort_values('GAME_DATE')
                
                sim = monte_carlo_prop_simulation(
                    player_df=player_df,
                    modelPred=predictions[i]['predicted_stat'],
                    prop_line=lines[i],
                    std_dev=std_devs[i],
                    num_simulations=simulations
                )
                sims.append(sim)
            
            # Calculate probabilities
            sim1_over = sims[0]['prob_over']
            sim1_under = sims[0]['prob_under']
            sim2_over = sims[1]['prob_over']
            sim2_under = sims[1]['prob_under']
            sim3_over = sims[2]['prob_over']
            sim3_under = sims[2]['prob_under']
            
            combo_probs = {
                'OVER/OVER/OVER': sim1_over * sim2_over * sim3_over,
                'OVER/OVER/UNDER': sim1_over * sim2_over * sim3_under,
                'OVER/UNDER/OVER': sim1_over * sim2_under * sim3_over,
                'OVER/UNDER/UNDER': sim1_over * sim2_under * sim3_under,
                'UNDER/OVER/OVER': sim1_under * sim2_over * sim3_over,
                'UNDER/OVER/UNDER': sim1_under * sim2_over * sim3_under,
                'UNDER/UNDER/OVER': sim1_under * sim2_under * sim3_over,
                'UNDER/UNDER/UNDER': sim1_under * sim2_under * sim3_under
            }
            
            # Calculate EVs
            evs = {k: round((combo_probs[k] * payout) - stake, 2) for k in combo_probs}
            
            # Find best combination
            best_type = max(evs, key=evs.get)
            best_ev = evs[best_type]
            best_prob = combo_probs[best_type]
            
            return {
                'players': players,
                'categories': categories,
                'stat_lines': stat_lines,
                'lines': lines,
                'predictions': [pred['predicted_stat'] for pred in predictions],
                'best_type': best_type,
                'best_ev': best_ev,
                'best_prob': best_prob
            }
            
        except Exception as e:
            print(f"Error processing combination {players}: {e}")
            return None
    
    # Process combinations in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp
    
    results = []
    max_workers = min(mp.cpu_count(), len(valid_combinations))
    
    print(f"Processing {len(valid_combinations)} combinations with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {
            executor.submit(process_combination, combo): i 
            for i, combo in enumerate(valid_combinations)
        }
        
        completed = 0
        for future in as_completed(future_to_combo):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                completed += 1
                
                if completed % 100 == 0:
                    print(f"Completed {completed}/{len(valid_combinations)} combinations")
                    
            except Exception as e:
                print(f"Error in future: {e}")
    
    print(f"Successfully processed {len(results)} combinations")
    
    # Build final results DataFrame
    print("Building final results...")
    all_trios = []
    
    for result in results:
        all_trios.append({
            'PLAYER 1': result['players'][0],
            'CATEGORY 1': result['categories'][0],
            'STAT TYPE 1': result['stat_lines'][0],
            'PLAYER 1 LINE': result['lines'][0],
            'PLAYER 1 PREDICTION': result['predictions'][0],
            'PLAYER 2': result['players'][1],
            'CATEGORY 2': result['categories'][1],
            'STAT TYPE 2': result['stat_lines'][1],
            'PLAYER 2 LINE': result['lines'][1],
            'PLAYER 2 PREDICTION': result['predictions'][1],
            'PLAYER 3': result['players'][2],
            'CATEGORY 3': result['categories'][2],
            'STAT TYPE 3': result['stat_lines'][2],
            'PLAYER 3 LINE': result['lines'][2],
            'PLAYER 3 PREDICTION': result['predictions'][2],
            'TYPE': result['best_type'],
            'EV': round(result['best_ev'], 2),
            'PROBABILITY': round(result['best_prob'], 4),
            'KELLY CRITERION': kelly_criterion(result['best_prob'], payout, stake)
        })
    
    return pd.DataFrame(all_trios)