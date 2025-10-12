import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import truncnorm
from nba_api.stats.endpoints import scoreboardv2


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

def findOpp(playerName, players_df, gameDate):
    player_team_id = players_df.loc[
        players_df['PLAYER_NAME'] == playerName, 'TEAM_ID'
    ].iloc[-1]
    
    scoreboard = scoreboardv2.ScoreboardV2(
    game_date=gameDate,
    league_id='00').get_data_frames()[2]
    
    row = scoreboard[
        (scoreboard['HOME_TEAM_ID'] == player_team_id) |
        (scoreboard['VISITOR_TEAM_ID'] == player_team_id)]
    
    if row.empty:
        print(f"No game found for {playerName} on {gameDate}")
        return None
    
    row = row.iloc[0]
    
    if player_team_id == row['HOME_TEAM_ID']:
        opp_team_id = row['VISITOR_TEAM_ID']
    else:
        opp_team_id = row['HOME_TEAM_ID']
    
    return int(opp_team_id)

def calculate_opponent_features(player_name, game_date, historical_data, player_data_row):    
    # Get opponent team ID
    opp_team_id = findOpp(player_name, historical_data, game_date)
    if opp_team_id is None:
        print(f"Could not find opponent for {player_name} on {game_date}")
        return player_data_row
    
    # Filter opponent's historical data (games before the prediction date)
    opp_history = historical_data[
        (historical_data['TEAM_ID'] == opp_team_id) & 
        (historical_data['GAME_DATE'] < game_date)
    ].copy()
    
    if opp_history.empty:
        print(f"No historical data for opponent team {opp_team_id}")
        return player_data_row
    
    updated_row = player_data_row.copy()
    
    # ==========================================
    # 1. OPPONENT POSITION-SPECIFIC DEFENSE
    # ==========================================
    positions = ['GUARD', 'FORWARD', 'CENTER']
    def_metrics = ['DEF_FG_PCT_ALLOWED', 'DEF_3PT_PCT_ALLOWED', 'PTS_ALLOWED_PER_MIN']
    
    for pos in positions:
        pos_players = opp_history[opp_history[pos] == 1]
        
        for metric in def_metrics:
            if metric in pos_players.columns:
                # Calculate mean from all available games
                avg_val = pos_players[metric].mean()
                feature_name = f'OPP_{pos}_{metric}'
                updated_row[feature_name] = round(avg_val, 3) if not pd.isna(avg_val) else updated_row.get(feature_name, 0)
    
    # ==========================================
    # 2. OPPONENT TEAM AGGREGATED STATS
    # ==========================================
    team_stats = {
        'OPP_DEF_RATING_AVG_TO_DATE': 'TEAM_DEF_RATING',
        'OPP_PACE_AVG_TO_DATE': 'TEAM_PACE',
        'OPP_PTS_AVG_TO_DATE': 'TEAM_PTS',
        'OPP_PTS_PAINT_AVG_TO_DATE': 'PTS_PAINT',
        'OPP_FGA_AVG_TO_DATE': 'TEAM_FGA',
        'OPP_REB_AVG_TO_DATE': 'TEAM_REB',
        'OPP_AST_AVG_TO_DATE': 'TEAM_AST',
        'OPP_TOV_AVG_TO_DATE': 'TEAM_TOV',
        'OPP_BLK_AVG_TO_DATE': 'TEAM_BLK',
        'OPP_STL_AVG_TO_DATE': 'TEAM_STL'
    }
    
    # Get unique games for the opponent team
    opp_games = opp_history.drop_duplicates(subset=['GAME_ID']).sort_values('GAME_DATE')
    
    for feature_name, source_col in team_stats.items():
        if source_col in opp_games.columns:
            avg_val = opp_games[source_col].mean()
            updated_row[feature_name] = round(avg_val, 2) if not pd.isna(avg_val) else updated_row.get(feature_name, 0)
    
    # Paint defense specifically (average of all opponent players)
    if 'OPP_PTS_PAINT' in opp_history.columns:
        updated_row['OPP_PTS_PAINT'] = round(opp_history['OPP_PTS_PAINT'].mean(), 2)
    
    # ==========================================
    # 3. OPPONENT MATCHUP-SPECIFIC FEATURES
    # ==========================================
    # Get player's previous games against this opponent
    player_vs_opp = historical_data[
        (historical_data['PLAYER_NAME'] == player_name) &
        (historical_data['OPP_TEAM_ID'] == opp_team_id) &
        (historical_data['GAME_DATE'] < game_date)
    ].sort_values('GAME_DATE')
    
    if not player_vs_opp.empty:
        # Last 3 games USG_PCT average
        last_3 = player_vs_opp.tail(3)
        if 'USG_PCT' in last_3.columns:
            updated_row['MATCHUP_AVG_USG_PCT_LAST_3_TO_DATE'] = round(last_3['USG_PCT'].mean(), 2)
        
        # Last 5 games USG_PCT average
        last_5 = player_vs_opp.tail(5)
        if 'USG_PCT' in last_5.columns:
            updated_row['MATCHUP_AVG_USG_PCT_LAST_5_TO_DATE'] = round(last_5['USG_PCT'].mean(), 2)
    
    # ==========================================
    # 4. INTERACTION FEATURES
    # ==========================================
    # Recalculate interaction features using new opponent stats
    if 'percentageFieldGoalsAttempted3pt_AVG_TO_DATE' in updated_row.index and 'OPP_GUARD_DEF_3PT_PCT_ALLOWED' in updated_row.index:
        updated_row['PLAYER_3PT_X_OPP_3PT_DEF'] = (
            updated_row['percentageFieldGoalsAttempted3pt_AVG_TO_DATE'] * 
            updated_row['OPP_GUARD_DEF_3PT_PCT_ALLOWED']
        )
    
    if 'percentageFieldGoalsAttempted3pt_ROLLING_AVG_5' in updated_row.index and 'OPP_GUARD_DEF_3PT_PCT_ALLOWED' in updated_row.index:
        updated_row['PLAYER_3PT_X_OPP_3PT_DEF_RECENT'] = (
            updated_row['percentageFieldGoalsAttempted3pt_ROLLING_AVG_5'] * 
            updated_row['OPP_GUARD_DEF_3PT_PCT_ALLOWED']
        )
    
    if 'percentagePointsPaint_AVG_TO_DATE' in updated_row.index and 'OPP_PTS_PAINT' in updated_row.index:
        updated_row['PLAYER_PAINT_X_OPP_PAINT_DEF'] = (
            updated_row['percentagePointsPaint_AVG_TO_DATE'] * 
            updated_row['OPP_PTS_PAINT']
        )
    
    if 'percentagePointsPaint_ROLLING_AVG_5' in updated_row.index and 'OPP_PTS_PAINT' in updated_row.index:
        updated_row['PLAYER_PAINT_X_OPP_PAINT_DEF_RECENT'] = (
            updated_row['percentagePointsPaint_ROLLING_AVG_5'] * 
            updated_row['OPP_PTS_PAINT']
        )
    
    if 'percentagePointsMidrange2pt_AVG_TO_DATE' in updated_row.index and 'OPP_FORWARD_DEF_FG_PCT_ALLOWED' in updated_row.index:
        updated_row['PLAYER_MID_X_OPP_MID_DEF'] = (
            updated_row['percentagePointsMidrange2pt_AVG_TO_DATE'] * 
            updated_row['OPP_FORWARD_DEF_FG_PCT_ALLOWED']
        )
    
    # Team offensive vs opponent defensive rating
    if 'TEAM_OFF_RATING_AVG_TO_DATE' in updated_row.index and 'OPP_DEF_RATING_AVG_TO_DATE' in updated_row.index:
        updated_row['TEAM_OFF_MINUS_OPP_DEF'] = (
            updated_row['TEAM_OFF_RATING_AVG_TO_DATE'] - 
            updated_row['OPP_DEF_RATING_AVG_TO_DATE']
        )
    
    return updated_row


def predictPTS(playerName, data, model, features, gameDate):
    # Get player's most recent data
    playerData = data[data['PLAYER_NAME'] == playerName].copy()
    latestRow = playerData.sort_values(by='GAME_DATE').iloc[-1]
    
    # Recalculate opponent features for the upcoming game
    updated_row = calculate_opponent_features(
        player_name=playerName,
        game_date=gameDate,
        historical_data=data,
        player_data_row=latestRow
    )
    
    # Filter to only the features needed by the model
    available_features = [f for f in features if f in updated_row.index]
    playerInput = updated_row[available_features]
    
    # Convert to DataFrame for prediction
    playerInput_df = pd.DataFrame([playerInput.values], columns=available_features)
    
    # Handle data types
    for col in playerInput_df.columns:
        if playerInput_df[col].dtype == 'object':
            playerInput_df[col] = pd.to_numeric(playerInput_df[col], errors='coerce').fillna(0)
        elif playerInput_df[col].dtype == 'bool':
            playerInput_df[col] = playerInput_df[col].astype(int)
    
    # Make prediction
    pred = model.predict(playerInput_df)[0]
    return round(float(pred), 3)

def predictPTS(playerName, data, model, features):
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

def monteCarloSim(player_df, modelPred, prop_line, std_dev, num_simulations=1000, min_std=2.0, max_std=8.5):
    baseSTD = std_dev
    volAdj = 1.0  
    
    if 'PTS_EXPANDING_VOLATILITY_TO_DATE' in player_df.columns and len(player_df) >= 5:
        recent_vol = player_df['PTS'].tail(5).std()  
        season_vol = player_df['PTS_EXPANDING_VOLATILITY_TO_DATE'].iloc[-1]  

        if season_vol > 0 and not pd.isna(season_vol) and not pd.isna(recent_vol):
            volAdj = recent_vol / season_vol  
            volAdj = 1 + (volAdj - 1) * 0.3 
            volAdj = np.clip(volAdj, 0.7, 2.0)  
        
        stdDev = baseSTD * volAdj
    else:
        stdDev = baseSTD
    
    stdDev = float(np.clip(stdDev, min_std, max_std))  
    
    # Create truncated normal distribution (lower bound = 0)
    a = -modelPred / stdDev if stdDev > 0 else 0
    b = np.inf
    simulated_points = truncnorm.rvs(a, b, loc=modelPred, scale=stdDev, size=num_simulations)
    
    prob_over = np.mean(simulated_points > prop_line)
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


def single_bet(data, bookmakers, model, features, edge_threshold=4.5, stake=100, simulations=10000, 
               std_window=10, min_std=2.0, max_std=8.5, stat_col='PTS'):

    print("Processing single bets...")

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

    def impliedProb(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    results = []

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

        # Get prediction using your predictPTS function
        try:
            prediction = predictPTS(name, data, model, features)
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue

        std_dev = get_player_std(player_df, stat_col)

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

    
def prizepickspairsEV(data, bookmakers, model, features, edge_threshold=4.5, stake=100,
                      simulations=10000, std_window=10, min_std=2.0, max_std=8.5, stat_col='PTS'):
    
    print("Processing pairs...")

    def get_player_std(player_df, stat_col):
        s = player_df[stat_col].dropna()
        if s.empty:
            return 5.0
        sd = s.tail(std_window).std(ddof=1) if len(s) >= std_window else s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            sd = 5.0
        return float(np.clip(sd, min_std, max_std))

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

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        player_team = player_df['TEAM_ABBREVIATION'].iloc[0]

        # Get prediction using your predictPTS function
        try:
            prediction = predictPTS(name, data, model, features)
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue

        std_dev = get_player_std(player_df, stat_col)
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

        edge = model_prob - line
        if abs(edge) > edge_threshold:
            recommendation = 1
        else:
            recommendation = 0

        legs.append({
            'NAME': name,
            'TEAM': player_team,
            'CATEGORY': category,
            'BOOKMAKER': bookmaker,
            'ODDS': odds,
            'LINE': line,
            'SIDE': side,  
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

            ev_per_unit = p_both * b - (1 - p_both)
            ev_percent = ev_per_unit

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
                'SIDE 1': leg1['SIDE'],  
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
                'SIDE 2': leg2['SIDE'],  
                'PREDICTION 2': round(leg2['PREDICTION'], 2),
                'MODEL_SIDE 2': leg2['MODEL_SIDE'],
                'OVER% 2': round(leg2['OVER%'], 3),
                'UNDER% 2': round(leg2['UNDER%'], 3),
                'CONFIDENCE INTERVAL 2': f"({leg2['CI'][0]:.1f}, {leg2['CI'][1]:.1f})",
                'RECOMMENDED_TYPE': f"{leg1['MODEL_SIDE']}/{leg2['MODEL_SIDE']}",
                'RECOMMENDATION': recommendation,
                'PROBABILITY': round(p_both, 4),
                'EV%': round(ev_percent, 3),
                'KELLY': round(kelly_full, 3),
            })

    return pd.DataFrame(pair_results)

def prizepicks3LegEV(data, bookmakers, model, features, edge_threshold=4.5, stake=100,
                     simulations=10000, std_window=10, min_std=2.0, max_std=8.5, stat_col='PTS'):
    
    print("Processing 3-leg parlays...")

    def get_player_std(player_df, stat_col):
        s = player_df[stat_col].dropna()
        if s.empty:
            return 5.0
        sd = s.tail(std_window).std(ddof=1) if len(s) >= std_window else s.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            sd = 5.0
        return float(np.clip(sd, min_std, max_std))

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

        player_df = data[data['PLAYER_NAME'] == name].sort_values(by='GAME_DATE', ascending=False)
        if player_df.empty or stat_col not in player_df.columns:
            continue

        player_team = player_df['TEAM_ABBREVIATION'].iloc[0]

        # Get prediction
        try:
            prediction = predictPTS(name, data, model, features)
        except Exception as e:
            print(f"Error getting prediction for {name}: {e}")
            continue

        std_dev = get_player_std(player_df, stat_col)
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

        edge = model_prob - line
        legs.append({
            'NAME': name,
            'TEAM': player_team,
            'CATEGORY': category,
            'BOOKMAKER': bookmaker,
            'ODDS': odds,
            'LINE': line,
            'SIDE': side,  
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

                # Expected value calculation
                ev_per_unit = p_all_three * b - (1 - p_all_three)
                ev_percent = ev_per_unit

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
                    'SIDE 1': leg1['SIDE'],
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
                    'SIDE 2': leg2['SIDE'],
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
                    'SIDE 3': leg3['SIDE'],
                    'PREDICTION 3': round(leg3['PREDICTION'], 2),
                    'MODEL_SIDE 3': leg3['MODEL_SIDE'],
                    'OVER% 3': round(leg3['OVER%'], 3),
                    'UNDER% 3': round(leg3['UNDER%'], 3),
                    'CONFIDENCE INTERVAL 3': f"({leg3['CI'][0]:.1f}, {leg3['CI'][1]:.1f})",
                    
                    'RECOMMENDED_TYPE': f"{leg1['MODEL_SIDE']}/{leg2['MODEL_SIDE']}/{leg3['MODEL_SIDE']}",
                    'RECOMMENDATION': recommendation,
                    'PROBABILITY': round(p_all_three, 4),
                    'EV%': round(ev_percent, 3),
                    'KELLY': round(kelly_full, 3),
                })

    return pd.DataFrame(parlay_results)