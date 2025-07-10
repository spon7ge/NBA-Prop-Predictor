# Grab the top evs from 10-22-24 to 1-12-25
# Then they will be used to backtest and see how they would have done
import pandas as pd
from datetime import datetime, timedelta
from Models.model import loadXGBModel
from Models.pipeline import get_espn_games
from PrizePicks.prizePicksPairsEV import prizePicksPairsEV

def format_date_for_csv(date):
    return date.strftime('%m_%d_%Y')

def format_date_for_espn(date):
    return date.strftime('%Y%m%d')

def process_date_range(start_date, end_date, propDict, models):
    current_date = start_date
    while current_date <= end_date:
        print(f"\nProcessing {current_date.date()}")
        
        # Format dates for different uses
        espn_date = format_date_for_espn(current_date)
        csv_date = format_date_for_csv(current_date)
        db_date = current_date.strftime('%Y-%m-%d')
        
        try:
            # Get games for the current date
            games = get_espn_games(date_str=espn_date)
            
            # Check if there are any games for this date
            if not games or len(games) == 0:
                print(f"No games found for {current_date.date()}")
                current_date += timedelta(days=1)
                continue
            
            # Load bookmaker data for the current date
            bookmakers = pd.read_csv('CSV_FILES/HISTORICAL_ODDS/ALL_HISTORICAL_ODDS.csv')
            PrizePicks = bookmakers[(bookmakers['BOOKMAKER'] == 'PrizePicks') & 
                                  (bookmakers['GAME_DATE'] == db_date)]
            
            # Check if we have any PrizePicks data for this date
            if PrizePicks.empty:
                print(f"No PrizePicks data found for {current_date.date()}")
                current_date += timedelta(days=1)
                continue
            
            # Calculate pairs and get top 5 EVs
            pairs = prizePicksPairsEV(PrizePicks, propDict, models, games, 
                                    simulations=10000, stake=100, payout=300)
            
            # Check if we got any pairs
            if pairs.empty:
                print(f"No valid pairs found for {current_date.date()}")
                current_date += timedelta(days=1)
                continue
                
            pairs = pairs.sort_values('EV', ascending=False).head(5).reset_index(drop=True)
            
            # Save to CSV
            output_file = f'CSV_FILES/HISTORICAL_PROP_PAIRS/{espn_date}_PAIRS.csv'
            pairs.to_csv(output_file)
            print(f"Saved top 5 EVs to {output_file}")
            
        except Exception as e:
            print(f"Error processing {current_date.date()}: {str(e)}")
        
        current_date += timedelta(days=1)

# Define the prop dictionary and load models
propDict = {
    'player_points': 'PTS',
    'player_rebounds': 'REB',
    'player_assists': 'AST',
}

models = {
    'PTS': loadXGBModel('PTS'),
    'REB': loadXGBModel('REB'),
    'AST': loadXGBModel('AST'),
}

# Set date range
start_date = datetime(2024, 10, 22)
end_date = datetime(2025, 1, 12)

# Run the processing
process_date_range(start_date, end_date, propDict, models)

import pandas as pd
from Models.model import loadXGBModel
from Models.pipeline import make_prediction

def add_predictions_to_historical(model_type='PTS'):
    """
    Add predicted values to historical features dataset.
    """
    print(f"Loading {model_type} model and data...")
    
    # Load the model
    model = loadXGBModel(model_type)
    
    # Load historical data
    historical_data = pd.read_csv(f'CSV_FILES/REGULAR_DATA/historical_24_{model_type}_features.csv')
    
    # Create a new column for predictions
    historical_data['MODEL_PREDICTION'] = None
    
    # Group by game date to process each day's games together
    for date, group in historical_data.groupby('GAME_DATE'):
        print(f"Processing {date}...")
        
        # Process each player in this date's games
        for idx, row in group.iterrows():
            player = row['PLAYER_NAME']
            opponent = row['OPP_ABBREVIATION']  # Get opponent directly from the data
            
            # Create a mock games structure that make_prediction expects
            games = [{
                'home_team': row['TEAM_ABBREVIATION'] if row['HOME_GAME'] == 1 else opponent,
                'away_team': opponent if row['HOME_GAME'] == 1 else row['TEAM_ABBREVIATION']
            }]
            
            # Create temporary props DataFrame for prediction
            temp_props = pd.DataFrame({
                'NAME': [player],
                'LINE': [row[model_type]],  # Use actual stat as the line
                'CATEGORY': [f'player_{model_type.lower()}']
            })
            
            try:
                # Make prediction
                pred = make_prediction(
                    player_name=player,
                    bookmakers=temp_props,
                    opponent=opponent,
                    model=model,
                    data=historical_data,
                    games=games,
                    is_playoff=0,
                    stat_line=model_type
                )
                
                # Store prediction
                historical_data.loc[idx, 'MODEL_PREDICTION'] = pred['predicted_stat']
                
            except Exception as e:
                print(f"Error predicting for {player} on {date}: {e}")
                continue
    
    # Save the updated dataset
    output_file = f'CSV_FILES/REGULAR_DATA/historical_25_{model_type}_features_with_predictions.csv'
    historical_data.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")
    
    # Print some statistics
    prediction_stats = historical_data[['MODEL_PREDICTION', model_type]].describe()
    print("\nPrediction Statistics:")
    print(prediction_stats)
    
    # Calculate accuracy metrics
    actual = historical_data[model_type]
    predicted = historical_data['MODEL_PREDICTION'].fillna(actual.mean())  # Fill NaN with mean for metrics
    mae = abs(actual - predicted).mean()
    mse = ((actual - predicted) ** 2).mean()
    rmse = mse ** 0.5
    
    print(f"\nAccuracy Metrics:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Square Error: {rmse:.2f}")

# Usage:
add_predictions_to_historical('PTS')  # For points