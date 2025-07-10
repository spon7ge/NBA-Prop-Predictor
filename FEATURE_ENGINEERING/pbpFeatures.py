import pandas as pd
import numpy as np
import re
from nba_api.stats.endpoints import playbyplayv2, boxscoreusagev2

def PlayByPlayOrangized(game_id):
    """
    Enhanced version that includes player IDs and removes team abbreviations
    """
    # Get play-by-play data
    df = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]
    
    # Fill in descriptions, prioritizing home team then visitor
    df['DESCRIPTION'] = df['HOMEDESCRIPTION'].fillna(df['VISITORDESCRIPTION'])
    df['DESCRIPTION'] = df['DESCRIPTION'].fillna(df['NEUTRALDESCRIPTION'])
    
    # Select only the columns we need
    scores = df[['GAME_ID', 'PERIOD', 'PCTIMESTRING', 'DESCRIPTION', 'SCORE',
                 'PLAYER1_NAME', 'PLAYER2_NAME', 'PLAYER3_NAME',
                 'PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID',
                 'EVENTMSGTYPE']].reset_index(drop=True)  # Added EVENTMSGTYPE for better event tracking
    
    # Convert time string to seconds
    scores['SECONDS_REMAINING'] = scores['PCTIMESTRING'].apply(lambda x: 
        int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    
    # Parse and process scores
    scores['HOME_SCORE'] = scores['SCORE'].str.split('-').str[0].astype(float)
    scores['AWAY_SCORE'] = scores['SCORE'].str.split('-').str[1].astype(float)
    
    # Forward fill scores and convert to integers
    scores['HOME_SCORE'] = scores['HOME_SCORE'].ffill().fillna(0).astype(int)
    scores['AWAY_SCORE'] = scores['AWAY_SCORE'].ffill().fillna(0).astype(int)
    
    # Convert player IDs to integers where they exist
    id_columns = ['PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID']
    for col in id_columns:
        scores[col] = pd.to_numeric(scores[col], errors='coerce').astype('Int64')
    
    # Drop original SCORE column since we now have HOME_SCORE and AWAY_SCORE
    scores = scores.drop(columns=['SCORE'])
    
    # Reorder columns for better readability
    scores = scores[['GAME_ID', 'PERIOD', 'PCTIMESTRING', 'SECONDS_REMAINING', 
                    'DESCRIPTION', 'HOME_SCORE', 'AWAY_SCORE',
                    'PLAYER1_NAME', 'PLAYER1_ID',
                    'PLAYER2_NAME', 'PLAYER2_ID',
                    'PLAYER3_NAME', 'PLAYER3_ID',
                    'EVENTMSGTYPE']]
    
    return scores

def parseDescription(pbp_df):
    """
    Parse play descriptions to extract shot details and player actions
    """
    # Initialize columns for shot-related info
    pbp_df = pbp_df.copy()
    pbp_df['PLAYER1_ACTION'] = None
    pbp_df['PLAYER1_SHOT_TYPE'] = None
    pbp_df['PLAYER1_SHOT_OUTCOME'] = None
    pbp_df['PLAYER1_SHOT_DISTANCE'] = None
    pbp_df['PLAYER2_ACTION'] = None
    pbp_df['PLAYER3_ACTION'] = None
    
    for idx, row in pbp_df.iterrows():
        if pd.isna(row['DESCRIPTION']):
            continue
            
        description = str(row['DESCRIPTION'])
        
        # Shot detection
        shot_match = re.search(r"(\w+)\s+(\d+)'\s+([\w\s]+)(?=\s+\()", description)
        if shot_match:
            pbp_df.at[idx, 'PLAYER1_ACTION'] = 'SHOT'
            pbp_df.at[idx, 'PLAYER1_SHOT_DISTANCE'] = int(shot_match.group(2))
            pbp_df.at[idx, 'PLAYER1_SHOT_TYPE'] = shot_match.group(3)
            pbp_df.at[idx, 'PLAYER1_SHOT_OUTCOME'] = 1 if '(PTS)' in description else 0
            
            # Check for assist
            assist_match = re.search(r"\((\w+)\s+(\d+)\s+AST\)", description)
            if assist_match:
                pbp_df.at[idx, 'PLAYER2_ACTION'] = 'ASSIST'
    
    return pbp_df

def get_shot_profile_features(player_id, recent_games_pbp):
    """
    Extract shot selection and efficiency patterns
    """
    features = {}
    
    # Shot distance analysis
    shots = recent_games_pbp[recent_games_pbp['PLAYER1_ID'] == player_id]
    shots = shots[shots['PLAYER1_ACTION'] == 'SHOT']
    
    features['avg_shot_distance_l5'] = shots['PLAYER1_SHOT_DISTANCE'].tail(50).mean()
    features['shot_efficiency_close'] = shots[shots['PLAYER1_SHOT_DISTANCE'] <= 5]['PLAYER1_SHOT_OUTCOME'].mean()
    features['shot_efficiency_mid'] = shots[(shots['PLAYER1_SHOT_DISTANCE'] > 5) & (shots['PLAYER1_SHOT_DISTANCE'] <= 20)]['PLAYER1_SHOT_OUTCOME'].mean()
    features['shot_efficiency_three'] = shots[shots['PLAYER1_SHOT_DISTANCE'] > 20]['PLAYER1_SHOT_OUTCOME'].mean()
    
    # Shot type preferences
    for shot_type in ['Jump Shot', 'Layup', '3PT Field Goal', 'Pullup Jump Shot']:
        type_shots = shots[shots['PLAYER1_SHOT_TYPE'].str.contains(shot_type, na=False)]
        features[f'{shot_type.lower().replace(" ", "_")}_attempts_pg'] = len(type_shots) / shots['GAME_ID'].nunique()
        features[f'{shot_type.lower().replace(" ", "_")}_efficiency'] = type_shots['PLAYER1_SHOT_OUTCOME'].mean()
    
    return features

def get_contextual_features(player_id, recent_games_pbp):
    """
    Performance in different game situations
    Returns detailed quarter-by-quarter stats and other contextual features
    """
    features = {}
    
    # Filter for player's shots
    player_data = recent_games_pbp[
        (recent_games_pbp['PLAYER1_ID'] == player_id) & 
        (recent_games_pbp['PLAYER1_ACTION'] == 'SHOT')
    ]
    
    if len(player_data) == 0:
        return {
            'efficiency_when_leading': 0,
            'efficiency_when_trailing': 0,
            'clutch_efficiency': 0,
            # Quarter stats
            'q1_fg_pct': 0,
            'q1_fga_per_game': 0,
            'q1_fgm_per_game': 0,
            'q2_fg_pct': 0,
            'q2_fga_per_game': 0,
            'q2_fgm_per_game': 0,
            'q3_fg_pct': 0,
            'q3_fga_per_game': 0,
            'q3_fgm_per_game': 0,
            'q4_fg_pct': 0,
            'q4_fga_per_game': 0,
            'q4_fgm_per_game': 0
        }
    
    # Calculate score differential for all plays at once
    player_data['score_diff'] = player_data['HOME_SCORE'] - player_data['AWAY_SCORE']
    
    # Get number of unique games for per-game calculations
    num_games = player_data['GAME_ID'].nunique()
    if num_games == 0:
        num_games = 1  # Prevent division by zero
    
    # Performance when leading/trailing
    leading_shots = player_data[player_data['score_diff'] > 5]
    trailing_shots = player_data[player_data['score_diff'] < -5]
    
    features['efficiency_when_leading'] = round(
        leading_shots['PLAYER1_SHOT_OUTCOME'].mean() 
        if len(leading_shots) > 0 else 0,
        2
    )
    features['efficiency_when_trailing'] = round(
        trailing_shots['PLAYER1_SHOT_OUTCOME'].mean() 
        if len(trailing_shots) > 0 else 0,
        2
    )
    
    # Clutch time performance (last 5 minutes of 4th quarter)
    clutch_data = player_data[
        (player_data['PERIOD'] == 4) & 
        (player_data['SECONDS_REMAINING'] <= 300)
    ]
    features['clutch_efficiency'] = round(
        clutch_data['PLAYER1_SHOT_OUTCOME'].mean() 
        if len(clutch_data) > 0 else 0,
        2
    )
    
    # Quarter-by-quarter detailed stats
    for quarter in range(1, 5):
        quarter_data = player_data[player_data['PERIOD'] == quarter]
        
        # Field Goal Attempts per game
        fga = len(quarter_data)
        fga_per_game = round(fga / num_games, 2)
        
        # Field Goals Made per game
        fgm = quarter_data['PLAYER1_SHOT_OUTCOME'].sum()
        fgm_per_game = round(fgm / num_games, 2)
        
        # Field Goal Percentage
        fg_pct = round(fgm / fga if fga > 0 else 0, 2)
        
        # Store the features
        features[f'q{quarter}_fg_pct'] = fg_pct
        features[f'q{quarter}_fga_per_game'] = fga_per_game
        features[f'q{quarter}_fgm_per_game'] = fgm_per_game
    
    return features

def get_momentum_features(recent_games_pbp, player_id):
    """
    Scoring streaks and momentum indicators
    All values are converted to standard Python types and rounded to 2 decimals
    """
    features = {}
    
    # Filter for player's shots
    shots = recent_games_pbp[
        (recent_games_pbp['PLAYER1_ID'] == player_id) & 
        (recent_games_pbp['PLAYER1_ACTION'] == 'SHOT')
    ].sort_values(['GAME_ID', 'PERIOD', 'SECONDS_REMAINING'])
    
    if len(shots) == 0:
        return {
            'hot_hand_frequency': 0.0,
            'cold_streak_frequency': 0.0,
            'recent_shooting_trend': 0.0,
            'last_5_shots_made': 0,
            'avg_consecutive_makes_entering_game': 0.0
        }
    
    # Hot hand detection (3+ consecutive makes)
    shots['rolling_makes'] = shots['PLAYER1_SHOT_OUTCOME'].rolling(window=3, min_periods=1).sum()
    features['hot_hand_frequency'] = float(round(
        (shots['rolling_makes'] >= 3).mean() if len(shots) >= 3 else 0,
        2
    ))
    
    # Cold streak detection (3+ consecutive misses)
    shots['rolling_misses'] = (1 - shots['PLAYER1_SHOT_OUTCOME']).rolling(window=3, min_periods=1).sum()
    features['cold_streak_frequency'] = float(round(
        (shots['rolling_misses'] >= 3).mean() if len(shots) >= 3 else 0,
        2
    ))
    
    # Recent shooting trend (last 10 shots vs previous 10)
    if len(shots) >= 20:
        last_10 = shots.tail(10)['PLAYER1_SHOT_OUTCOME'].mean()
        prev_10 = shots.tail(20).head(10)['PLAYER1_SHOT_OUTCOME'].mean()
        features['recent_shooting_trend'] = float(round(last_10 - prev_10, 2))
    else:
        features['recent_shooting_trend'] = 0.0
    
    # Last 5 shots made
    features['last_5_shots_made'] = int(round(
        shots.tail(5)['PLAYER1_SHOT_OUTCOME'].sum() if len(shots) >= 5 else 0,
        0
    ))
    
    # Consecutive makes entering each game
    if len(shots) > 0:
        game_starts = []
        for game_id in shots['GAME_ID'].unique():
            game_shots = shots[shots['GAME_ID'] == game_id]
            if len(game_shots) > 0:
                consecutive = 0
                for outcome in game_shots['PLAYER1_SHOT_OUTCOME']:
                    if outcome == 1:
                        consecutive += 1
                    else:
                        break
                game_starts.append(consecutive)
        features['avg_consecutive_makes_entering_game'] = float(round(
            np.mean(game_starts) if game_starts else 0,
            2
        ))
    else:
        features['avg_consecutive_makes_entering_game'] = 0.0
    
    return features