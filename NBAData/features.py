import pandas as pd
import numpy as np

#grabbing play by play data
import re
from nba_api.stats.endpoints import PlayByPlayV2
def PlayByPlayOrangized(game_id):
    df = PlayByPlayV2(game_id=game_id).get_data_frames()[0]
    df['DESCRIPTION'] = df['HOMEDESCRIPTION'].fillna(df['VISITORDESCRIPTION'])
    df['DESCRIPTION'] = df['DESCRIPTION'].fillna(df['NEUTRALDESCRIPTION'])
    
    scores = df[['GAME_ID','PERIOD','PCTIMESTRING','DESCRIPTION','SCORE', 'PLAYER1_NAME', 'PLAYER2_NAME', 'PLAYER3_NAME','PLAYER1_TEAM_ABBREVIATION','PLAYER2_TEAM_ABBREVIATION','PLAYER3_TEAM_ABBREVIATION','PLAYER1_ID','PLAYER2_ID','PLAYER3_ID']].reset_index(drop=True)
    scores['SECONDS_REMAINING'] = scores['PCTIMESTRING'].apply(lambda x: 
    int(x.split(':')[0]) * 60 + int(x.split(':')[1])
    )
    scores['HOME_SCORE'] = scores['SCORE'].str.split('-').str[0].astype(float)
    scores['AWAY_SCORE'] = scores['SCORE'].str.split('-').str[1].astype(float)

    scores['HOME_SCORE'] = scores['HOME_SCORE'].ffill().fillna(0).astype(int)
    scores['AWAY_SCORE'] = scores['AWAY_SCORE'].ffill().fillna(0).astype(int)

    scores = scores.drop(columns=['SCORE'])
    scores = scores[['GAME_ID', 'PERIOD', 'PCTIMESTRING', 'SECONDS_REMAINING', 'DESCRIPTION', 'HOME_SCORE', 'AWAY_SCORE', 'PLAYER1_NAME', 'PLAYER2_NAME', 'PLAYER3_NAME','PLAYER1_TEAM_ABBREVIATION','PLAYER2_TEAM_ABBREVIATION','PLAYER3_TEAM_ABBREVIATION']]
    return scores

#grabbing action types from description
def parseDescription(pbp_df):
    # Initialize columns - only PLAYER1 gets shot-related columns
    pbp_df['PLAYER1_NAME'] = None
    pbp_df['PLAYER1_ACTION'] = None
    pbp_df['PLAYER1_SHOT_TYPE'] = None
    pbp_df['PLAYER1_SHOT_OUTCOME'] = None
    pbp_df['PLAYER1_DISTANCE'] = None
    
    # Other players just need name and action
    pbp_df['PLAYER2_NAME'] = None
    pbp_df['PLAYER2_ACTION'] = None
    
    pbp_df['PLAYER3_NAME'] = None
    pbp_df['PLAYER3_ACTION'] = None
    
    for idx, description in enumerate(pbp_df['DESCRIPTION']):
        if pd.isna(description):
            continue
            
        # Primary action (shots)
        shot_match = re.search(r"(\w+)\s+(\d+)'\s+([\w\s]+)(?=\s+\()", str(description))
        if shot_match:
            pbp_df.loc[idx, 'PLAYER1_NAME'] = shot_match.group(1)  # Player name
            pbp_df.loc[idx, 'PLAYER1_DISTANCE'] = int(shot_match.group(2))  # Shot distance
            pbp_df.loc[idx, 'PLAYER1_SHOT_TYPE'] = shot_match.group(3)  # Shot type
            pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'SHOT'
            pbp_df.loc[idx, 'PLAYER1_SHOT_OUTCOME'] = 1 if re.search(r"\(\d+ PTS\)", description) else 0
            
            # Check for assist
            assist_match = re.search(r"\((\w+)\s+(\d+)\s+AST\)", str(description))
            if assist_match:
                pbp_df.loc[idx, 'PLAYER2_NAME'] = assist_match.group(1)
                pbp_df.loc[idx, 'PLAYER2_ACTION'] = 'ASSIST'
    
        
        # Handle non-shot actions
        elif 'REBOUND' in str(description):
            rebound_match = re.search(r"(\w+)\s+REBOUND", str(description))
            if rebound_match:
                pbp_df.loc[idx, 'PLAYER1_NAME'] = rebound_match.group(1)
                pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'REBOUND'
                
        elif 'STEAL' in str(description):
            steal_match = re.search(r"(\w+)\s+STEAL", str(description))
            if steal_match:
                pbp_df.loc[idx, 'PLAYER1_NAME'] = steal_match.group(1)
                pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'STEAL'
                
        elif 'BLOCK' in str(description):
            block_match = re.search(r"(\w+)\s+BLOCK", str(description))
            if block_match:
                pbp_df.loc[idx, 'PLAYER1_NAME'] = block_match.group(1)
                pbp_df.loc[idx, 'PLAYER1_ACTION'] = 'BLOCK'
            
    return pbp_df

#grabs players rest days between games
def calculateDaysOfRest(df, player_id_col='PLAYER_ID', game_date_col='GAME_DATE'):
    df[game_date_col] = pd.to_datetime(df[game_date_col], format='%Y-%m-%d')
    df = df.sort_values(by=[player_id_col, game_date_col])
    df['DAYS_OF_REST'] = df.groupby(player_id_col)[game_date_col].diff().dt.days
    return df

def convert_height_to_inches(height_str):
    if pd.isna(height_str):
        return np.nan
    # Split the string into feet and inches
    feet, inches = map(int, height_str.split('-'))
    # Convert to total inches
    return (feet * 12) + inches
    
# only for the playoffs
def assign_playoff_series_info(df):
    df = df.copy()
    # Ensure GAME_DATE is in datetime format
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Create a consistent matchup key regardless of home/away
    df['MATCHUP_KEY'] = df.apply(
        lambda x: '-'.join(sorted([x['TEAM_ABBREVIATION'], x['OPP_ABBREVIATION']])), axis=1
    )

    # Get unique games to avoid player duplicates
    unique_games = df.drop_duplicates(subset=['GAME_ID'])
    
    # Sort by matchup and date
    unique_games = unique_games.sort_values(by=['GAME_DATE', 'MATCHUP_KEY'])
    
    # Initialize team tracking
    team_series = {}
    team_opponents = {}
    
    # First pass: Track all opponents for each team
    for _, game in unique_games.iterrows():
        team1, team2 = game['TEAM_ABBREVIATION'], game['OPP_ABBREVIATION']
        date = game['GAME_DATE']
        
        # Initialize if not exists
        if team1 not in team_opponents:
            team_opponents[team1] = []
        if team2 not in team_opponents:
            team_opponents[team2] = []
            
        # Add opponent to list if it's a new matchup
        if team2 not in team_opponents[team1]:
            team_opponents[team1].append(team2)
        if team1 not in team_opponents[team2]:
            team_opponents[team2].append(team1)
    
    # Determine series number for each team
    for team in team_opponents:
        team_series[team] = len(team_opponents[team])
    
    # Apply series numbers to the full dataset
    def get_series_number(row):
        team = row['TEAM_ABBREVIATION']
        opp = row['OPP_ABBREVIATION']
        
        # Get the opponent's index in the team's opponent list (1-based)
        try:
            series_num = team_opponents[team].index(opp) + 1
        except ValueError:
            series_num = 0
            
        return series_num
    
    # Apply series number and game number within series
    df['Series'] = df.apply(get_series_number, axis=1)
    df['GameInSeries'] = df.groupby(['MATCHUP_KEY'])['GAME_DATE'].rank(method='dense').astype(int)
    
    # Add series name for better context
    def get_series_name(series_num):
        series_names = {
            1: 'First Round',
            2: 'Conference Semifinals',
            3: 'Conference Finals',
            4: 'NBA Finals'
        }
        return series_names.get(series_num, 'Unknown')
    
    df['SeriesName'] = df['Series'].map(get_series_name)
    
    return df

#rolling averages for points against each team
def statAgainstTeam(player_data, player_id_col='PLAYER_ID', opp_col='OPP_ABBREVIATION', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, 'GAME_DATE'])
    
    # Calculate recent average points (last 3 games)
    player_data[f'MATCHUP_AVG_{stat_line}_LAST_3'] = (
        player_data.groupby([player_id_col, opp_col])[stat_line]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean().round(2))
    )
    
    # Count number of games against this team
    player_data['GAMES_VS_OPP'] = player_data.groupby([player_id_col, opp_col]).cumcount() + 1
    
    return player_data

########################################################################################
#rolling averages
########################################################################################

def rollingAverages(player_data, rolling_windows, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, date_col]).copy()
    rolling_features = {
        'PTS': [
            'MIN', 'PTS', 'FGA', 'FGM', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'USG_PCT', 'TS_PCT', 'EFG_PCT',
            'OREB', 'DREB', 'REB', 'PLUS_MINUS', 'PIE',
            'TEAM_FGA', 'TEAM_FG_PCT', 'TEAM_FG3A', 'TEAM_FG3_PCT',
            'TEAM_FTM', 'TEAM_FTA', 'TEAM_FT_PCT',
            'TEAM_PTS', 'TEAM_PACE', 'TEAM_OFF_RATING',
            'OPP_DEF_RATING', 'OPP_PACE', 'OPP_FG_PCT'
        ],
        'AST': [
            'MIN', 'AST', 'FGA', 'FGM', 'FG_PCT', 'FG3A', 'FG3M', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'USG_PCT', 'AST_PCT', 'AST_TOV', 'TS_PCT', 'EFG_PCT', 'PIE', 'PLUS_MINUS',
            'TEAM_FG_PCT', 'TEAM_FGM', 'TEAM_AST', 'TEAM_TOV', 'TEAM_PACE', 'TEAM_PTS',
            'OPP_DEF_RATING', 'OPP_STL', 'OPP_PACE'
        ],
        'REB': [
            'MIN', 'OREB', 'DREB', 'REB', 'FGA', 'FGM', 'FG_PCT',
            'FG3A', 'FG3M', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'PIE', 'PLUS_MINUS',
            'USG_PCT', 'TS_PCT', 'EFG_PCT', 'PACE', 'POSS',
            'TEAM_FG_PCT', 'TEAM_FG3_PCT', 'TEAM_FGA', 'TEAM_FG3A',
            'OPP_REB', 'OPP_FG_PCT', 'OPP_DEF_RATING', 'OPP_PACE'
        ]
    }

    if stat_line not in rolling_features:
        raise ValueError(f"Invalid stat_line: {stat_line}. Must be one of {list(rolling_features.keys())}")

    for window in rolling_windows:
        for feature in rolling_features[stat_line]:
            roll_avg_col = f'{feature}_ROLL_AVG_{window}'

            # Compute rolling average (no leakage)
            player_data[roll_avg_col] = (
                player_data
                .groupby(player_id_col)[feature]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean().round(2))
            )

            # Fill missing values: bfill → ffill → global mean
            global_mean = player_data[roll_avg_col].mean()
            player_data[roll_avg_col] = (
                player_data[roll_avg_col]
                .bfill()
                .ffill()
                .fillna(global_mean)
            )

            # Add rolling std for PTS only
            if stat_line == 'PTS' and feature == 'PTS':
                std_col = f'{feature}_STD_AVG_{window}'
                player_data[std_col] = (
                    player_data
                    .groupby(player_id_col)[feature]
                    .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().round(2))
                )
                global_std = player_data[std_col].mean()
                player_data[std_col] = (
                    player_data[std_col]
                    .bfill()
                    .ffill()
                    .fillna(global_std)
                )

    return player_data

def HomeAwayAverages(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, date_col]).copy()
    
    for home_away in ['HOME', 'AWAY']:
        avg_column_name = f'PLAYER_{home_away}_AVG_{stat_line}'

        def compute_expanding_avg(group):
            mask = group['HOME_GAME'] == (1 if home_away == 'HOME' else 0)
            result = pd.Series(index=group.index, dtype='float64')
            result[mask] = group.loc[mask, stat_line].expanding().mean().round(2)
            if home_away == 'AWAY':
                result = result.bfill().ffill()
            elif home_away == 'HOME':
                result = result.bfill().ffill()
            return result

        player_data[avg_column_name] = (
            player_data.groupby(player_id_col)
            .apply(compute_expanding_avg)
            .reset_index(level=0, drop=True)
        )

    return player_data

def addLagFeatures(player_data, player_id_col='PLAYER_ID', date_col='GAME_DATE', stat_line='PTS'):
    player_data = player_data.sort_values([player_id_col, date_col])
    for lag in range(1,5):
        player_data[f'{stat_line}_LAG_{lag}'] = player_data.groupby(player_id_col)[stat_line].shift(lag)
    return player_data

def defenseRollingAverage(df):
    df = df.sort_values(by='GAME_DATE', ascending=True).reset_index(drop=True)
    rolling_df = (
    df.groupby('OPP_ABBREVIATION')
      .apply(lambda group: group.assign(
          ROLL_OPP_DEF_RATING=group['OPP_DEF_RATING']
              .expanding(min_periods=1)
              .mean()
      ))
      .reset_index(drop=True)
    )
    return rolling_df

def categorize_by_rank(df, opp_col='OPP_ABBREVIATION', strength_col='ROLL_OPP_DEF_RATING', top_n=10):
    """
    Assigns a binary defense category (1=strong, 0=weak) based on team-level defensive strength rank.
    Each unique opponent gets one consistent label across all games.
    """
    # Calculate average defense strength per team
    team_strength = df.groupby(opp_col)[strength_col].mean().reset_index()
    # Rank teams: 1 is strongest defense
    team_strength['DEF_RANK'] = team_strength[strength_col].rank(ascending=True, method='min')
    # Assign category: top_n strongest defenses → 1
    team_strength['DEF_CATEGORY'] = (team_strength['DEF_RANK'] <= top_n).astype(int)
    # Merge back to main dataframe so every row for same team gets consistent label
    df = df.merge(team_strength[[opp_col, 'DEF_RANK', 'DEF_CATEGORY']], on=opp_col, how='left')
    return df

def CalculatePlayerVsDefense(player_data, player_id_col='PLAYER_ID', stat_line='PTS'):
    """
    Calculate how players perform against different defensive categories
    """
    # Calculate average performance against each defensive category
    metrics = {
        'PTS': ['PTS','FGA', 'FTA', 'FG3A', 'USG_PCT','TOV'],
        'AST': ['AST','AST_PCT', 'AST_TOV', 'USG_PCT', 'PACE', 'POSS', 'OFF_RATING','TOV'],
        'REB': ['REB','OREB', 'DREB', 'REB_PCT', 'USG_PCT', 'GAME_PACE','TOV']
    }
    
    for metric in metrics[stat_line]:
        # Calculate average against strong and weak defenses
        avg_by_def = (
            player_data.groupby([player_id_col, 'DEF_CATEGORY'])[metric]
            .transform('mean')
            .round(2)
        )
        player_data[f'{metric}_VS_DEF'] = (
            (player_data[metric] - avg_by_def) / avg_by_def
        ).round(3)

    return player_data

def add_all_opponent_features(player_data, stat_line='PTS'):
    """
    Wrapper function to add all opponent-related features
    """
    player_data = defenseRollingAverage(player_data)
    player_data = categorize_by_rank(player_data)
    player_data = CalculatePlayerVsDefense(player_data, stat_line=stat_line)
    return player_data


########################################################################################
#lineup composition features
########################################################################################
def teamUsualStarters(df):
    """
    Adds to df:
    - TEAM_STARTER_OFF_RATING_AVG, TEAM_STARTER_DEF_RATING_AVG, TEAM_STARTER_USG_PCT_AVG
    - NUM_USUAL_STARTERS_PRESENT: number of usual starters present for own team
    """
    # 1) Compute usual starters: top 5 most frequent starters per team
    player_starts = (
        df[df['STARTING'] == 1]
        .groupby(['TEAM_ID', 'PLAYER_ID'])
        .size()
        .reset_index(name='NUM_STARTS')
    )
    usual_starters = (
        player_starts
        .sort_values(['TEAM_ID', 'NUM_STARTS'], ascending=[True, False])
        .groupby('TEAM_ID')
        .head(5)
    )
    usual_starters_dict = (
        usual_starters
        .groupby('TEAM_ID')['PLAYER_ID']
        .apply(set)
        .to_dict()
    )

    # 2) Compute NUM_USUAL_STARTERS_PRESENT for each game-team
    starters_per_game = (
        df[df['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PLAYER_ID': list})
        .reset_index()
    )
    def count_usual_starters(row):
        team_id = row['TEAM_ID']
        actual_starters = set(row['PLAYER_ID'])
        usual_starters = usual_starters_dict.get(team_id, set())
        return len(actual_starters & usual_starters)
    
    starters_per_game['NUM_USUAL_STARTERS_PRESENT'] = starters_per_game.apply(count_usual_starters, axis=1)
    
    # 3) Compute team starter averages (OFF/DEF/USG)
    starters_df = df[df['STARTING'] == 1].copy()
    team_starter_features = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({
            'OFF_RATING': 'mean',
            'DEF_RATING': 'mean',
            'USG_PCT': 'mean'
        })
        .rename(columns={
            'OFF_RATING': 'TEAM_STARTER_OFF_RATING_AVG',
            'DEF_RATING': 'TEAM_STARTER_DEF_RATING_AVG',
            'USG_PCT': 'TEAM_STARTER_USG_PCT_AVG'
        })
        .reset_index()
    )

    # 4) Merge NUM_USUAL_STARTERS_PRESENT into team starter features
    team_starter_features = team_starter_features.merge(
        starters_per_game[['GAME_ID', 'TEAM_ID', 'NUM_USUAL_STARTERS_PRESENT']],
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    # 5) Merge combined features into main df
    df = df.merge(
        team_starter_features,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    return df

def oppTeamUsualStarters(df):
    """
    Adds opponent-side starter features:
    - NUM_USUAL_STARTERS_PRESENT_OPP: count of opponent usual starters present
    - OPP_STARTER_AVG_DEF_RATING: average DEF_RATING of opponent starters
    - OPP_GUARDS_AVG_DEF_RATING_OPP: average DEF_RATING of opponent starting guards
    - OPP_FORWARDS_AVG_DEF_RATING_OPP: average DEF_RATING of opponent starting forwards
    """

    # 1) Compute usual starters: top 5 most frequent starters per team
    player_starts = (
        df[df['STARTING'] == 1]
        .groupby(['TEAM_ID', 'PLAYER_ID'])
        .size()
        .reset_index(name='NUM_STARTS')
    )
    usual_starters = (
        player_starts
        .sort_values(['TEAM_ID', 'NUM_STARTS'], ascending=[True, False])
        .groupby('TEAM_ID')
        .head(5)
    )
    usual_starters_dict = (
        usual_starters
        .groupby('TEAM_ID')['PLAYER_ID']
        .apply(set)
        .to_dict()
    )

    # 2) Compute NUM_USUAL_STARTERS_PRESENT for opponent team
    opp_actual_starters_per_game = (
        df[df['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PLAYER_ID': list})
        .reset_index()
    )
    def count_opp_usual_starters(row):
        team_id = row['TEAM_ID']
        actual_starters = set(row['PLAYER_ID'])
        usual_starters = usual_starters_dict.get(team_id, set())
        return len(actual_starters & usual_starters)
    
    opp_actual_starters_per_game['NUM_USUAL_STARTERS_PRESENT'] = opp_actual_starters_per_game.apply(count_opp_usual_starters, axis=1)
    df = df.merge(
        opp_actual_starters_per_game[['GAME_ID', 'TEAM_ID', 'NUM_USUAL_STARTERS_PRESENT']],
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    # 3) Compute overall opponent starter average DEF_RATING
    starters_df = df[df['STARTING'] == 1].copy()
    opp_starter_def_rating = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'DEF_RATING': 'mean'})
        .rename(columns={'DEF_RATING': 'OPP_STARTER_AVG_DEF_RATING'})
        .reset_index()
    )
    df = df.merge(
        opp_starter_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    # 4) Compute average DEF_RATING of opponent starting guards and forwards directly using existing categories
    guards_df = starters_df[starters_df['GUARD'] == 1].copy()
    opp_guards_def_rating = (
        guards_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'DEF_RATING': 'mean'})
        .rename(columns={'DEF_RATING': 'OPP_GUARDS_AVG_DEF_RATING_OPP'})
        .reset_index()
    )
    df = df.merge(
        opp_guards_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    forwards_df = starters_df[starters_df['FORWARD'] == 1].copy()
    opp_forwards_def_rating = (
        forwards_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'DEF_RATING': 'mean'})
        .rename(columns={'DEF_RATING': 'OPP_FORWARDS_AVG_DEF_RATING_OPP'})
        .reset_index()
    )
    df = df.merge(
        opp_forwards_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )

    centers_df = starters_df[starters_df['CENTER'] == 1].copy()
    opp_centers_def_rating = (
    centers_df
    .groupby(['GAME_ID', 'TEAM_ID'])
    .agg({'DEF_RATING': 'mean'})
    .rename(columns={'DEF_RATING': 'OPP_CENTERS_AVG_DEF_RATING_OPP'})
    .reset_index()
    )
    df = df.merge(
        opp_centers_def_rating,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )
    return df

def team_starter_spacing(df):
    starters_df = df[df['STARTING'] == 1].copy()
    
    team_spacing = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'FG3_PCT': 'mean'})
        .rename(columns={'FG3_PCT': 'TEAM_STARTER_SPACING_METRIC'})
        .reset_index()
    )
    
    df = df.merge(
        team_spacing,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    return df

def pace_expectation(df):
    starters_df = df[df['STARTING'] == 1].copy()
    
    # Team starter average pace
    team_pace = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PACE': 'mean'})
        .rename(columns={'PACE': 'TEAM_STARTER_PACE'})
        .reset_index()
    )
    df = df.merge(
        team_pace,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )
    
    # Opponent starter average pace
    opp_pace = (
        starters_df
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg({'PACE': 'mean'})
        .rename(columns={'PACE': 'OPP_STARTER_PACE'})
        .reset_index()
    )
    df = df.merge(
        opp_pace,
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_OPP')
    )
    
    # Calculate expected pace as average of team + opponent starters
    df['PACE_EXPECTATION'] = (df['TEAM_STARTER_PACE'] + df['OPP_STARTER_PACE']) / 2
    
    return df

def allLineupFeatures(df):
    df = teamUsualStarters(df)
    df = oppTeamUsualStarters(df)
    df = team_starter_spacing(df)
    df = pace_expectation(df)
    return df

