import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

def getGameLogs(season, season_type):
    # Get initial game logs
    df = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type
    ).get_data_frames()[0]
    
    def fn(df):
        if len(df) != 2:
            return df
        a, b = df.iloc[0], df.iloc[1]
        
        df = df.copy()
        
        # Calculate possessions for each team
        p1 = a['FGA'] + 0.44 * a['FTA'] - a['OREB'] + a['TOV']
        p2 = b['FGA'] + 0.44 * b['FTA'] - b['OREB'] + b['TOV']
        
        # Use the same team minutes (game minutes) for both teams
        game_minutes = float(a['MIN'])
        
        # CORRECT: Single pace calculation for the entire game
        game_pace = ((240.0 / game_minutes) * (p1 + p2) / 2.0) if game_minutes > 0 else 0
        game_pace = round(game_pace, 1)
        
        # Offensive and Defensive Ratings
        off_rating_a = round((a['PTS'] / p1) * 100 if p1 > 0 else 0, 1)
        off_rating_b = round((b['PTS'] / p2) * 100 if p2 > 0 else 0, 1)
        def_rating_a = round((b['PTS'] / p1) * 100 if p1 > 0 else 0, 1)
        def_rating_b = round((a['PTS'] / p2) * 100 if p2 > 0 else 0, 1)
        
        # Team A assignments
        df.loc[df.index[0], 'TEAM_PACE'] = game_pace
        df.loc[df.index[0], 'GAME_PACE'] = game_pace  
        df.loc[df.index[0], 'OPP_PACE'] = game_pace
        
        # Team B assignments  
        df.loc[df.index[1], 'TEAM_PACE'] = game_pace
        df.loc[df.index[1], 'GAME_PACE'] = game_pace
        df.loc[df.index[1], 'OPP_PACE'] = game_pace
        
        # Team A
        df.loc[df.index[0], 'OPP_TEAM_ID'] = b['TEAM_ID']
        df.loc[df.index[0], 'TEAM_OFF_RATING'] = off_rating_a
        df.loc[df.index[0], 'TEAM_DEF_RATING'] = def_rating_a
        df.loc[df.index[0], 'OPP_DEF_RATING'] = def_rating_b
        df.loc[df.index[0], 'OPP_OFF_RATING'] = off_rating_b
        df.loc[df.index[0], 'OPP_PTS'] = b['PTS']
        df.loc[df.index[0], 'OPP_FGM'] = b['FGM']
        df.loc[df.index[0], 'OPP_FGA'] = b['FGA']
        df.loc[df.index[0], 'OPP_FG_PCT'] = b['FG_PCT']
        df.loc[df.index[0], 'OPP_REB'] = b['REB']
        df.loc[df.index[0], 'OPP_AST'] = b['AST']
        df.loc[df.index[0], 'OPP_STL'] = b['STL']
        df.loc[df.index[0], 'OPP_BLK'] = b['BLK']
        df.loc[df.index[0], 'OPP_TOV'] = b['TOV']
        
        # Team B
        df.loc[df.index[1], 'OPP_TEAM_ID'] = a['TEAM_ID']
        df.loc[df.index[1], 'TEAM_OFF_RATING'] = off_rating_b
        df.loc[df.index[1], 'TEAM_DEF_RATING'] = def_rating_b
        df.loc[df.index[1], 'OPP_DEF_RATING'] = def_rating_a
        df.loc[df.index[1], 'OPP_OFF_RATING'] = off_rating_a
        df.loc[df.index[1], 'OPP_PTS'] = a['PTS']
        df.loc[df.index[1], 'OPP_FGM'] = a['FGM']
        df.loc[df.index[1], 'OPP_FGA'] = a['FGA']
        df.loc[df.index[1], 'OPP_FG_PCT'] = a['FG_PCT']
        df.loc[df.index[1], 'OPP_REB'] = a['REB']
        df.loc[df.index[1], 'OPP_AST'] = a['AST']
        df.loc[df.index[1], 'OPP_STL'] = a['STL']
        df.loc[df.index[1], 'OPP_BLK'] = a['BLK']
        df.loc[df.index[1], 'OPP_TOV'] = a['TOV']
        
        return df
    
    # Process each game
    data = []
    for game_id in df['GAME_ID'].unique():
        game_df = df.loc[df['GAME_ID'] == game_id]
        processed_game = fn(game_df)
        data.append(processed_game)
    
    # Combine all processed games
    df = pd.concat(data, ignore_index=True)
    df['OPP_TEAM_ID'] = df['OPP_TEAM_ID'].astype(int)
    
    # Dynamically prefix TEAM_ to own team stats
    exclude_prefixes = ['OPP_', 'TEAM_']  # Skip already prefixed columns
    exclude_cols = ['GAME_ID', 'GAME_PACE', 'VIDEO_AVAILABLE']
    cols_to_prefix = [
    col for col in df.columns 
    if not any(col.startswith(p) for p in exclude_prefixes) and col not in exclude_cols
    ]
    
    rename_dict = {col: f'TEAM_{col}' for col in cols_to_prefix}
    df = df.rename(columns=rename_dict)
    
    return df


def mergeTeamtoPlayer(player_df, season='2024-25', season_type='Regular Season'):
    team_df = getGameLogs(season=season, season_type=season_type)
    player_df[['TEAM_ID', 'GAME_ID']] = player_df[['TEAM_ID', 'GAME_ID']].astype(int)
    team_df[['TEAM_ID', 'GAME_ID']] = team_df[['TEAM_ID', 'GAME_ID']].astype(int)
    
    # Drop TEAM_ABBREVIATION from team_df to avoid duplicates
    team_df = team_df.drop(columns=['TEAM_ABBREVIATION'], errors='ignore')
    
    return player_df.merge(team_df, on=['GAME_ID', 'TEAM_ID'], how='left')
