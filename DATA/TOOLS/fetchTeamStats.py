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
        p1 = round(a['FGA'] + 0.44 * a['FTA'] - a['OREB'] + a['TOV'], 2)  # Team A possessions
        p2 = round(b['FGA'] + 0.44 * b['FTA'] - b['OREB'] + b['TOV'], 2)  # Team B possessions
        avg_pace = (p1 + p2) / 2  # Average game pace
        
        # Calculate offensive ratings (points per 100 possessions)
        off_rating_a = round((a['PTS'] / p1) * 100 if p1 > 0 else 0, 2)  # Team A's offensive rating
        off_rating_b = round((b['PTS'] / p2) * 100 if p2 > 0 else 0, 2)  # Team B's offensive rating
        
        # Calculate defensive ratings (opponent points per 100 possessions)
        def_rating_a = round((b['PTS'] / p2) * 100 if p2 > 0 else 0, 2)  # Team A's defensive rating
        def_rating_b = round((a['PTS'] / p1) * 100 if p1 > 0 else 0, 2)  # Team B's defensive rating
        
        # Add stats for team A
        df.loc[df.index[0], 'OPP_TEAM_ID'] = b['TEAM_ID']
        df.loc[df.index[0], 'TEAM_OFF_RATING'] = off_rating_a
        df.loc[df.index[0], 'TEAM_DEF_RATING'] = def_rating_a
        df.loc[df.index[0], 'TEAM_PACE'] = p1
        df.loc[df.index[0], 'GAME_PACE'] = avg_pace
        df.loc[df.index[0], 'OPP_PACE'] = p2
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
        
        # Add stats for team B
        df.loc[df.index[1], 'OPP_TEAM_ID'] = a['TEAM_ID']
        df.loc[df.index[1], 'TEAM_OFF_RATING'] = off_rating_b
        df.loc[df.index[1], 'TEAM_DEF_RATING'] = def_rating_b
        df.loc[df.index[1], 'TEAM_PACE'] = p2
        df.loc[df.index[1], 'GAME_PACE'] = avg_pace
        df.loc[df.index[1], 'OPP_PACE'] = p1
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
    return pd.concat(data, ignore_index=True)