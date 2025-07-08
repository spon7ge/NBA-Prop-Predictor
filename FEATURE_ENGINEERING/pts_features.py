import pandas as pd

def analyzeTeammateSplitsPTS(game_data, min_minutes=10):
    """
    Analyze player's performance splits based on presence of star teammates
    with optimized performance.
    """
    # Filter minutes once at the start
    game_data = game_data[game_data['MIN'] >= min_minutes].copy()
    
    # Precompute starters once for all players
    game_starters = (
        game_data[game_data['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg(PLAYER_NAMES=('PLAYER_NAME', list))
        .reset_index()
    )
    
    # Precompute star players lookup dictionary
    star_players_dict = (
        game_data[game_data['IS_STAR'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])['PLAYER_NAME']
        .apply(set)
        .to_dict()
    )
    
    def get_star_info_vectorized(group):
        game_ids = group['GAME_ID']
        team_ids = group['TEAM_ID']
        player_name = group['PLAYER_NAME'].iloc[0]
        
        # Create a mask for star teammates
        star_counts = []
        for game_id, team_id in zip(game_ids, team_ids):
            star_set = star_players_dict.get((game_id, team_id), set())
            # Remove the current player if they're a star
            star_count = len(star_set - {player_name})
            star_counts.append(star_count)
            
        group = group.copy()
        group['NUM_STAR_TEAMMATES'] = star_counts
        group['HAS_STAR_TEAMMATE'] = (group['NUM_STAR_TEAMMATES'] > 0).astype(int)
        
        # Vectorized calculations for per-36 stats
        group['PTS_PER_36'] = round(group['PTS'] * (36 / group['MIN']), 2)
        group['FGA_PER_36'] = round(group['FGA'] * (36 / group['MIN']), 2)
        
        # Vectorized calculations for with/without star stats
        with_star_mask = group['HAS_STAR_TEAMMATE'] == 1
        without_star_mask = ~with_star_mask
        
        # Calculate averages once
        with_star_pts36 = group.loc[with_star_mask, 'PTS_PER_36'].mean() if with_star_mask.any() else 0
        without_star_pts36 = group.loc[without_star_mask, 'PTS_PER_36'].mean() if without_star_mask.any() else 0
        with_star_usg = group.loc[with_star_mask, 'USG_PCT'].mean() if with_star_mask.any() else 0
        without_star_usg = group.loc[without_star_mask, 'USG_PCT'].mean() if without_star_mask.any() else 0
        
        # Assign values using vectorized operations
        group['AVG_PTS_PER_36_WITH_STAR'] = round(with_star_pts36, 2)
        group['AVG_PTS_PER_36_WITHOUT_STAR'] = round(without_star_pts36, 2)
        group['AVG_USG_PCT_WITH_STAR'] = round(with_star_usg, 2)
        group['AVG_USG_PCT_WITHOUT_STAR'] = round(without_star_usg, 2)
        
        return group
    
    # Process all players at once using groupby
    result = (
        game_data
        .groupby('PLAYER_NAME', group_keys=False)
        .apply(get_star_info_vectorized)
    )
    
    return result
def analyzeTeammateSplitsWhenAnyStarSitsPTS(game_data, min_minutes=10):
    """
    Optimized version to calculate historical averages when any star teammate sits.
    """
    # Initial filtering
    game_data = game_data[game_data['MIN'] >= min_minutes].copy()
    
    # Precompute starters once
    game_starters = (
        game_data[game_data['STARTING'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .agg(PLAYER_NAMES=('PLAYER_NAME', set))  # Using set instead of list for faster lookups
        .reset_index()
    )
    
    # Precompute star players by team
    team_stars = (
        game_data[game_data['IS_STAR'] == 1]
        .groupby('TEAM_ID')['PLAYER_NAME']
        .agg(set)
        .to_dict()
    )
    
    # Precompute starter sets for quick lookup
    starter_dict = (
        game_starters
        .set_index(['GAME_ID', 'TEAM_ID'])['PLAYER_NAMES']
        .to_dict()
    )
    
    def process_player_games(group):
        player_name = group['PLAYER_NAME'].iloc[0]
        team_id = group['TEAM_ID'].iloc[0]
        
        # Get star teammates for this player's team
        team_star_teammates = team_stars.get(team_id, set()) - {player_name}
        
        if not team_star_teammates:  # No star teammates on team
            group['STAR_OUT'] = 0
            group['AVG_PTS_WHEN_STAR_OUT'] = 0
            group['AVG_USG_PCT_WHEN_STAR_OUT'] = 0
            group['AVG_MIN_WHEN_STAR_OUT'] = 0
            return group
            
        # Vectorized operation to check for missing stars
        def check_missing_stars(row):
            game_starters_set = starter_dict.get((row['GAME_ID'], row['TEAM_ID']), set())
            return int(any(star not in game_starters_set for star in team_star_teammates))
        
        # Apply the check vectorized across all games
        group['STAR_OUT'] = group.apply(check_missing_stars, axis=1)
        
        # Calculate averages when star is out using boolean indexing
        star_out_mask = group['STAR_OUT'] == 1
        
        if star_out_mask.any():
            star_out_games = group[star_out_mask]
            avg_pts = round(star_out_games['PTS'].mean(), 2)
            avg_usg = round(star_out_games['USG_PCT'].mean(), 2)
            avg_min = round(star_out_games['MIN'].mean(), 2)
        else:
            avg_pts = avg_usg = avg_min = 0
            
        # Assign values using vectorized operations
        group['AVG_PTS_WHEN_STAR_OUT'] = avg_pts
        group['AVG_USG_PCT_WHEN_STAR_OUT'] = avg_usg
        group['AVG_MIN_WHEN_STAR_OUT'] = avg_min
        
        # Calculate HAS_STAR_TEAMMATE using vectorized operations
        def check_active_stars(row):
            game_starters_set = starter_dict.get((row['GAME_ID'], row['TEAM_ID']), set())
            return int(any(star in game_starters_set for star in team_star_teammates))
            
        group['HAS_STAR_TEAMMATE'] = group.apply(check_active_stars, axis=1)
        
        return group
    
    # Process all players at once using groupby
    result = (
        game_data
        .groupby(['PLAYER_NAME', 'TEAM_ID'], group_keys=False)
        .apply(process_player_games)
    )
    
    return result
