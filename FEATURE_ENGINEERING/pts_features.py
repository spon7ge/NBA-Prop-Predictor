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


def check_all_defensive_players_at_position(df, all_defensive_players, position, year=2025):
    """
    Check if there are all-defensive players at a specific position (guard, forward, or center).
    """
    # Create copy and validate inputs
    df = df.copy()
    position = position.lower()
    
    if position not in ['guard', 'forward', 'center']:
        raise ValueError("Position must be 'guard', 'forward', or 'center'")
    
    if year not in all_defensive_players:
        raise ValueError(f"Year {year} not found in all_defensive_players dictionary")
    
    # Get all-defensive players for the specified year
    def_players = all_defensive_players[year]
    
    # Create position column mapping
    position_mapping = {
        'guard': 'GUARD',
        'forward': 'FORWARD', 
        'center': 'CENTER'
    }
    
    position_col = position_mapping[position]
    
    # Ensure required columns exist
    required_cols = ['PLAYER_NAME', 'GAME_ID', 'TEAM_ID', position_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Mark if player is an all-defensive player (only add if not already exists)
    if 'IS_ALL_DEFENSIVE' not in df.columns:
        df['IS_ALL_DEFENSIVE'] = df['PLAYER_NAME'].isin(def_players).astype(int)
    
    # Mark if player is an all-defensive player at the specified position
    df[f'IS_ALL_DEF_{position.upper()}'] = (
        (df['IS_ALL_DEFENSIVE'] == 1) & (df[position_col] == 1)
    ).astype(int)
    
    # Group by game and team to count all-defensive players at position
    team_def_counts = (
        df[df[f'IS_ALL_DEF_{position.upper()}'] == 1]
        .groupby(['GAME_ID', 'TEAM_ID'])
        .size()
        .reset_index(name=f'ALL_DEF_{position.upper()}_COUNT')
    )
    
    # Initialize count columns with zeros
    df[f'ALL_DEF_{position.upper()}_COUNT_TEAM'] = 0
    
    # Merge counts back to main dataframe for player's team (only if there are counts)
    if not team_def_counts.empty:
        df = df.merge(
            team_def_counts,
            on=['GAME_ID', 'TEAM_ID'],
            how='left'
        )
        # Update the team count column with actual values, keeping 0 for NaN
        df[f'ALL_DEF_{position.upper()}_COUNT_TEAM'] = df[f'ALL_DEF_{position.upper()}_COUNT'].fillna(0).astype(int)
        # Clean up the temporary column
        df.drop(columns=[f'ALL_DEF_{position.upper()}_COUNT'], inplace=True, errors='ignore')
    
    # Add opponent team counts (if OPP_TEAM_ID exists)
    if 'OPP_TEAM_ID' in df.columns:
        # Initialize opponent count column
        df[f'ALL_DEF_{position.upper()}_COUNT_OPP'] = 0
        
        # Only merge if there are defensive players at this position
        if not team_def_counts.empty:
            opp_counts = team_def_counts.rename(columns={
                'TEAM_ID': 'OPP_TEAM_ID', 
                f'ALL_DEF_{position.upper()}_COUNT': f'ALL_DEF_{position.upper()}_COUNT_OPP'
            })
            
            df = df.merge(
                opp_counts,
                on=['GAME_ID', 'OPP_TEAM_ID'],
                how='left',
                suffixes=('', '_merge')
            )
            
            # Update opponent count, handling the case where merge column exists
            if f'ALL_DEF_{position.upper()}_COUNT_OPP_merge' in df.columns:
                df[f'ALL_DEF_{position.upper()}_COUNT_OPP'] = df[f'ALL_DEF_{position.upper()}_COUNT_OPP_merge'].fillna(0).astype(int)
                df.drop(columns=[f'ALL_DEF_{position.upper()}_COUNT_OPP_merge'], inplace=True)
        
        # Binary flag for facing all-defensive player at position
        df[f'FACING_ALL_DEF_{position.upper()}'] = (df[f'ALL_DEF_{position.upper()}_COUNT_OPP'] > 0).astype(int)
    
    # Binary flag for having all-defensive player at position in game
    opp_count = df.get(f'ALL_DEF_{position.upper()}_COUNT_OPP', 0)
    df[f'ALL_DEF_{position.upper()}_IN_GAME'] = (
        (df[f'ALL_DEF_{position.upper()}_COUNT_TEAM'] > 0) | 
        (opp_count > 0)
    ).astype(int)
    
    return df

def add_all_defensive_features(df, all_defensive_players, year=2025):
    """
    Add comprehensive all-defensive player features for all positions.
    
    Parameters:
    - df: DataFrame containing player data
    - all_defensive_players: Dictionary with years as keys and lists of all-defensive player names as values
    - year: Integer - Year to get all-defensive players list from (default: 2025)
    
    Returns:
    - DataFrame with all-defensive features for guards, forwards, and centers
    """
    # Apply for all three positions
    for position in ['guard', 'forward', 'center']:
        df = check_all_defensive_players_at_position(df, all_defensive_players, position, year)
    
    # Add summary features
    df['TOTAL_ALL_DEF_TEAM'] = (
        df['ALL_DEF_GUARD_COUNT_TEAM'] + 
        df['ALL_DEF_FORWARD_COUNT_TEAM'] + 
        df['ALL_DEF_CENTER_COUNT_TEAM']
    )
    
    if 'OPP_TEAM_ID' in df.columns:
        df['TOTAL_ALL_DEF_OPP'] = (
            df['ALL_DEF_GUARD_COUNT_OPP'] + 
            df['ALL_DEF_FORWARD_COUNT_OPP'] + 
            df['ALL_DEF_CENTER_COUNT_OPP']
        )
        
        df['ALL_DEF_MATCHUP_TOTAL'] = df['TOTAL_ALL_DEF_TEAM'] + df['TOTAL_ALL_DEF_OPP']
    
    return df
