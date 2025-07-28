import pandas as pd
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams


class TeamStatsFetcher:
    def __init__(self, default_season='2023-24'):
        """
        Initialize the TeamStatsFetcher.
        
        Args:
            default_season (str): Default season in format 'YYYY-YY'. Defaults to '2023-24'.
        """
        self.default_season = default_season

    def getTeamData(self, season=None, season_type='Regular Season'):
        """
        Fetch team game log data for all teams in a given season.
        
        Args:
            season (str, optional): Season in format 'YYYY-YY'. Defaults to self.default_season.
            season_type (str, optional): Type of season. Defaults to 'Regular Season'.
            
        Returns:
            pd.DataFrame: Concatenated DataFrame of all teams' game logs.
        """
        season = season or self.default_season
        tlist = teams.get_teams()
        data = []
        for t in tlist:
            try:
                df = teamgamelog.TeamGameLog(
                    team_id=t['id'], season=season,
                    season_type_all_star=season_type
                ).get_data_frames()[0]
                df.columns = df.columns.str.upper()
                drop = ['MATCHUP','WL','W','L','W_PCT','GAMEDATE']
                df.drop(columns=[c for c in drop if c in df], errors='ignore', inplace=True)
                df.rename(columns={c: f'TEAM_{c}' for c in df.columns if c not in ['GAME_ID','TEAM_ID']}, inplace=True)
                data.append(df)
            except:
                continue
        return pd.concat(data, ignore_index=True)

    def addOpponentStats(self, df):
        """
        Add opponent defensive stats to the team game log DataFrame.
        
        Args:
            df (pd.DataFrame): Team game log DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with opponent stats added.
        """
        def fn(g):
            if len(g) != 2: 
                return g
            a, b = g.iloc[0], g.iloc[1]
            
            # Calculate defensive ratings
            def1 = (b['TEAM_PTS']/(b['TEAM_FGA']+0.44*b['TEAM_FTA']-b['TEAM_OREB']+b['TEAM_TOV']))*100
            def2 = (a['TEAM_PTS']/(a['TEAM_FGA']+0.44*a['TEAM_FTA']-a['TEAM_OREB']+a['TEAM_TOV']))*100
            
            # Create a copy to avoid SettingWithCopyWarning
            g = g.copy()
            
            # Add opponent stats
            g.loc[g.index[0], 'OPP_DEF_RATING'] = def2
            g.loc[g.index[0], 'OPP_STL'] = b['TEAM_STL']
            g.loc[g.index[0], 'OPP_BLK'] = b['TEAM_BLK']
            g.loc[g.index[0], 'OPP_REB'] = b['TEAM_OREB'] + b['TEAM_DREB']
            g.loc[g.index[0], 'OPP_FG_PCT'] = b['TEAM_FGM'] / b['TEAM_FGA'] if b['TEAM_FGA'] > 0 else 0
            g.loc[g.index[0], 'OPP_TEAM_ID'] = b['TEAM_ID']
            
            g.loc[g.index[1], 'OPP_DEF_RATING'] = def1
            g.loc[g.index[1], 'OPP_STL'] = a['TEAM_STL']
            g.loc[g.index[1], 'OPP_BLK'] = a['TEAM_BLK']
            g.loc[g.index[1], 'OPP_REB'] = a['TEAM_OREB'] + a['TEAM_DREB']
            g.loc[g.index[1], 'OPP_FG_PCT'] = a['TEAM_FGM'] / a['TEAM_FGA'] if a['TEAM_FGA'] > 0 else 0
            g.loc[g.index[1], 'OPP_TEAM_ID'] = a['TEAM_ID']
            
            return g
        
        return df.groupby('GAME_ID', group_keys=False).apply(fn)

    def addOffensiveRating(self, df):
        """
        Add offensive rating stats to the team game log DataFrame.
        
        Args:
            df (pd.DataFrame): Team game log DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with offensive rating added.
        """
        def fn(g):
            if len(g) != 2: 
                return g
            a, b = g.iloc[0], g.iloc[1]
            
            p1 = a['TEAM_FGA'] + 0.44*a['TEAM_FTA'] - a['TEAM_OREB'] + a['TEAM_TOV']
            p2 = b['TEAM_FGA'] + 0.44*b['TEAM_FTA'] - b['TEAM_OREB'] + b['TEAM_TOV']
            
            g = g.copy()
            g.loc[g.index[0], 'TEAM_OFF_RATING'] = (a['TEAM_PTS']/p1)*100 if p1 > 0 else 0
            g.loc[g.index[1], 'TEAM_OFF_RATING'] = (b['TEAM_PTS']/p2)*100 if p2 > 0 else 0
            
            return g
        
        return df.groupby('GAME_ID', group_keys=False).apply(fn)

    def add_pace_stats(self, df):
        """
        Add pace stats to the team game log DataFrame.
        
        Args:
            df (pd.DataFrame): Team game log DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with pace stats added.
        """
        def fn(g):
            if len(g) != 2: 
                return g
            a, b = g.iloc[0], g.iloc[1]
            
            p1 = a['TEAM_FGA'] + 0.44*a['TEAM_FTA'] - a['TEAM_OREB'] + a['TEAM_TOV']
            p2 = b['TEAM_FGA'] + 0.44*b['TEAM_FTA'] - b['TEAM_OREB'] + b['TEAM_TOV']
            avg = (p1 + p2) / 2
            
            g = g.copy()
            g.loc[g.index[0], 'TEAM_PACE'] = p1
            g.loc[g.index[0], 'GAME_PACE'] = avg
            g.loc[g.index[0], 'OPP_PACE'] = p2
            
            g.loc[g.index[1], 'TEAM_PACE'] = p2
            g.loc[g.index[1], 'GAME_PACE'] = avg
            g.loc[g.index[1], 'OPP_PACE'] = p1
            
            return g
        
        return df.groupby('GAME_ID', group_keys=False).apply(fn)

    def process_team_stats(self, season=None, season_type='Regular Season'):
        """
        Get and process team stats with all additional metrics.
        
        Args:
            season (str, optional): Season in format 'YYYY-YY'. Defaults to self.default_season.
            season_type (str, optional): Type of season. Defaults to 'Regular Season'.
            
        Returns:
            pd.DataFrame: Processed team stats DataFrame with all metrics.
        """
        df = self.getTeamData(season, season_type)
        df = self.addOpponentStats(df)
        df = self.addOffensiveRating(df)
        df = self.add_pace_stats(df)
        return df


def add_team_and_opp_stats(player_df, team_stats_df):
    """
    Merge team and opponent team stats onto a player-level DataFrame.
    
    Args:
        player_df (pd.DataFrame): Player-level DataFrame with GAME_ID and TEAM_ID columns.
        team_stats_df (pd.DataFrame): Team-level DataFrame with GAME_ID, TEAM_ID, and team stats columns.
        
    Returns:
        pd.DataFrame: Player DataFrame with team and opponent team stats merged in.
    """
    
    # First, merge player's own team stats
    player_with_team = player_df.merge(
        team_stats_df,
        on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_DUPLICATE')
    )
    
    # Drop any duplicate columns that may have been created
    duplicate_cols = [col for col in player_with_team.columns if col.endswith('_DUPLICATE')]
    if duplicate_cols:
        player_with_team = player_with_team.drop(columns=duplicate_cols)
    
    # Create opponent stats mapping
    # Since addOpponentStats already adds OPP_TEAM_ID, we can use it directly
    if 'OPP_TEAM_ID' not in player_with_team.columns:
        print("Warning: OPP_TEAM_ID not found in team_stats_df. Make sure addOpponentStats was called.")
        return player_with_team
    
    # Prepare opponent team stats with OPP_ prefix (excluding already prefixed columns)
    opp_stats_cols = [col for col in team_stats_df.columns 
                      if col.startswith('TEAM_') and not col.startswith('OPP_')]
    opp_stats_cols.extend(['GAME_ID', 'TEAM_ID'])  # Include join keys
    
    opp_team_stats = team_stats_df[opp_stats_cols].copy()
    
    # Rename team stats columns to OPP_ prefix
    rename_dict = {}
    for col in opp_team_stats.columns:
        if col.startswith('TEAM_'):
            rename_dict[col] = f'OPP_{col}'
    opp_team_stats = opp_team_stats.rename(columns=rename_dict)
    
    # Rename TEAM_ID to OPP_TEAM_ID for the merge
    opp_team_stats = opp_team_stats.rename(columns={'TEAM_ID': 'OPP_TEAM_ID'})
    
    # Merge opponent team stats
    result = player_with_team.merge(
        opp_team_stats,
        on=['GAME_ID', 'OPP_TEAM_ID'],
        how='left',
        suffixes=('', '_OPP_DUPLICATE')
    )
    
    # Clean up any duplicate columns from the opponent merge
    opp_duplicate_cols = [col for col in result.columns if col.endswith('_OPP_DUPLICATE')]
    if opp_duplicate_cols:
        result = result.drop(columns=opp_duplicate_cols)
    
    return result


def diagnose_merge_issues(player_df, team_stats_df):
    """
    Helper function to diagnose merge issues and identify sources of NaN values.
    
    Args:
        player_df (pd.DataFrame): Player-level DataFrame
        team_stats_df (pd.DataFrame): Team-level DataFrame
        
    Returns:
        dict: Dictionary with diagnostic information
    """
    diagnostics = {}
    
    # Check for missing GAME_IDs
    player_games = set(player_df['GAME_ID'].unique())
    team_games = set(team_stats_df['GAME_ID'].unique())
    
    diagnostics['missing_games_in_team_stats'] = player_games - team_games
    diagnostics['missing_games_in_player_stats'] = team_games - player_games
    
    # Check for missing TEAM_IDs
    player_teams = set(player_df['TEAM_ID'].unique())
    team_teams = set(team_stats_df['TEAM_ID'].unique())
    
    diagnostics['missing_teams_in_team_stats'] = player_teams - team_teams
    diagnostics['missing_teams_in_player_stats'] = team_teams - player_teams
    
    # Check for games with only one team (incomplete games)
    game_team_counts = team_stats_df.groupby('GAME_ID')['TEAM_ID'].count()
    incomplete_games = game_team_counts[game_team_counts != 2].index.tolist()
    diagnostics['incomplete_games'] = incomplete_games
    
    # Check for duplicate GAME_ID + TEAM_ID combinations
    duplicates = team_stats_df.duplicated(subset=['GAME_ID', 'TEAM_ID']).sum()
    diagnostics['duplicate_game_team_combinations'] = duplicates
    
    return diagnostics


# Example usage with diagnostics:
def process_with_diagnostics(player_df, season='2023-24'):
    """
    Process team stats and merge with player data, including diagnostics.
    """
    fetcher = TeamStatsFetcher(default_season=season)
    team_stats = fetcher.process_team_stats()
    
    # Run diagnostics
    diagnostics = diagnose_merge_issues(player_df, team_stats)
    
    print("Merge Diagnostics:")
    for key, value in diagnostics.items():
        if value:  # Only print non-empty issues
            print(f"{key}: {len(value) if isinstance(value, (list, set)) else value}")
    
    # Perform the merge
    result = add_team_and_opp_stats(player_df, team_stats)
    
    # Check for NaN values in key opponent columns
    opp_cols = [col for col in result.columns if col.startswith('OPP_TEAM_')]
    nan_counts = result[opp_cols].isnull().sum()
    
    print(f"\nNaN counts in opponent stats:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"{col}: {count}")
    
    return result, diagnostics