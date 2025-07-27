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
            if len(g)!=2: return g
            a,b = g.iloc
            def1 = (b['TEAM_PTS']/(b['TEAM_FGA']+0.44*b['TEAM_FTA']-b['TEAM_OREB']+b['TEAM_TOV']))*100
            def2 = (a['TEAM_PTS']/(a['TEAM_FGA']+0.44*a['TEAM_FTA']-a['TEAM_OREB']+a['TEAM_TOV']))*100
            
            # Create columns if they don't exist
            for col in ['OPP_DEF_RATING', 'OPP_STL', 'OPP_BLK', 'OPP_REB', 'OPP_FG_PCT', 'OPP_TEAM_ID']:
                if col not in g.columns:
                    g[col] = 0.0
            
            # Get indices after ensuring columns exist
            idx = g.columns.get_indexer(['OPP_DEF_RATING','OPP_STL','OPP_BLK','OPP_REB','OPP_FG_PCT','OPP_TEAM_ID'])
            g.iloc[0, idx]=[def2,b['TEAM_STL'],b['TEAM_BLK'],b['TEAM_OREB']+b['TEAM_DREB'],b['TEAM_FGM']/b['TEAM_FGA'],b['TEAM_ID']]
            g.iloc[1, idx]=[def1,a['TEAM_STL'],a['TEAM_BLK'],a['TEAM_OREB']+a['TEAM_DREB'],a['TEAM_FGM']/a['TEAM_FGA'],a['TEAM_ID']]
            return g
        return df.groupby('GAME_ID',group_keys=False).apply(fn)

    def addOffensiveRating(self, df):
        """
        Add offensive rating stats to the team game log DataFrame.
        
        Args:
            df (pd.DataFrame): Team game log DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with offensive rating added.
        """
        def fn(g):
            if len(g)!=2: return g
            a,b = g.iloc
            p1=a['TEAM_FGA']+0.44*a['TEAM_FTA']-a['TEAM_OREB']+a['TEAM_TOV']
            p2=b['TEAM_FGA']+0.44*b['TEAM_FTA']-b['TEAM_OREB']+b['TEAM_TOV']
            
            # Create column if it doesn't exist
            if 'TEAM_OFF_RATING' not in g.columns:
                g['TEAM_OFF_RATING'] = 0.0
                
            g.iloc[0,g.columns.get_indexer(['TEAM_OFF_RATING'])]=[(a['TEAM_PTS']/p1)*100]
            g.iloc[1,g.columns.get_indexer(['TEAM_OFF_RATING'])]=[(b['TEAM_PTS']/p2)*100]
            return g
        return df.groupby('GAME_ID',group_keys=False).apply(fn)

    def add_pace_stats(self, df):
        """
        Add pace stats to the team game log DataFrame.
        
        Args:
            df (pd.DataFrame): Team game log DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with pace stats added.
        """
        def fn(g):
            if len(g)!=2: return g
            a,b=g.iloc
            p1=a['TEAM_FGA']+0.44*a['TEAM_FTA']-a['TEAM_OREB']+a['TEAM_TOV']
            p2=b['TEAM_FGA']+0.44*b['TEAM_FTA']-b['TEAM_OREB']+b['TEAM_TOV']
            avg=(p1+p2)/2
            
            # Create columns if they don't exist
            for col in ['TEAM_PACE', 'GAME_PACE', 'OPP_PACE']:
                if col not in g.columns:
                    g[col] = 0.0
            
            # Get indices after ensuring columns exist
            idx=g.columns.get_indexer(['TEAM_PACE','GAME_PACE','OPP_PACE'])
            g.iloc[0,idx]=[p1,avg,p2]
            g.iloc[1,idx]=[p2,avg,p1]
            return g
        return df.groupby('GAME_ID',group_keys=False).apply(fn)

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
    # Merge player's own team stats
    player_df = player_df.merge(
        team_stats_df,
        on=['GAME_ID', 'TEAM_ID'],
        how='left',
        suffixes=('', '_TEAM')
    )

    # Get mapping for opponent team
    opp_map = team_stats_df[['GAME_ID', 'TEAM_ID']].copy()
    opp_map = opp_map.rename(columns={'TEAM_ID': 'OPP_TEAM_ID'})
    opp_map = opp_map.merge(
        team_stats_df[['GAME_ID', 'TEAM_ID']],
        on='GAME_ID'
    )
    opp_map = opp_map[opp_map['TEAM_ID'] != opp_map['OPP_TEAM_ID']]

    # Merge to get OPP_TEAM_ID for each player row
    player_df = player_df.merge(
        opp_map[['GAME_ID', 'TEAM_ID', 'OPP_TEAM_ID']],
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )

    # Merge opponent team stats
    player_df = player_df.merge(
        team_stats_df.add_prefix('OPP_'),
        left_on=['GAME_ID', 'OPP_TEAM_ID'],
        right_on=['OPP_GAME_ID', 'OPP_TEAM_ID'],
        how='left'
    )

    return player_df


    # player_stats_with_team = add_team_and_opp_stats(player_df, team_stats)
