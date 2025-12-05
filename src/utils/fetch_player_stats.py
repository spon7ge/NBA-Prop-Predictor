import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoreadvancedv3,
    teamgamelog,
    boxscoreplayertrackv3,
    boxscoremiscv3,
    boxscorematchupsv3,
)
from nba_api.stats.static import teams
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.features.features_v2 import engineerPlayerPlaybyPlayBasics, cleanPlaybyPlay

class FetchPlayersStats:
    def __init__(self, default_season='2024-25', sleep_time=0.1):
        self.default_season = default_season
        self.sleep_time = sleep_time

    def normalize_game_id(self, game_id):
        """Normalize game ID to 10-digit string format with leading zeros"""
        if pd.isna(game_id):
            return None
        game_id_str = str(int(float(game_id)))  # Convert to int first to remove any decimal points
        return game_id_str.zfill(10)  # Pad with leading zeros to make it 10 digits

    def getGameLogs(self, season=None, season_type='Regular Season'):
        """Get team game logs with calculated pace, offensive/defensive ratings, and opponent stats"""
        season = season or self.default_season
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

    def mergeTeamtoPlayer(self, player_df, season=None, season_type='Regular Season'):
        """Merge team stats with player stats"""
        season = season or self.default_season
        team_df = self.getGameLogs(season=season, season_type=season_type)
        
        # Ensure GAME_IDs are in consistent format (normalized strings)
        player_df['GAME_ID'] = player_df['GAME_ID'].apply(self.normalize_game_id)
        player_df = player_df.dropna(subset=['GAME_ID'])  # Remove any invalid game IDs
        team_df['GAME_ID'] = team_df['GAME_ID'].astype(str).apply(self.normalize_game_id)
        team_df = team_df.dropna(subset=['GAME_ID'])  # Remove any invalid game IDs
        
        # Ensure TEAM_IDs are int for merging
        player_df['TEAM_ID'] = player_df['TEAM_ID'].astype(int)
        team_df['TEAM_ID'] = team_df['TEAM_ID'].astype(int)
        
        # Drop TEAM_ABBREVIATION from team_df to avoid duplicates
        team_df = team_df.drop(columns=['TEAM_ABBREVIATION'], errors='ignore')
        
        return player_df.merge(team_df, on=['GAME_ID', 'TEAM_ID'], how='left')

    def fetchPlayerStats(self, season=None, season_type='Regular Season'):
        season = season or self.default_season
        df = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation='P',
            season_type_all_star=season_type
        ).get_data_frames()[0]

        df['OPP_ABBREVIATION'] = df['MATCHUP'].str.extract(r'(?:vs\.|@) ([A-Z]+)')
        df['HOME_GAME'] = df['MATCHUP'].str.contains(r'vs\.').astype(int)
        fga, fta, pts, fgm, fg3m = df['FGA'], df['FTA'], df['PTS'], df['FGM'], df['FG3M']
        df['POINT_PER_SHOT'] = np.where(fga == 0, 0.0, pts / (fga + 0.44 * fta)).round(3)
        df['EFG'] = (fgm + 0.5 * fg3m) / fga

        cols = [
            'PLAYER_NAME', 'PLAYER_ID', 'MATCHUP', 'TEAM_ABBREVIATION', 'TEAM_ID',
            'OPP_ABBREVIATION', 'HOME_GAME', 'GAME_ID', 'GAME_DATE', 'WL',
            'PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 
            'PLUS_MINUS', 'FANTASY_PTS', 'POINT_PER_SHOT', 'EFG'
        ]
        return df[cols]

    def fetchAdvancedStats(self, game_id, sleep_time=None, max_retries=5, timeout=60):
        sleep_time = sleep_time or self.sleep_time
        normalized_game_id = self.normalize_game_id(game_id)
        if not normalized_game_id:
            print(f"[ERROR] Invalid game ID: {game_id}")
            return pd.DataFrame()
            
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))  # Exponential backoff
                df = boxscoreadvancedv3.BoxScoreAdvancedV3(
                    game_id=normalized_game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                
                # Map new camelCase column names to old UPPERCASE format
                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personId': 'PLAYER_ID',
                    'position': 'START_POSITION',
                    'comment': 'COMMENT',
                    'offensiveRating': 'OFF_RATING',
                    'estimatedOffensiveRating': 'E_OFF_RATING',
                    'defensiveRating': 'DEF_RATING',
                    'estimatedDefensiveRating': 'E_DEF_RATING',
                    'netRating': 'NET_RATING',
                    'offensiveReboundPercentage': 'OREB_PCT',
                    'defensiveReboundPercentage': 'DREB_PCT',
                    'reboundPercentage': 'REB_PCT',
                    'assistPercentage': 'AST_PCT',
                    'effectiveFieldGoalPercentage': 'EFG_PCT',
                    'assistToTurnover': 'AST_TOV',
                    'usagePercentage': 'USG_PCT',
                    'trueShootingPercentage': 'TS_PCT',
                    'estimatedPace': 'E_PACE',
                    'pace': 'PACE',
                    'PIE': 'PIE',
                    'possessions': 'POSS',
                    'pacePer40': 'PACE_PER40',
                    'estimatedUsagePercentage': 'E_USG_PCT'
                }
                
                # Rename columns to match old format
                df = df.rename(columns=column_mapping)
                
                # List of columns we want to keep
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'START_POSITION', 'COMMENT',
                    'OFF_RATING', 'E_OFF_RATING', 'DEF_RATING',
                    'E_DEF_RATING', 'NET_RATING', 'OREB_PCT', 'DREB_PCT',
                    'REB_PCT', 'AST_PCT', 'EFG_PCT', 'AST_TOV', 'USG_PCT',
                    'TS_PCT', 'E_PACE', 'PACE', 'PIE', 'POSS',
                    'PACE_PER40', 'E_USG_PCT'
                ]
                
                # Only select columns that exist in the dataframe
                existing_cols = [col for col in cols if col in df.columns]
                return df[existing_cols]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n[RETRY {attempt+1}/{max_retries}] Game {normalized_game_id}: {e}")
                else:
                    print(f"\n[FAILED] Game {normalized_game_id} after {max_retries} attempts: {e}")
                    return pd.DataFrame()

    def fetchTrackingStats(self, game_id, sleep_time=None, max_retries=3, timeout=60):
        sleep_time = sleep_time or self.sleep_time
        normalized_game_id = self.normalize_game_id(game_id)
        if not normalized_game_id:
            print(f"[ERROR] Invalid game ID: {game_id}")
            return pd.DataFrame()
            
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))
                df = boxscoreplayertrackv3.BoxScorePlayerTrackV3(
                    game_id=normalized_game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                
                # Map new column names to old column names
                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personId': 'PLAYER_ID',
                    'minutes': 'MIN',
                    'speed': 'SPD',
                    'distance': 'DIST',
                    'reboundChancesOffensive': 'ORBC',
                    'reboundChancesDefensive': 'DRBC',
                    'reboundChancesTotal': 'RBC',
                    'touches': 'TCHS',
                    'secondaryAssists': 'SAST',
                    'freeThrowAssists': 'FTAST',
                    'passes': 'PASS',
                    'contestedFieldGoalsMade': 'CFGM',
                    'contestedFieldGoalsAttempted': 'CFGA',
                    'contestedFieldGoalPercentage': 'CFG_PCT',
                    'uncontestedFieldGoalsMade': 'UFGM',
                    'uncontestedFieldGoalsAttempted': 'UFGA',
                    'uncontestedFieldGoalsPercentage': 'UFG_PCT',
                    'defendedAtRimFieldGoalsMade': 'DFGM',
                    'defendedAtRimFieldGoalsAttempted': 'DFGA',
                    'defendedAtRimFieldGoalPercentage': 'DFG_PCT'
                }
                
                # Rename columns to match old format
                df = df.rename(columns=column_mapping)
                
                # Select the columns we want to keep
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'MIN', 'SPD', 'DIST', 'ORBC', 'DRBC', 'RBC',
                    'TCHS', 'SAST', 'FTAST', 'PASS', 'CFGM', 'CFGA', 'CFG_PCT',
                    'UFGM', 'UFGA', 'UFG_PCT', 'DFGM', 'DFGA', 'DFG_PCT'
                ]
                
                # Only select columns that exist in the dataframe
                existing_cols = [col for col in cols if col in df.columns]
                return df[existing_cols]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Tracking stats for {normalized_game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching tracking for {normalized_game_id}: {e}")
                    return pd.DataFrame()


    def fetchBoxScoreScoring(self, game_id, sleep_time=None, max_retries=3, timeout=60):
        """Fetch BoxScoreScoringV3 data for a single game"""
        sleep_time = sleep_time or self.sleep_time
        normalized_game_id = self.normalize_game_id(game_id)
        if not normalized_game_id:
            print(f"[ERROR] Invalid game ID: {game_id}")
            return pd.DataFrame()
            
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))
                from nba_api.stats.endpoints import boxscorescoringv3
                df = boxscorescoringv3.BoxScoreScoringV3(
                    game_id=normalized_game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                
                if df.empty:
                    return pd.DataFrame()
                    
                # Map new column names to standardized names
                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personId': 'PLAYER_ID'
                }
                
                # Rename columns to match old format
                df = df.rename(columns=column_mapping)
                
                # Select the columns we want to keep
                cols = [
                    'GAME_ID', 'PLAYER_ID',
                    'percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt',
                    'percentagePoints2pt', 'percentagePointsMidrange2pt', 'percentagePoints3pt',
                    'percentagePointsFastBreak', 'percentagePointsFreeThrow', 'percentagePointsOffTurnovers',
                    'percentagePointsPaint', 'percentageAssisted2pt', 'percentageUnassisted2pt',
                    'percentageAssisted3pt', 'percentageUnassisted3pt', 'percentageAssistedFGM',
                    'percentageUnassistedFGM'
                ]
                
                # Only select columns that exist in the dataframe
                existing_cols = [col for col in cols if col in df.columns]
                return df[existing_cols]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Scoring stats for {normalized_game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching scoring for {normalized_game_id}: {e}")
                    return pd.DataFrame()

    def fetchBoxScoreMisc(self, game_id, sleep_time=None, max_retries=3, timeout=60):
        """Fetch BoxScoreMiscV3 data for a single game"""
        sleep_time = sleep_time or self.sleep_time
        normalized_game_id = self.normalize_game_id(game_id)
        if not normalized_game_id:
            print(f"[ERROR] Invalid game ID: {game_id}")
            return pd.DataFrame()

        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))
                misc_endpoint = boxscoremiscv3.BoxScoreMiscV3(
                    game_id=normalized_game_id,
                    timeout=timeout
                )
                if hasattr(misc_endpoint, "player_stats") and misc_endpoint.player_stats is not None:
                    df = misc_endpoint.player_stats.get_data_frame()
                else:
                    data_frames = misc_endpoint.get_data_frames()
                    if not data_frames:
                        return pd.DataFrame()
                    df = data_frames[0]

                if df.empty:
                    return pd.DataFrame()

                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personId': 'PLAYER_ID',
                    'pointsOffTurnovers': 'PTS_OFF_TOV',
                    'pointsSecondChance': 'PTS_2ND_CHANCE',
                    'pointsFastBreak': 'PTS_FB',
                    'pointsPaint': 'PTS_PAINT',
                    'oppPointsOffTurnovers': 'OPP_PTS_OFF_TOV',
                    'oppPointsSecondChance': 'OPP_PTS_2ND_CHANCE',
                    'oppPointsFastBreak': 'OPP_PTS_FB',
                    'oppPointsPaint': 'OPP_PTS_PAINT',
                    'blocks': 'BLK',
                    'blocksAgainst': 'BLKA',
                    'foulsPersonal': 'PF',
                    'foulsDrawn': 'PFD',
                }

                cols = [col for col in column_mapping if col in df.columns]
                misc_df = df[cols].rename(columns=column_mapping)
                return misc_df
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Misc stats for {normalized_game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching misc for {normalized_game_id}: {e}")
                    return pd.DataFrame()

        return pd.DataFrame()

    def fetchBoxScoreMatchups(self, game_id, sleep_time=None, max_retries=3, timeout=60):
        """Fetch BoxScoreMatchupsV3 data for a single game with defensive aggregations"""
        sleep_time = sleep_time or self.sleep_time
        normalized_game_id = self.normalize_game_id(game_id)
        if not normalized_game_id:
            print(f"[ERROR] Invalid game ID: {game_id}")
            return pd.DataFrame()
            
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))
                from nba_api.stats.endpoints import boxscorematchupsv3
                df = boxscorematchupsv3.BoxScoreMatchupsV3(
                    game_id=normalized_game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                
                if df.empty:
                    return pd.DataFrame()
                
                # Convert minutes from seconds to decimal format
                df['matchupMinutes'] = round(df['matchupMinutesSort'] / 60, 2)
                
                # Group by defender (personIdDef) to get defensive stats
                def_df = (
                    df.groupby('personIdDef')
                    .agg({
                        'gameId': 'first',
                        'teamId': 'first',
                        'matchupFieldGoalsMade': 'sum',
                        'matchupFieldGoalsAttempted': 'sum',
                        'matchupThreePointersMade': 'sum',
                        'matchupThreePointersAttempted': 'sum',
                        'playerPoints': 'sum',
                        'matchupMinutes': 'sum',
                        'matchupTurnovers': 'sum',
                        'matchupBlocks': 'sum',
                        'shootingFouls': 'sum',
                        'matchupAssists': 'sum'
                    })
                    .reset_index()
                )
                
                # Rename columns to match standard naming convention
                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personIdDef': 'PLAYER_ID',
                    'teamId': 'TEAM_ID'
                }
                def_df = def_df.rename(columns=column_mapping)
                
                # Also calculate matchup field goal and 3pt percentages
                def_df['matchupFieldGoalsPercentage'] = def_df.apply(
                    lambda row: round(row['matchupFieldGoalsMade'] / row['matchupFieldGoalsAttempted'], 3) 
                    if row['matchupFieldGoalsAttempted'] > 0 else 0, axis=1
                )
                def_df['matchupThreePointersPercentage'] = def_df.apply(
                    lambda row: round(row['matchupThreePointersMade'] / row['matchupThreePointersAttempted'], 3) 
                    if row['matchupThreePointersAttempted'] > 0 else 0, axis=1
                )
                
                # Select the columns we want to keep
                cols = [
                    'GAME_ID', 'PLAYER_ID',
                    'matchupFieldGoalsMade', 'matchupFieldGoalsAttempted',
                    'matchupThreePointersMade', 'matchupThreePointersAttempted',
                    'playerPoints', 'matchupMinutes', 'matchupFieldGoalsPercentage',
                    'matchupThreePointersPercentage'
                ]
                
                # Only select columns that exist in the dataframe
                existing_cols = [col for col in cols if col in def_df.columns]
                return def_df[existing_cols]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Matchup stats for {normalized_game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching matchup for {normalized_game_id}: {e}")
                    return pd.DataFrame()

    def fetchPlayByPlayStats(self, game_id, player_ids, sleep_time=None, max_retries=3, timeout=60):
        """Fetch play-by-play stats for all players in a game"""
        sleep_time = sleep_time or self.sleep_time
        normalized_game_id = self.normalize_game_id(game_id)
        if not normalized_game_id:
            print(f"[ERROR] Invalid game ID: {game_id}")
            return pd.DataFrame()
        
        all_pbp_stats = []
        
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))  # Exponential backoff
                
                # Fetch play-by-play data ONCE per game
                try:
                    game_pbp_data = cleanPlaybyPlay(normalized_game_id)
                    print(f"[SUCCESS] Fetched play-by-play data for game {normalized_game_id}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"[RETRY {attempt+1}/{max_retries}] Play-by-play data for {normalized_game_id}: {e}")
                        continue
                    else:
                        print(f"[ERROR] Failed to fetch play-by-play data for game {normalized_game_id}: {e}")
                        return pd.DataFrame()
                
                # Process each player using the same game data
                for player_id in player_ids:
                    try:
                        # No need to pre-check - engineerPlayerPlaybyPlayBasics handles it now
                        pbp_data = engineerPlayerPlaybyPlayBasics(normalized_game_id, player_id, game_pbp_data)
                        all_pbp_stats.append(pbp_data)
                    except Exception as e:
                        print(f"[ERROR] Processing player {player_id} in game {normalized_game_id}: {e}")
                        continue
                
                if all_pbp_stats:
                    return pd.DataFrame(all_pbp_stats)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}/{max_retries}] Play-by-play for game {normalized_game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching play-by-play for {normalized_game_id}: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()

        
    def getAdvancedStats(self, player_data, sleep_time=None, max_workers=None, cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv'):
        sleep_time = sleep_time or self.sleep_time
        max_workers = max_workers or min(10, os.cpu_count() or 4)
        game_ids = player_data['GAME_ID'].unique()
        total_games = len(game_ids)
        
        # Key advanced stats columns that should be present
        adv_cols = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT', 'USG_PCT']
        
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, dtype={'GAME_ID': str})
            # Check if all required columns exist and have data
            if all(col in cached.columns for col in adv_cols):
                cached = cached[cached[adv_cols].notna().any(axis=1)]
                cached_ids = cached['GAME_ID'].unique()
            else:
                cached, cached_ids = pd.DataFrame(), []
        else:
            cached, cached_ids = pd.DataFrame(), []
            
        missing = [gid for gid in game_ids if gid not in cached_ids]
        stats = [cached]
        
        if missing:
            print(f"\nFetching advanced stats for {len(missing)} out of {total_games} games...")
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.fetchAdvancedStats, gid, sleep_time): gid for gid in missing}
                for f in as_completed(futures):
                    df = f.result()
                    if not df.empty: 
                        stats.append(df)
                    completed += 1
                    print(f"\rProgress: {completed}/{len(missing)} games processed", end="", flush=True)
            print("\nFinished fetching advanced stats.")
        else:
            print(f"\nAll {total_games} games found in cache.")
            
        combined = pd.concat(stats, ignore_index=True).drop_duplicates(subset=['GAME_ID','PLAYER_ID'])
        combined.to_csv(cache_file, index=False)
        return combined

    def getTrackingStats(self, player_data, sleep_time=None, max_workers=None, cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv'):
        sleep_time = sleep_time or self.sleep_time
        max_workers = max_workers or min(10, os.cpu_count() or 4)
        game_ids = player_data['GAME_ID'].unique()
        total_games = len(game_ids)
        
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, dtype={'GAME_ID': str})
            track_cols = ['SPD','DIST','ORBC','DRBC','RBC']
            if all(c in cached for c in track_cols):
                cached = cached[cached[track_cols].notna().any(axis=1)]
                cached_ids = cached['GAME_ID'].unique()
            else:
                cached, cached_ids = pd.DataFrame(), []
        else:
            cached, cached_ids = pd.DataFrame(), []
            
        missing = [gid for gid in game_ids if gid not in cached_ids]
        stats = [cached]
        
        if missing:
            print(f"\nFetching tracking stats for {len(missing)} out of {total_games} games...")
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.fetchTrackingStats, gid, sleep_time): gid for gid in missing}
                for f in as_completed(futures):
                    df = f.result()
                    if not df.empty: stats.append(df)
                    completed += 1
                    print(f"\rProgress: {completed}/{len(missing)} games processed", end="", flush=True)
            print("\nFinished fetching tracking stats.")
        else:
            print(f"\nAll {total_games} tracking stats found in cache.")
            
        combined = pd.concat(stats, ignore_index=True).drop_duplicates(subset=['GAME_ID','PLAYER_ID'])
        combined.to_csv(cache_file, index=False)
        return combined

    def getScoringStats(self, player_data, sleep_time=None, max_workers=None, cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv'):
        """Fetch BoxScoreScoringV3 stats for multiple games with caching and threading"""
        sleep_time = sleep_time or self.sleep_time
        max_workers = max_workers or min(10, os.cpu_count() or 4)
        game_ids = player_data['GAME_ID'].unique()
        total_games = len(game_ids)
        
        # Key scoring stats columns that should be present
        scoring_cols = ['percentageFieldGoalsAttempted2pt', 'percentageFieldGoalsAttempted3pt', 
                       'percentagePoints2pt', 'percentagePoints3pt', 'percentagePointsPaint']
        
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, dtype={'GAME_ID': str})
            # Check if all required columns exist and have data
            if all(col in cached.columns for col in scoring_cols):
                cached = cached[cached[scoring_cols].notna().any(axis=1)]
                cached_ids = cached['GAME_ID'].unique()
            else:
                cached, cached_ids = pd.DataFrame(), []
        else:
            cached, cached_ids = pd.DataFrame(), []
            
        missing = [gid for gid in game_ids if gid not in cached_ids]
        stats = [cached]
        
        if missing:
            print(f"\nFetching scoring stats for {len(missing)} out of {total_games} games...")
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.fetchBoxScoreScoring, gid, sleep_time): gid for gid in missing}
                for f in as_completed(futures):
                    df = f.result()
                    if not df.empty: 
                        stats.append(df)
                    completed += 1
                    print(f"\rProgress: {completed}/{len(missing)} games processed", end="", flush=True)
            print("\nFinished fetching scoring stats.")
        else:
            print(f"\nAll {total_games} scoring stats found in cache.")
            
        combined = pd.concat(stats, ignore_index=True).drop_duplicates(subset=['GAME_ID','PLAYER_ID'])
        combined.to_csv(cache_file, index=False)
        return combined

    def getMatchupStats(self, player_data, sleep_time=None, max_workers=None, cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv'):
        """Fetch BoxScoreMatchupsV3 stats for multiple games with caching and threading"""
        sleep_time = sleep_time or self.sleep_time
        max_workers = max_workers or min(10, os.cpu_count() or 4)
        game_ids = player_data['GAME_ID'].unique()
        total_games = len(game_ids)
        
        # Key matchup stats columns that should be present
        matchup_cols = ['matchupFieldGoalsMade', 'matchupFieldGoalsAttempted', 
                       'matchupFieldGoalsPercentage', 'matchupMinutes']
        
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, dtype={'GAME_ID': str})
            # Check if all required columns exist and have data
            if all(col in cached.columns for col in matchup_cols):
                cached = cached[cached[matchup_cols].notna().any(axis=1)]
                cached_ids = cached['GAME_ID'].unique()
            else:
                cached, cached_ids = pd.DataFrame(), []
        else:
            cached, cached_ids = pd.DataFrame(), []
            
        missing = [gid for gid in game_ids if gid not in cached_ids]
        stats = [cached]
        
        if missing:
            print(f"\nFetching matchup stats for {len(missing)} out of {total_games} games...")
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.fetchBoxScoreMatchups, gid, sleep_time): gid for gid in missing}
                for f in as_completed(futures):
                    df = f.result()
                    if not df.empty: 
                        stats.append(df)
                    completed += 1
                    print(f"\rProgress: {completed}/{len(missing)} games processed", end="", flush=True)
            print("\nFinished fetching matchup stats.")
        else:
            print(f"\nAll {total_games} matchup stats found in cache.")
            
        combined = pd.concat(stats, ignore_index=True).drop_duplicates(subset=['GAME_ID','PLAYER_ID'])
        combined.to_csv(cache_file, index=False)
        return combined

    def getCompleteStats(self, season=None, season_type='Regular Season',
                         sleep_time=2, max_workers=3, batch_limit=None,
                         complete_cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv',
                         include_playbyplay=False):
        
        # Your existing cache loading code
        if os.path.exists(complete_cache_file):
            cache = pd.read_csv(complete_cache_file, dtype={'GAME_ID':str})
            # FIXED: Normalize existing game IDs to match normalization of new game IDs
            cache['GAME_ID'] = cache['GAME_ID'].apply(self.normalize_game_id)
            cache = cache.dropna(subset=['GAME_ID'])  # Remove any invalid game IDs
            existing_ids = set(cache['GAME_ID'].unique())
        else:
            cache, existing_ids = pd.DataFrame(), set()

        # Fetch full player stats and identify new games
        all_stats = self.fetchPlayerStats(season, season_type)
        all_stats['GAME_ID'] = all_stats['GAME_ID'].astype(str)
        
        # Normalize all game IDs to ensure consistent format
        all_stats['GAME_ID'] = all_stats['GAME_ID'].apply(self.normalize_game_id)
        all_stats = all_stats.dropna(subset=['GAME_ID'])  # Remove any invalid game IDs
        
        new_stats = all_stats[~all_stats['GAME_ID'].isin(existing_ids)]

        if new_stats.empty:
            print("No new games to update, returning cached data.")
            return cache

        new_game_ids = new_stats['GAME_ID'].unique()
        
        # Apply batch limit if specified
        if batch_limit and batch_limit > 0:
            new_game_ids = new_game_ids[:batch_limit]
            new_stats = new_stats[new_stats['GAME_ID'].isin(new_game_ids)]
            print(f"Processing {len(new_game_ids)} games (limited by batch_limit={batch_limit})")
        else:
            print(f"Processing all {len(new_game_ids)} new games")

        # Process games in batches
        batch_size = max_workers * 2  # Process 2 games per worker at a time
        game_batches = [new_game_ids[i:i + batch_size] for i in range(0, len(new_game_ids), batch_size)]
        
        all_advanced = []
        all_tracking = []
        all_scoring = []
        all_misc = []
        all_matchups = []
        all_playbyplay = []  # Add play-by-play stats array

        total_batches = len(game_batches)
        for batch_idx, game_batch in enumerate(game_batches, 1):
            print(f"\nProcessing batch {batch_idx}/{total_batches} ({len(game_batch)} games)")
            
            # Process each game in the batch
            for game_id in game_batch:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all stat types for this game concurrently
                    futures = {
                        'advanced': executor.submit(self.fetchAdvancedStats, game_id, sleep_time),
                        'tracking': executor.submit(self.fetchTrackingStats, game_id, sleep_time),
                        'scoring': executor.submit(self.fetchBoxScoreScoring, game_id, sleep_time),
                        'misc': executor.submit(self.fetchBoxScoreMisc, game_id, sleep_time),
                        'matchups': executor.submit(self.fetchBoxScoreMatchups, game_id, sleep_time),
                    }
                    
                    # Add play-by-play if requested
                    if include_playbyplay:
                        # Get player IDs for this game
                        game_player_ids = new_stats[new_stats['GAME_ID'] == game_id]['PLAYER_ID'].unique()
                        futures['playbyplay'] = executor.submit(self.fetchPlayByPlayStats, game_id, game_player_ids, sleep_time)
                    
                    # Collect results
                    results = {}
                    for stat_type, future in futures.items():
                        try:
                            result = future.result()
                            if not result.empty:
                                if stat_type == 'advanced': all_advanced.append(result)
                                elif stat_type == 'tracking': all_tracking.append(result)
                                elif stat_type == 'scoring': all_scoring.append(result)
                                elif stat_type == 'misc': all_misc.append(result)
                                elif stat_type == 'matchups': all_matchups.append(result)
                                elif stat_type == 'playbyplay': all_playbyplay.append(result)  # Add play-by-play handling
                        except Exception as e:
                            print(f"Error fetching {stat_type} stats for game {game_id}: {e}")

                # Sleep between games within a batch
                time.sleep(sleep_time)
            
            print(f"Completed batch {batch_idx}/{total_batches}")
            # Sleep between batches
            time.sleep(sleep_time * 2)

        # Combine all stats
        if all_advanced:
            advanced_stats = pd.concat(all_advanced, ignore_index=True)
            merged_player = self.mergeData(new_stats, advanced_stats)
        else:
            merged_player = new_stats

        if all_tracking:
            tracking_stats = pd.concat(all_tracking, ignore_index=True)
            merged_player = pd.merge(
                merged_player, 
                tracking_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )


        if all_scoring:
            scoring_stats = pd.concat(all_scoring, ignore_index=True)
            merged_player = pd.merge(
                merged_player, 
                scoring_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )

        # Add this new section for matchup stats
        if all_matchups:
            matchup_stats = pd.concat(all_matchups, ignore_index=True)
            merged_player = pd.merge(
                merged_player, 
                matchup_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )

        if all_misc:
            misc_stats = pd.concat(all_misc, ignore_index=True)
            # Drop TEAM_ID from misc_stats to avoid conflicts (already in base player stats)
            misc_stats = misc_stats.drop(columns=['TEAM_ID'], errors='ignore')
            merged_player = pd.merge(
                merged_player,
                misc_stats,
                on=['GAME_ID', 'PLAYER_ID'],
                how='left'
            )

        # Add play-by-play stats if requested
        if include_playbyplay and all_playbyplay:
            playbyplay_stats = pd.concat(all_playbyplay, ignore_index=True)
            merged_player = pd.merge(
                merged_player, 
                playbyplay_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )

        # Merge team stats with player stats
        print("\nMerging team stats...")
        merged_player = self.mergeTeamtoPlayer(merged_player, season=season, season_type=season_type)

        # Combine with existing cache and save
        if not cache.empty:
            cache = cache[~cache['GAME_ID'].isin(merged_player['GAME_ID'].unique())]
        combined = pd.concat([cache, merged_player], ignore_index=True)
        combined = combined.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID'], keep='last')
        # Ensure all game IDs are normalized before saving
        combined['GAME_ID'] = combined['GAME_ID'].apply(self.normalize_game_id)
        combined.to_csv(complete_cache_file, index=False)
        print(f"Cache updated. Total games now: {combined['GAME_ID'].nunique()}")
        
        return combined

    def mergeData(self, player_data, advanced_stats):
        player_data['GAME_ID'] = player_data['GAME_ID'].astype(str)
        advanced_stats['GAME_ID'] = advanced_stats['GAME_ID'].astype(str)
        advanced_stats['PLAYER_ID'] = advanced_stats['PLAYER_ID'].astype(int)

        adv_cols = [
            'GAME_ID', 'PLAYER_ID', 'START_POSITION', 'COMMENT',
            'OFF_RATING', 'E_OFF_RATING', 'DEF_RATING',
            'E_DEF_RATING', 'NET_RATING', 'OREB_PCT', 'DREB_PCT', 
            'REB_PCT', 'AST_PCT', 'EFG_PCT', 'AST_TOV', 'USG_PCT', 
            'TS_PCT', 'E_PACE', 'PACE', 'PIE', 'POSS',
            'PACE_PER40', 'E_USG_PCT', 'OPP_DEF_RATING', 'OPP_PACE'
        ]
        
        # Only keep columns that exist in advanced_stats
        existing_adv_cols = [col for col in adv_cols if col in advanced_stats.columns]
        
        return pd.merge(
            player_data, 
            advanced_stats[existing_adv_cols], 
            on=['GAME_ID', 'PLAYER_ID'], 
            how='left'
        )