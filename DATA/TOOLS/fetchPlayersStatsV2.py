import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2, teamgamelog, boxscoreplayertrackv3
from nba_api.stats.static import teams
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from FEATURES.featuresV2 import engineerPlayerPlaybyPlayBasics, quarterStatsDiff

class FetchPlayersStatsV2:
    def __init__(self, default_season='2024-25', sleep_time=0.1):
        self.default_season = default_season
        self.sleep_time = sleep_time

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
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))  # Exponential backoff
                df = boxscoreadvancedv2.BoxScoreAdvancedV2(
                    game_id=game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'START_POSITION', 'COMMENT',
                    'OFF_RATING', 'E_OFF_RATING', 'DEF_RATING',
                    'E_DEF_RATING', 'NET_RATING', 'OREB_PCT', 'DREB_PCT',
                    'REB_PCT', 'AST_PCT', 'EFG_PCT', 'AST_TOV', 'USG_PCT',
                    'TS_PCT', 'E_PACE', 'PACE', 'PIE', 'POSS',
                    'PACE_PER40', 'E_USG_PCT'
                ]
                return df[cols]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n[RETRY {attempt+1}/{max_retries}] Game {game_id}: {e}")
                else:
                    print(f"\n[FAILED] Game {game_id} after {max_retries} attempts: {e}")
                    return pd.DataFrame()

    def fetchTrackingStats(self, game_id, sleep_time=None, max_retries=3, timeout=60):
        sleep_time = sleep_time or self.sleep_time
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))
                df = boxscoreplayertrackv3.BoxScorePlayerTrackV3(
                    game_id=game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                
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
                
                df = df.rename(columns=column_mapping)
                
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'MIN', 'SPD', 'DIST', 'ORBC', 'DRBC', 'RBC',
                    'TCHS', 'SAST', 'FTAST', 'PASS', 'CFGM', 'CFGA', 'CFG_PCT',
                    'UFGM', 'UFGA', 'UFG_PCT', 'DFGM', 'DFGA', 'DFG_PCT'
                ]
                
                existing_cols = [col for col in cols if col in df.columns]
                return df[existing_cols]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Tracking stats for {game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching tracking for {game_id}: {e}")
                    return pd.DataFrame()

    def fetchPlayByPlayStats(self, game_id, player_ids, sleep_time=None, max_retries=3):
        """Fetch play-by-play stats for all players in a game"""
        sleep_time = sleep_time or self.sleep_time
        all_pbp_stats = []
        
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))
                
                for player_id in player_ids:
                    try:
                        # Get play-by-play features
                        pbp_data = engineerPlayerPlaybyPlayBasics(game_id, player_id)
                        pbp_data = quarterStatsDiff(pbp_data)
                        all_pbp_stats.append(pbp_data)
                    except Exception as e:
                        print(f"[ERROR] Play-by-play for player {player_id} in game {game_id}: {e}")
                        continue
                
                if all_pbp_stats:
                    return pd.DataFrame(all_pbp_stats)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Play-by-play for game {game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching play-by-play for {game_id}: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def getCompleteStats(self, season=None, season_type='Regular Season',
                         sleep_time=2, max_workers=3, batch_limit=None,
                         complete_cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA_V2.csv',
                         include_playbyplay=True):
        
        # Load cache
        if os.path.exists(complete_cache_file):
            cache = pd.read_csv(complete_cache_file, dtype={'GAME_ID':str})
            existing_ids = set(cache['GAME_ID'].astype(str).unique())
        else:
            cache, existing_ids = pd.DataFrame(), set()

        # Fetch full player stats and identify new games
        all_stats = self.fetchPlayerStats(season, season_type)
        all_stats['GAME_ID'] = all_stats['GAME_ID'].astype(str)
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
        batch_size = max_workers * 2
        game_batches = [new_game_ids[i:i + batch_size] for i in range(0, len(new_game_ids), batch_size)]
        
        all_advanced = []
        all_tracking = []
        all_playbyplay = []

        total_batches = len(game_batches)
        for batch_idx, game_batch in enumerate(game_batches, 1):
            print(f"\nProcessing batch {batch_idx}/{total_batches} ({len(game_batch)} games)")
            
            # Process each game in the batch
            for game_id in game_batch:
                print(f"  Processing game {game_id}...")
                
                # Get player IDs for this game
                game_player_ids = new_stats[new_stats['GAME_ID'] == game_id]['PLAYER_ID'].unique()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all stat types for this game concurrently
                    futures = {
                        'advanced': executor.submit(self.fetchAdvancedStats, game_id, sleep_time),
                        'tracking': executor.submit(self.fetchTrackingStats, game_id, sleep_time),
                    }
                    
                    # Add play-by-play if requested
                    if include_playbyplay:
                        futures['playbyplay'] = executor.submit(
                            self.fetchPlayByPlayStats, game_id, game_player_ids, sleep_time
                        )
                    
                    # Collect results
                    for stat_type, future in futures.items():
                        try:
                            result = future.result()
                            if not result.empty:
                                if stat_type == 'advanced': all_advanced.append(result)
                                elif stat_type == 'tracking': all_tracking.append(result)
                                elif stat_type == 'playbyplay': all_playbyplay.append(result)
                        except Exception as e:
                            print(f"  Error fetching {stat_type} stats for game {game_id}: {e}")

                # Sleep between games within a batch
                time.sleep(sleep_time)
            
            print(f"Completed batch {batch_idx}/{total_batches}")
            # Sleep between batches
            time.sleep(sleep_time * 2)

        # Combine all stats
        merged_player = new_stats.copy()
        
        if all_advanced:
            print("\nMerging advanced stats...")
            advanced_stats = pd.concat(all_advanced, ignore_index=True)
            merged_player = self.mergeData(merged_player, advanced_stats)

        if all_tracking:
            print("Merging tracking stats...")
            tracking_stats = pd.concat(all_tracking, ignore_index=True)
            merged_player = pd.merge(
                merged_player, 
                tracking_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )

        if all_playbyplay:
            print("Merging play-by-play stats...")
            playbyplay_stats = pd.concat(all_playbyplay, ignore_index=True)
            # Convert PLAYER_ID to int for matching
            playbyplay_stats['PLAYER_ID'] = playbyplay_stats['PLAYER_ID'].astype(int)
            playbyplay_stats['GAME_ID'] = playbyplay_stats['GAME_ID'].astype(str)
            merged_player = pd.merge(
                merged_player, 
                playbyplay_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )

        # Combine with existing cache and save
        combined = pd.concat([cache, merged_player], ignore_index=True)
        combined.to_csv(complete_cache_file, index=False)
        print(f"\nCache updated. Total games now: {combined['GAME_ID'].nunique()}")
        
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

