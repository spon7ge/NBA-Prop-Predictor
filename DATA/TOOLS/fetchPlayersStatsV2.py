import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2, boxscoreplayertrackv3
from nba_api.stats.static import teams
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from FEATURES.featuresV2 import engineerPlayerPlaybyPlayBasics, quarterStatsDiff, cleanPlaybyPlay

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

        cols = [
            'PLAYER_NAME', 'PLAYER_ID', 'MATCHUP', 'TEAM_ABBREVIATION', 'TEAM_ID',
            'OPP_ABBREVIATION', 'HOME_GAME', 'GAME_ID', 'GAME_DATE', 'WL',
            'PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 
            'PLUS_MINUS', 'FANTASY_PTS'
        ]
        return df[cols]

    def fetchAdvancedStats(self, game_id, sleep_time=None, timeout=60):
        sleep_time = sleep_time or self.sleep_time
        for attempt in range(2):  # Single retry (2 attempts total)
            try:
                if attempt > 0:
                    time.sleep(sleep_time)
                df = boxscoreadvancedv2.BoxScoreAdvancedV2(
                    game_id=game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'START_POSITION', 
                    'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'OREB_PCT', 'DREB_PCT',
                    'REB_PCT', 'AST_PCT', 'EFG_PCT', 'AST_TOV', 'USG_PCT', 'TS_PCT',
                    'PACE', 'PIE', 'POSS', 'PACE_PER40'
                ]
                return df[cols]
            except Exception as e:
                if attempt == 0:
                    print(f"[RETRY] Game {game_id}: {e}")
                else:
                    print(f"[FAILED] Game {game_id}: {e}")
                    return pd.DataFrame()

    def fetchTrackingStats(self, game_id, sleep_time=None, timeout=60):
        sleep_time = sleep_time or self.sleep_time
        for attempt in range(2):  # Single retry
            try:
                if attempt > 0:
                    time.sleep(sleep_time)
                df = boxscoreplayertrackv3.BoxScorePlayerTrackV3(
                    game_id=game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                
                # Only rename the necessary columns
                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personId': 'PLAYER_ID',
                    'minutes': 'MIN'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Keep only the key columns you need
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'MIN', 'speed', 'distance',
                    'touches', 'secondaryAssists', 'freeThrowAssists',
                    'passes', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted',
                    'uncontestedFieldGoalsMade', 'uncontestedFieldGoalsAttempted',
                    'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted'
                ]
                
                existing_cols = [col for col in cols if col in df.columns]
                return df[existing_cols]
            
            except Exception as e:
                if attempt == 0:
                    print(f"[RETRY] Tracking stats for {game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching tracking for {game_id}: {e}")
                    return pd.DataFrame()

    def fetchPlayByPlayStats(self, game_id, player_ids, sleep_time=None):
        """Fetch play-by-play stats for all players in a game"""
        sleep_time = sleep_time or self.sleep_time
        all_pbp_stats = []
        
        for attempt in range(2):  # Single retry
            try:
                if attempt > 0:
                    time.sleep(sleep_time)
                
                # Fetch play-by-play data ONCE per game
                try:
                    game_pbp_data = cleanPlaybyPlay(game_id)
                    print(f"[SUCCESS] Fetched play-by-play data for game {game_id}")
                except Exception as e:
                    if attempt == 0:
                        raise  # Retry on first attempt
                    print(f"[ERROR] Failed to fetch play-by-play data for game {game_id}: {e}")
                    return pd.DataFrame()
                
                # Process each player using the same game data
                for player_id in player_ids:
                    try:
                        # Pass the pre-fetched data to avoid redundant API calls
                        pbp_data = engineerPlayerPlaybyPlayBasics(game_id, player_id, game_pbp_data)
                        all_pbp_stats.append(pbp_data)
                    except Exception as e:
                        print(f"[ERROR] Processing player {player_id} in game {game_id}: {e}")
                        continue
                
                if all_pbp_stats:
                    return pd.DataFrame(all_pbp_stats)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                if attempt == 0:
                    print(f"[RETRY] Play-by-play for game {game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching play-by-play for {game_id}: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()

    def getCompleteStats(self, season=None, season_type='Regular Season',
                         sleep_time=0.1, max_workers=3, batch_limit=None,
                         cache_dir='../DATA/CSV_FILES/REGULAR_DATA',
                         include_playbyplay=True):
        
        # Generate cache file path based on season
        season = season or self.default_season
        cache_file = os.path.join(cache_dir, f'complete_data_{season.replace("-", "_")}.csv')
        
        # Load cache
        if os.path.exists(cache_file):
            cache = pd.read_csv(cache_file, dtype={'GAME_ID':str})
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

        # Process all games concurrently with worker pool throttling
        all_advanced = []
        all_tracking = []
        all_playbyplay = []

        def process_game(game_id):
            """Process a single game and return all stat types"""
            results = {'game_id': game_id, 'advanced': None, 'tracking': None, 'playbyplay': None}
            
            # Get player IDs for this game
            game_player_ids = new_stats[new_stats['GAME_ID'] == game_id]['PLAYER_ID'].unique()
            
            try:
                results['advanced'] = self.fetchAdvancedStats(game_id, sleep_time)
            except Exception as e:
                print(f"  Error fetching advanced stats for game {game_id}: {e}")
            
            try:
                results['tracking'] = self.fetchTrackingStats(game_id, sleep_time)
            except Exception as e:
                print(f"  Error fetching tracking stats for game {game_id}: {e}")
            
            if include_playbyplay:
                try:
                    results['playbyplay'] = self.fetchPlayByPlayStats(game_id, game_player_ids, sleep_time)
                except Exception as e:
                    print(f"  Error fetching play-by-play for game {game_id}: {e}")
            
            return results

        # Submit all games to worker pool
        print(f"\nProcessing {len(new_game_ids)} games with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_game = {executor.submit(process_game, game_id): game_id for game_id in new_game_ids}
            
            for future in as_completed(future_to_game):
                game_id = future_to_game[future]
                try:
                    results = future.result()
                    print(f"  Completed game {game_id}")
                    
                    if results['advanced'] is not None and not results['advanced'].empty:
                        all_advanced.append(results['advanced'])
                    if results['tracking'] is not None and not results['tracking'].empty:
                        all_tracking.append(results['tracking'])
                    if results['playbyplay'] is not None and not results['playbyplay'].empty:
                        all_playbyplay.append(results['playbyplay'])
                except Exception as e:
                    print(f"  Error processing game {game_id}: {e}")

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
        os.makedirs(cache_dir, exist_ok=True)
        combined.to_csv(cache_file, index=False)
        print(f"\nCache updated: {cache_file}")
        print(f"Total games now: {combined['GAME_ID'].nunique()}")
        
        return combined

    def mergeData(self, player_data, advanced_stats):
        player_data['GAME_ID'] = player_data['GAME_ID'].astype(str)
        advanced_stats['GAME_ID'] = advanced_stats['GAME_ID'].astype(str)
        advanced_stats['PLAYER_ID'] = advanced_stats['PLAYER_ID'].astype(int)

        adv_cols = [
        'GAME_ID', 'PLAYER_ID', 'START_POSITION',
        'OFF_RATING', 'DEF_RATING', 'NET_RATING',
        'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'AST_PCT',
        'AST_TOV', 'USG_PCT', 'TS_PCT', 'EFG_PCT',
        'PACE', 'PIE', 'POSS', 'PACE_PER40'
    ]
        
        # Only keep columns that exist in advanced_stats
        existing_adv_cols = [col for col in adv_cols if col in advanced_stats.columns]
        
        return pd.merge(
            player_data, 
            advanced_stats[existing_adv_cols], 
            on=['GAME_ID', 'PLAYER_ID'], 
            how='left'
        )

