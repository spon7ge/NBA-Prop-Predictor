import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2, teamgamelog, boxscoreplayertrackv2
from nba_api.stats.static import teams
from concurrent.futures import ThreadPoolExecutor, as_completed

class FetchPlayersStats:
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
        """
        Fetches advanced stats for a specific game with retry logic.
        
        Args:
            game_id (str): NBA game ID
            sleep_time (float, optional): Time to sleep between API calls
            max_retries (int, optional): Maximum number of retry attempts
            timeout (int, optional): Timeout for the API request in seconds
            
        Returns:
            pd.DataFrame: Advanced statistics
        """
        sleep_time = sleep_time or self.sleep_time
        for attempt in range(max_retries):
            try:
                time.sleep(sleep_time * (attempt + 1))  # Exponential backoff
                df = boxscoreadvancedv2.BoxScoreAdvancedV2(
                    game_id=game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                # List of columns we want to keep
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
                df = boxscoreplayertrackv2.BoxScorePlayerTrackV2(
                    game_id=game_id,
                    timeout=timeout
                ).get_data_frames()[0]
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'MIN', 'SPD', 'DIST', 'ORBC', 'DRBC', 'RBC',
                    'TCHS', 'SAST', 'FTAST', 'PASS', 'CFGM', 'CFGA', 'CFG_PCT',
                    'UFGM', 'UFGA', 'UFG_PCT', 'DFGM', 'DFGA', 'DFG_PCT'
                ]
                return df[cols]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt+1}] Tracking stats for {game_id}: {e}")
                else:
                    print(f"[ERROR] Failed fetching tracking for {game_id}: {e}")
                    return pd.DataFrame()

    def fetchBoxScoreMisc(self, game_id, sleep_time=None):
        """
        Fetches miscellaneous boxscore stats for a specific game.
        
        Args:
            game_id (str): NBA game ID
            sleep_time (float, optional): Time to sleep between API calls
            
        Returns:
            pd.DataFrame: Miscellaneous boxscore statistics
        """
        sleep_time = sleep_time or self.sleep_time
        try:
            time.sleep(sleep_time)
            from nba_api.stats.endpoints import boxscoremiscv2
            df = boxscoremiscv2.BoxScoreMiscV2(game_id=game_id).get_data_frames()[0]
            cols = [
                'GAME_ID', 'PLAYER_ID', 'PTS_OFF_TOV', 'PTS_2ND_CHANCE',
                'PTS_FB', 'PTS_PAINT', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE',
                'OPP_PTS_FB', 'OPP_PTS_PAINT', 'BLKA', 'PF', 'PFD'
            ]
            return df[cols]
        except Exception as e:
            print(f"[ERROR] Misc stats for game {game_id}: {e}")
            return pd.DataFrame()

    def getMiscStats(self, player_data, sleep_time=None, max_workers=None, cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv'):
        """
        Gets miscellaneous stats for all games in player_data.
        
        Args:
            player_data (pd.DataFrame): Player statistics data
            sleep_time (float, optional): Time to sleep between API calls
            max_workers (int, optional): Maximum number of concurrent workers
            cache_file (str, optional): Path to cache file
            
        Returns:
            pd.DataFrame: Combined miscellaneous statistics
        """
        sleep_time = sleep_time or self.sleep_time
        max_workers = max_workers or min(10, os.cpu_count() or 4)
        game_ids = player_data['GAME_ID'].unique()
        total_games = len(game_ids)
        
        # Check cache
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file, dtype={'GAME_ID': str})
            misc_cols = ['PTS_OFF_TOV', 'PTS_2ND_CHANCE', 'PTS_FB', 'PTS_PAINT']
            if all(c in cached for c in misc_cols):
                cached = cached[cached[misc_cols].notna().any(axis=1)]
                cached_ids = cached['GAME_ID'].unique()
            else:
                cached, cached_ids = pd.DataFrame(), []
        else:
            cached, cached_ids = pd.DataFrame(), []
            
        # Get missing games
        missing = [gid for gid in game_ids if gid not in cached_ids]
        stats = [cached]
        
        if missing:
            print(f"\nFetching misc stats for {len(missing)} out of {total_games} games...")
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.fetchBoxScoreMisc, gid, sleep_time): gid for gid in missing}
                for f in as_completed(futures):
                    df = f.result()
                    if not df.empty: stats.append(df)
                    completed += 1
                    print(f"\rProgress: {completed}/{len(missing)} games processed", end="", flush=True)
            print("\nFinished fetching misc stats.")
        else:
            print(f"\nAll {total_games} misc stats found in cache.")
            
        combined = pd.concat(stats, ignore_index=True).drop_duplicates(subset=['GAME_ID','PLAYER_ID'])
        combined.to_csv(cache_file, index=False)
        return combined
        
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

    def getCompleteStats(self, season=None, season_type='Regular Season',
                         sleep_time=2, max_workers=3, batch_limit=None,
                         complete_cache_file='../DATA/CSV_FILES/REGULAR_DATA/ALL_COMPLETE_DATA.csv'):
        
        # Your existing cache loading code
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
        batch_size = max_workers * 2  # Process 2 games per worker at a time
        game_batches = [new_game_ids[i:i + batch_size] for i in range(0, len(new_game_ids), batch_size)]
        
        all_advanced = []
        all_tracking = []
        all_misc = []

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
                        'misc': executor.submit(self.fetchBoxScoreMisc, game_id, sleep_time),
                    }
                    
                    # Collect results
                    results = {}
                    for stat_type, future in futures.items():
                        try:
                            result = future.result()
                            if not result.empty:
                                if stat_type == 'advanced': all_advanced.append(result)
                                elif stat_type == 'tracking': all_tracking.append(result)
                                elif stat_type == 'misc': all_misc.append(result)
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

        if all_misc:
            misc_stats = pd.concat(all_misc, ignore_index=True)
            merged_player = pd.merge(
                merged_player, 
                misc_stats, 
                on=['GAME_ID', 'PLAYER_ID'], 
                how='left'
            )

        
        # Remove getTeamData, addOpponentStats, addOffensiveRating, add_pace_stats, and all team_data usage
        # Remove getTeamData
        # Remove addOpponentStats
        # Remove addOffensiveRating
        # Remove add_pace_stats
        # Remove all lines assigning or using team_data in getCompleteStats and elsewhere
        # Remove merging with team_data
        # Remove any references to these functions in the file
        # The rest of the code remains unchanged

        # Combine with existing cache and save
        combined = pd.concat([cache, merged_player], ignore_index=True)
        combined.to_csv(complete_cache_file, index=False)
        print(f"Cache updated. Total games now: {combined['GAME_ID'].nunique()}")
        
        return combined

    def mergeData(self, player_data, advanced_stats):
        """
        Merges basic player data with advanced stats.
        """
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
