import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.endpoints import boxscorematchupsv3

class MatchupDataFetcher:
    def __init__(self, cache_dir='matchup_cache', max_workers=3, sleep_time=1.5):
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.failed_games = []
        self.completed_games = set()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self.load_cache_status()
    
    def get_cache_file(self, game_id):
        """Get cache file path for a specific game"""
        return os.path.join(self.cache_dir, f'game_{game_id}.pkl')
    
    def is_cached(self, game_id):
        """Check if a game is already cached"""
        return os.path.exists(self.get_cache_file(game_id))
    
    def save_game_data(self, game_id, data):
        """Save game data to cache"""
        cache_file = self.get_cache_file(game_id)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        self.completed_games.add(game_id)
    
    def load_game_data(self, game_id):
        """Load game data from cache"""
        cache_file = self.get_cache_file(game_id)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_cache_status(self):
        """Load which games are already completed"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('game_') and f.endswith('.pkl')]
        self.completed_games = set(f.replace('game_', '').replace('.pkl', '') for f in cache_files)
        print(f"Found {len(self.completed_games)} cached games")
    
    def fetch_single_game(self, game_id, retry_count=3):
        """Fetch data for a single game with retry logic"""
        for attempt in range(retry_count):
            try:
                time.sleep(self.sleep_time)
                
                boxscore = boxscorematchupsv3.BoxScoreMatchupsV3(
                    game_id=f'00{game_id}',
                    timeout=60
                )
                data_frames = boxscore.get_data_frames()
                
                if data_frames and len(data_frames) > 0 and not data_frames[0].empty:
                    # Process the data
                    boxscore_df = data_frames[0]
                    boxscore_df['matchupMinutes'] = round(boxscore_df['matchupMinutesSort'] / 60, 2)
                    
                    def_df = (
                        boxscore_df.groupby('personIdDef')
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
                    
                    # Calculate percentages safely
                    def_df['DEF_FG_PCT_ALLOWED'] = def_df.apply(
                        lambda row: round(row['matchupFieldGoalsMade'] / row['matchupFieldGoalsAttempted'], 3) 
                        if row['matchupFieldGoalsAttempted'] > 0 else 0, axis=1
                    )
                    def_df['DEF_3PT_PCT_ALLOWED'] = def_df.apply(
                        lambda row: round(row['matchupThreePointersMade'] / row['matchupThreePointersAttempted'], 3) 
                        if row['matchupThreePointersAttempted'] > 0 else 0, axis=1
                    )
                    def_df['PTS_ALLOWED_PER_MIN'] = def_df.apply(
                        lambda row: round(row['playerPoints'] / row['matchupMinutes'], 2) 
                        if row['matchupMinutes'] > 0 else 0, axis=1
                    )
                    def_df['DEF_TOV_FORCED_PER_MIN'] = def_df.apply(
                    lambda row: round(row['matchupTurnovers'] / row['matchupMinutes'], 2) 
                    if row['matchupMinutes'] > 0 else 0, axis=1
                )
                    def_df['DEF_BLOCKS_PER_MIN'] = def_df.apply(
                        lambda row: round(row['matchupBlocks'] / row['matchupMinutes'], 2) 
                        if row['matchupMinutes'] > 0 else 0, axis=1
                    )
                    def_df['DEF_SHOOTING_FOULS_PER_MIN'] = def_df.apply(
                        lambda row: round(row['shootingFouls'] / row['matchupMinutes'], 2) 
                        if row['matchupMinutes'] > 0 else 0, axis=1
                    )
                    def_df['DEF_AST_ALLOWED_PER_MIN'] = def_df.apply(
                        lambda row: round(row['matchupAssists'] / row['matchupMinutes'], 2) 
                        if row['matchupMinutes'] > 0 else 0, axis=1
                    )
                    return def_df, None
                else:
                    return None, "No data available"
                    
            except ReadTimeout:
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None, "Timeout after retries"
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                return None, str(e)
        
        return None, "Max retries exceeded"
    
    def fetch_games_bulk(self, game_ids, resume=True):
        """Fetch games in bulk with caching and resume capability"""
        if resume:
            # Filter out already completed games
            remaining_games = [gid for gid in game_ids if str(gid) not in self.completed_games]
            print(f"Resuming: {len(remaining_games)} games remaining out of {len(game_ids)}")
        else:
            remaining_games = game_ids
            print(f"Fetching all {len(game_ids)} games (not resuming)")
        
        if not remaining_games:
            print("All games already cached!")
            return self.load_all_cached_data(game_ids)
        
        print(f"Processing {len(remaining_games)} games using ThreadPoolExecutor...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for remaining games
            future_to_game = {
                executor.submit(self.fetch_single_game, game_id): game_id 
                for game_id in remaining_games
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_game), 1):
                game_id = future_to_game[future]
                
                try:
                    result, error = future.result()
                    
                    if result is not None:
                        # Save to cache
                        self.save_game_data(str(game_id), result)
                        print(f"✓ Game {game_id} completed and cached ({i}/{len(remaining_games)})")
                    else:
                        self.failed_games.append((game_id, error))
                        print(f"✗ Game {game_id} failed: {error} ({i}/{len(remaining_games)})")
                        
                except Exception as e:
                    self.failed_games.append((game_id, str(e)))
                    print(f"✗ Game {game_id} exception: {str(e)} ({i}/{len(remaining_games)})")
        
        print(f"\nCompleted!")
        print(f"Successfully processed: {len(remaining_games) - len(self.failed_games)} games")
        print(f"Failed games: {len(self.failed_games)}")
        
        return self.load_all_cached_data(game_ids)
    
    def load_all_cached_data(self, game_ids):
        """Load all cached data for the given game IDs"""
        all_data = []
        missing_games = []
        
        for game_id in game_ids:
            cached_data = self.load_game_data(str(game_id))
            if cached_data is not None:
                all_data.append(cached_data)
            else:
                missing_games.append(game_id)
        
        if missing_games:
            print(f"Warning: {len(missing_games)} games not found in cache")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(all_data)} games, total shape: {combined_data.shape}")
            return combined_data
        else:
            print("No cached data found!")
            return pd.DataFrame()
    
    def retry_failed_games(self):
        """Retry games that previously failed"""
        if not self.failed_games:
            print("No failed games to retry")
            return pd.DataFrame()
        
        print(f"Retrying {len(self.failed_games)} failed games...")
        failed_game_ids = [game_id for game_id, _ in self.failed_games]
        self.failed_games = []  # Clear the failed list
        
        return self.fetch_games_bulk(failed_game_ids, resume=False)
    
    def get_cache_stats(self):
        """Get statistics about the cache"""
        total_cached = len(self.completed_games)
        failed_count = len(self.failed_games)
        
        print(f"Cache Statistics:")
        print(f"  Cached games: {total_cached}")
        print(f"  Failed games: {failed_count}")
        print(f"  Cache directory: {self.cache_dir}")
        
        return {
            'cached_games': total_cached,
            'failed_games': failed_count,
            'cache_dir': self.cache_dir
        }

# Usage example:
# Initialize the fetcher
fetcher = MatchupDataFetcher(cache_dir='matchup_cache', max_workers=4, sleep_time=1.5)

# Get all game IDs
gameIds = s19_regular['GAME_ID'].unique()

# Fetch all games (will resume from cache if interrupted)
gameLogsTotal = fetcher.fetch_games_bulk(gameIds, resume=True)

# # Check cache statistics
fetcher.get_cache_stats()



def mergeMatchupDATA(data, matchup_data):
    data = data.copy()
    matchup_data = matchup_data.copy()
    matchup_data.rename(columns={'personIdDef': 'PLAYER_ID', 'teamId': 'TEAM_ID', 'gameId': 'GAME_ID'}, inplace=True)

    if 'GAME_ID' in matchup_data.columns:
        matchup_data['GAME_ID'] = matchup_data['GAME_ID'].astype(int)
    
    # Ensure main data PLAYER_ID is also string
    if 'PLAYER_ID' in data.columns:
        data['PLAYER_ID'] = data['PLAYER_ID']
    
    merge_columns = [
        'GAME_ID', 'PLAYER_ID',
        'matchupFieldGoalsMade', 'matchupFieldGoalsAttempted',
        'matchupThreePointersMade', 'matchupThreePointersAttempted',
        'playerPoints', 'matchupMinutes', 'matchupFieldGoalsPercentage',
        'matchupThreePointersPercentage', 'DEF_FG_PCT_ALLOWED',
        'DEF_3PT_PCT_ALLOWED', 'PTS_ALLOWED_PER_MIN',
        'DEF_TOV_FORCED_PER_MIN', 'DEF_BLOCKS_PER_MIN', 'DEF_SHOOTING_FOULS_PER_MIN', 'DEF_AST_ALLOWED_PER_MIN'
    ]
    
    available_columns = [col for col in merge_columns if col in matchup_data.columns]
    matchup_subset = matchup_data[available_columns].copy()
    
    merged_data = data.merge(
        matchup_subset,
        on=['GAME_ID', 'PLAYER_ID'],
        how='left',
        suffixes=('', '_matchup')
    )
    return merged_data

df = mergeMatchupDATA(s19_regular, gameLogsTotal)
df.head()
# If you want to retry failed games later
# gameLogsTotal = fetcher.retry_failed_games()