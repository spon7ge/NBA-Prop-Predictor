from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

def starters(data):
    starters = ['G','F','C']
    if data['START_POSITION'] in starters:
        return 1
    else:
        return 0
    
def assign_position(data, max_workers=4, delay_between_requests=0.5):
    """
    Optimized version with parallel processing and caching
    
    Parameters:
    - data: DataFrame containing PLAYER_ID column
    - max_workers: Number of parallel threads (keep low to respect API limits)
    - delay_between_requests: Delay between requests to avoid rate limiting
    """
    
    print("Extracting unique player IDs...")
    unique_ids = data['PLAYER_ID'].unique()
    total_players = len(unique_ids)
    
    print(f"Found {total_players} unique players to process...")
    
    # Cache for storing results
    position_cache = {}
    
    def fetch_player_position(player_id):
        """Fetch position for a single player"""
        try:
            # Add small delay to respect API rate limits
            time.sleep(delay_between_requests)
            
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            
            if not player_info.empty:
                position = player_info.iloc[0]['POSITION']
                return player_id, position
            else:
                return player_id, None
                
        except Exception as e:
            print(f"Error fetching data for PLAYER_ID {player_id}: {e}")
            return player_id, None
    
    # Process players in parallel with controlled concurrency
    print(f"Fetching player positions using {max_workers} threads...")
    
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_player = {
            executor.submit(fetch_player_position, player_id): player_id 
            for player_id in unique_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_player):
            try:
                player_id, position = future.result()
                position_cache[player_id] = position
                completed += 1
                
                print(f"Gathered data for PLAYER_ID {player_id}... ({completed}/{total_players})")
                
                # Show progress every 10 completions or at the end
                if completed % 10 == 0 or completed == total_players:
                    print(f"Progress: {completed}/{total_players} players processed")
                    
            except Exception as e:
                player_id = future_to_player[future]
                print(f"Error processing PLAYER_ID {player_id}: {e}")
                position_cache[player_id] = None
    
    print(f"Successfully processed {len([v for v in position_cache.values() if v is not None])} players")
    
    # Apply positions to data
    print("Applying positions to dataset...")
    data['POSITION'] = data['PLAYER_ID'].map(position_cache)
    
    # Create binary flags for simplified position types
    print("Creating position flags...")
    data['GUARD'] = data['POSITION'].str.contains('G', na=False).astype(int)
    data['FORWARD'] = data['POSITION'].str.contains('F', na=False).astype(int)
    data['CENTER'] = data['POSITION'].str.contains('C', na=False).astype(int)
    
    # Drop the original POSITION column
    data = data.drop('POSITION', axis=1)
    
    print("Position assignment completed!")
    return data

def assign_position_with_cache(data, cache_file='player_positions_cache.csv', max_workers=4, delay_between_requests=0.5):
    """
    Enhanced version with persistent caching to avoid re-fetching known players
    
    Parameters:
    - data: DataFrame containing PLAYER_ID column
    - cache_file: Path to CSV file for caching player positions
    - max_workers: Number of parallel threads
    - delay_between_requests: Delay between requests
    """
    
    print("Loading position cache...")
    
    # Try to load existing cache
    try:
        cache_df = pd.read_csv(cache_file)
        position_cache = dict(zip(cache_df['PLAYER_ID'], cache_df['POSITION']))
        print(f"Loaded {len(position_cache)} players from cache")
    except FileNotFoundError:
        print("No cache file found, starting fresh")
        position_cache = {}
    
    # Find players not in cache
    unique_ids = data['PLAYER_ID'].unique()
    uncached_ids = [pid for pid in unique_ids if pid not in position_cache]
    
    print(f"Found {len(unique_ids)} unique players, {len(uncached_ids)} need to be fetched")
    
    if uncached_ids:
        def fetch_player_position(player_id):
            """Fetch position for a single player"""
            try:
                time.sleep(delay_between_requests)
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                
                if not player_info.empty:
                    position = player_info.iloc[0]['POSITION']
                    return player_id, position
                else:
                    return player_id, None
                    
            except Exception as e:
                print(f"Error fetching data for PLAYER_ID {player_id}: {e}")
                return player_id, None
        
        # Process uncached players in parallel
        print(f"Fetching {len(uncached_ids)} new players using {max_workers} threads...")
        
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_player = {
                executor.submit(fetch_player_position, player_id): player_id 
                for player_id in uncached_ids
            }
            
            for future in as_completed(future_to_player):
                try:
                    player_id, position = future.result()
                    position_cache[player_id] = position
                    completed += 1
                    
                    print(f"Fetched PLAYER_ID {player_id}... ({completed}/{len(uncached_ids)})")
                    
                except Exception as e:
                    player_id = future_to_player[future]
                    print(f"Error processing PLAYER_ID {player_id}: {e}")
                    position_cache[player_id] = None
        
        # Save updated cache
        print("Saving updated cache...")
        cache_data = [{'PLAYER_ID': pid, 'POSITION': pos} for pid, pos in position_cache.items()]
        pd.DataFrame(cache_data).to_csv(cache_file, index=False)
        print(f"Cache saved with {len(position_cache)} players")
    
    # Apply positions to data
    print("Applying positions to dataset...")
    data['POSITION'] = data['PLAYER_ID'].map(position_cache)
    
    # Create binary flags
    print("Creating position flags...")
    data['GUARD'] = data['POSITION'].str.contains('G', na=False).astype(int)
    data['FORWARD'] = data['POSITION'].str.contains('F', na=False).astype(int)
    data['CENTER'] = data['POSITION'].str.contains('C', na=False).astype(int)
    
    data = data.drop('POSITION', axis=1)
    
    print("Position assignment completed!")
    return data
