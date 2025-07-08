from nba_api.stats.endpoints import commonplayerinfo
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime

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
        """Fetch position, height, and weight for a single player"""
        try:
            time.sleep(delay_between_requests)
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            
            if not player_info.empty:
                position = player_info.iloc[0]['POSITION']
                height = player_info.iloc[0]['HEIGHT']
                weight = player_info.iloc[0]['WEIGHT']
                return player_id, position, height, weight
            else:
                return player_id, None, None, None
                
        except Exception as e:
            print(f"Error fetching data for PLAYER_ID {player_id}: {e}")
            return player_id, None, None, None
    
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
                player_id, position, height, weight = future.result()
                position_cache[player_id] = (position, height, weight)
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
    data['POSITION'] = data['PLAYER_ID'].map(lambda pid: position_cache.get(pid, (None, None, None))[0])
    data['HEIGHT'] = data['PLAYER_ID'].map(lambda pid: position_cache.get(pid, (None, None, None))[1])
    data['WEIGHT'] = data['PLAYER_ID'].map(lambda pid: position_cache.get(pid, (None, None, None))[2])
    
    # Create binary flags for simplified position types
    print("Creating position flags...")
    data['GUARD'] = data['POSITION'].str.contains('G', na=False).astype(int)
    data['FORWARD'] = data['POSITION'].str.contains('F', na=False).astype(int)
    data['CENTER'] = data['POSITION'].str.contains('C', na=False).astype(int)
    
    # Drop the original POSITION column
    data = data.drop('POSITION', axis=1)
    
    print("Position assignment completed!")
    
    # After all modifications to the DataFrame
    data = data.copy()
    return data

def assign_position_with_cache(data, cache_file='playerInfo.csv', max_workers=4, delay_between_requests=0.5):
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
        position_cache = dict(zip(cache_df['PLAYER_ID'], zip(cache_df['POSITION'], cache_df['HEIGHT'], cache_df['WEIGHT'])))
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
            """Fetch position, height, and weight for a single player"""
            try:
                time.sleep(delay_between_requests)
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                
                if not player_info.empty:
                    position = player_info.iloc[0]['POSITION']
                    height = player_info.iloc[0]['HEIGHT']
                    weight = player_info.iloc[0]['WEIGHT']
                    return player_id, position, height, weight
                else:
                    return player_id, None, None, None
                    
            except Exception as e:
                print(f"Error fetching data for PLAYER_ID {player_id}: {e}")
                return player_id, None, None, None
        
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
                    player_id, position, height, weight = future.result()
                    position_cache[player_id] = (position, height, weight)
                    completed += 1
                    
                    print(f"Fetched PLAYER_ID {player_id}... ({completed}/{len(uncached_ids)})")
                    
                except Exception as e:
                    player_id = future_to_player[future]
                    print(f"Error processing PLAYER_ID {player_id}: {e}")
                    position_cache[player_id] = None
    
    # Save updated cache
    print("Saving updated cache...")
    cache_data = [{'PLAYER_ID': pid, 'POSITION': pos, 'HEIGHT': ht, 'WEIGHT': wt} 
                  for pid, (pos, ht, wt) in position_cache.items()]
    pd.DataFrame(cache_data).to_csv(cache_file, index=False)
    print(f"Cache saved with {len(position_cache)} players")
    
    # Apply positions to data
    print("Applying positions to dataset...")
    data['POSITION'] = data['PLAYER_ID'].map(lambda pid: position_cache.get(pid, (None, None, None))[0])
    data['HEIGHT'] = data['PLAYER_ID'].map(lambda pid: position_cache.get(pid, (None, None, None))[1])
    data['WEIGHT'] = data['PLAYER_ID'].map(lambda pid: position_cache.get(pid, (None, None, None))[2])
    
    # Create binary flags for simplified position types
    print("Creating position flags...")
    data['GUARD'] = data['POSITION'].str.contains('G', na=False).astype(int)
    data['FORWARD'] = data['POSITION'].str.contains('F', na=False).astype(int)
    data['CENTER'] = data['POSITION'].str.contains('C', na=False).astype(int)
    
    data = data.drop('POSITION', axis=1)
    
    print("Position assignment completed!")
    return data
