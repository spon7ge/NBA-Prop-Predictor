import json
import requests
import time
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Add parent directory to path to import Supplier
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Supplier import Supplier

class PrizePicks_Scraper():
    def __init__(self):
        supplier = Supplier()
        self.lines = []
        directory = supplier.getDirectory()
        
        if not directory:
            raise ValueError("PROJECTIONS_DIRECTORY environment variable is not set. Please set it in your .env file.")
        
        # If directory is a path without filename, append a filename
        # Check if it ends with a slash (directory) or has no extension (assume directory)
        if directory.endswith(('/', '\\')) or (os.path.exists(directory) and os.path.isdir(directory)):
            self.directory = os.path.join(directory, 'prizepicks_projections.json')
        elif not os.path.splitext(directory)[1]:  # No extension, assume it's a directory
            self.directory = os.path.join(directory, 'prizepicks_projections.json')
        else:
            self.directory = directory
        
        print(f"Using file path: {self.directory}")
        self.getJSON()
        self.loadJSON()
    
    def getJSON(self):
        """
        Fetch JSON data directly from PrizePicks API using HTTP request.
        This replaces the fragile Selenium + pyautogui approach.
        """
        url = "https://api.prizepicks.com/projections?league_id=7"
        
        # Headers to mimic a browser request (may help avoid blocking)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://app.prizepicks.com/',
            'Origin': 'https://app.prizepicks.com'
        }
        
        try:
            # Make HTTP request to API
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Get JSON data
            data = response.json()
            
            # Ensure directory exists
            directory_path = os.path.dirname(self.directory)
            if directory_path and not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
            
            # Save JSON to file (maintains compatibility with loadJSON method)
            with open(self.directory, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            
            print(f"✓ Successfully fetched PrizePicks data and saved to {self.directory}")
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching data from PrizePicks API: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON response: {e}")
            raise
        except IOError as e:
            print(f"✗ Error saving file to {self.directory}: {e}")
            raise

    def loadJSON(self):
        """
        Load and parse the JSON file to extract player projections.
        """
        try:
            with open(self.directory, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.directory}")
            raise
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON in file {self.directory}: {e}")
            raise

        player_names = {}
        for elem in data.get('included', []):
            if elem.get('type') == 'new_player':
                player_names[elem['id']] = elem['attributes']['name']

        player_projections = []
        for proj in data.get('data', []):
            if proj.get('type') == 'projection':
                player_id = proj['relationships']['new_player']['data']['id']
                player_name = player_names.get(player_id)

                line_score = proj['attributes']['line_score']
                stat_type = proj['attributes']['stat_type']
                flash_sale = proj['attributes']['odds_type']

                start_time = proj['attributes']['start_time']
                dt = datetime.fromisoformat(start_time)
                pacific_time = dt.astimezone(ZoneInfo("America/Los_Angeles"))
                month = pacific_time.strftime("%b")
                day = pacific_time.strftime("%d").lstrip('0')
                formatted_date = pacific_time.strftime(f"{month}-{day}-%Y %I:%M %p")
                
                player_projections.append((player_name, stat_type, line_score, flash_sale, formatted_date))
        
        self.lines = player_projections
        print(f"✓ Loaded {len(self.lines)} player projections")


if __name__ == "__main__":
    print("Starting PrizePicks Scraper...")
    try:
        scraper = PrizePicks_Scraper()
        print(f"\n✓ Scraping complete! Found {len(scraper.lines)} projections.")
        print("\nSample projections:")
        for i, (name, stat, line, odds, date) in enumerate(scraper.lines[:5], 1):
            print(f"  {i}. {name}: {stat} {line} ({odds}) - {date}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
