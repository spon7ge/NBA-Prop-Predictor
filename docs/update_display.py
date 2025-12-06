#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

# Get project root by navigating up from this file's location
# This file is in docs/, so go up 1 level to reach project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
project_root_str = str(project_root)

def csv_to_js_array(csv_file, array_name):
    """Convert CSV file to JavaScript array format."""
    data = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV data to the format expected by the dashboard
            data_point = {
                'name': row['NAME'],
                'line': float(row['LINE']),
                'l5': float(row['L-5']),
                'l10': float(row['L-10']),
                'l15': float(row['L-15']),
                'overPct': float(row['OVER%']),
                'underPct': float(row['UNDER%'])
            }
            data.append(data_point)
    
    # Convert to JavaScript array format
    js_code = f"const {array_name} = [\n"
    for item in data:
        js_code += "    " + json.dumps(item) + ",\n"
    js_code += "];"
    
    return js_code

def csv_to_js_array_combo(csv_file, array_name):
    """Convert combo prop CSV file to JavaScript array format (no OVER%/UNDER% columns)."""
    data = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV data to the format expected by the dashboard
            # Combo props don't have OVER%/UNDER% from Poisson, only historical hit rates
            data_point = {
                'name': row['NAME'],
                'line': float(row['LINE']),
                'l5': float(row['L-5']),
                'l10': float(row['L-10']),
                'l15': float(row['L-15']),
                'overPct': float(row['L-10']),  # Use L-10 as a proxy for over percentage
                'underPct': 1 - float(row['L-10'])  # Complement of L-10
            }
            data.append(data_point)
    
    # Convert to JavaScript array format
    js_code = f"const {array_name} = [\n"
    for item in data:
        js_code += "    " + json.dumps(item) + ",\n"
    js_code += "];"
    
    return js_code


def csv_to_js_pairs(csv_file, array_name):
    """Convert pairs CSV file to JavaScript array format."""
    data = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name1 = row['NAME 1']
            name2 = row['NAME 2']
            line1 = float(row['LINE 1'])
            line2 = float(row['LINE 2'])
            side1 = row.get('SIDE 1', row.get('MODEL SIDE 1', '')).lower()
            side2 = row.get('SIDE 2', row.get('MODEL SIDE 2', '')).lower()
            
            data_point = {
                'name1': name1,
                'name2': name2,
                'line1': line1,
                'line2': line2,
                'prediction1': float(row.get('PREDICTION 1', 0.0)),
                'prediction2': float(row.get('PREDICTION 2', 0.0)),
                'side1': side1,
                'side2': side2,
                'edge1': float(row.get('EDGE 1', 0.0)),
                'edge2': float(row.get('EDGE 2', 0.0)),
                'impliedProb1': float(row.get('IMPLIED_PROB 1', 0.0)),
                'impliedProb2': float(row.get('IMPLIED_PROB 2', 0.0))
            }
            data.append(data_point)
    
    # Convert to JavaScript array format
    js_code = f"const {array_name} = [\n"
    for item in data:
        js_code += "    " + json.dumps(item) + ",\n"
    js_code += "];"
    
    return js_code

def csv_to_js_trios(csv_file, array_name):
    """Convert trios CSV file to JavaScript array format."""
    data = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name1 = row['NAME 1']
            name2 = row['NAME 2']
            name3 = row['NAME 3']
            line1 = float(row['LINE 1'])
            line2 = float(row['LINE 2'])
            line3 = float(row['LINE 3'])
            side1 = row.get('SIDE 1', row.get('MODEL SIDE 1', '')).lower()
            side2 = row.get('SIDE 2', row.get('MODEL SIDE 2', '')).lower()
            side3 = row.get('SIDE 3', row.get('MODEL SIDE 3', '')).lower()
            
            data_point = {
                'name1': name1,
                'name2': name2,
                'name3': name3,
                'line1': line1,
                'line2': line2,
                'line3': line3,
                'prediction1': float(row.get('PREDICTION 1', 0.0)),
                'prediction2': float(row.get('PREDICTION 2', 0.0)),
                'prediction3': float(row.get('PREDICTION 3', 0.0)),
                'side1': side1,
                'side2': side2,
                'side3': side3,
                'edge1': float(row.get('EDGE 1', 0.0)),
                'edge2': float(row.get('EDGE 2', 0.0)),
                'edge3': float(row.get('EDGE 3', 0.0)),
                'impliedProb1': float(row.get('IMPLIED_PROB 1', 0.0)),
                'impliedProb2': float(row.get('IMPLIED_PROB 2', 0.0)),
                'impliedProb3': float(row.get('IMPLIED_PROB 3', 0.0))
            }
            data.append(data_point)
    
    # Convert to JavaScript array format
    js_code = f"const {array_name} = [\n"
    for item in data:
        js_code += "    " + json.dumps(item) + ",\n"
    js_code += "];"
    
    return js_code

def csv_to_js_singles(csv_file, array_name):
    """Convert singles CSV file to JavaScript array format."""
    data = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV data to the format expected by the dashboard
            data_point = {
                'name': row['NAME'],
                'bookmaker': row['BOOKMAKER'],
                'line': float(row['LINE']),
                'prediction': float(row['PREDICTION']),
                'side': row['SIDE'],
                'odds': int(row['ODDS']),
                'recommendation': int(row['RECOMMENDATION']),
                'ev': float(row.get('EV%', row.get('EV$', 0.0))),
                'kelly': float(row['KELLY_FRACTION']),
                'sigma': row['SIGMA FLAG']
            }
            data.append(data_point)
    
    # Convert to JavaScript array format
    js_code = f"const {array_name} = [\n"
    for item in data:
        js_code += "    " + json.dumps(item) + ",\n"
    js_code += "];"
    
    return js_code

def update_dashboard(csv_files_map, script_file=None):
    """
    Update the script.js file with new CSV data.
    
    Args:
        csv_files_map: Dict mapping array names to CSV file paths
                      e.g., {'prizepicksPointsHitRates': 'player_points.csv'}
        script_file: Path to the JavaScript file (defaults to docs/script.js)
    """
    if script_file is None:
        script_file = os.path.join(project_root_str, 'docs', 'script.js')
    
    # Read the current script.js file
    with open(script_file, 'r', encoding='utf-8', errors='replace') as f:
        script_content = f.read()
    
    # Generate JavaScript code for each CSV file
    js_arrays = {}
    for array_name, csv_file in csv_files_map.items():
        if Path(csv_file).exists():
            print(f"Processing {csv_file}...")
            # Determine which converter to use based on array name
            if 'Pairs' in array_name:
                js_arrays[array_name] = csv_to_js_pairs(csv_file, array_name)
            elif 'Trios' in array_name:
                js_arrays[array_name] = csv_to_js_trios(csv_file, array_name)
            elif any(combo in array_name for combo in ['PRA', 'PR', 'PA', 'RA', 'Turnovers', 'BlocksSteals']):
                js_arrays[array_name] = csv_to_js_array_combo(csv_file, array_name)
            else:
                js_arrays[array_name] = csv_to_js_array(csv_file, array_name)
        else:
            print(f"Warning: {csv_file} not found, skipping...")
    
    # Replace the data in the script.js file
    for array_name, js_code in js_arrays.items():
        # Find and replace the array definition
        # Looking for pattern: const arrayName = [...];
        start_marker = f"const {array_name} = ["
        
        start_idx = script_content.find(start_marker)
        if start_idx == -1:
            print(f"Warning: Could not find {array_name} in script.js file")
            continue
        
        # Find the matching closing bracket
        # Need to handle ]; with optional semicolons after, and placeholder comments
        # start_marker includes the opening '[', so we start counting from there
        bracket_count = 1  # We already found the opening bracket in start_marker
        idx = start_idx + len(start_marker)  # Start after the opening bracket
        end_idx = -1
        
        while idx < len(script_content):
            char = script_content[idx]
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    # Found the closing bracket, now find where the statement ends
                    end_idx = idx + 1
                    # Skip semicolons and whitespace
                    while end_idx < len(script_content) and script_content[end_idx] in [';', '\n', '\r', ' ', '\t']:
                        if script_content[end_idx] == ';':
                            end_idx += 1
                            # Skip whitespace after semicolon
                            while end_idx < len(script_content) and script_content[end_idx] in ['\n', '\r', ' ', '\t']:
                                end_idx += 1
                            break
                        end_idx += 1
                    break
            idx += 1
        
        if end_idx == -1 or bracket_count != 0:
            print(f"Warning: Could not find matching bracket for {array_name}")
            continue
        
        # Replace the old array with new data
        script_content = script_content[:start_idx] + js_code + script_content[end_idx:]
        print(f"✓ Updated {array_name}")
    
    # Write the updated script.js file
    with open(script_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write(script_content)
    
    print(f"\n✓ Dashboard updated successfully!")
    print(f"✓ Output saved to: {script_file}")
    print(f"\nYou can now open display.html in your browser.")

if __name__ == "__main__":
    # Build paths relative to project root
    csv_files_map = {
    'prizepicksPointsHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_points.csv'),
    'prizepicksAssistsHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_assists.csv'),  
    'prizepicksReboundsHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_rebounds.csv'),
    'prizepicksBlocksHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_blocks.csv'),
    'prizepicksStealsHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_steals.csv'),
    'prizepicksPRAHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_points_rebounds_assists.csv'),
    'prizepicksPRHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_points_rebounds.csv'),
    'prizepicksPAHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_points_assists.csv'),
    'prizepicksRAHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_rebounds_assists.csv'),
    'prizepicksTurnoversHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_turnovers.csv'),
    'prizepicksBlocksStealsHitRates': os.path.join(project_root_str, 'data', 'props', 'prizepicks', 'player_blocks_steals.csv'),
    
    # Underdog
    'underdogPointsHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_points.csv'),
    'underdogAssistsHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_assists.csv'),
    'underdogReboundsHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_rebounds.csv'),
    'underdogBlocksHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_blocks.csv'),
    'underdogStealsHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_steals.csv'),
    'underdogPRAHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_points_rebounds_assists.csv'),
    'underdogPRHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_points_rebounds.csv'),
    'underdogPAHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_points_assists.csv'),
    'underdogRAHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_rebounds_assists.csv'),
    'underdogTurnoversHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_turnovers.csv'),
    'underdogBlocksStealsHitRates': os.path.join(project_root_str, 'data', 'props', 'underdog', 'player_blocks_steals.csv'),

    # Pairs and Trios
    'prizepicksPairsData': os.path.join(project_root_str, 'data', 'props', 'ev_analysis', 'prizepicksPairs.csv'),  
    'prizepicksTriosData': os.path.join(project_root_str, 'data', 'props', 'ev_analysis', 'prizepicksTrios.csv'),  
    'underdogPairsData': os.path.join(project_root_str, 'data', 'props', 'ev_analysis', 'underdogPairs.csv'),  
    'underdogTriosData': os.path.join(project_root_str, 'data', 'props', 'ev_analysis', 'underdogTrios.csv'),  
}
    
    print("PrizePicks Dashboard Data Updater")
    print("=" * 50)
    
    # Update the dashboard
    script_file = os.path.join(project_root_str, 'docs', 'script.js')
    update_dashboard(csv_files_map, script_file)