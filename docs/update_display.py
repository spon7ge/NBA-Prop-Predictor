#!/usr/bin/env python3
"""
Script to update the PrizePicks dashboard with new CSV data.
Place your CSV files in the same directory and run this script.
"""

import csv
import json
from pathlib import Path

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

def load_hit_rate_lookup():
    """Load all hit rate CSV files into lookup dictionaries by platform.
    Returns: dict with 'prizepicks' and 'underdog' keys, each containing
             a lookup dict with key (name, line) -> {overPct, underPct, l5, l10, l15}
    """
    lookups = {'prizepicks': {}, 'underdog': {}}
    prop_types = ['points', 'assists', 'rebounds', 'blocks', 'steals']
    
    for prop_type in prop_types:
        for platform_key, platform_dir in [('prizepicks', 'PRIZEPICKS'), ('underdog', 'UNDERDOG')]:
            csv_path = f'../DATA/CSV_FILES/PROP_DATA/OVER_RATES_{platform_dir}/player_{prop_type}.csv'
            if Path(csv_path).exists():
                try:
                    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            name = row['NAME']
                            line = float(row['LINE'])
                            key = (name, line)
                            # Prioritize points, but allow other prop types if points not found
                            if key not in lookups[platform_key] or prop_type == 'points':
                                lookups[platform_key][key] = {
                                    'overPct': float(row['OVER%']),
                                    'underPct': float(row['UNDER%']),
                                    'l5': float(row['L-5']),
                                    'l10': float(row['L-10']),
                                    'l15': float(row['L-15'])
                                }
                except Exception as e:
                    print(f"Warning: Could not load {csv_path}: {e}")
    
    return lookups

def csv_to_js_pairs(csv_file, array_name, hit_rate_lookups=None):
    """Convert pairs CSV file to JavaScript array format."""
    data = []
    
    # Determine platform from array name
    platform = 'prizepicks' if 'prizepicks' in array_name.lower() else 'underdog'
    hit_rate_lookup = hit_rate_lookups.get(platform, {}) if hit_rate_lookups else {}
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Look up hit rate data for each player
            name1 = row['NAME 1']
            name2 = row['NAME 2']
            line1 = float(row['LINE 1'])
            line2 = float(row['LINE 2'])
            side1 = row['MODEL SIDE 1'].lower()
            side2 = row['MODEL SIDE 2'].lower()
            
            # Get hit rate data - try exact match first, then try with small tolerance for floating point
            hr1 = hit_rate_lookup.get((name1, line1), {})
            hr2 = hit_rate_lookup.get((name2, line2), {})
            
            # If exact match not found, try with tolerance (for floating point precision)
            if not hr1:
                for (name, line), hr_data in hit_rate_lookup.items():
                    if name == name1 and abs(line - line1) < 0.01:
                        hr1 = hr_data
                        break
            
            if not hr2:
                for (name, line), hr_data in hit_rate_lookup.items():
                    if name == name2 and abs(line - line2) < 0.01:
                        hr2 = hr_data
                        break
            
            # Calculate hit rate based on side (over or under)
            hit_rate1 = (hr1.get('overPct', 0.0) * 100) if side1 == 'over' else (hr1.get('underPct', 0.0) * 100)
            hit_rate2 = (hr2.get('overPct', 0.0) * 100) if side2 == 'over' else (hr2.get('underPct', 0.0) * 100)
            
            data_point = {
                'name1': name1,
                'name2': name2,
                'line1': line1,
                'line2': line2,
                'prediction1': float(row.get('PREDICTION 1', 0.0)),
                'prediction2': float(row.get('PREDICTION 2', 0.0)),
                'side1': side1,
                'side2': side2,
                'recommendation': int(row['RECOMMENDATION']),
                'ev': float(row.get('EV%', row.get('EV$', 0.0))),
                'kelly': float(row['KELLY FULL']),
                'sigma1': row['SIGMA FLAG 1'],
                'sigma2': row['SIGMA FLAG 2'],
                'prob1': float(row.get('PROB 1', 0.0)),
                'prob2': float(row.get('PROB 2', 0.0)),
                'hitRate1': round(hit_rate1, 1),
                'l5_1': hr1.get('l5', 0.0),
                'l15_1': hr1.get('l15', 0.0),
                'hitRate2': round(hit_rate2, 1),
                'l5_2': hr2.get('l5', 0.0),
                'l15_2': hr2.get('l15', 0.0)
            }
            data.append(data_point)
    
    # Convert to JavaScript array format
    js_code = f"const {array_name} = [\n"
    for item in data:
        js_code += "    " + json.dumps(item) + ",\n"
    js_code += "];"
    
    return js_code

def csv_to_js_trios(csv_file, array_name, hit_rate_lookups=None):
    """Convert trios CSV file to JavaScript array format."""
    data = []
    
    # Determine platform from array name
    platform = 'prizepicks' if 'prizepicks' in array_name.lower() else 'underdog'
    hit_rate_lookup = hit_rate_lookups.get(platform, {}) if hit_rate_lookups else {}
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Look up hit rate data for each player
            name1 = row['NAME 1']
            name2 = row['NAME 2']
            name3 = row['NAME 3']
            line1 = float(row['LINE 1'])
            line2 = float(row['LINE 2'])
            line3 = float(row['LINE 3'])
            side1 = row['MODEL SIDE 1'].lower()
            side2 = row['MODEL SIDE 2'].lower()
            side3 = row['MODEL SIDE 3'].lower()
            
            # Get hit rate data - try exact match first, then try with small tolerance for floating point
            hr1 = hit_rate_lookup.get((name1, line1), {})
            hr2 = hit_rate_lookup.get((name2, line2), {})
            hr3 = hit_rate_lookup.get((name3, line3), {})
            
            # If exact match not found, try with tolerance (for floating point precision)
            if not hr1:
                for (name, line), hr_data in hit_rate_lookup.items():
                    if name == name1 and abs(line - line1) < 0.01:
                        hr1 = hr_data
                        break
            
            if not hr2:
                for (name, line), hr_data in hit_rate_lookup.items():
                    if name == name2 and abs(line - line2) < 0.01:
                        hr2 = hr_data
                        break
            
            if not hr3:
                for (name, line), hr_data in hit_rate_lookup.items():
                    if name == name3 and abs(line - line3) < 0.01:
                        hr3 = hr_data
                        break
            
            # Calculate hit rate based on side (over or under)
            hit_rate1 = (hr1.get('overPct', 0.0) * 100) if side1 == 'over' else (hr1.get('underPct', 0.0) * 100)
            hit_rate2 = (hr2.get('overPct', 0.0) * 100) if side2 == 'over' else (hr2.get('underPct', 0.0) * 100)
            hit_rate3 = (hr3.get('overPct', 0.0) * 100) if side3 == 'over' else (hr3.get('underPct', 0.0) * 100)
            
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
                'recommendation': int(row['RECOMMENDATION']),
                'ev': float(row.get('EV%', row.get('EV$', 0.0))),
                'kelly': float(row['KELLY FULL']),
                'sigma1': row['SIGMA FLAG 1'],
                'sigma2': row['SIGMA FLAG 2'],
                'sigma3': row['SIGMA FLAG 3'],
                'prob1': float(row.get('PROB 1', 0.0)),
                'prob2': float(row.get('PROB 2', 0.0)),
                'prob3': float(row.get('PROB 3', 0.0)),
                'hitRate1': round(hit_rate1, 1),
                'l5_1': hr1.get('l5', 0.0),
                'l15_1': hr1.get('l15', 0.0),
                'hitRate2': round(hit_rate2, 1),
                'l5_2': hr2.get('l5', 0.0),
                'l15_2': hr2.get('l15', 0.0),
                'hitRate3': round(hit_rate3, 1),
                'l5_3': hr3.get('l5', 0.0),
                'l15_3': hr3.get('l15', 0.0)
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

def update_dashboard(csv_files_map, script_file='script.js'):
    """
    Update the script.js file with new CSV data.
    
    Args:
        csv_files_map: Dict mapping array names to CSV file paths
                      e.g., {'prizepicksPointsHitRates': 'player_points.csv'}
        script_file: Path to the JavaScript file
    """
    
    # Read the current script.js file
    with open(script_file, 'r', encoding='utf-8', errors='replace') as f:
        script_content = f.read()
    
    # Load hit rate lookup for pairs/trios
    print("Loading hit rate data...")
    hit_rate_lookups = load_hit_rate_lookup()
    print(f"Loaded {len(hit_rate_lookups.get('prizepicks', {}))} PrizePicks hit rate entries")
    print(f"Loaded {len(hit_rate_lookups.get('underdog', {}))} Underdog hit rate entries")
    
    # Generate JavaScript code for each CSV file
    js_arrays = {}
    for array_name, csv_file in csv_files_map.items():
        if Path(csv_file).exists():
            print(f"Processing {csv_file}...")
            # Determine which converter to use based on array name
            if 'Pairs' in array_name:
                js_arrays[array_name] = csv_to_js_pairs(csv_file, array_name, hit_rate_lookups)
            elif 'Trios' in array_name:
                js_arrays[array_name] = csv_to_js_trios(csv_file, array_name, hit_rate_lookups)
            elif 'singleBets' in array_name or 'Singles' in array_name:
                js_arrays[array_name] = csv_to_js_singles(csv_file, array_name)
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
    csv_files_map = {
    'prizepicksPointsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_points.csv',
    'prizepicksAssistsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_assists.csv',  
    'prizepicksReboundsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_rebounds.csv',
    'prizepicksBlocksHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_blocks.csv',
    'prizepicksStealsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_steals.csv',
    'prizepicksPRAHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_points_rebounds_assists.csv',
    'prizepicksPRHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_points_rebounds.csv',
    'prizepicksPAHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_points_assists.csv',
    'prizepicksRAHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_rebounds_assists.csv',
    'prizepicksTurnoversHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_turnovers.csv',
    'prizepicksBlocksStealsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_PRIZEPICKS/player_blocks_steals.csv',
    
    # Underdog
    'underdogPointsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_points.csv',
    'underdogAssistsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_assists.csv',
    'underdogReboundsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_rebounds.csv',
    'underdogBlocksHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_blocks.csv',
    'underdogStealsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_steals.csv',
    'underdogPRAHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_points_rebounds_assists.csv',
    'underdogPRHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_points_rebounds.csv',
    'underdogPAHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_points_assists.csv',
    'underdogRAHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_rebounds_assists.csv',
    'underdogTurnoversHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_turnovers.csv',
    'underdogBlocksStealsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_blocks_steals.csv',

    # Pairs and Trios
    'prizepicksPairsData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksPairs.csv',  
    'prizepicksTriosData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksTrios.csv',  
    'underdogPairsData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogPairs.csv',  
    'underdogTriosData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogTrios.csv',  
    'prizepicksSinglesData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/singleBets.csv',
}
    
    print("PrizePicks Dashboard Data Updater")
    print("=" * 50)
    
    # Update the dashboard
    update_dashboard(csv_files_map, 'script.js')