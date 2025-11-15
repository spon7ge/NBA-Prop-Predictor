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
    
    with open(csv_file, 'r') as f:
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

def csv_to_js_pairs(csv_file, array_name):
    """Convert pairs CSV file to JavaScript array format."""
    data = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV data to the format expected by the dashboard
            # Note: hitRate and l5/l15 data not in CSV, setting defaults
            data_point = {
                'name1': row['NAME 1'],
                'name2': row['NAME 2'],
                'line1': float(row['LINE 1']),
                'line2': float(row['LINE 2']),
                'side1': row['MODEL SIDE 1'].lower(),
                'side2': row['MODEL SIDE 2'].lower(),
                'recommendation': int(row['RECOMMENDATION']),
                'ev': float(row['EV$']),
                'kelly': float(row['KELLY FULL']),
                'sigma1': row['SIGMA FLAG 1'],
                'sigma2': row['SIGMA FLAG 2'],
                'hitRate1': 0.0,  # Not in CSV - would need to be fetched separately
                'l5_1': 0.0,       # Not in CSV - would need to be fetched separately
                'l15_1': 0.0,      # Not in CSV - would need to be fetched separately
                'hitRate2': 0.0,   # Not in CSV - would need to be fetched separately
                'l5_2': 0.0,       # Not in CSV - would need to be fetched separately
                'l15_2': 0.0       # Not in CSV - would need to be fetched separately
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
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert CSV data to the format expected by the dashboard
            # Note: hitRate and l5/l15 data not in CSV, setting defaults
            data_point = {
                'name1': row['NAME 1'],
                'name2': row['NAME 2'],
                'name3': row['NAME 3'],
                'line1': float(row['LINE 1']),
                'line2': float(row['LINE 2']),
                'line3': float(row['LINE 3']),
                'side1': row['MODEL SIDE 1'].lower(),
                'side2': row['MODEL SIDE 2'].lower(),
                'side3': row['MODEL SIDE 3'].lower(),
                'recommendation': int(row['RECOMMENDATION']),
                'ev': float(row['EV$']),
                'kelly': float(row['KELLY FULL']),
                'sigma1': row['SIGMA FLAG 1'],
                'sigma2': row['SIGMA FLAG 2'],
                'sigma3': row['SIGMA FLAG 3'],
                'hitRate1': 0.0,   # Not in CSV - would need to be fetched separately
                'l5_1': 0.0,       # Not in CSV - would need to be fetched separately
                'l15_1': 0.0,      # Not in CSV - would need to be fetched separately
                'hitRate2': 0.0,   # Not in CSV - would need to be fetched separately
                'l5_2': 0.0,       # Not in CSV - would need to be fetched separately
                'l15_2': 0.0,      # Not in CSV - would need to be fetched separately
                'hitRate3': 0.0,   # Not in CSV - would need to be fetched separately
                'l5_3': 0.0,       # Not in CSV - would need to be fetched separately
                'l15_3': 0.0       # Not in CSV - would need to be fetched separately
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
    
    with open(csv_file, 'r') as f:
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
                'ev': float(row['EV$']),
                'roi': float(row['EXPECTED ROI']),  # script.js uses 'roi' not 'expectedRoi'
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
    with open(script_file, 'r') as f:
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
            elif 'singleBets' in array_name or 'Singles' in array_name:
                js_arrays[array_name] = csv_to_js_singles(csv_file, array_name)
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
    with open(script_file, 'w') as f:
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
        'underdogPointsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_points.csv',
        'underdogAssistsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_assists.csv',
        'underdogReboundsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_rebounds.csv',
        'underdogBlocksHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_blocks.csv',
        'underdogStealsHitRates': '../DATA/CSV_FILES/PROP_DATA/OVER_RATES_UNDERDOG/player_steals.csv',
        'prizepicksPairsData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksPairs.csv',  
        'prizepicksTriosData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksTrios.csv',  
        'underdogPairsData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogPairs.csv',  
        'underdogTriosData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/underdogTrios.csv',  
        'prizepicksSinglesData': '../DATA/CSV_FILES/PROP_DATA/PROPS_EV/prizepicksSingles.csv',
    } 
    
    print("PrizePicks Dashboard Data Updater")
    print("=" * 50)
    
    # Update the dashboard
    update_dashboard(csv_files_map, 'script.js')