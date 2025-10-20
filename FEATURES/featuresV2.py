import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplayv3
import isodate
import re

def cleanPlaybyPlay(game_id):
    df = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
    df['MIN_SECONDS'] = df['clock'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    df['scoreHome'] = df['scoreHome'].replace('', np.nan).ffill()
    df['scoreAway'] = df['scoreAway'].replace('', np.nan).ffill()
    return df

import numpy as np
import re

def get_minutes_played(quarter_df, player_nameI):
    pass # work on getting minutes played by player for each quarter



def engineerPlayerPlaybyPlayBasics(game_id, player_id, game_pbp_data=None):
    # Use provided data or fetch it
    if game_pbp_data is not None:
        df = game_pbp_data
    else:
        df = cleanPlaybyPlay(game_id)
    
    player_data = df[df['personId'] == player_id]
    player_id = player_data['personId'].iloc[0]
    player_name = player_data['playerName'].iloc[0]
    team_id = player_data['teamId'].iloc[0]
    game_id = player_data['gameId'].iloc[0]
    location = player_data['location'].iloc[0]

    result = {
        'PLAYER_ID': player_id,
        'NAME': player_name,
        'TEAM_ID': team_id,
        'GAME_ID': game_id,
    }

    # Get all periods in the game (including OT)
    all_periods = sorted(player_data['period'].unique())

    for period in all_periods:
        quarter = period
        quarterData = player_data[player_data['period'] == period]
        if quarterData.empty:
            continue

        # player team stats
        teamQuarterData = df[(df['period'] == quarter) & (df['teamId'] == player_data['teamId'].iloc[0])]
        team_possessions = len(teamQuarterData[teamQuarterData['actionType'].isin(['Made Shot', 'Missed Shot', 'Free Throw', 
    'Turnover', 'Rebound', 'Foul', 'Substitution'])])
        teamFgAttempted = len(teamQuarterData[teamQuarterData['isFieldGoal'] == 1])
        teamFta = len(teamQuarterData[teamQuarterData['actionType'] == 'Free Throw'])
        teamTurnovers = len(teamQuarterData[teamQuarterData['actionType'] == 'Turnover'])

        # score margin - get from full game data for this quarter
        fullQuarterData = df[df['period'] == quarter]
        scoreMarginStart, scoreMarginEnd, scoreMarginChange = 0, 0, 0
        leadingTeam, trailingTeam = 0, 0
        if len(fullQuarterData) > 0:
            if location == 'h':
                scoreMarginStart = int(fullQuarterData['scoreHome'].iloc[0]) - int(fullQuarterData['scoreAway'].iloc[0])
                scoreMarginEnd = int(fullQuarterData['scoreHome'].iloc[-1]) - int(fullQuarterData['scoreAway'].iloc[-1])
                if scoreMarginEnd > 0:
                    leadingTeam = 1
                    trailingTeam = 0
                else:
                    trailingTeam = 1
                    leadingTeam = 0
            else:
                scoreMarginStart = int(fullQuarterData['scoreAway'].iloc[0]) - int(fullQuarterData['scoreHome'].iloc[0])
                scoreMarginEnd = int(fullQuarterData['scoreAway'].iloc[-1]) - int(fullQuarterData['scoreHome'].iloc[-1])
                if scoreMarginEnd > 0:
                    leadingTeam = 1
                    trailingTeam = 0
                else:
                    trailingTeam = 1
                    leadingTeam = 0
            scoreMarginChange = abs(scoreMarginEnd) - abs(scoreMarginStart)
        
        #checking to see if player was active during the quarter
        totalActions = len(quarterData[quarterData['actionType'].isin(['Made Shot', 'Missed Shot', 'Free Throw', 
    'Turnover', 'Rebound', 'Foul', 'Substitution'])])
        wasActive = 0
        if totalActions > 0:
            wasActive = 1
        else:
            wasActive = 0

        # minutes played
        # minutesPlayed_q, non_active_q = get_minutes_played(teamQuarterData, player_name)
        
        # free throws
        ftEvents = quarterData[quarterData['actionType'] == 'Free Throw']
        FTM_q, FTA_q = 0, 0
        for _,row in ftEvents.iterrows():
            FTA_q += 1
            if 'MISS' not in row['description']:
                FTM_q += 1
        ftPct_q = round(FTM_q / FTA_q, 2) if FTA_q > 0 else 0
        
        # field goals - separate 2-pointers and 3-pointers
        fg2m, fg2a = 0, 0
        fg3m, fg3a = 0, 0

        for _, row in quarterData.iterrows():
            if row['isFieldGoal'] == 1:  
                if '3PT' in row['description'].split(' '):
                    fg3a += 1
                    if row['shotResult'] == 'Made':
                        fg3m += 1
                else:
                    fg2a += 1
                    if row['shotResult'] == 'Made':
                        fg2m += 1

        totalFG = fg2a + fg3a
        fgMade = fg2m + fg3m  
        FG_PCT = round(fgMade / totalFG, 2) if totalFG > 0 else 0
        
        # field goal distance averages
        fgMissedDistAvg = quarterData[quarterData['shotResult'] == 'Missed']['shotDistance'].mean()
        fgMadeDistAvg = quarterData[quarterData['shotResult'] == 'Made']['shotDistance'].mean()
        overDistAvg = (fgMissedDistAvg + fgMadeDistAvg) / 2
        
        # 3-pointers
        FG3_PCT = round(fg3m / fg3a, 2) if fg3a > 0 else 0
        
        # close shots
        closeShotsMade, closeShotsAttempted = 0, 0
        for _,row in quarterData.iterrows():
            if row['shotResult'] == 'Made' and row['shotDistance'] <= 8:
                closeShotsMade += 1
                closeShotsAttempted += 1
            elif row['shotResult'] == 'Missed' and row['shotDistance'] <= 8:
                closeShotsAttempted += 1
        closeShotsPCT = round(closeShotsMade / closeShotsAttempted, 2) if closeShotsAttempted > 0 else 0
        
        # mid range shots 
        midRangeMade, midRangeAttempted = 0, 0
        for _,row in quarterData.iterrows():
            if row['shotResult'] == 'Made' and row['shotDistance'] > 8 and row['shotDistance'] <= 22:
                midRangeMade += 1
                midRangeAttempted += 1
            elif row['shotResult'] == 'Missed' and row['shotDistance'] > 8 and row['shotDistance'] <= 22:
                midRangeAttempted += 1
        midRangePCT = round(midRangeMade / midRangeAttempted, 2) if midRangeAttempted > 0 else 0
        
        # extra additions
        assists = (quarterData['actionType'] == 'Assist').sum()
        turnovers = (quarterData['actionType'] == 'Turnover').sum()
        offReb = quarterData['description'].str.contains('Off', na=False).sum()
        defReb = quarterData['description'].str.contains('Def', na=False).sum()
        steals = (quarterData['description'].str.contains('Steal', na=False)).sum()
        blocks = (quarterData['description'].str.contains('Block', na=False)).sum()
        fouls = (quarterData['actionType'] == 'Foul').sum()
        
        # early fouls 
        early_fouls = (quarterData[quarterData['actionType'] == 'Foul']['MIN_SECONDS'] > 600).sum()
        
        # calculate usage percentage 
        # denominator = minutesPlayed_q * (teamFgAttempted + 0.44 * teamFta + teamTurnovers)
        # if denominator > 0 and minutesPlayed_q > 0:
        #     usg_pct = 100 * ((totalFG + 0.44 * FTA_q + turnovers) * (12.0)) / denominator
        # else:
        #     usg_pct = 0

        result[f'ACTIVE_Q{quarter}'] = wasActive
        result[f'PLAYER_TOTAL_ACTIONS_Q{quarter}'] = totalActions
        result[f'SCORE_MARGIN_START_Q{quarter}'] = scoreMarginStart
        result[f'SCORE_MARGIN_END_Q{quarter}'] = scoreMarginEnd
        result[f'SCORE_MARGIN_CHANGE_Q{quarter}'] = scoreMarginChange
        result[f'POINTS_Q{quarter}'] = 2 * fg2m + 3 * fg3m + FTM_q
        result[f'eFG%_Q{quarter}'] = round((fgMade + 0.5 * fg3m) / totalFG, 2) if totalFG > 0 else 0
        result[f'TS%_Q{quarter}'] = round((fgMade * 2 + fg3m + FTM_q) / (2 * (totalFG + 0.44 * FTA_q)), 2) if (2 * (totalFG + 0.44 * FTA_q)) > 0 else 0
        # result[f'USG%_Q{quarter}'] = round(usg_pct, 2)
        result[f'totalShare%_Q{quarter}'] = round(totalActions / team_possessions, 3) if team_possessions > 0 else 0
        result[f'FG3M_Q{quarter}'] = fg3m
        result[f'FG3A_Q{quarter}'] = fg3a
        result[f'FG3%_Q{quarter}'] = FG3_PCT
        result[f'FGM_Q{quarter}'] = fgMade 
        result[f'FGA_Q{quarter}'] = totalFG 
        result[f'FG%_Q{quarter}'] = FG_PCT
        result[f'FTM_Q{quarter}'] = FTM_q
        result[f'FTA_Q{quarter}'] = FTA_q
        result[f'FT%_Q{quarter}'] = ftPct_q
        result[f'closeShotsMade_Q{quarter}'] = closeShotsMade
        result[f'closeShotsAttempted_Q{quarter}'] = closeShotsAttempted
        result[f'closeShotsPCT_Q{quarter}'] = closeShotsPCT
        result[f'midRangeMade_Q{quarter}'] = midRangeMade
        result[f'midRangeAttempted_Q{quarter}'] = midRangeAttempted
        result[f'midRangePCT_Q{quarter}'] = midRangePCT
        result[f'fgMissedDistAvg_Q{quarter}'] = round(fgMissedDistAvg, 2)
        result[f'fgMadeDistAvg_Q{quarter}'] = round(fgMadeDistAvg, 2)
        result[f'overallDistAvg_Q{quarter}'] = round(overDistAvg, 2)
        result[f'assists_Q{quarter}'] = assists
        result[f'turnovers_Q{quarter}'] = turnovers
        result[f'offReb_Q{quarter}'] = offReb
        result[f'defReb_Q{quarter}'] = defReb
        result[f'steals_Q{quarter}'] = steals
        result[f'blocks_Q{quarter}'] = blocks
        result[f'fouls_Q{quarter}'] = fouls
        result[f'earlyFouls_Q{quarter}'] = early_fouls
        result[f'leadingTeam_Q{quarter}'] = leadingTeam
        result[f'trailingTeam_Q{quarter}'] = trailingTeam
        result[f'clutch_Q{quarter}'] = 1 if scoreMarginEnd <= 5 else 0
        result[f'blowout_Q{quarter}'] = 1 if scoreMarginEnd >= 15 else 0
    return result
    
def quarterStatsDiff(df):
    diff_stats = [
        'points', 'eFG%', 'TS%', 'totalShare%', 'FG3%', 'FG%', 'FT%',
        'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM',
        'closeShotsMade', 'closeShotsAttempted', 
        'midRangeMade', 'midRangeAttempted',
        'USG%'
    ]
    
    # Calculate differences for Q2, Q3, Q4 (each compared to previous quarter)
    for i in range(2, 5):  # Quarters 2, 3, 4
        prev_quarter = i - 1
        
        for stat in diff_stats:
            current_key = f'{stat}_Q{i}'
            previous_key = f'{stat}_Q{prev_quarter}'
            diff_key = f'{stat}Diff_Q{i}'
            
            # Only calculate if both quarters exist in dict
            if current_key in df and previous_key in df:
                current_val = df[current_key]
                previous_val = df[previous_key]
                
                # Handle NaN/None values
                if pd.notna(current_val) and pd.notna(previous_val):
                    df[diff_key] = round(current_val - previous_val, 2)
                else:
                    df[diff_key] = None
    
    return df
    