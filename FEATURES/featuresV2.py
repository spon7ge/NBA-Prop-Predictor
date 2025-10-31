import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplayv3
import isodate
import numpy as np
import re

def cleanPlaybyPlay(game_id):
    df = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
    df['MIN_SECONDS'] = df['clock'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    df['scoreHome'] = df['scoreHome'].replace('', np.nan).ffill()
    df['scoreAway'] = df['scoreAway'].replace('', np.nan).ffill()
    return df

def engineerPlayerPlaybyPlayBasics(game_id, player_id, game_pbp_data=None):
    # Use provided data or fetch it
    if game_pbp_data is not None:
        df = game_pbp_data
    else:
        df = cleanPlaybyPlay(game_id)
    
    player_data = df[df['personId'] == player_id]
    
    # Check if player played in this game
    if player_data.empty:
        # Player didn't play - return all zeros
        return {
            'PLAYER_ID': player_id,
            'NAME': 'Unknown',
            'TEAM_ID': 0,
            'GAME_ID': game_id,
            # Add all quarter-based features with zeros (only first 4 quarters)
            **{f'ACTIVE_Q{q}': 0 for q in range(1, 5)},
            **{f'PLAYER_TOTAL_ACTIONS_Q{q}': 0 for q in range(1, 5)},
            **{f'PLAYER_TOTAL_ACTIONS_PER_MIN_Q{q}': 0 for q in range(1, 5)},
            **{f'SCORE_MARGIN_START_Q{q}': 0 for q in range(1, 5)},
            **{f'SCORE_MARGIN_END_Q{q}': 0 for q in range(1, 5)},
            **{f'SCORE_MARGIN_CHANGE_Q{q}': 0 for q in range(1, 5)},
            **{f'POINTS_Q{q}': 0 for q in range(1, 5)},
            **{f'POINTS_PER_MIN_Q{q}': 0 for q in range(1, 5)},
            **{f'eFG%_Q{q}': 0 for q in range(1, 5)},
            **{f'TS%_Q{q}': 0 for q in range(1, 5)},
            **{f'USG%_Q{q}': 0 for q in range(1, 5)},
            **{f'playerPossessionShare%_Q{q}': 0 for q in range(1, 5)},
            **{f'make_streak_Q{q}': 0 for q in range(1, 5)},
            **{f'miss_streak_Q{q}': 0 for q in range(1, 5)},
            **{f'points_in_burst_Q{q}': 0 for q in range(1, 5)},
            **{f'points_after_turnover_Q{q}': 0 for q in range(1, 5)},
            **{f'margin_when_scoring_avg_Q{q}': 0 for q in range(1, 5)},
            **{f'momentum_run_Q{q}': 0 for q in range(1, 5)},
            **{f'lead_change_involvement_Q{q}': 0 for q in range(1, 5)},
            **{f'stint_length_avg_Q{q}': 0 for q in range(1, 5)},
            **{f'first_sub_time_Q{q}': 0 for q in range(1, 5)},
            **{f'minutes_played_in_q_Q{q}': 0 for q in range(1, 5)},
            **{f'early_points_share_Q{q}': 0 for q in range(1, 5)},
            **{f'late_points_share_Q{q}': 0 for q in range(1, 5)},
            **{f'clutch_fg_pct_Q{q}': 0 for q in range(1, 5)},
            **{f'FG3M_Q{q}': 0 for q in range(1, 5)},
            **{f'FG3A_Q{q}': 0 for q in range(1, 5)},
            **{f'FG3A_PER_MIN_Q{q}': 0 for q in range(1, 5)},
            **{f'FG3%_Q{q}': 0 for q in range(1, 5)},
            **{f'FGM_Q{q}': 0 for q in range(1, 5)},
            **{f'FGA_Q{q}': 0 for q in range(1, 5)},
            **{f'FGA_PER_MIN_Q{q}': 0 for q in range(1, 5)},
            **{f'FG%_Q{q}': 0 for q in range(1, 5)},
            **{f'FTM_Q{q}': 0 for q in range(1, 5)},
            **{f'FTA_PER_MIN_Q{q}': 0 for q in range(1, 5)},
            **{f'FTA_Q{q}': 0 for q in range(1, 5)},
            **{f'FT%_Q{q}': 0 for q in range(1, 5)},
            **{f'closeShotsMade_Q{q}': 0 for q in range(1, 5)},
            **{f'closeShotsAttempted_Q{q}': 0 for q in range(1, 5)},
            **{f'closeShotsPCT_Q{q}': 0 for q in range(1, 5)},
            **{f'midRangeMade_Q{q}': 0 for q in range(1, 5)},
            **{f'midRangeAttempted_Q{q}': 0 for q in range(1, 5)},
            **{f'midRangePCT_Q{q}': 0 for q in range(1, 5)},
            **{f'fgMissedDistAvg_Q{q}': 0 for q in range(1, 5)},
            **{f'fgMadeDistAvg_Q{q}': 0 for q in range(1, 5)},
            **{f'overallDistAvg_Q{q}': 0 for q in range(1, 5)},
            **{f'assists_Q{q}': 0 for q in range(1, 5)},
            **{f'turnovers_Q{q}': 0 for q in range(1, 5)},
            **{f'offReb_Q{q}': 0 for q in range(1, 5)},
            **{f'defReb_Q{q}': 0 for q in range(1, 5)},
            **{f'steals_Q{q}': 0 for q in range(1, 5)},
            **{f'blocks_Q{q}': 0 for q in range(1, 5)},
            **{f'fouls_Q{q}': 0 for q in range(1, 5)},
            **{f'earlyFouls_Q{q}': 0 for q in range(1, 5)},
            **{f'leadingTeam_Q{q}': 0 for q in range(1, 5)},
            **{f'trailingTeam_Q{q}': 0 for q in range(1, 5)},
            **{f'clutch_Q{q}': 0 for q in range(1, 5)},
            **{f'blowout_Q{q}': 0 for q in range(1, 5)},
        }
    
    # Player did play - continue with normal processing
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

    # Get only the first 4 quarters (exclude OT)
    all_periods = sorted([p for p in player_data['period'].unique() if p <= 4])

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
        
        # ================================
        # SCORING RHYTHM AND STREAKS
        # ================================
        
        # Get field goal attempts in chronological order
        fg_attempts = quarterData[quarterData['isFieldGoal'] == 1].sort_values('MIN_SECONDS').reset_index(drop=True)
        
        # Make streak and miss streak
        make_streak = 0
        make_streak_max = 0
        miss_streak = 0
        miss_streak_max = 0
        
        for _, row in fg_attempts.iterrows():
            if row['shotResult'] == 'Made':
                make_streak += 1
                make_streak_max = max(make_streak_max, make_streak)
                miss_streak = 0
            elif row['shotResult'] == 'Missed':
                miss_streak += 1
                miss_streak_max = max(miss_streak_max, miss_streak)
                make_streak = 0
        
        # Points in burst (max points in any 60-second span)
        points_in_burst = 0
        
        if len(fg_attempts) > 0:
            for i in range(len(fg_attempts)):
                burst_points = 0
                current_time = fg_attempts.iloc[i]['MIN_SECONDS']
                
                # Look ahead 60 seconds
                for j in range(i, len(fg_attempts)):
                    event_time = fg_attempts.iloc[j]['MIN_SECONDS']
                    
                    if event_time <= current_time + 60:
                        # Check if this is a made shot and add points
                        if fg_attempts.iloc[j]['shotResult'] == 'Made':
                            shot_value = fg_attempts.iloc[j]['shotValue']
                            if pd.notna(shot_value):
                                burst_points += int(shot_value)
                    else:
                        break
                
                points_in_burst = max(points_in_burst, burst_points)
            
            # Also include free throws in the burst calculation
            ft_events = quarterData[quarterData['actionType'] == 'Free Throw'].sort_values('MIN_SECONDS').reset_index(drop=True)
            
            for i in range(len(ft_events)):
                if 'MISS' not in ft_events.iloc[i]['description']:
                    current_time = ft_events.iloc[i]['MIN_SECONDS']
                    ft_burst_points = 1  # Count this FT
                    
                    # Check for FGs within 60 seconds
                    for j in range(len(fg_attempts)):
                        event_time = fg_attempts.iloc[j]['MIN_SECONDS']
                        if current_time <= event_time <= current_time + 60:
                            if fg_attempts.iloc[j]['shotResult'] == 'Made':
                                shot_value = fg_attempts.iloc[j]['shotValue']
                                if pd.notna(shot_value):
                                    ft_burst_points += int(shot_value)
                    
                    points_in_burst = max(points_in_burst, ft_burst_points)
        
        # Points after turnover (points immediately following opponent turnovers)
        points_after_turnover = 0
        
        # Get all turnovers from opponents in this quarter
        opponent_data = df[(df['period'] == quarter) & (df['teamId'] != player_data['teamId'].iloc[0])]
        opponent_turnovers = opponent_data[opponent_data['actionType'] == 'Turnover'].sort_values('MIN_SECONDS').reset_index(drop=True)
        
        # Get player's made shots and made FTs
        player_made_shots = quarterData[
            (quarterData['isFieldGoal'] == 1) & (quarterData['shotResult'] == 'Made')
        ].sort_values('MIN_SECONDS').reset_index(drop=True)
        
        ft_made = quarterData[
            (quarterData['actionType'] == 'Free Throw') & 
            (~quarterData['description'].str.contains('MISS', na=False))
        ].sort_values('MIN_SECONDS').reset_index(drop=True)
        
        # Find made shots and FTs that occurred after opponent turnovers
        for _, turnover in opponent_turnovers.iterrows():
            turnover_time = turnover['MIN_SECONDS']
            found_shot = False
            
            # Look for the next made shot by player
            for _, shot in player_made_shots.iterrows():
                shot_time = shot['MIN_SECONDS']
                
                if shot_time > turnover_time and not found_shot:
                    # Count points scored
                    shot_value = shot['shotValue']
                    if pd.notna(shot_value):
                        points_after_turnover += int(shot_value)
                    found_shot = True
                    break
            
            # Look for the next made FT by player
            found_ft = False
            for _, ft in ft_made.iterrows():
                ft_time = ft['MIN_SECONDS']
                
                if ft_time > turnover_time and not found_ft:
                    points_after_turnover += 1
                    found_ft = True
                    break
        
        # ================================
        # MOMENTUM AND GAME STATE RESPONSE
        # ================================
        
        # margin_when_scoring_avg_Q{q} - average scoreMargin at each make
        margin_when_scoring_sum = 0
        margin_when_scoring_count = 0
        
        # Get all made shots with score margins
        made_shots_with_margin = quarterData[
            (quarterData['isFieldGoal'] == 1) & (quarterData['shotResult'] == 'Made')
        ].sort_values('MIN_SECONDS')
        
        # Also include made free throws
        made_fts_with_margin = quarterData[
            (quarterData['actionType'] == 'Free Throw') & 
            (~quarterData['description'].str.contains('MISS', na=False))
        ].sort_values('MIN_SECONDS')
        
        # Calculate margin for each made shot
        for _, shot in made_shots_with_margin.iterrows():
            # Find the margin at the time of this shot
            shot_time = shot['MIN_SECONDS']
            # Get the closest row in fullQuarterData that has score info
            full_quarter_sorted = fullQuarterData.sort_values('MIN_SECONDS')
            relevant_rows = full_quarter_sorted[full_quarter_sorted['MIN_SECONDS'] <= shot_time]
            
            if len(relevant_rows) > 0:
                last_row = relevant_rows.iloc[-1]
                if location == 'h':
                    margin = int(last_row['scoreHome']) - int(last_row['scoreAway'])
                else:
                    margin = int(last_row['scoreAway']) - int(last_row['scoreHome'])
                
                margin_when_scoring_sum += margin
                margin_when_scoring_count += 1
        
        # Calculate margin for each made free throw
        for _, ft in made_fts_with_margin.iterrows():
            ft_time = ft['MIN_SECONDS']
            full_quarter_sorted = fullQuarterData.sort_values('MIN_SECONDS')
            relevant_rows = full_quarter_sorted[full_quarter_sorted['MIN_SECONDS'] <= ft_time]
            
            if len(relevant_rows) > 0:
                last_row = relevant_rows.iloc[-1]
                if location == 'h':
                    margin = int(last_row['scoreHome']) - int(last_row['scoreAway'])
                else:
                    margin = int(last_row['scoreAway']) - int(last_row['scoreHome'])
                
                margin_when_scoring_sum += margin
                margin_when_scoring_count += 1
        
        margin_when_scoring_avg = round(margin_when_scoring_sum / margin_when_scoring_count, 2) if margin_when_scoring_count > 0 else 0
        
        # momentum_run_Q{q} - max score differential change while player on court
        # Track maximum positive score change (momentum swing in player's favor)
        momentum_run = 0
        max_positive_change = 0
        
        if len(fullQuarterData) > 0:
            # Get score margins throughout the quarter
            margins = []
            
            for _, row in fullQuarterData.sort_values('MIN_SECONDS').iterrows():
                if location == 'h':
                    margin = int(row['scoreHome']) - int(row['scoreAway'])
                else:
                    margin = int(row['scoreAway']) - int(row['scoreHome'])
                margins.append(margin)
            
            # Find the maximum positive change (momentum swing in player's favor)
            # Look for the largest increase in score margin
            if len(margins) > 0:
                start_margin = margins[0]
                max_increase = 0
                
                for i in range(1, len(margins)):
                    # Calculate the increase from the start to this point
                    change = margins[i] - start_margin
                    if change > max_increase:
                        max_increase = change
                    
                    # Reset start if we hit a new low (building a new run)
                    if margins[i] < start_margin:
                        start_margin = margins[i]
                
                momentum_run = max_increase
        
        # lead_change_involvement_Q{q} - 1 if player scored or assisted during lead change
        lead_change_involvement = 0
        
        # Track when lead changes occur
        prev_leader = None
        
        for _, row in fullQuarterData.sort_values('MIN_SECONDS').iterrows():
            if location == 'h':
                margin = int(row['scoreHome']) - int(row['scoreAway'])
            else:
                margin = int(row['scoreAway']) - int(row['scoreHome'])
            
            current_leader = 'home' if margin > 0 else ('away' if margin < 0 else 'tied')
            
            if prev_leader is not None and current_leader != prev_leader and current_leader != 'tied':
                # Lead changed! Check if player was involved
                change_time = row['MIN_SECONDS']
                
                # Check if player had a score or assist within 5 seconds of the lead change
                player_actions_near_change = quarterData[
                    (quarterData['MIN_SECONDS'] >= change_time - 5) & 
                    (quarterData['MIN_SECONDS'] <= change_time + 5) &
                    (
                        ((quarterData['isFieldGoal'] == 1) & (quarterData['shotResult'] == 'Made')) |
                        (quarterData['actionType'] == 'Assist') |
                        ((quarterData['actionType'] == 'Free Throw') & (~quarterData['description'].str.contains('MISS', na=False)))
                    )
                ]
                
                if len(player_actions_near_change) > 0:
                    lead_change_involvement = 1
                    break
            
            prev_leader = current_leader
        
        # ================================
        # SUBSTITUTION AND FATIGUE
        # ================================
        
        # Get all substitution events in the quarter for this player's team
        team_subs = df[(df['period'] == quarter) & 
                        (df['teamId'] == player_data['teamId'].iloc[0]) &
                        (df['actionType'] == 'Substitution')].sort_values('MIN_SECONDS').reset_index(drop=True)
        
        # Parse substitutions to find when our player enters and exits
        player_events = []  # List of (time, 'in' or 'out')
        
        for _, sub in team_subs.iterrows():
            # Format: "SUB: player_in FOR player_out"
            sub_desc = sub['description']
            sub_time = sub['MIN_SECONDS']
            
            # Check if player is coming IN
            if sub_desc.startswith('SUB: ' + player_name + ' FOR '):
                player_events.append((sub_time, 'in'))
            # Check if player is going OUT
            elif 'FOR ' + player_name in sub_desc:
                player_events.append((sub_time, 'out'))
        
        # Sort by time descending (forward time order: 720 -> 0)
        # Earlier events (larger MIN_SECONDS) come first
        player_events.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate stint lengths and minutes played
        stint_lengths = []
        first_sub_time = None
        minutes_played_in_q = 0
        quarter_start = 720
        quarter_end = 0
        
        if len(player_events) == 0:
            # No substitutions - player played entire quarter (or didn't play at all)
            # Check if player had activity
            if len(quarterData) > 0:
                stint_lengths.append(quarter_start)
                first_sub_time = None
                minutes_played_in_q = 12.0
            else:
                stint_lengths.append(0)
                first_sub_time = None
                minutes_played_in_q = 0
        else:
            # Determine starting position based on first event
            # If first event is 'in', player came off the bench
            # If first event is 'out', player started on the court
            if player_events[0][1] == 'in':
                # Started off court
                on_court = False
                stint_start = None
            else:
                # Started on court (first event is 'out')
                on_court = True
                stint_start = quarter_start
            
            first_exit = None
            
            # Process events chronologically
            for event_time, event_type in player_events:
                if event_type == 'in':
                    if not on_court:
                        # Player entering the game from the bench
                        on_court = True
                        stint_start = event_time
                
                elif event_type == 'out':
                    if on_court:
                        # Player exiting the game
                        if stint_start is not None:
                            stint_length = stint_start - event_time  # forward time
                            stint_lengths.append(stint_length)
                            
                            # Record first exit time
                            if first_exit is None:
                                first_exit = event_time
                                first_sub_time = quarter_start - event_time
                        
                        on_court = False
                        stint_start = None
            
            # If player ended quarter on court, add final stint
            if on_court and stint_start is not None:
                stint_length = stint_start - quarter_end
                stint_lengths.append(stint_length)
        
        # Calculate average stint length (moved outside if/else to always execute)
        stint_length_avg = round(sum(stint_lengths) / len(stint_lengths), 2) if len(stint_lengths) > 0 else 0
        
        # Calculate total minutes played (sum of all stints)
        total_seconds_played = sum(stint_lengths)
        if len(player_events) > 0:
            minutes_played_in_q = round(total_seconds_played / 60, 2)
        
        # ================================
        # TEMPORAL DISTRIBUTION
        # ================================
        
        # early_points_share_Q{q} - share of points scored in first 3 minutes of quarter
        # late_points_share_Q{q} - share in final 2 minutes
        # clutch_fg_pct_Q{q} - FG% in final 2 minutes within 5-point margin
        
        early_points = 0
        late_points = 0
        total_points = 0
        
        clutch_fgm = 0
        clutch_fga = 0
        
        # First 3 minutes: 720 down to 540 (first 3 minutes = 180 seconds)
        early_cutoff = 720 - 180  # 540
        
        # Final 2 minutes: 120 down to 0
        late_cutoff = 120
        
        # Get all scoring events
        made_shots = quarterData[
            (quarterData['isFieldGoal'] == 1) & (quarterData['shotResult'] == 'Made')
        ].sort_values('MIN_SECONDS')
        
        made_fts = quarterData[
            (quarterData['actionType'] == 'Free Throw') & 
            (~quarterData['description'].str.contains('MISS', na=False))
        ].sort_values('MIN_SECONDS')
        
        # Calculate early points (first 3 minutes)
        early_made_shots = made_shots[made_shots['MIN_SECONDS'] >= early_cutoff]
        for _, shot in early_made_shots.iterrows():
            shot_value = shot['shotValue']
            if pd.notna(shot_value):
                points = int(shot_value)
                early_points += points
                total_points += points
        
        early_made_fts = made_fts[made_fts['MIN_SECONDS'] >= early_cutoff]
        early_points += len(early_made_fts)
        total_points += len(early_made_fts)
        
        # Calculate late points (final 2 minutes)
        late_made_shots = made_shots[made_shots['MIN_SECONDS'] <= late_cutoff]
        for _, shot in late_made_shots.iterrows():
            shot_value = shot['shotValue']
            if pd.notna(shot_value):
                points = int(shot_value)
                late_points += points
                total_points += points
        
        late_made_fts = made_fts[made_fts['MIN_SECONDS'] <= late_cutoff]
        late_points += len(late_made_fts)
        total_points += len(late_made_fts)
        
        # Calculate clutch FG% (final 2 minutes within 5-point margin)
        for _, shot in made_shots.iterrows():
            shot_time = shot['MIN_SECONDS']
            if shot_time <= late_cutoff:
                # Check if within 5-point margin at time of shot
                shot_time_floor = shot_time
                full_quarter_sorted = fullQuarterData.sort_values('MIN_SECONDS')
                relevant_rows = full_quarter_sorted[full_quarter_sorted['MIN_SECONDS'] <= shot_time_floor]
                
                if len(relevant_rows) > 0:
                    last_row = relevant_rows.iloc[-1]
                    if location == 'h':
                        margin = abs(int(last_row['scoreHome']) - int(last_row['scoreAway']))
                    else:
                        margin = abs(int(last_row['scoreAway']) - int(last_row['scoreHome']))
                    
                    if margin <= 5:
                        clutch_fga += 1
                        if shot['shotResult'] == 'Made':
                            clutch_fgm += 1
        
        # Calculate shares
        early_points_share = round(early_points / total_points, 3) if total_points > 0 else 0
        late_points_share = round(late_points / total_points, 3) if total_points > 0 else 0
        clutch_fg_pct = round(clutch_fgm / clutch_fga, 2) if clutch_fga > 0 else 0
        
        # calculate usage percentage 
        denominator = minutes_played_in_q * (teamFgAttempted + 0.44 * teamFta + teamTurnovers)
        if denominator > 0 and minutes_played_in_q > 0:
            usg_pct = 100 * ((totalFG + 0.44 * FTA_q + turnovers) * (12.0)) / denominator
        else:
            usg_pct = 0

        points = 2 * fg2m + 3 * fg3m + FTM_q

        result[f'ACTIVE_Q{quarter}'] = wasActive
        result[f'PLAYER_TOTAL_ACTIONS_Q{quarter}'] = totalActions
        result[f'PLAYER_TOTAL_ACTIONS_PER_MIN_Q{quarter}'] = round(totalActions / minutes_played_in_q, 2) if minutes_played_in_q > 0 else 0
        result[f'SCORE_MARGIN_START_Q{quarter}'] = scoreMarginStart
        result[f'SCORE_MARGIN_END_Q{quarter}'] = scoreMarginEnd
        result[f'SCORE_MARGIN_CHANGE_Q{quarter}'] = scoreMarginChange
        result[f'POINTS_Q{quarter}'] = points
        result[f'POINTS_PER_MIN_Q{quarter}'] = round(points / minutes_played_in_q, 2) if minutes_played_in_q > 0 else 0
        result[f'eFG%_Q{quarter}'] = round((fgMade + 0.5 * fg3m) / totalFG, 2) if totalFG > 0 else 0
        result[f'TS%_Q{quarter}'] = round((fgMade * 2 + fg3m + FTM_q) / (2 * (totalFG + 0.44 * FTA_q)), 2) if (2 * (totalFG + 0.44 * FTA_q)) > 0 else 0
        result[f'USG%_Q{quarter}'] = round(usg_pct, 2)
        result[f'playerPossessionShare%_Q{quarter}'] = round(totalActions / team_possessions, 3) if team_possessions > 0 else 0
        
        # Add the new streak features
        result[f'make_streak_Q{quarter}'] = make_streak_max
        result[f'miss_streak_Q{quarter}'] = miss_streak_max
        result[f'points_in_burst_Q{quarter}'] = points_in_burst
        result[f'points_after_turnover_Q{quarter}'] = points_after_turnover
        
        # Add momentum and game state features
        result[f'margin_when_scoring_avg_Q{quarter}'] = margin_when_scoring_avg
        result[f'momentum_run_Q{quarter}'] = momentum_run
        result[f'lead_change_involvement_Q{quarter}'] = lead_change_involvement
        
        # Add substitution and fatigue features
        result[f'stint_length_avg_Q{quarter}'] = stint_length_avg
        result[f'first_sub_time_Q{quarter}'] = first_sub_time
        result[f'minutes_played_in_q_Q{quarter}'] = minutes_played_in_q
        
        # Add temporal distribution features
        result[f'early_points_share_Q{quarter}'] = early_points_share
        result[f'late_points_share_Q{quarter}'] = late_points_share
        result[f'clutch_fg_pct_Q{quarter}'] = clutch_fg_pct
        
        result[f'FG3M_Q{quarter}'] = fg3m
        result[f'FG3A_Q{quarter}'] = fg3a
        result[f'FG3A_PER_MIN_Q{quarter}'] = round(fg3a / minutes_played_in_q, 2) if minutes_played_in_q > 0 else 0
        result[f'FG3%_Q{quarter}'] = FG3_PCT
        result[f'FGM_Q{quarter}'] = fgMade 
        result[f'FGA_Q{quarter}'] = totalFG 
        result[f'FGA_PER_MIN_Q{quarter}'] = round(totalFG / minutes_played_in_q, 2) if minutes_played_in_q > 0 else 0
        result[f'FG%_Q{quarter}'] = FG_PCT
        result[f'FTM_Q{quarter}'] = FTM_q
        result[f'FTA_PER_MIN_Q{quarter}'] = round(FTA_q / minutes_played_in_q, 2) if minutes_played_in_q > 0 else 0
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
    
    # Final NaN cleanup for any remaining NaN values
    for key, value in result.items():
        if pd.isna(value):
            result[key] = 0
            
    return result
    
def quarterStatsDiff(df):
    diff_stats = [
        'points', 'eFG%', 'TS%', 'playerPossessionShare%', 'FG3%', 'FG%', 'FT%', 'USG%',
        'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'closeShotsMade', 'closeShotsAttempted', 'midRangeMade', 'midRangeAttempted',
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
    

