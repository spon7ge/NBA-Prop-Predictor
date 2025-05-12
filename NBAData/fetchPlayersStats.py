import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2 
from nba_api.stats.static import players
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams

class FetchPlayersStats:
    def __init__(self, default_season='2024-25',sleep_time=0.1):
        self.default_season = default_season
        self.sleep_time = sleep_time

    #grabs basic stats from gamelogs and adds other stats
    def fetchPlayerStats(self, season=None, sample_size=None):
        season = season or self.default_season 
        game_logs = leaguegamelog.LeagueGameLog(
        season=season, 
        player_or_team_abbreviation='P'
        ).get_data_frames()[0]
    
        # Take a sample if specified
        if sample_size:
            # Get unique players
            unique_players = game_logs['PLAYER_ID'].unique()
            # Take a sample of players
            sample_players = unique_players[:min(sample_size, len(unique_players))]
            # Filter to only include those players
            game_logs = game_logs[game_logs['PLAYER_ID'].isin(sample_players)]
            print(f"Using sample of {len(sample_players)} players")
    
        def extract_opponent(row):
            matchup = row['MATCHUP']
            team_abbr = row['TEAM_ABBREVIATION']
            if ' vs. ' in matchup:
                opponent = matchup.split(' vs. ')[1]
            elif ' @ ' in matchup:
                opponent = matchup.split(' @ ')[1]
            else:
                opponent = None
            return opponent
        
        def PointsPerShotAttempt(PTS,FGA,FTA):
            res = np.where(FGA == 0, 0.0,PTS/(FGA+0.44*FTA))
            return res.round(3)

        game_logs['OPP_ABBREVIATION'] = game_logs.apply(extract_opponent, axis=1)
        game_logs['HOME_GAME'] = game_logs['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
        game_logs['AWAY_GAME'] = game_logs['MATCHUP'].apply(lambda x: 1 if '@' in x else 0)
        game_logs['PointsPerShot'] = PointsPerShotAttempt(game_logs['PTS'], game_logs['FGA'], game_logs['FTA'])

        columns = [
            'PLAYER_NAME', 'PLAYER_ID', 'MATCHUP', 'TEAM_ABBREVIATION', 'TEAM_ID', 
            'OPP_ABBREVIATION', 'HOME_GAME','AWAY_GAME', 'GAME_ID', 'GAME_DATE', 'WL', 
            'MIN', 'PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT', 
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 
            'PLUS_MINUS', 'FANTASY_PTS', 'PointsPerShot'
        ]
        game_logs = game_logs[columns]
        print(f"Basic data completed for {season}")
        return game_logs
    
    def fetchAdvancedStats(self, game_id, sleep_time=None):
        sleep_time = sleep_time or self.sleep_time
        try:
            time.sleep(sleep_time)  # Sleep to respect rate limits
            boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
            advanced_stats = boxscore.get_data_frames()[0]  
            return advanced_stats
        except Exception as e:
            print(f"Error fetching advanced stats for game {game_id}: {e}")
            return pd.DataFrame()
    
    def getAdvancedStats(self, player_data,sleep_time=None,max_workers=10):
        game_ids = player_data['GAME_ID'].unique()
        total_games = len(game_ids)
        all_advanced_stats = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetchAdvancedStats, id, sleep_time=sleep_time): id for id in game_ids}
            for idx, future in enumerate(as_completed(futures)):
                game_id = futures[future]
                try:
                    advanced_stats = future.result()
                    if not advanced_stats.empty:
                        all_advanced_stats.append(advanced_stats)
                    print(f"Fetched advanced stats for {idx+1} of {total_games} games")
                except Exception as e:
                    print(f"Error fetching advanced stats for game {game_id}: {e}")

        if all_advanced_stats:
            adv_df = pd.concat(all_advanced_stats,ignore_index=True)
            return adv_df
        else:
            print("No advanced stats found")
            return pd.DataFrame()
            
    
    def mergeData(self,player_data, advanced_stats):
        player_data['GAME_ID'] = player_data['GAME_ID'].astype(str)
        advanced_stats['GAME_ID'] = advanced_stats['GAME_ID'].astype(str)
        advanced_stats['PLAYER_ID'] = advanced_stats['PLAYER_ID'].astype(int)
        
        adv_columns = [
            'GAME_ID', 'PLAYER_ID', 'START_POSITION', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'OREB_PCT',
            'DREB_PCT','REB_PCT','AST_PCT', 'AST_TOV', 'USG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PIE', 'PACE_PER40'
        ]
        advanced_stats_subset = advanced_stats[adv_columns]
        merged_data = pd.merge(player_data, advanced_stats_subset, on=['GAME_ID', 'PLAYER_ID'], how='left')
        return merged_data

    def getTeamData(self, season=None):
        all_teams = teams.get_teams()
        team_ids = [team['id'] for team in all_teams]
        team_names = [team['full_name'] for team in all_teams]
        team_data = []
        total_teams = len(team_ids)
        columns_removed = ['MATCHUP','WL','W','L','W_PCT','GAMEDATE']
        for i,id in enumerate(team_ids):
            try:
                print(f"Fetching team data for {team_names[i]} ({i+1} of {total_teams})")
                team_logs = teamgamelog.TeamGameLog(team_id=id, season=season).get_data_frames()[0]
                team_logs.columns = team_logs.columns.str.upper()
                columns_to_remove_upper = [col.upper() for col in columns_removed]
                team_logs = team_logs.drop(columns=columns_to_remove_upper, errors='ignore')
                # Identify columns to prefix (exclude 'GAME_ID' and 'TEAM_ID')
                columns_to_prefix = [col for col in team_logs.columns if col not in ['GAME_ID', 'TEAM_ID']]
                # Add 'team_' prefix to the identified columns
                team_logs = team_logs.rename(columns=lambda x: f"TEAM_{x}" if x in columns_to_prefix else x)
                # Append the processed DataFrame to the list
                team_data.append(team_logs)
            except Exception as e:
                print(f"Error fetching team data for {team_names[i]}: {e}")
        
        if team_data:
            return pd.concat(team_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def addOpponentStats(self,teams_df):
        game_ids = teams_df['GAME_ID'].unique()
        for i in game_ids:
            game_data = teams_df[teams_df['GAME_ID'] == i]
            if len(game_data) != 2:
                continue
            team_one = game_data.iloc[0]
            team_two = game_data.iloc[1]
            team_one_DRTG = (team_two['TEAM_PTS'] / (team_two['TEAM_FGA'] + 0.44 * team_two['TEAM_FTA'] - team_two['TEAM_OREB'] + team_two['TEAM_TOV'])) * 100
            team_two_DRTG = (team_one['TEAM_PTS'] / (team_one['TEAM_FGA'] + 0.44 * team_one['TEAM_FTA'] - team_one['TEAM_OREB'] + team_one['TEAM_TOV'])) * 100

            # Update teams_df with the calculated DRTG values for both teams
            teams_df.loc[game_data.index[0], 'OPP_DEF_RATING'] = team_two_DRTG  # Update team one with team two's DRTG
            teams_df.loc[game_data.index[1], 'OPP_DEF_RATING'] = team_one_DRTG  # Update team two with team one's DRTG

            # Update team one's OPP stats based on team two's stats
            teams_df.loc[game_data.index[0], 'OPP_STL'] = team_two['TEAM_STL']  # Team one's opponent steals
            teams_df.loc[game_data.index[0], 'OPP_BLK'] = team_two['TEAM_BLK']  # Team one's opponent blocks
            teams_df.loc[game_data.index[0], 'OPP_REB'] = team_two['TEAM_OREB'] + team_two['TEAM_DREB']  # Team one's opponent total rebounds
            teams_df.loc[game_data.index[0], 'OPP_FG_PCT'] = team_two['TEAM_FGM'] / team_two['TEAM_FGA']  # Team one's opponent FG%

            # Update team two's OPP stats based on team one's stats
            teams_df.loc[game_data.index[1], 'OPP_STL'] = team_one['TEAM_STL']  # Team two's opponent steals
            teams_df.loc[game_data.index[1], 'OPP_BLK'] = team_one['TEAM_BLK']  # Team two's opponent blocks
            teams_df.loc[game_data.index[1], 'OPP_REB'] = team_one['TEAM_OREB'] + team_one['TEAM_DREB']  # Team two's opponent total rebounds
            teams_df.loc[game_data.index[1], 'OPP_FG_PCT'] = team_one['TEAM_FGM'] / team_one['TEAM_FGA']  # Team two's opponent FG%

            # Add OPP_TEAM_ID column
            teams_df.loc[game_data.index[0], 'OPP_TEAM_ID'] = team_two['TEAM_ID']  # Team one's opponent team ID
            teams_df.loc[game_data.index[1], 'OPP_TEAM_ID'] = team_one['TEAM_ID']  # Team two's opponent team ID
    
        return teams_df
    
    def addOffensiveRating(self,teams_df):
        game_ids = teams_df['GAME_ID'].unique()
        
        for i in game_ids:
            # Filter data for the current game
            game_data = teams_df[teams_df['GAME_ID'] == i]

            if len(game_data) != 2:
                continue 
            
            # Split into two teams based on the matchup
            team_one = game_data.iloc[0]  # First team in the matchup
            team_two = game_data.iloc[1]  # Second team in the matchup

            # Calculate possessions for both teams
            team_one_possessions = team_one['TEAM_FGA'] + 0.44 * team_one['TEAM_FTA'] - team_one['TEAM_OREB'] + team_one['TEAM_TOV']
            team_two_possessions = team_two['TEAM_FGA'] + 0.44 * team_two['TEAM_FTA'] - team_two['TEAM_OREB'] + team_two['TEAM_TOV']

            # Calculate team one's Offensive Rating (based on team one's own offensive stats)
            team_one_off_rating = (team_one['TEAM_PTS'] / team_one_possessions) * 100

            # Calculate team two's Offensive Rating (based on team two's own offensive stats)
            team_two_off_rating = (team_two['TEAM_PTS'] / team_two_possessions) * 100

            # Update teams_df with the calculated OFF_RATING values for both teams
            teams_df.loc[game_data.index[0], 'TEAM_OFF_RATING'] = team_one_off_rating  # Team one's offensive rating
            teams_df.loc[game_data.index[1], 'TEAM_OFF_RATING'] = team_two_off_rating  # Team two's offensive rating
        
        return teams_df
    
    def add_pace_stats(self,teams_df):
        game_ids = teams_df['GAME_ID'].unique()
        
        for i in game_ids:
            game_data = teams_df[teams_df['GAME_ID'] == i]
            if len(game_data) != 2:
                continue 
            team_one = game_data.iloc[0]
            team_two = game_data.iloc[1]

            # Calculate possessions for each team
            team_one_possessions = team_one['TEAM_FGA'] + 0.44 * team_one['TEAM_FTA'] - team_one['TEAM_OREB'] + team_one['TEAM_TOV']
            team_two_possessions = team_two['TEAM_FGA'] + 0.44 * team_two['TEAM_FTA'] - team_two['TEAM_OREB'] + team_two['TEAM_TOV']

            # Total possessions for the game (average of both teams)
            total_possessions = (team_one_possessions + team_two_possessions) / 2

            # Assume total game time is 48 minutes for NBA (could be adjusted for other leagues)
            game_pace = 48 * total_possessions / 48  # Total minutes is 48 for NBA regulation time

            # Calculate pace for each team
            team_one_pace = 48 * team_one_possessions / 48
            team_two_pace = 48 * team_two_possessions / 48

            # Update teams_df with the calculated team pace values
            teams_df.loc[game_data.index[0], 'TEAM_PACE'] = team_one_pace  # Update team one's pace
            teams_df.loc[game_data.index[1], 'TEAM_PACE'] = team_two_pace  # Update team two's pace

            # Update teams_df with the overall game pace (same for both teams)
            teams_df.loc[game_data.index[0], 'GAME_PACE'] = game_pace  # Game pace (same for both teams)
            teams_df.loc[game_data.index[1], 'GAME_PACE'] = game_pace  # Game pace (same for both teams)

            # Update teams_df with the opponent's pace
            teams_df.loc[game_data.index[0], 'OPP_PACE'] = team_two_pace  # Opponent pace for team one is team two's pace
            teams_df.loc[game_data.index[1], 'OPP_PACE'] = team_one_pace  # Opponent pace for team two is team one's pace
        
        return teams_df
    
    def mergeWithTeam(self,player_data, team_data):
    # Perform a left merge to add all team details to each player
        merged_data = pd.merge(
            player_data,
            team_data,
            on=['GAME_ID', 'TEAM_ID'],
            how='left'
        )
        return merged_data

    def getCompleteStats(self, season=None, sample_size=None, sleep_time=None, max_workers=10):
        """Complete workflow to fetch basic stats, advanced stats, and merge them with team data"""
        season = season or self.default_season
        
        # Step 1: Get basic player stats
        print("Fetching player stats...")
        player_stats = self.fetchPlayerStats(season, sample_size=sample_size)
        
        # Step 2: Get advanced stats for all games
        print("Fetching advanced player stats...")
        advanced_stats = self.getAdvancedStats(player_stats, sleep_time, max_workers)
        
        # Step 3: Merge basic and advanced stats
        print("Merging player data...")
        merged_player_stats = self.mergeData(player_stats, advanced_stats)
        
        # Step 4: Get team data
        print("Fetching team data...")
        team_data = self.getTeamData(season)
        
        # Step 5: Process team data with additional metrics
        print("Adding opponent statistics...")
        team_data = self.addOpponentStats(team_data)
        
        print("Adding offensive ratings...")
        team_data = self.addOffensiveRating(team_data)
        
        print("Adding pace statistics...")
        team_data = self.add_pace_stats(team_data)
        
        # Step 6: Merge player data with team data
        print("Merging player and team data...")
        complete_stats = self.mergeWithTeam(merged_player_stats, team_data)
        
        print("Complete stats processing finished!")
        return complete_stats
    

    

