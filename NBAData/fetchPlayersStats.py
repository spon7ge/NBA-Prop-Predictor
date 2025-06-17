import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv2, teamgamelog
from nba_api.stats.static import teams
from concurrent.futures import ThreadPoolExecutor, as_completed

class FetchPlayersStats:
    def __init__(self, default_season='2024-25', sleep_time=0.1):
        self.default_season = default_season
        self.sleep_time = sleep_time

    def fetchPlayerStats(self, season=None, season_type='Regular Season'):
        season = season or self.default_season
        df = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation='P',
            season_type_all_star=season_type
        ).get_data_frames()[0]

        df['OPP_ABBREVIATION'] = df['MATCHUP'].str.extract(r'(?:vs\.|@) ([A-Z]+)')
        df['HOME_GAME'] = df['MATCHUP'].str.contains('vs\.').astype(int)
        fga, fta, pts, fgm, fg3m = df['FGA'], df['FTA'], df['PTS'], df['FGM'], df['FG3M']
        df['PointsPerShot'] = np.where(fga == 0, 0.0, pts / (fga + 0.44 * fta)).round(3)
        df['eFG'] = (fgm + 0.5 * fg3m) / fga 
        

        cols = [
            'PLAYER_NAME', 'PLAYER_ID', 'MATCHUP', 'TEAM_ABBREVIATION', 'TEAM_ID',
            'OPP_ABBREVIATION', 'HOME_GAME', 'GAME_ID', 'GAME_DATE', 'WL',
            'MIN', 'PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF',
            'PLUS_MINUS', 'FANTASY_PTS', 'PointsPerShot', 'eFG'
        ]
        return df[cols]

    def fetchAdvancedStats(self, game_id, sleep_time=None):
        sleep_time = sleep_time or self.sleep_time
        try:
            time.sleep(sleep_time)
            df = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id).get_data_frames()[0]
            return df
        except Exception as e:
            print(f"[ERROR] Game {game_id}: {e}")
            return pd.DataFrame()

    def getAdvancedStats(self, player_data, sleep_time=None, max_workers=None, cache_file ='PLAYOFF_DATA/PLAYOFFS_25.csv'):
        sleep_time = sleep_time or self.sleep_time
        max_workers = max_workers or min(10, os.cpu_count() or 4)
        game_ids = player_data['GAME_ID'].unique()
        
        if os.path.exists(cache_file):
            cached_df = pd.read_csv(cache_file, dtype={'GAME_ID': str})
            cached_game_ids = cached_df['GAME_ID'].unique()
        else:
            cached_df = pd.DataFrame()
            cached_game_ids = []

        missing_ids = [gid for gid in game_ids if gid not in cached_game_ids]
        print(f"Total games: {len(game_ids)}, Cached: {len(cached_game_ids)}, To fetch: {len(missing_ids)}")
        
        new_stats = []
        if missing_ids:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.fetchAdvancedStats, gid, sleep_time): gid for gid in missing_ids}
                for i, future in enumerate(as_completed(futures)):
                    gid = futures[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            new_stats.append(df)
                        print(f"[{i+1}/{len(missing_ids)}] Fetched {gid}")
                    except Exception as e:
                        print(f"[ERROR] Fetching game {gid}: {e}")

        # Combine cached + new
        if new_stats:
            new_df = pd.concat(new_stats, ignore_index=True)
            combined = pd.concat([cached_df, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['GAME_ID', 'PLAYER_ID'], inplace=True)
            combined.to_csv(cache_file, index=False)
        else:
            combined = cached_df

        return combined

    def mergeData(self, player_data, advanced_stats):
        player_data['GAME_ID'] = player_data['GAME_ID'].astype(str)
        advanced_stats['GAME_ID'] = advanced_stats['GAME_ID'].astype(str)
        advanced_stats['PLAYER_ID'] = advanced_stats['PLAYER_ID'].astype(int)

        adv_cols = [
            'GAME_ID', 'PLAYER_ID', 'START_POSITION', 'OFF_RATING', 'DEF_RATING',
            'NET_RATING', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'AST_PCT',
            'AST_TOV', 'USG_PCT', 'TS_PCT', 'E_PACE', 'PACE', 'PIE', 'PACE_PER40'
        ]
        return pd.merge(player_data, advanced_stats[adv_cols], on=['GAME_ID', 'PLAYER_ID'], how='left')

    def getTeamData(self, season=None, season_type='Regular Season'):
        season = season or self.default_season
        teams_list = teams.get_teams()
        team_data = []

        for i, team in enumerate(teams_list):
            try:
                print(f"[{i+1}/{len(teams_list)}] Fetching data for {team['full_name']}")
                df = teamgamelog.TeamGameLog(team_id=team['id'], season=season, season_type_all_star=season_type).get_data_frames()[0]
                df.columns = df.columns.str.upper()
                drop_cols = ['MATCHUP', 'WL', 'W', 'L', 'W_PCT', 'GAMEDATE']
                df.drop(columns=[c for c in drop_cols if c in df], inplace=True, errors='ignore')
                df.rename(columns={col: f'TEAM_{col}' for col in df.columns if col not in ['GAME_ID', 'TEAM_ID']}, inplace=True)
                team_data.append(df)
            except Exception as e:
                print(f"[ERROR] {team['full_name']}: {e}")

        return pd.concat(team_data, ignore_index=True) if team_data else pd.DataFrame()

    def addOpponentStats(self, df):
        def calc(group):
            if len(group) != 2:
                return group
            t1, t2 = group.iloc[0], group.iloc[1]

            defrtg_1 = (t2['TEAM_PTS'] / (t2['TEAM_FGA'] + 0.44 * t2['TEAM_FTA'] - t2['TEAM_OREB'] + t2['TEAM_TOV'])) * 100
            defrtg_2 = (t1['TEAM_PTS'] / (t1['TEAM_FGA'] + 0.44 * t1['TEAM_FTA'] - t1['TEAM_OREB'] + t1['TEAM_TOV'])) * 100

            group.loc[group.index[0], ['OPP_DEF_RATING', 'OPP_STL', 'OPP_BLK', 'OPP_REB', 'OPP_FG_PCT', 'OPP_TEAM_ID']] = [
                defrtg_2, t2['TEAM_STL'], t2['TEAM_BLK'], t2['TEAM_OREB'] + t2['TEAM_DREB'], t2['TEAM_FGM'] / t2['TEAM_FGA'], t2['TEAM_ID']
            ]
            group.loc[group.index[1], ['OPP_DEF_RATING', 'OPP_STL', 'OPP_BLK', 'OPP_REB', 'OPP_FG_PCT', 'OPP_TEAM_ID']] = [
                defrtg_1, t1['TEAM_STL'], t1['TEAM_BLK'], t1['TEAM_OREB'] + t1['TEAM_DREB'], t1['TEAM_FGM'] / t1['TEAM_FGA'], t1['TEAM_ID']
            ]
            return group

        return df.groupby('GAME_ID', group_keys=False).apply(calc)

    def addOffensiveRating(self, df):
        def calc(group):
            if len(group) != 2:
                return group
            t1, t2 = group.iloc[0], group.iloc[1]
            pos1 = t1['TEAM_FGA'] + 0.44 * t1['TEAM_FTA'] - t1['TEAM_OREB'] + t1['TEAM_TOV']
            pos2 = t2['TEAM_FGA'] + 0.44 * t2['TEAM_FTA'] - t2['TEAM_OREB'] + t2['TEAM_TOV']
            off1 = (t1['TEAM_PTS'] / pos1) * 100
            off2 = (t2['TEAM_PTS'] / pos2) * 100
            group.loc[group.index[0], 'TEAM_OFF_RATING'] = off1
            group.loc[group.index[1], 'TEAM_OFF_RATING'] = off2
            return group

        return df.groupby('GAME_ID', group_keys=False).apply(calc)

    def add_pace_stats(self, df):
        def calc(group):
            if len(group) != 2:
                return group
            t1, t2 = group.iloc[0], group.iloc[1]
            pos1 = t1['TEAM_FGA'] + 0.44 * t1['TEAM_FTA'] - t1['TEAM_OREB'] + t1['TEAM_TOV']
            pos2 = t2['TEAM_FGA'] + 0.44 * t2['TEAM_FTA'] - t2['TEAM_OREB'] + t2['TEAM_TOV']
            avg_pos = (pos1 + pos2) / 2
            group.loc[group.index[0], ['TEAM_PACE', 'GAME_PACE', 'OPP_PACE']] = [pos1, avg_pos, pos2]
            group.loc[group.index[1], ['TEAM_PACE', 'GAME_PACE', 'OPP_PACE']] = [pos2, avg_pos, pos1]
            return group

        return df.groupby('GAME_ID', group_keys=False).apply(calc)

    def mergeWithTeam(self, player_data, team_data):
        return pd.merge(player_data, team_data, on=['GAME_ID', 'TEAM_ID'], how='left')

    def getCompleteStats(self, season=None, season_type='Regular Season', sleep_time=None, max_workers=None, cache_file = 'PLAYOFF_DATA/PLAYOFFS_25.csv'):
        season = season or self.default_season
        print("[1] Fetching basic player stats...")
        player_stats = self.fetchPlayerStats(season, season_type)

        print("[2] Fetching advanced player stats...")
        adv_stats = self.getAdvancedStats(player_stats, sleep_time, max_workers, cache_file)

        print("[3] Merging player data...")
        merged_player_stats = self.mergeData(player_stats, adv_stats)

        print("[4] Fetching and processing team data...")
        team_data = self.getTeamData(season, season_type)
        team_data = self.addOpponentStats(team_data)
        team_data = self.addOffensiveRating(team_data)
        team_data = self.add_pace_stats(team_data)

        print("[5] Final player-team merge...")
        complete_stats = self.mergeWithTeam(merged_player_stats, team_data)
        print("✅ Complete stats processing finished.")
        return complete_stats
