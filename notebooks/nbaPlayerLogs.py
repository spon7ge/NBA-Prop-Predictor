import time
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import playergamelogs, teamgamelogs
from nba_api.stats.endpoints import boxscoreplayertrackv3


class NBAGameLogs:
    """
    Fetches and merges player base, player advanced, team base, team advanced,
    opponent team stats, START_POSITION, and player tracking stats into a
    single flat DataFrame.
    """

    def __init__(self, season: str, season_type: str = 'Regular Season'):
        self.season = season
        self.season_type = season_type
        self.final = None

        self._player_base = None
        self._player_adv = None
        self._team_base = None
        self._team_adv = None
        self._start_positions = None

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def fetch(
        self,
        start_position_delay: float = 0.6,
        batch_size: int = 100,
        checkpoint_path: str = 'start_positions_checkpoint.csv',
        skip_start_positions: bool = False
    ) -> 'NBAGameLogs':
        print(f"Fetching data for {self.season} {self.season_type}...")

        self._player_base = playergamelogs.PlayerGameLogs(
            season_nullable=self.season,
            season_type_nullable=self.season_type,
            measure_type_player_game_logs_nullable='Base'
        ).get_data_frames()[0]
        time.sleep(0.6)
        print("✓ Player base")

        self._player_adv = playergamelogs.PlayerGameLogs(
            season_nullable=self.season,
            season_type_nullable=self.season_type,
            measure_type_player_game_logs_nullable='Advanced'
        ).get_data_frames()[0]
        time.sleep(0.6)
        print("✓ Player advanced")

        self._team_base = teamgamelogs.TeamGameLogs(
            season_nullable=self.season,
            season_type_nullable=self.season_type,
            measure_type_player_game_logs_nullable='Base'
        ).get_data_frames()[0]
        time.sleep(0.6)
        print("✓ Team base")

        self._team_adv = teamgamelogs.TeamGameLogs(
            season_nullable=self.season,
            season_type_nullable=self.season_type,
            measure_type_player_game_logs_nullable='Advanced'
        ).get_data_frames()[0]
        time.sleep(0.6)
        print("✓ Team advanced")

        if skip_start_positions:
            print("⚡ Skipping START_POSITION fetch")
        else:
            self._start_positions = self._fetch_start_positions(
                delay=start_position_delay,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path
            )

        return self

    def _fetch_start_positions(
        self,
        delay: float,
        batch_size: int,
        checkpoint_path: str
    ) -> pd.DataFrame:
        all_game_ids = self._player_base['GAME_ID'].unique()
        total = len(all_game_ids)

        # ── Resume from checkpoint if it exists ───────────────────────────────
        if os.path.exists(checkpoint_path):
            checkpoint = pd.read_csv(checkpoint_path, dtype=str)
            done_ids = set(checkpoint['GAME_ID'].unique())
            remaining = [gid for gid in all_game_ids if str(gid).zfill(10) not in done_ids]
            frames = [checkpoint]
            print(f"✓ Checkpoint loaded — {len(done_ids)}/{total} games already done, {len(remaining)} remaining")
        else:
            remaining = list(all_game_ids)
            frames = []
            print(f"Fetching START_POSITION for {total} games (batch size: {batch_size})...")

        if not remaining:
            print("✓ All games already fetched from checkpoint")
            return self._finalize_start_positions(pd.concat(frames, ignore_index=True))

        # ── Batch loop ────────────────────────────────────────────────────────
        failed = []
        batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            print(f"\n── Batch {batch_num}/{len(batches)} ({len(batch)} games) ──")
            batch_frames = []

            def fetch_one(game_id):
                normalized = str(game_id).zfill(10)
                time.sleep(delay)
                df = boxscoreplayertrackv3.BoxScorePlayerTrackV3(
                    game_id=normalized,
                    timeout=60
                ).get_data_frames()[0]

                column_mapping = {
                    'gameId': 'GAME_ID',
                    'personId': 'PLAYER_ID',
                    'position': 'START_POSITION',
                    'minutes': 'MIN',
                    'speed': 'SPD',
                    'distance': 'DIST',
                    'reboundChancesOffensive': 'ORBC',
                    'reboundChancesDefensive': 'DRBC',
                    'reboundChancesTotal': 'RBC',
                    'touches': 'TCHS',
                    'secondaryAssists': 'SAST',
                    'freeThrowAssists': 'FTAST',
                    'passes': 'PASS',
                    'contestedFieldGoalsMade': 'CFGM',
                    'contestedFieldGoalsAttempted': 'CFGA',
                    'contestedFieldGoalPercentage': 'CFG_PCT',
                    'uncontestedFieldGoalsMade': 'UFGM',
                    'uncontestedFieldGoalsAttempted': 'UFGA',
                    'uncontestedFieldGoalsPercentage': 'UFG_PCT',
                    'defendedAtRimFieldGoalsMade': 'DFGM',
                    'defendedAtRimFieldGoalsAttempted': 'DFGA',
                    'defendedAtRimFieldGoalPercentage': 'DFG_PCT'
                }
                df = df.rename(columns=column_mapping)

                # MIN is intentionally excluded — already present on player_base.
                cols = [
                    'GAME_ID', 'PLAYER_ID', 'START_POSITION',
                    'SPD', 'DIST', 'ORBC', 'DRBC', 'RBC',
                    'TCHS', 'SAST', 'FTAST', 'PASS',
                    'CFGM', 'CFGA', 'CFG_PCT',
                    'UFGM', 'UFGA', 'UFG_PCT',
                    'DFGM', 'DFGA', 'DFG_PCT'
                ]
                existing_cols = [c for c in cols if c in df.columns]
                return df[existing_cols]

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(fetch_one, gid): gid for gid in batch}

                for i, future in enumerate(as_completed(futures), 1):
                    game_id = futures[future]
                    try:
                        batch_frames.append(future.result())
                    except Exception as e:
                        failed.append(game_id)
                        print(f"  ⚠ Skipped game {str(game_id).zfill(10)}: {e}")

                    if i % 25 == 0:
                        print(f"  … {i}/{len(batch)} games in batch")

            if batch_frames:
                frames.extend(batch_frames)

                checkpoint_so_far = pd.concat(frames, ignore_index=True)
                checkpoint_so_far.to_csv(checkpoint_path, index=False)
                print(f"  ✓ Batch {batch_num} done — checkpoint saved ({len(checkpoint_so_far)} rows total)")

            print(f"  → Done for now. Re-run fetch() to continue with the next batch.")
            break  # <-- Remove this line to run all batches in one go

        if failed:
            print(f"\n  ✗ {len(failed)} games failed across all batches: {failed}")

        result = pd.concat(frames, ignore_index=True)
        return self._finalize_start_positions(result)

    def _finalize_start_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        df['GAME_ID'] = df['GAME_ID'].astype(self._player_base['GAME_ID'].dtype)
        df['PLAYER_ID'] = df['PLAYER_ID'].astype(self._player_base['PLAYER_ID'].dtype)
        print(f"✓ START_POSITION ready ({len(df)} rows)")
        return df

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> 'NBAGameLogs':
        if any(df is None for df in [self._player_base, self._player_adv,
                                      self._team_base, self._team_adv]):
            raise RuntimeError("Call .fetch() before .build()")

        # 1. Player base + advanced
        player_merged = self._player_base.merge(
            self._player_adv,
            on=['GAME_ID', 'PLAYER_ID', 'TEAM_ID'],
            suffixes=('', '_adv')
        )

        # 2. Attach START_POSITION only if available
        if self._start_positions is not None:
            player_merged = player_merged.merge(
                self._start_positions,
                on=['GAME_ID', 'PLAYER_ID'],
                how='left'
            )

        # 3. Team base + advanced
        team_merged = self._team_base.merge(
            self._team_adv,
            on=['GAME_ID', 'TEAM_ID'],
            suffixes=('_base', '_adv')
        )

        # 4. Build OPP df before prefixing team_merged
        opp_team = team_merged.copy()
        opp_team = opp_team.add_prefix('TEAM_').rename(columns={
            'TEAM_GAME_ID': 'GAME_ID',
            'TEAM_TEAM_ID': 'OPP_TEAM_ID'
        })
        opp_team.columns = [
            col.replace('TEAM_', 'OPP_') if col.startswith('TEAM_') else col
            for col in opp_team.columns
        ]

        # 5. Prefix team_merged and restore join keys
        team_merged = team_merged.add_prefix('TEAM_').rename(columns={
            'TEAM_GAME_ID': 'GAME_ID',
            'TEAM_TEAM_ID': 'TEAM_ID'
        })

        # 6. Join team + opp onto player rows
        final = player_merged.merge(team_merged, on=['GAME_ID', 'TEAM_ID'], how='left')
        # opp_team rows are "team T's box score in game G" → OPP_TEAM_ID == T.
        # Joining on (GAME_ID, TEAM_ID) == (GAME_ID, OPP_TEAM_ID) re-attaches the *player's*
        # team; we need the *other* team in the same game.
        final = final.merge(opp_team, on='GAME_ID', how='left')
        final = final.loc[final['TEAM_ID'] != final['OPP_TEAM_ID']]

        self.final = self._clean(final)
        print(f"✓ Built — shape: {self.final.shape}")
        return self

    # ── Clean ─────────────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in df.columns if '_RANK' in c])
        df = df.drop(columns=[c for c in df.columns if c.endswith('_adv')])
        df = df.drop(columns=[c for c in self._redundant_cols('TEAM_') if c in df.columns])
        df = df.drop(columns=[c for c in self._redundant_cols('OPP_') if c in df.columns])
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    @staticmethod
    def _redundant_cols(prefix: str) -> list:
        suffixes = [
            'SEASON_YEAR_base', 'TEAM_ABBREVIATION_base', 'TEAM_NAME_base',
            'GAME_DATE_base', 'MATCHUP_base', 'WL_base', 'MIN_base',
            'SEASON_YEAR_adv', 'TEAM_ABBREVIATION_adv', 'TEAM_NAME_adv',
            'GAME_DATE_adv', 'MATCHUP_adv', 'WL_adv', 'MIN_adv',
            'AVAILABLE_FLAG_base', 'AVAILABLE_FLAG_adv'
        ]
        return [f"{prefix}{s}" for s in suffixes]

    # ── Access ────────────────────────────────────────────────────────────────

    def get_df(self) -> pd.DataFrame:
        if self.final is None:
            raise RuntimeError("Call .fetch().build() first")
        return self.final

    def filter_player(self, player_name: str) -> pd.DataFrame:
        return self.final[self.final['PLAYER_NAME'].str.lower() == player_name.lower()]

    def filter_team(self, team_abbreviation: str) -> pd.DataFrame:
        return self.final[self.final['TEAM_ABBREVIATION'].str.upper() == team_abbreviation.upper()]