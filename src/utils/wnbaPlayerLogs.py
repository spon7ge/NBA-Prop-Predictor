"""Fetch WNBA game-log endpoints and upsert each into its ``raw.wnba_*`` table.

One endpoint → one DataFrame → one ``raw.wnba_*`` table. No merging here.

WNBA uses ``league_id_nullable='10'`` on PlayerGameLogs / TeamGameLogs.
``start_positions`` uses BoxScorePlayerTrackV3 per game (same endpoint as NBA).

Schema: ``db/migrations/009_wnba_raw_gamelogs.sql``,
``db/migrations/010_wnba_start_positions.sql``.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, teamgamelogs

from src.utils.nbaPlayerLogs import (
    _TRACK_V3_COLUMN_MAP,
    _call_with_retry,
    _fetch_track_v3_for_game,
    _finalize_track_v3_df,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

WNBA_LEAGUE_ID = '10'

# Dataset key (in-memory / CLI) → raw.wnba_* table in Supabase.
RAW_DATASETS: tuple[str, ...] = (
    'player_base',
    'player_adv',
    'team_base',
    'team_adv',
    'start_positions',
)
RAW_TABLE_BY_DATASET: dict[str, str] = {
    'player_base': 'wnba_player_base',
    'player_adv': 'wnba_player_adv',
    'team_base': 'wnba_team_base',
    'team_adv': 'wnba_team_adv',
    'start_positions': 'wnba_start_positions',
}
LEAGUE_WIDE_DATASETS: frozenset[str] = frozenset({
    'player_base', 'player_adv', 'team_base', 'team_adv',
})


def _normalize_datasets(datasets: str | Iterable[str] | None) -> list[str]:
    if datasets is None:
        return list(RAW_DATASETS)
    if isinstance(datasets, str):
        requested = [datasets]
    else:
        requested = list(datasets)
    unknown = set(requested) - set(RAW_DATASETS)
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {sorted(unknown)}. "
            f"Choose from: {', '.join(RAW_DATASETS)}"
        )
    return [name for name in RAW_DATASETS if name in requested]


def _upsert_raw(table: str, df: pd.DataFrame) -> None:
    try:
        from src.utils.db import upsert_df
        if table == 'wnba_start_positions':
            raw_cols = [c for c in _TRACK_V3_COLUMN_MAP.values() if c in df.columns]
            upsert_df(table, df[raw_cols])
        else:
            upsert_df(table, df)
    except Exception as exc:
        logger.warning("raw upsert failed for %s: %s", table, exc)


class WNBAGameLogs:
    """Fetch WNBA API game-log endpoints; one dataset per ``raw.wnba_*`` table."""

    def __init__(self, season: str, season_type: str = 'Regular Season'):
        self.season = season
        self.season_type = season_type
        self._data: dict[str, pd.DataFrame] = {}

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        return dict(self._data)

    def get(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._data:
            raise KeyError(
                f"Dataset '{dataset}' not fetched. "
                f"Available: {sorted(self._data) or '(none)'}"
            )
        return self._data[dataset]

    def fetch(
        self,
        datasets: str | Iterable[str] | None = None,
        *,
        parallel: bool = True,
        start_position_delay: float = 0.3,
        batch_size: int = 100,
        checkpoint_path: str = 'wnba_start_positions_checkpoint.csv',
        start_position_workers: int = 8,
        run_all_batches: bool = True,
        db_upsert: bool = False,
        game_ids: Iterable | None = None,
    ) -> WNBAGameLogs:
        """Fetch one or more endpoints and optionally upsert into ``raw.wnba_*``.

        Parameters
        ----------
        datasets:
            ``None`` → all five endpoints. Pass a single name or list to run a subset.
        parallel:
            When multiple league-wide endpoints are requested, fetch them concurrently.
        db_upsert:
            Upsert each fetched frame into its matching ``raw.wnba_*`` table immediately.
        game_ids:
            Optional explicit game list for ``start_positions`` when ``player_base``
            was not fetched in this call (otherwise ``player_base`` GAME_IDs are used).
        """
        requested = _normalize_datasets(datasets)
        print(
            f"Fetching WNBA {', '.join(requested)} for {self.season} {self.season_type}..."
        )

        league_requested = [d for d in requested if d in LEAGUE_WIDE_DATASETS]
        if league_requested:
            self._fetch_league_wide(league_requested, parallel=parallel, db_upsert=db_upsert)

        if 'start_positions' in requested:
            self._fetch_start_positions(
                delay=start_position_delay,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
                workers=start_position_workers,
                run_all_batches=run_all_batches,
                db_upsert=db_upsert,
                game_ids=game_ids,
            )

        return self

    def _player_logs(self, measure: str) -> pd.DataFrame:
        return playergamelogs.PlayerGameLogs(
            season_nullable=self.season,
            league_id_nullable=WNBA_LEAGUE_ID,
            season_type_nullable=self.season_type,
            measure_type_player_game_logs_nullable=measure,
        ).get_data_frames()[0]

    def _team_logs(self, measure: str) -> pd.DataFrame:
        return teamgamelogs.TeamGameLogs(
            season_nullable=self.season,
            league_id_nullable=WNBA_LEAGUE_ID,
            season_type_nullable=self.season_type,
            measure_type_player_game_logs_nullable=measure,
        ).get_data_frames()[0]

    def _fetch_league_wide(
        self,
        datasets: list[str],
        *,
        parallel: bool,
        db_upsert: bool,
    ) -> None:
        league_specs = {
            'player_base': (self._player_logs, 'Base'),
            'player_adv': (self._player_logs, 'Advanced'),
            'team_base': (self._team_logs, 'Base'),
            'team_adv': (self._team_logs, 'Advanced'),
        }

        def _run_one(name: str) -> tuple[str, pd.DataFrame]:
            fetcher, measure = league_specs[name]
            df = _call_with_retry(fetcher, measure, label=name)
            return name, df

        if parallel and len(datasets) > 1:
            with ThreadPoolExecutor(max_workers=len(datasets)) as ex:
                futures = {ex.submit(_run_one, name): name for name in datasets}
                for future in as_completed(futures):
                    name, df = future.result()
                    self._store(name, df, db_upsert=db_upsert)
        else:
            for name in datasets:
                _, df = _run_one(name)
                self._store(name, df, db_upsert=db_upsert)

    def _store(self, dataset: str, df: pd.DataFrame, *, db_upsert: bool) -> None:
        self._data[dataset] = df
        print(f"✓ {dataset} — {len(df):,} rows")
        if db_upsert:
            _upsert_raw(RAW_TABLE_BY_DATASET[dataset], df)

    def _resolve_game_ids(self, game_ids: Iterable | None) -> list:
        if game_ids is not None:
            return list(game_ids)
        if 'player_base' in self._data:
            return list(self._data['player_base']['GAME_ID'].unique())
        raise ValueError(
            "start_positions needs game IDs — fetch player_base in the same call, "
            "call fetch('player_base') first, or pass game_ids=."
        )

    def _fetch_start_positions(
        self,
        *,
        delay: float,
        batch_size: int,
        checkpoint_path: str,
        workers: int,
        run_all_batches: bool,
        db_upsert: bool,
        game_ids: Iterable | None,
    ) -> None:
        all_game_ids = self._resolve_game_ids(game_ids)
        total = len(all_game_ids)

        if os.path.exists(checkpoint_path):
            checkpoint = pd.read_csv(checkpoint_path, dtype=str)
            done_ids = set(checkpoint['GAME_ID'].unique())
            remaining = [gid for gid in all_game_ids if str(gid).zfill(10) not in done_ids]
            frames = [checkpoint]
            print(
                f"✓ Checkpoint loaded — {len(done_ids)}/{total} games done, "
                f"{len(remaining)} remaining"
            )
        else:
            remaining = list(all_game_ids)
            frames = []
            print(f"Fetching start_positions for {total} games (batch size: {batch_size})...")

        if not remaining:
            print("✓ All games already fetched from checkpoint")
            result = self._finalize_start_positions(pd.concat(frames, ignore_index=True))
            self._data['start_positions'] = result
            if db_upsert:
                _upsert_raw('wnba_start_positions', result)
            return

        failed: list = []
        batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

        for batch_num, batch in enumerate(batches, 1):
            print(f"\n── Batch {batch_num}/{len(batches)} ({len(batch)} games) ──")
            batch_frames: list[pd.DataFrame] = []

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_fetch_track_v3_for_game, gid, delay): gid
                    for gid in batch
                }
                for i, future in enumerate(as_completed(futures), 1):
                    game_id = futures[future]
                    try:
                        result_df = future.result()
                        if result_df is None:
                            failed.append(game_id)
                        else:
                            batch_frames.append(result_df)
                    except Exception as exc:
                        failed.append(game_id)
                        logger.warning(
                            "Skipped game %s after retries: %s",
                            str(game_id).zfill(10), exc,
                        )
                    if i % 25 == 0:
                        print(f"  … {i}/{len(batch)} games in batch")

            if batch_frames:
                frames.extend(batch_frames)
                checkpoint_so_far = pd.concat(frames, ignore_index=True)
                checkpoint_so_far.to_csv(checkpoint_path, index=False)
                print(
                    f"  ✓ Batch {batch_num} done — checkpoint saved "
                    f"({len(checkpoint_so_far):,} rows total)"
                )

            if not run_all_batches:
                print(f"  → Stopping after batch {batch_num} (run_all_batches=False).")
                break

        if failed:
            print(f"\n  ✗ {len(failed)} games failed: {failed}")

        result = self._finalize_start_positions(pd.concat(frames, ignore_index=True))
        self._data['start_positions'] = result
        if db_upsert:
            _upsert_raw('wnba_start_positions', result)

    def _finalize_start_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        result = _finalize_track_v3_df(df)
        if 'player_base' in self._data:
            result['GAME_ID'] = result['GAME_ID'].astype(self._data['player_base']['GAME_ID'].dtype)
            result['PLAYER_ID'] = result['PLAYER_ID'].astype(
                self._data['player_base']['PLAYER_ID'].dtype
            )
        print(f"✓ start_positions — {len(result):,} rows")
        return result


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch WNBA game-log endpoints into raw.wnba_* tables (no merging).",
    )
    p.add_argument('--season', default='2025')
    p.add_argument('--season-type', default='Regular Season')
    p.add_argument(
        '--datasets',
        nargs='+',
        choices=RAW_DATASETS,
        default=None,
        help='Endpoints to fetch (default: all five)',
    )
    p.add_argument(
        '--sequential',
        action='store_true',
        help='Fetch league-wide endpoints one at a time instead of in parallel',
    )
    p.add_argument('--db-upsert', action='store_true')
    p.add_argument('--checkpoint', default='wnba_start_positions_checkpoint.csv')
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--start-position-delay', type=float, default=0.3)
    p.add_argument('--start-position-workers', type=int, default=8)
    p.add_argument(
        '--one-batch',
        action='store_true',
        help='Stop start_positions after the first batch (run_all_batches=False)',
    )
    return p.parse_args()


def main() -> int:
    args = _parse_cli()
    logs = WNBAGameLogs(season=args.season, season_type=args.season_type)
    logs.fetch(
        datasets=args.datasets,
        parallel=not args.sequential,
        db_upsert=args.db_upsert,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        start_position_delay=args.start_position_delay,
        start_position_workers=args.start_position_workers,
        run_all_batches=not args.one_batch,
    )
    for name, df in logs.data.items():
        print(f"  {name}: {df.shape}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
