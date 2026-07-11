"""Fetch NBA or WNBA game-log endpoints and upsert into ``raw.*`` tables.

One endpoint → one DataFrame → one ``raw`` (or ``raw.wnba_*``) table.
Only ``game_id`` / ``player_id`` / ``team_id`` are renamed; all other columns
stay as the API returns them.

Examples::

    from src.utils.fetch import GameLogs
    GameLogs("2025-26", season_type="Regular Season", league="nba").fetch()
    GameLogs("2025", season_type="Playoffs", league="wnba").fetch()
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd
from nba_api.stats.endpoints import boxscoreplayertrackv3, playergamelogs, teamgamelogs

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

LeagueKey = Literal['nba', 'wnba']

RAW_DATASETS: tuple[str, ...] = (
    'player_base',
    'player_adv',
    'team_base',
    'team_adv',
    'start_positions',
)
LEAGUE_WIDE_DATASETS: frozenset[str] = frozenset({
    'player_base', 'player_adv', 'team_base', 'team_adv',
})

# Only identity keys are renamed; everything else keeps the endpoint's names.
_ID_COLUMN_ALIASES: dict[str, str] = {
    'GAME_ID': 'game_id',
    'PLAYER_ID': 'player_id',
    'TEAM_ID': 'team_id',
    'gameId': 'game_id',
    'personId': 'player_id',  # BoxScorePlayerTrackV3
    'teamId': 'team_id',
}


@dataclass(frozen=True)
class LeagueConfig:
    key: LeagueKey
    label: str
    league_id: str | None
    raw_table_by_dataset: dict[str, str]
    default_season: str
    default_checkpoint: str


LEAGUES: dict[LeagueKey, LeagueConfig] = {
    'nba': LeagueConfig(
        key='nba',
        label='NBA',
        league_id=None,
        raw_table_by_dataset={
            'player_base': 'nba_player_base',
            'player_adv': 'nba_player_adv',
            'team_base': 'nba_team_base',
            'team_adv': 'nba_team_adv',
            'start_positions': 'nba_player_tracking',
        },
        default_season='2025-26',
        default_checkpoint='data/raw/cache/start_positions_checkpoint.csv',
    ),
    'wnba': LeagueConfig(
        key='wnba',
        label='WNBA',
        league_id='10',
        raw_table_by_dataset={
            'player_base': 'wnba_player_base',
            'player_adv': 'wnba_player_adv',
            'team_base': 'wnba_team_base',
            'team_adv': 'wnba_team_adv',
            'start_positions': 'wnba_start_positions',
        },
        default_season='2025',
        default_checkpoint='data/raw/cache/wnba_start_positions_checkpoint.csv',
    ),
}


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


def _rename_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename only game/player/team id columns; leave all other API columns alone."""
    rename = {src: dst for src, dst in _ID_COLUMN_ALIASES.items() if src in df.columns}
    if not rename:
        return df
    return df.rename(columns=rename)


def _normalize_id_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'game_id' in out.columns:
        out['game_id'] = out['game_id'].astype(str).str.zfill(10)
    for col in ('player_id', 'team_id'):
        if col in out.columns:
            # Int64 keeps whole IDs as integers (not 203827.0) for Postgres BIGINT.
            out[col] = pd.to_numeric(out[col], errors='coerce').astype('Int64')
    return out


def _scope_start_positions(df: pd.DataFrame, game_ids: set[str]) -> pd.DataFrame:
    """Keep only rows for the game IDs requested in this fetch."""
    if df.empty:
        return df
    id_col = next((c for c in ('game_id', 'GAME_ID', 'gameId') if c in df.columns), None)
    if id_col is None:
        return df
    if id_col != 'game_id':
        df = df.rename(columns={id_col: 'game_id'})
    mask = df['game_id'].astype(str).str.zfill(10).isin(game_ids)
    return df.loc[mask].copy()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        'rate limit', 'too many requests', '429',
        'timeout', 'timed out', 'connection', 'read timed out',
    ))


def _call_with_retry(fn, *args, label: str = '', max_retries: int = 5,
                     base_delay: float = 1.0, max_delay: float = 60.0, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt >= max_retries:
                raise
            sleep_for = min(max_delay, base_delay * (2 ** attempt))
            sleep_for += random.uniform(0, 0.5 * sleep_for)
            logger.warning(
                "Rate limit / transient error on %s (attempt %d/%d): %s — retrying in %.1fs",
                label or fn.__name__, attempt + 1, max_retries, e, sleep_for,
            )
            time.sleep(sleep_for)
            attempt += 1


def _upsert_raw(table: str, df: pd.DataFrame) -> None:
    try:
        from src.utils.db import upsert_df
        upsert_df(table, df)
    except Exception as exc:
        logger.warning("raw upsert failed for %s: %s", table, exc)


def _fetch_track_v3_for_game(game_id, delay: float) -> pd.DataFrame | None:
    normalized = str(game_id).zfill(10)
    time.sleep(delay)

    def _call():
        return boxscoreplayertrackv3.BoxScorePlayerTrackV3(
            game_id=normalized,
            timeout=60,
        ).get_data_frames()[0]

    df = _call_with_retry(_call, label=f"game {normalized}")
    if df is None or df.empty:
        logger.warning("Skipping game %s: no tracking data returned", normalized)
        return None

    df = _rename_id_columns(df)
    missing_required = {'game_id', 'player_id'} - set(df.columns)
    if missing_required:
        logger.warning(
            "Skipping game %s: missing required columns %s",
            normalized, sorted(missing_required),
        )
        return None
    return df


class GameLogs:
    """Fetch NBA or WNBA API game-log endpoints; one dataset per ``raw.*`` table."""

    def __init__(
        self,
        season: str,
        season_type: str = 'Regular Season',
        *,
        league: LeagueKey = 'nba',
    ):
        self.season = season
        self.season_type = season_type
        self._cfg = LEAGUES[league]
        self._data: dict[str, pd.DataFrame] = {}

    @property
    def league(self) -> LeagueKey:
        return self._cfg.key

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        return dict(self._data)

    def raw_table(self, dataset: str) -> str:
        return self._cfg.raw_table_by_dataset[dataset]

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
        checkpoint_path: str | None = None,
        start_position_workers: int = 8,
        run_all_batches: bool = True,
        db_upsert: bool = True,
        game_ids: Iterable | None = None,
    ) -> GameLogs:
        """Fetch endpoints and upsert into Supabase ``raw.*`` (default on)."""
        requested = _normalize_datasets(datasets)
        checkpoint_path = checkpoint_path or self._cfg.default_checkpoint
        print(
            f"Fetching {self._cfg.label} {', '.join(requested)} "
            f"for {self.season} {self.season_type}..."
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
        kwargs: dict[str, str] = {
            'season_nullable': self.season,
            'season_type_nullable': self.season_type,
            'measure_type_player_game_logs_nullable': measure,
        }
        if self._cfg.league_id is not None:
            kwargs['league_id_nullable'] = self._cfg.league_id
        return playergamelogs.PlayerGameLogs(**kwargs).get_data_frames()[0]

    def _team_logs(self, measure: str) -> pd.DataFrame:
        kwargs: dict[str, str] = {
            'season_nullable': self.season,
            'season_type_nullable': self.season_type,
            'measure_type_player_game_logs_nullable': measure,
        }
        if self._cfg.league_id is not None:
            kwargs['league_id_nullable'] = self._cfg.league_id
        return teamgamelogs.TeamGameLogs(**kwargs).get_data_frames()[0]

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
            df = _normalize_id_dtypes(_rename_id_columns(df))
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
        table = self.raw_table(dataset)
        print(f"✓ {dataset} → {table} — {len(df):,} rows")
        if db_upsert:
            _upsert_raw(table, df)

    def _resolve_game_ids(self, game_ids: Iterable | None) -> list:
        if game_ids is not None:
            return list(game_ids)
        if 'player_base' in self._data:
            return list(self._data['player_base']['game_id'].unique())
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
        table = self.raw_table('start_positions')
        all_game_ids = self._resolve_game_ids(game_ids)
        all_game_ids_norm = {str(gid).zfill(10) for gid in all_game_ids}
        total = len(all_game_ids)

        if os.path.exists(checkpoint_path):
            checkpoint = pd.read_csv(checkpoint_path, dtype=str)
            # Prefer renamed id; fall back to older checkpoint column names.
            id_col = next(
                (c for c in ('game_id', 'GAME_ID', 'gameId') if c in checkpoint.columns),
                None,
            )
            if id_col is None:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} missing game_id / GAME_ID / gameId"
                )
            if id_col != 'game_id':
                checkpoint = checkpoint.rename(columns={id_col: 'game_id'})
            done_ids = set(checkpoint['game_id'].astype(str).str.zfill(10).unique())
            remaining = [
                gid for gid in all_game_ids
                if str(gid).zfill(10) not in done_ids
            ]
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
            result = _normalize_id_dtypes(pd.concat(frames, ignore_index=True))
            self._data['start_positions'] = _scope_start_positions(
                result, all_game_ids_norm
            )
            if db_upsert:
                _upsert_raw(table, self._data['start_positions'])
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

        result = _normalize_id_dtypes(pd.concat(frames, ignore_index=True))
        scoped = _scope_start_positions(result, all_game_ids_norm)
        self._data['start_positions'] = scoped
        print(f"✓ start_positions — {len(scoped):,} rows ({len(all_game_ids_norm)} games)")
        if db_upsert:
            _upsert_raw(table, scoped)


class NBAGameLogs(GameLogs):
    def __init__(self, season: str, season_type: str = 'Regular Season'):
        super().__init__(season, season_type, league='nba')


class WNBAGameLogs(GameLogs):
    def __init__(self, season: str, season_type: str = 'Regular Season'):
        super().__init__(season, season_type, league='wnba')
