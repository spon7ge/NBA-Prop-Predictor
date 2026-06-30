"""Bronze layer: fetch one NBA API source → upsert into raw.* (no merges)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, teamgamelogs

from src.utils.db import upsert_df
from src.utils.nbaPlayerLogs import (
    _TRACK_V3_COLUMN_MAP,
    _call_with_retry,
    _fetch_track_v3_for_game,
    _finalize_track_v3_df,
)

DEFAULT_CHECKPOINT = Path("data/raw/cache/tracking_checkpoint.csv")
_START_POSITIONS_TABLE = "start_positions"


def _upsert_start_positions(df: pd.DataFrame) -> None:
    raw_cols = [c for c in _TRACK_V3_COLUMN_MAP.values() if c in df.columns]
    upsert_df(_START_POSITIONS_TABLE, df[raw_cols])


def _game_ids_for_season(season: str, season_type: str) -> list[str]:
    """Unique GAME_IDs from PlayerGameLogs(Base) for the given season."""
    df = _call_with_retry(
        lambda: playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            measure_type_player_game_logs_nullable="Base",
        ).get_data_frames()[0],
        label="player_base (game ids)",
    )
    return df["GAME_ID"].astype(str).str.zfill(10).unique().tolist()


def fetch_player_base(season: str, season_type: str) -> None:
    """PlayerGameLogs(Base) → raw.player_base."""
    data = _call_with_retry(
        lambda: playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            measure_type_player_game_logs_nullable="Base",
        ).get_data_frames()[0],
        label="player_base",
    )
    upsert_df("player_base", data)


def fetch_player_adv(season: str, season_type: str) -> None:
    """PlayerGameLogs(Advanced) → raw.player_adv."""
    data = _call_with_retry(
        lambda: playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            measure_type_player_game_logs_nullable="Advanced",
        ).get_data_frames()[0],
        label="player_adv",
    )
    upsert_df("player_adv", data)


def fetch_team_base(season: str, season_type: str) -> None:
    """TeamGameLogs(Base) → raw.team_base."""
    data = _call_with_retry(
        lambda: teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            # nba_api quirk: TeamGameLogs uses the *player* measure param name.
            measure_type_player_game_logs_nullable="Base",
        ).get_data_frames()[0],
        label="team_base",
    )
    upsert_df("team_base", data)


def fetch_team_adv(season: str, season_type: str) -> None:
    """TeamGameLogs(Advanced) → raw.team_adv."""
    data = _call_with_retry(
        lambda: teamgamelogs.TeamGameLogs(
            season_nullable=season,
            season_type_nullable=season_type,
            measure_type_player_game_logs_nullable="Advanced",
        ).get_data_frames()[0],
        label="team_adv",
    )
    upsert_df("team_adv", data)


def fetch_all(season: str, season_type: str) -> None:
    """Fetch all four league-wide endpoints (base + advanced)."""
    fetch_player_base(season, season_type)
    fetch_player_adv(season, season_type)
    fetch_team_base(season, season_type)
    fetch_team_adv(season, season_type)


def fetch_boxscoreplayertrackv3(
    season: str,
    season_type: str,
    *,
    game_ids: Iterable[str] | None = None,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    delay: float = 0.3,
    batch_size: int = 100,
    workers: int = 8,
    run_all_batches: bool = True,
) -> None:
    """BoxScorePlayerTrackV3 for every game in the season → raw.start_positions.

    Game IDs come from PlayerGameLogs(Base) for ``season`` / ``season_type``
    unless ``game_ids`` is passed explicitly. Resumes from ``checkpoint_path``.
    """
    checkpoint_path = Path(checkpoint_path)
    all_game_ids = list(game_ids) if game_ids is not None else _game_ids_for_season(
        season, season_type
    )
    total = len(all_game_ids)
    print(f"fetch_boxscoreplayertrackv3 — {total:,} games ({season} {season_type})")

    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path, dtype=str, low_memory=False)
        done_ids = set(checkpoint["GAME_ID"].astype(str).str.zfill(10))
        remaining = [gid for gid in all_game_ids if str(gid).zfill(10) not in done_ids]
        frames = [checkpoint]
        print(f"  checkpoint — {len(done_ids):,}/{total:,} done, {len(remaining):,} remaining")
    else:
        remaining = all_game_ids
        frames = []
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  no checkpoint — fetching {total:,} games (batch size {batch_size})")

    if not remaining:
        print("  all games already in checkpoint — upserting to raw.start_positions")
        result = _finalize_track_v3_df(pd.concat(frames, ignore_index=True))
        _upsert_start_positions(result)
        return

    failed: list[str] = []
    batches = [remaining[i : i + batch_size] for i in range(0, len(remaining), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n── batch {batch_num}/{len(batches)} ({len(batch)} games) ──")
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
                        failed.append(str(game_id))
                    else:
                        batch_frames.append(result_df)
                except Exception as exc:
                    failed.append(str(game_id))
                    print(f"  skipped game {str(game_id).zfill(10)}: {exc}")
                if i % 25 == 0:
                    print(f"  … {i}/{len(batch)} games in batch")

        if batch_frames:
            frames.extend(batch_frames)
            checkpoint_so_far = _finalize_track_v3_df(pd.concat(frames, ignore_index=True))
            checkpoint_so_far.to_csv(checkpoint_path, index=False)
            _upsert_start_positions(_finalize_track_v3_df(pd.concat(batch_frames, ignore_index=True)))
            print(
                f"  ✓ batch {batch_num} — checkpoint saved "
                f"({len(checkpoint_so_far):,} rows total)"
            )

        if not run_all_batches:
            print(f"  → stopping after batch {batch_num} (run_all_batches=False)")
            break

    if failed:
        sample = failed[:20]
        suffix = "…" if len(failed) > 20 else ""
        print(f"\n  ✗ {len(failed)} games failed: {sample}{suffix}")

    if not frames:
        raise RuntimeError("fetch_boxscoreplayertrackv3 finished with no rows")

    result = _finalize_track_v3_df(pd.concat(frames, ignore_index=True))
    print(f"✓ start_positions — {len(result):,} rows from {len(result['GAME_ID'].unique()):,} games")


if __name__ == "__main__":
    for season_type in ("Regular Season", "Playoffs"):
        fetch_all("2025-26", season_type)
        fetch_boxscoreplayertrackv3("2025-26", season_type)
