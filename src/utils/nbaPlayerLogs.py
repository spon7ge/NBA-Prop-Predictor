"""Backwards-compatible re-exports — use ``src.utils.playerLogs`` instead."""

from src.utils.playerLogs import (  # noqa: F401
    LEAGUE_WIDE_DATASETS,
    NBAGameLogs,
    RAW_DATASETS,
    GameLogs,
    _TRACK_V3_COLUMN_MAP,
    _TRACK_V3_LEGACY_ALIASES,
    _call_with_retry,
    _fetch_track_v3_for_game,
    _finalize_track_v3_df,
    refresh_tracking_checkpoint,
)

__all__ = [
    'LEAGUE_WIDE_DATASETS',
    'NBAGameLogs',
    'RAW_DATASETS',
    'GameLogs',
    '_TRACK_V3_COLUMN_MAP',
    '_TRACK_V3_LEGACY_ALIASES',
    '_call_with_retry',
    '_fetch_track_v3_for_game',
    '_finalize_track_v3_df',
    'refresh_tracking_checkpoint',
]

if __name__ == '__main__':
    from src.utils.playerLogs import main
    raise SystemExit(main())
