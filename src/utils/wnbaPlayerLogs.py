"""Backwards-compatible re-exports — use ``src.utils.playerLogs`` instead."""

from src.utils.playerLogs import (  # noqa: F401
    RAW_DATASETS,
    RAW_TABLE_BY_DATASET,
    WNBAGameLogs,
    GameLogs,
    _TRACK_V3_COLUMN_MAP,
    _call_with_retry,
    _fetch_track_v3_for_game,
    _finalize_track_v3_df,
)

__all__ = [
    'RAW_DATASETS',
    'RAW_TABLE_BY_DATASET',
    'WNBAGameLogs',
    'GameLogs',
    '_TRACK_V3_COLUMN_MAP',
    '_call_with_retry',
    '_fetch_track_v3_for_game',
    '_finalize_track_v3_df',
]

if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0], '--league', 'wnba', *sys.argv[1:]]
    from src.utils.playerLogs import main
    raise SystemExit(main())
