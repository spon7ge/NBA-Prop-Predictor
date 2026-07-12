"""Fetch NBA/WNBA raw endpoints, then optionally clean into silver.

Default workflow: raw fetch → merge endpoints → ``pos`` (tracking + CSV) →
Rotowire team odds (NBA) → silver upsert.

Examples::

    python scripts/fetch_raw.py --league nba --season 2025-26
    python scripts/fetch_raw.py --league nba --season 2025-26 --season-type Playoffs
    python scripts/fetch_raw.py --league wnba --season 2025
    python scripts/fetch_raw.py --league nba --season 2025-26 --raw-only
    python scripts/fetch_raw.py --league nba --season 2025-26 --silver-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch game-log endpoints → raw.*, then clean → silver.*",
    )
    p.add_argument("--league", choices=("nba", "wnba"), default="nba")
    p.add_argument("--season", default=None)
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help='e.g. "Regular Season" or "Playoffs"',
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=(
            "player_base",
            "player_adv",
            "team_base",
            "team_adv",
            "start_positions",
        ),
        default=None,
        help="Endpoints to fetch (default: all five)",
    )
    p.add_argument("--sequential", action="store_true")
    p.add_argument("--no-db-upsert", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--start-position-delay", type=float, default=0.3)
    p.add_argument("--start-position-workers", type=int, default=8)
    p.add_argument("--one-batch", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--raw-only",
        action="store_true",
        help="Fetch raw endpoints only (skip silver clean)",
    )
    mode.add_argument(
        "--silver-only",
        action="store_true",
        help="Skip fetch; rebuild silver from existing raw.* tables",
    )
    p.add_argument(
        "--auto-scrape-rotowire",
        action="store_true",
        help="If Rotowire CSV is missing, scrape it (NBA only)",
    )
    return p.parse_args()


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.pipeline.clean import build_silver, fetch_and_build_silver
    from src.pipeline.fetch import LEAGUES, GameLogs

    args = _parse_args()
    season = args.season or LEAGUES[args.league].default_season
    db_upsert = not args.no_db_upsert

    if args.silver_only:
        build_silver(
            season,
            args.season_type,
            league=args.league,
            db_upsert=db_upsert,
            auto_scrape_rotowire=args.auto_scrape_rotowire,
        )
        return 0

    if args.raw_only:
        logs = GameLogs(season=season, season_type=args.season_type, league=args.league)
        logs.fetch(
            datasets=args.datasets,
            parallel=not args.sequential,
            db_upsert=db_upsert,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            start_position_delay=args.start_position_delay,
            start_position_workers=args.start_position_workers,
            run_all_batches=not args.one_batch,
        )
        for name, df in logs.data.items():
            print(f"  {name}: {df.shape}")
        return 0

    fetch_and_build_silver(
        season,
        args.season_type,
        league=args.league,
        db_upsert=db_upsert,
        auto_scrape_rotowire=args.auto_scrape_rotowire,
        datasets=args.datasets,
        parallel=not args.sequential,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        start_position_delay=args.start_position_delay,
        start_position_workers=args.start_position_workers,
        run_all_batches=not args.one_batch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
