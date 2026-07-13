"""Fetch player prop odds from The Odds API and upsert to raw.* tables.

Examples::

    python scripts/PropFinder.py
    python scripts/PropFinder.py --league wnba
    python scripts/PropFinder.py --league nba
    python scripts/PropFinder.py --league all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.scrapers.NBAPropFinder import NBAPropFinder
from src.scrapers.WNBAPropFinder import WNBAPropFinder

REGIONS = ("us,eu", "us_dfs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape Odds API player props → raw.nba_props_* / raw.wnba_props_*",
    )
    p.add_argument(
        "--league",
        choices=("nba", "wnba", "all"),
        default="wnba",
        help="Which league(s) to scrape (default: wnba — NBA out of season)",
    )
    return p.parse_args()


def _run_nba() -> None:
    for region in REGIONS:
        NBAPropFinder(region=region, db_upsert=True)


def _run_wnba() -> None:
    for region in REGIONS:
        WNBAPropFinder(region=region, db_upsert=True)


def main() -> int:
    args = _parse_args()
    if args.league in ("nba", "all"):
        _run_nba()
    if args.league in ("wnba", "all"):
        _run_wnba()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
