"""
Build silver.player_gamelogs from raw.* tables and upsert to Supabase.

Run after NBA bronze ingest (and ideally Rotowire scrape).

Examples (from repo root):

  python scripts/upload_silver.py
  python scripts/upload_silver.py --season 2025-26 --season-type "Regular Season"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge raw.* → silver.player_gamelogs.")
    p.add_argument("--season", default="2025-26")
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help="Regular Season | Playoffs | PlayIn",
    )
    p.add_argument(
        "--skip-rotowire",
        action="store_true",
        help="Do not require Rotowire CSV (matchups may be incomplete)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.db import upsert_silver
    from src.utils.silver import build_gamelogs_silver

    df = build_gamelogs_silver(
        args.season,
        args.season_type,
        skip_rotowire=args.skip_rotowire,
        scrape_rotowire=False,
    )
    upsert_silver(df, season_type=args.season_type)
    print(f"✓ Uploaded {len(df):,} rows → silver.player_gamelogs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
