"""
Build silver.wnba_player_gamelogs from raw.wnba_* tables and upsert to Supabase.

Examples (from repo root):

  python scripts/upload_wnba_silver.py
  python scripts/upload_wnba_silver.py --season 2025 --season-type "Regular Season"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge raw.wnba_* → silver.wnba_player_gamelogs.")
    p.add_argument("--season", default="2025")
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help="Regular Season | Playoffs",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.db import upsert_silver
    from src.utils.wnba_silver import build_wnba_gamelogs_silver

    df = build_wnba_gamelogs_silver(args.season, args.season_type)
    upsert_silver(df, season_type=args.season_type, table="wnba_player_gamelogs")
    print(f"✓ Uploaded {len(df):,} rows → silver.wnba_player_gamelogs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
