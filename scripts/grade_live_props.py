"""
Grade last night's (or a lookback window of) live props after box scores land.

Joins ml.*_live_prop_predictions → silver.*_player_gamelogs and writes
ml.*_live_prop_grades (actual_stat, hit, miss_reason, abs_error).

Examples (run from repository root)
-------------------------------------
    python scripts/grade_live_props.py --league wnba
    python scripts/grade_live_props.py --league nba --date 2026-07-17
    python scripts/grade_live_props.py --league wnba --lookback-days 7 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.grade_live_props import run_grade_live_props


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--league", choices=("nba", "wnba"), required=True)
    p.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Slate date YYYY-MM-DD (default: yesterday, or lookback window).",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=1,
        help="When --date is omitted, grade this many calendar days ending yesterday.",
    )
    p.add_argument("--dry-run", action="store_true", help="Score but do not write.")
    args = p.parse_args()

    print(f"\nGrading live props — {args.league.upper()}")
    out = run_grade_live_props(
        league=args.league,
        game_date=args.date,
        lookback_days=args.lookback_days,
        dry_run=args.dry_run,
    )
    if out.empty:
        print("No grades written.")
        sys.exit(0)
    scored = out[out["miss_reason"] != "dnp"]
    if scored.empty:
        print(f"Done — {len(out)} rows (all DNP).")
    else:
        hits = int(scored["hit"].sum())
        print(f"Done — {hits}/{len(scored)} hit ({hits / len(scored):.0%} excl. DNP).")


if __name__ == "__main__":
    main()
