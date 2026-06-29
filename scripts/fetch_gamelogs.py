"""
End-to-end NBA game log fetch (bronze) + silver merge + gold features → CSV.

Bronze fetch: ``src.utils.nbaPlayerLogs`` / ``src.utils.bronze``
Silver merge: ``src.utils.silver``
Gold features: derived rates / encodings written at the end of this script

Examples (run from repository root):

  python scripts/fetch_gamelogs.py
  python scripts/fetch_gamelogs.py --silver-only
  python scripts/fetch_gamelogs.py --skip-rotowire
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _checkpoint_summary(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return False, 0, 0
    ck = pd.read_csv(path, dtype=str, low_memory=False)
    n_rows = len(ck)
    if "GAME_ID" not in ck.columns:
        return True, n_rows, 0
    return True, n_rows, ck["GAME_ID"].nunique()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch NBA logs → silver merge → gold CSV.")
    p.add_argument("--season", default="2025-26")
    p.add_argument("--season-type", default="Playoffs", help="Regular Season | Playoffs")
    p.add_argument("--positions", default="data/raw/player_positions/nba_2026_players.csv")
    p.add_argument("--reference-season-csv", default="data/raw/season_stats/S26.csv")
    p.add_argument("--out", default="data/raw/playoff_stats/P26.csv")
    p.add_argument("--checkpoint", default="data/raw/cache/tracking_checkpoint.csv")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--start-position-delay", type=float, default=2.5)
    p.add_argument("--start-position-workers", type=int, default=5)
    p.add_argument("--is-playoff-flag", type=int, default=1, choices=(0, 1))
    p.add_argument("--rotowire-season", default="2025")
    p.add_argument("--rotowire-csv", default="data/raw/rotowire/rotowire_nba_2025.csv")
    p.add_argument("--skip-rotowire", action="store_true")
    p.add_argument("--rotowire-headed", action="store_true")
    p.add_argument("--db-upsert", action="store_true")
    p.add_argument("--skip-nba-fetch", action="store_true")
    p.add_argument("--parquet", default="")
    p.add_argument("--save-parquet", default="")
    p.add_argument(
        "--silver-only",
        action="store_true",
        help="Skip NBA API fetch; build silver from existing raw.* Supabase tables",
    )
    return p.parse_args()


def _apply_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["STARTING"] = out["START_POSITION"].notna().astype(int)
    out["PTS_PER_MIN"] = out["PTS"] / out["MIN"].replace(0, np.nan)
    out["AST_PER_MIN"] = out["AST"] / out["MIN"].replace(0, np.nan)
    out["REB_PER_MIN"] = out["REB"] / out["MIN"].replace(0, np.nan)
    out["IS_HOME"] = out["MATCHUP"].str.contains("vs", na=False).astype(int)
    pos_fill = out["POS"].fillna("UNK").astype(str)
    out["POSITION_ENCODED"] = LabelEncoder().fit_transform(pos_fill)
    return out


def main() -> int:
    args = _parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.utils.nbaPlayerLogs import NBAGameLogs
    from src.utils.silver import build_gamelogs_silver, enrich_gamelogs_silver
    from src.scrapers.rotowire_scraper import run_scrape as run_rotowire_scrape

    pos_path = PROJECT_ROOT / args.positions
    ref_path = PROJECT_ROOT / args.reference_season_csv
    out_path = PROJECT_ROOT / args.out
    rotowire_path = (
        Path(args.rotowire_csv)
        if Path(args.rotowire_csv).is_absolute()
        else PROJECT_ROOT / args.rotowire_csv
    )
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint

    raw_frames = None

    if args.skip_nba_fetch:
        if not args.parquet:
            print("Error: --skip-nba-fetch requires --parquet", file=sys.stderr)
            return 1
        pq = Path(args.parquet)
        if not pq.is_absolute():
            pq = PROJECT_ROOT / pq
        print(f"Loading pre-merged frame from {pq}")
        df = enrich_gamelogs_silver(
            pd.read_parquet(pq),
            positions_path=pos_path,
            reference_season_csv=ref_path,
            rotowire_csv=rotowire_path,
            is_playoff=args.is_playoff_flag,
            skip_rotowire=args.skip_rotowire,
        )
    else:
        if not args.silver_only:
            print("── Bronze: NBA API fetch ──")
            ck_ok, ck_rows, ck_games = _checkpoint_summary(checkpoint.resolve())
            print(
                f"  checkpoint: {checkpoint.resolve()} "
                f"(exists={ck_ok}, rows={ck_rows}, games={ck_games})"
            )
            logs = NBAGameLogs(season=args.season, season_type=args.season_type)
            logs.fetch(
                batch_size=args.batch_size,
                start_position_delay=args.start_position_delay,
                start_position_workers=args.start_position_workers,
                checkpoint_path=str(checkpoint),
                db_upsert=args.db_upsert,
            )
            if not args.db_upsert:
                raw_frames = logs.data

        if not args.skip_rotowire:
            print("── Rotowire scrape ──")
            asyncio.run(
                run_rotowire_scrape(
                    season=args.rotowire_season,
                    output_file=rotowire_path,
                    headless=not args.rotowire_headed,
                )
            )

        df = build_gamelogs_silver(
            args.season,
            args.season_type,
            raw_frames=raw_frames,
            positions_path=pos_path,
            reference_season_csv=ref_path,
            rotowire_csv=rotowire_path,
            is_playoff=args.is_playoff_flag,
            skip_rotowire=args.skip_rotowire,
        )

    print("── Gold features ──")
    df = _apply_gold_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
