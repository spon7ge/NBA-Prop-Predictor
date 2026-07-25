"""
Run the live multi-leg slate pipeline and write parlays to ml.*_live_slates.

Workflow
--------
1. Load quantile model bundles (min + ppm/apm/rpm).
2. Load DFS + US sharp odds from raw Supabase tables.
3. For PTS / AST / REB: predict_rate → line_probs_for_market (union of DFS books).
4. enrich_dfs_picks (v2) + greedy 2/3/5/6-leg parlays via build_live_slates.
5. Upsert into ml.{league}_live_slates.

Examples (run from repository root)
-------------------------------------
    python scripts/run_live_slates.py --league nba
    python scripts/run_live_slates.py --league wnba --date 2026-07-17
    python scripts/run_live_slates.py --league nba --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.pipeline.build_slates import build_live_slates, silver_to_base_df
from src.pipeline.predict import (
    _load_silver,
    line_probs_for_market,
    load_latest_odds,
    load_opp_def_ratings,
    load_quantile_bundles,
    predict_rate,
)
from src.utils.db import upsert_live_slates
from src.utils.distributions import run_count_simulation, run_pts_simulation

_PROPS = ["player_points", "player_assists", "player_rebounds"]

_MARKET_LABEL = {
    "player_points": "PTS",
    "player_assists": "AST",
    "player_rebounds": "REB",
}

_SIM_FN = {
    "player_points": run_pts_simulation,
    "player_assists": run_count_simulation,
    "player_rebounds": run_count_simulation,
}

_RATE_BUNDLE_KEY = {
    "player_points": "ppm",
    "player_assists": "apm",
    "player_rebounds": "rpm",
}


def _team_odds_source() -> str | None:
    circa_dir = PROJECT_ROOT / "data" / "props" / "circa+betonline_team_lines"
    if not circa_dir.is_dir():
        return None
    files = list(circa_dir.glob("circa+betonline_*.json"))
    if not files:
        return None
    return str(max(files, key=lambda f: f.stat().st_mtime))


def _build_all_line_probs(
    *,
    league: str,
    game_date: str,
    season_type: str,
    bundles: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (all_line_probs, dfs_lines, us_lines)."""
    print(f"\nFetching defensive ratings ({season_type})…")
    try:
        def_ratings, league_avg_def_rtg, league_avg_pace = load_opp_def_ratings(
            season_type=season_type,
            league=league,
        )
    except Exception as exc:
        print(f"  WARNING: could not load def ratings ({exc}); context will be NaN")
        def_ratings, league_avg_def_rtg, league_avg_pace = {}, None, None

    print("\nLoading DFS + US odds…")
    dfs_all = load_latest_odds(league=league, region="dfs", prop=None)
    us_all = load_latest_odds(league=league, region="us", prop=None)
    if dfs_all.empty:
        print("  No DFS lines found.")
        return pd.DataFrame(), dfs_all, us_all

    frames: list[pd.DataFrame] = []
    for prop in _PROPS:
        market = _MARKET_LABEL[prop]
        sim_fn = _SIM_FN[prop]
        rate_key = _RATE_BUNDLE_KEY[prop]

        print(f"\n--- {market} ({prop}) ---")
        prop_lines = dfs_all[dfs_all["CATEGORY"] == prop] if "CATEGORY" in dfs_all.columns else dfs_all
        if prop_lines.empty:
            print(f"  No DFS lines for {prop}; skipping.")
            continue

        names = prop_lines["NAME"].dropna().unique().tolist()
        print(f"  {len(names)} unique players with DFS lines")

        preds = predict_rate(
            names,
            game_date,
            prop,
            league=league,
            min_bundle=bundles["min"],
            rate_bundle=bundles[rate_key],
            def_ratings=def_ratings,
            league_avg_def_rtg=league_avg_def_rtg,
            league_avg_pace=league_avg_pace,
        )
        if preds.empty:
            print(f"  No predictions for {prop}; skipping.")
            continue
        print(f"  {len(preds)} players predicted")

        # Score each bookmaker's line separately, then concat (enrich matches by name/market/line).
        for book, book_lines in prop_lines.groupby("BOOKMAKER"):
            results = line_probs_for_market(preds, book_lines, sim_fn)
            if results.empty:
                continue
            results["BOOKMAKER"] = book
            frames.append(results)

    if not frames:
        return pd.DataFrame(), dfs_all, us_all

    all_line_probs = pd.concat(frames, ignore_index=True)
    # enrich_dfs_picks matches on PLAYER_NAME / MARKET / LINE — drop book dupes by nearest line
    all_line_probs = (
        all_line_probs.dropna(subset=["PLAYER_NAME", "MARKET", "LINE"])
        .drop_duplicates(subset=["PLAYER_NAME", "MARKET", "LINE"], keep="first")
        .reset_index(drop=True)
    )
    print(f"\n  {len(all_line_probs)} unique model line-prob rows")
    return all_line_probs, dfs_all, us_all


def run_pipeline(
    league: str,
    game_date: str,
    season_type: str,
    models_dir: Path,
    *,
    dry_run: bool = False,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
) -> pd.DataFrame:
    run_at = datetime.now(timezone.utc)
    bundles = load_quantile_bundles(league, models_dir=models_dir)

    print("\nPre-loading silver gamelogs for enrich base_df…")
    silver = _load_silver(league)
    base_df = silver_to_base_df(silver)
    print(f"  base_df: {len(base_df):,} rows")

    all_line_probs, dfs_df, us_df = _build_all_line_probs(
        league=league,
        game_date=game_date,
        season_type=season_type,
        bundles=bundles,
    )
    if all_line_probs.empty or dfs_df.empty:
        print("\nNo line probs / DFS lines — nothing to write.")
        return pd.DataFrame()

    team_odds = _team_odds_source()
    if team_odds:
        print(f"\n  Team odds: {team_odds}")
    else:
        print("\n  Team odds: none (game_context spreads/totals may be empty)")

    upload = build_live_slates(
        dfs_df=dfs_df,
        us_df=us_df,
        base_df=base_df,
        all_line_probs=all_line_probs,
        game_date=game_date,
        league=league,
        run_at=run_at,
        team_odds_source=team_odds,
        output_dir=PROJECT_ROOT / "data" / "props" / "enriched",
        top_n=top_n,
        kelly_fraction=kelly_fraction,
        verbose=True,
    )

    if upload.empty:
        print("\nNo parlays built.")
        return upload

    if dry_run:
        print("\n[dry-run] Sample buckets:")
        for _, row in upload.head(8).iterrows():
            print(
                f"  {row['bookmaker']} {row['n_legs']}-leg: "
                f"{len(row['parlays'])} parlays"
            )
        return upload

    upsert_live_slates(upload, league=league)
    print(f"\nDone — results written to ml.{league}_live_slates.")
    return upload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate greedy multi-leg parlays and write to ml.*_live_slates."
    )
    p.add_argument(
        "--league",
        choices=("nba", "wnba"),
        default="nba",
        help="League to run (default: nba)",
    )
    p.add_argument(
        "--date",
        default=str(date.today()),
        help="Game date YYYY-MM-DD (default: today)",
    )
    p.add_argument(
        "--season-type",
        default="Regular Season",
        help='Season type, e.g. "Regular Season" | "Playoffs"',
    )
    p.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "models" / "saved_models"),
        help="Directory containing .joblib model bundles",
    )
    p.add_argument("--top-n", type=int, default=10, help="Parlays per book×leg (default 10)")
    p.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.5,
        help="Kelly fraction applied to slate rows (default 0.5)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline but do not write to the database",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_pipeline(
        league=args.league,
        game_date=args.date,
        season_type=args.season_type,
        models_dir=Path(args.models_dir),
        dry_run=args.dry_run,
        top_n=args.top_n,
        kelly_fraction=args.kelly_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
