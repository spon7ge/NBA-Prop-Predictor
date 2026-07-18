"""
build_slates.py — enrich DFS picks and build greedy multi-leg parlays.

Public API
----------
silver_to_base_df(silver)
    Map silver gamelog columns to the uppercase schema expected by
    ``enrich_dfs_picks``.

normalize_odds_for_enrich(df)
    Ensure OVER/UNDER column name for v2 pivot helpers.

build_live_slates(...)
    enrich_dfs_picks → greedy 2/3/5/6-leg parlays → DataFrame ready for
    ``upsert_live_slates``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.generalized_best_bets_v2 import enrich_dfs_picks
from src.utils.slates import SLATE_LEG_COUNTS, build_dfs_slate_records
from src.utils.slates_helper import load_sharp_aligned

# Platform label in odds / enrich → frontend Book slug
_PLATFORM_TO_BOOK: dict[str, str] = {
    "PrizePicks": "prizepicks",
    "Underdog": "underdog",
    "Betr DFS": "betr",
    "DraftKings Pick6": "draftkings",
}

_DFS_PLATFORMS: tuple[str, ...] = tuple(_PLATFORM_TO_BOOK.keys())

_SILVER_TO_BASE: dict[str, str] = {
    "player_name": "PLAYER_NAME",
    "team_abbreviation": "TEAM_ABBREVIATION",
    "game_date": "GAME_DATE",
    "pts": "PTS",
    "ast": "AST",
    "reb": "REB",
    "min": "MIN",
    "opp_team_abbreviation": "OPP_OPP_ABBREVIATION_base",
    "matchup": "MATCHUP",
}


def silver_to_base_df(silver: pd.DataFrame) -> pd.DataFrame:
    """Convert silver.*_player_gamelogs rows to enrich_dfs_picks base_df schema."""
    if silver.empty:
        return pd.DataFrame(columns=list(_SILVER_TO_BASE.values()))

    out = silver.rename(
        columns={k: v for k, v in _SILVER_TO_BASE.items() if k in silver.columns}
    ).copy()

    if "PLAYER_NAME" not in out.columns and "player_name" in silver.columns:
        out["PLAYER_NAME"] = silver["player_name"]

    # Prefer explicit opp abbr; fall back to parsing MATCHUP ("TEAM vs. OPP" / "TEAM @ OPP")
    if "OPP_OPP_ABBREVIATION_base" not in out.columns and "MATCHUP" in out.columns:
        out["OPP_OPP_ABBREVIATION_base"] = out["MATCHUP"].map(_opp_from_matchup)

    keep = [c for c in _SILVER_TO_BASE.values() if c in out.columns]
    # Always keep identity cols used by enrich
    for c in ("PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "PTS", "AST", "REB"):
        if c in out.columns and c not in keep:
            keep.append(c)
    if "OPP_OPP_ABBREVIATION_base" in out.columns and "OPP_OPP_ABBREVIATION_base" not in keep:
        keep.append("OPP_OPP_ABBREVIATION_base")

    return out[keep].copy()


def _opp_from_matchup(matchup: object) -> str | None:
    if matchup is None or (isinstance(matchup, float) and pd.isna(matchup)):
        return None
    s = str(matchup).strip()
    if " vs. " in s:
        parts = s.split(" vs. ")
        return parts[-1].strip() if len(parts) == 2 else None
    if " @ " in s:
        parts = s.split(" @ ")
        return parts[-1].strip() if len(parts) == 2 else None
    return None


def normalize_odds_for_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Rename OVER_UNDER → OVER/UNDER so pivot_odds_csv can consume DB/CSV odds."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    if "OVER/UNDER" not in out.columns and "OVER_UNDER" in out.columns:
        out = out.rename(columns={"OVER_UNDER": "OVER/UNDER"})
    return out


def book_slug(platform: str) -> str:
    """Map DFS platform display name → frontend Book slug."""
    if platform in _PLATFORM_TO_BOOK:
        return _PLATFORM_TO_BOOK[platform]
    key = platform.strip().lower()
    for name, slug in _PLATFORM_TO_BOOK.items():
        if name.lower() == key:
            return slug
    return key.replace(" ", "").replace("pick6", "").replace("dfs", "")


def build_live_slates(
    *,
    dfs_df: pd.DataFrame,
    us_df: pd.DataFrame,
    base_df: pd.DataFrame,
    all_line_probs: pd.DataFrame,
    game_date: str,
    league: str = "nba",
    run_at: datetime | None = None,
    team_odds_source: Any = None,
    dfs_platforms: list[str] | None = None,
    output_dir: str | Path = "data/props/enriched",
    stake_dollars: float = 10.0,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enrich DFS picks and build greedy multi-leg parlays for DB upsert.

    Returns a DataFrame with columns:
        run_at, game_date, bookmaker, n_legs, parlays
    where ``bookmaker`` is a frontend slug (prizepicks / underdog / …)
    and ``parlays`` is a list of FlatParlayRow-compatible dicts.
    """
    league = league.lower()
    run_at = run_at or datetime.now(timezone.utc)
    platforms = dfs_platforms or list(_DFS_PLATFORMS)

    dfs_norm = normalize_odds_for_enrich(dfs_df)
    us_norm = normalize_odds_for_enrich(us_df)

    if verbose:
        print("\n── Enriching DFS picks (v2) ──")

    enriched_path, aligned_path, _enriched_df = enrich_dfs_picks(
        dfs_df=dfs_norm,
        us_df=us_norm,
        base_df=base_df,
        all_line_probs=all_line_probs,
        team_odds_source=team_odds_source,
        dfs_platforms=platforms,
        current_date=game_date,
        output_dir=output_dir,
        verbose=verbose,
    )
    if verbose:
        print(f"  enriched → {enriched_path}")
        print(f"  aligned  → {aligned_path}")

    _, all_picks = load_sharp_aligned(aligned_path)
    if not all_picks:
        if verbose:
            print("  No sharp-aligned picks — no parlays to build")
        return pd.DataFrame(
            columns=["run_at", "game_date", "bookmaker", "n_legs", "parlays"]
        )

    if verbose:
        print("\n── Building greedy slates ──")

    rows: list[dict] = []
    for platform in platforms:
        records = build_dfs_slate_records(
            all_picks,
            platform,
            stake_dollars=stake_dollars,
            top_n=top_n,
            kelly_fraction=kelly_fraction,
            verbose=verbose,
        )
        slug = book_slug(platform)
        for n_legs in SLATE_LEG_COUNTS:
            parlays = records.get(n_legs) or []
            if not parlays:
                continue
            rows.append(
                {
                    "run_at": run_at,
                    "game_date": game_date,
                    "bookmaker": slug,
                    "n_legs": int(n_legs),
                    "parlays": parlays,
                }
            )

    out = pd.DataFrame(rows)
    if verbose:
        n_parlays = int(out["parlays"].map(len).sum()) if not out.empty else 0
        print(f"  {len(out)} book×leg buckets, {n_parlays} total parlays")
    return out
