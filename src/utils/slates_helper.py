"""
Utility helpers for building 2- and 3-leg DFS slates.

2-Leg strategy tiers (higher = better):
  Tier 3 — same high-total game, opposite teams (game-environment correlation)
  Tier 2 — same game opposite teams (any total) OR different games both high-total
  Tier 1 — standard independent (different games, any total)
  Same-team 2-leg combos are rejected outright.

3-Leg strategy tiers (higher = better):
  Tier 3 — 2 teammates in a high-total game + 1 independent from a different game
  Tier 2 — 2 teammates in any game + 1 independent from a different game
  Tier 1 — same-game opposite-team pair + 1 independent
  Tier 0 — skip (no useful structure)
  The independent leg is placed first (highest conviction anchor).
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_ready(obj: Any) -> Any:
    """Recursively convert slate rows for json.dump (tuples, numpy scalars, NaN)."""
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if math.isnan(x) else x
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_sharp_aligned(path: str | Path) -> tuple[dict, list[dict]]:
    """Load a dfs_sharp_aligned JSON.  Returns (metadata_dict, picks_list)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    picks = data.get("picks", []) if isinstance(data, dict) else data
    meta  = {k: v for k, v in data.items() if k != "picks"} if isinstance(data, dict) else {}
    return meta, picks


# ---------------------------------------------------------------------------
# Game-context accessor
# ---------------------------------------------------------------------------

def _game_ctx(raw: dict, key: str) -> Any:
    """Pull a value from raw['game_context'], returning None if missing."""
    gc = raw.get("game_context") or {}
    return gc.get(key)


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _high_total_threshold(legs: list[dict]) -> float:
    """Return the median game total across legs that have one, floored at 220."""
    totals = [
        float(leg["game_total"])
        for leg in legs
        if leg.get("game_total") is not None and not math.isnan(float(leg["game_total"]))
    ]
    if not totals:
        return 225.0
    return max(float(np.median(totals)), 220.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _valid_two_leg(chunk: list[dict]) -> bool:
    """Two legs must be different players, different teams, with positive win probability."""
    if len(chunk) != 2:
        return False
    a, b = chunk
    if a["prop_key"][0] == b["prop_key"][0]:  # same player
        return False
    if a["team"] == b["team"]:  # same team — avoid in 2-leg
        return False
    return a["p_win"] > 0 and b["p_win"] > 0


def _valid_three_leg(chunk: list[dict]) -> bool:
    """Three legs must be distinct players, ≥2 teams, all positive p_win."""
    if len(chunk) != 3:
        return False
    players = [leg["prop_key"][0] for leg in chunk]
    if len(set(players)) < 3:  # duplicate player
        return False
    teams = [leg["team"] for leg in chunk]
    if len(set(teams)) < 2:  # all same team — too correlated
        return False
    return all(leg["p_win"] > 0 for leg in chunk)


# ---------------------------------------------------------------------------
# Strategy tiering — 2 legs
# ---------------------------------------------------------------------------

def _strategy_tier_profile_2leg(
    a: dict, b: dict, hi_total: float
) -> tuple[int, str]:
    """Return (tier, profile_string) for a 2-leg combo.

    Same-team combos must already be rejected by _valid_two_leg before calling this.
    """
    same_game = (a["game_key"] is not None and a["game_key"] == b["game_key"])
    a_hi      = _is_high_total(a, hi_total)
    b_hi      = _is_high_total(b, hi_total)
    both_hi   = a_hi and b_hi

    if same_game and both_hi:
        # Best: game-environment correlation + high-scoring game
        tier, profile = 3, "opp-team/same-game/high-total"
    elif same_game:
        # Game-environment correlation, any total
        tier, profile = 2, "opp-team/same-game"
    elif both_hi:
        # Independent but both in high-pace/high-total matchups
        tier, profile = 2, "diff-game/both-high-total"
    else:
        # Pure independent — valid if EV is strong
        tier, profile = 1, "standard"

    return tier, profile


# ---------------------------------------------------------------------------
# Strategy tiering — 3 legs
# ---------------------------------------------------------------------------

def _strategy_tier_profile_3leg(
    chunk: list[dict], hi_total: float
) -> tuple[int, str, float, str | None]:
    """Return (tier, profile, anchor_p, anchor_name) for a 3-leg combo.

    Primary structure: 2 teammates + 1 independent from a different game.
    The independent leg is the highest-conviction anchor.
    """
    # Find a teammate pair (same team, different players already guaranteed)
    teammate_pair: tuple[dict, dict] | None = None
    independent: dict | None = None
    for a, b in combinations(chunk, 2):
        if a["team"] == b["team"]:
            teammate_pair = (a, b)
            independent = next(l for l in chunk if l is not a and l is not b)
            break

    if teammate_pair is not None:
        ta, tb = teammate_pair
        diff_game = (
            independent["game_key"] is None
            or ta["game_key"] is None
            or independent["game_key"] != ta["game_key"]
        )
        tm_hi = _is_high_total(ta, hi_total)  # both share same game total

        if diff_game and tm_hi:
            # Primary weapon: teammates in great matchup + independent anchor
            tier, profile = 3, "2+1/teammates/high-total"
        elif diff_game:
            # Teammates + independent from different game, any total
            tier, profile = 2, "2+1/teammates"
        else:
            # Independent in same game as teammates — weaker structure
            tier, profile = 1, "2+1/same-game"
    else:
        # No teammate pair — fall back to same-game opposite-team pair
        same_game_pairs = sum(
            1
            for a, b in combinations(chunk, 2)
            if a["game_key"] is not None and a["game_key"] == b["game_key"]
        )
        hi_count = sum(1 for leg in chunk if _is_high_total(leg, hi_total))

        if same_game_pairs >= 1 and hi_count >= 2:
            tier, profile = 1, "opp-team/same-game/high-total"
        elif same_game_pairs >= 1:
            tier, profile = 1, "opp-team/same-game"
        else:
            # All independent, no structure — skip
            return 0, "standard", 0.0, None

    anchor = max(chunk, key=lambda x: x["p_win"])
    return tier, profile, anchor["p_win"], anchor["player"]


# ---------------------------------------------------------------------------
# Leg ordering
# ---------------------------------------------------------------------------

def _order_three_leg_row(chunk: list[dict]) -> list[dict]:
    """Return chunk reordered: independent anchor first, then the 2 teammates.

    If no teammate pair exists, falls back to highest p_win first.
    """
    for a, b in combinations(chunk, 2):
        if a["team"] == b["team"]:
            indep = next(l for l in chunk if l is not a and l is not b)
            teammates = sorted([a, b], key=lambda x: x["player"])
            return [indep] + teammates
    # No teammate pair: highest p_win first, rest by team/player
    anchor = max(chunk, key=lambda x: x["p_win"])
    rest   = sorted(
        [leg for leg in chunk if leg is not anchor],
        key=lambda x: (x["team"], x["player"]),
    )
    return [anchor] + rest


# ---------------------------------------------------------------------------
# Greedy slate builder
# ---------------------------------------------------------------------------

def _greedy_slates(candidates: list[dict], top_n: int) -> list[dict]:
    """
    Select up to top_n slates such that no player appears in more than one
    selected slate.  Candidates must already be sorted best-first.
    """
    selected: list[dict] = []
    used_players: set[str] = set()

    for row in candidates:
        if len(selected) >= top_n:
            break
        players_in_row = {
            leg["player"]
            for leg in row.get("LEGS", [])
        }
        if players_in_row & used_players:
            continue
        selected.append(row)
        used_players |= players_in_row

    return selected


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _is_high_total(leg: dict, hi_total: float) -> bool:
    gt = leg.get("game_total")
    if gt is None:
        return False
    try:
        return float(gt) >= hi_total
    except (ValueError, TypeError):
        return False
