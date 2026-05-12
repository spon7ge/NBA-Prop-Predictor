"""
Utility helpers for building 2- and 3-leg DFS slates.

Strategy tiers (higher = better):
  Tier 3 — same-game pair + high total
  Tier 2 — same-game pair OR high total
  Tier 1 — everything else valid
  Tier 0 — skip (three-leg only: no same-game pair, no high total)
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
    """Two legs must be different players with positive win probability."""
    if len(chunk) != 2:
        return False
    a, b = chunk
    if a["prop_key"][0] == b["prop_key"][0]:  # same player
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
    """Return (tier, profile_string) for a 2-leg combo."""
    same_game   = (a["game_key"] is not None and a["game_key"] == b["game_key"])
    a_hi        = _is_high_total(a, hi_total)
    b_hi        = _is_high_total(b, hi_total)
    both_hi     = a_hi and b_hi

    if same_game and both_hi:
        tier, profile = 3, "same-game/high-total"
    elif same_game:
        tier, profile = 2, "same-game"
    elif both_hi:
        tier, profile = 2, "high-total"
    else:
        tier, profile = 1, "standard"

    return tier, profile


# ---------------------------------------------------------------------------
# Strategy tiering — 3 legs
# ---------------------------------------------------------------------------

def _strategy_tier_profile_3leg(
    chunk: list[dict], hi_total: float
) -> tuple[int, str, float, str | None]:
    """Return (tier, profile, anchor_p, anchor_name) for a 3-leg combo."""
    # Anchor = highest-confidence leg
    anchor = max(chunk, key=lambda x: x["p_win"])
    anchor_p    = anchor["p_win"]
    anchor_name = anchor["player"]

    # Count same-game pairs
    same_game_pairs = sum(
        1
        for a, b in combinations(chunk, 2)
        if a["game_key"] is not None and a["game_key"] == b["game_key"]
    )
    has_same_game_pair = same_game_pairs >= 1

    hi_count = sum(1 for leg in chunk if _is_high_total(leg, hi_total))
    all_hi   = hi_count == 3

    if has_same_game_pair and all_hi:
        tier, profile = 3, "2+1/same-game/high-total"
    elif has_same_game_pair:
        tier, profile = 2, "2+1/same-game"
    elif all_hi:
        tier, profile = 2, "high-total"
    elif hi_count >= 2:
        tier, profile = 1, "partial-high-total"
    else:
        # No same-game pair and no high-total advantage — skip
        return 0, "standard", anchor_p, anchor_name

    return tier, profile, anchor_p, anchor_name


# ---------------------------------------------------------------------------
# Leg ordering
# ---------------------------------------------------------------------------

def _order_three_leg_row(chunk: list[dict]) -> list[dict]:
    """Return chunk reordered: anchor (highest p_win) first, then by team."""
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
            row[f"NAME {i}"]
            for i in range(1, row["N_LEGS"] + 1)
            if f"NAME {i}" in row
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
