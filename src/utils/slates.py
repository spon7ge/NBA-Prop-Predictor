"""
slates.py — leg preparation and slate building for DFS platforms.

Public API:
  line_probs_for_market         – model predictions + book lines → P_OVER/P_UNDER per player
  build_dfs_slate_records       – in-memory 2-/3-/5-/6-leg parlays from aligned picks
  build_dfs_slates_from_aligned – same parlays written to JSON files

Internal helpers:
  _line_lookup_from_lines_df
  _json_ready
  _prep_legs_dfs
  _build_row_nleg_dfs
  _generate_candidates_dfs_nleg
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.slates_helper import (
    _greedy_slates,
    _high_total_threshold,
    _order_n_leg_row,
    _strategy_tier_profile_nleg,
    _valid_n_leg,
    load_sharp_aligned,
    _json_ready as _ud_json_ready,
    _game_ctx as _ud_game_ctx,
)

# Leg counts produced (4-leg intentionally skipped).
SLATE_LEG_COUNTS: tuple[int, ...] = (2, 3, 5, 6)

# Net-profit multipliers per platform and leg count (total payout − 1).
_DFS_PLATFORM_PAYOUTS: dict[str, dict[int, float]] = {
    "PrizePicks":       {2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5},
    "Underdog":         {2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5},
    "Betr DFS":         {2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5},
    "DraftKings Pick6": {2: 2.0, 3: 4.0, 5: 19.0, 6: 24.0},
}

_DEFAULT_PAYOUTS: dict[int, float] = {2: 2.0, 3: 4.5, 5: 19.0, 6: 36.5}

# Cap pool size for 5/6-leg enumeration (C(24,6) ≈ 134k combos).
_NLEG_COMBO_POOL_MAX = 24


# ─────────────────────────────────────────────────────────────────────────────
# LINE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _line_lookup_from_lines_df(ldf: pd.DataFrame) -> dict:
    """Build a direct name → line lookup from the lines DataFrame."""
    d = {}
    for _, r in ldf.iterrows():
        book_name = str(r["NAME"]).strip()
        d[book_name] = r["LINE"]
    return d


def line_probs_for_market(preds_df, lines_df, sim_fn, n_sims=10_000):
    """
    For each player in preds_df, run sim_fn and compute P_OVER / P_UNDER
    against the book line from lines_df.
    """
    line_by_name = _line_lookup_from_lines_df(lines_df)
    rows = []
    for _, row in preds_df.iterrows():
        name   = row["PLAYER_NAME"]
        market = row["MARKET"]
        line   = line_by_name.get(name)
        if line is None or np.isnan(row.get("MIN_Q50", np.nan)) or np.isnan(row.get("RATE_Q50", np.nan)):
            rows.append({
                "PLAYER_NAME": name, "MARKET": market, "LINE": np.nan,
                "MIN_Q50": np.nan, "STAT_Q50": np.nan,
                "P_OVER": np.nan, "P_UNDER": np.nan,
            })
            continue
        sims   = sim_fn(row, n_sims=n_sims)
        line_f = float(line)
        rows.append({
            "PLAYER_NAME": name, "MARKET": market, "LINE": line_f,
            "MIN_Q10":  round(row["MIN_Q10"],  2),
            "MIN_Q50":  round(row["MIN_Q50"],  2),
            "MIN_Q90":  round(row["MIN_Q90"],  2),
            "STAT_Q10": round(row["STAT_Q10"], 2),
            "STAT_Q50": round(row["STAT_Q50"], 2),
            "STAT_Q90": round(row["STAT_Q90"], 2),
            "P_OVER":   round(float(np.mean(sims > line_f)), 3),
            "P_UNDER":  round(float(np.mean(sims < line_f)), 3),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _json_ready(obj):
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
        return None if np.isnan(x) else x
    if isinstance(obj, (str, bool, int)) or obj is None:
        return obj
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# DFS LEG PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def _prep_legs_dfs(picks: list[dict], platform: str) -> list[dict]:
    """Convert enriched dfs_sharp_aligned picks for one platform into leg dicts
    compatible with the underdog_slates strategy-tiering helpers."""
    legs = []
    for raw in picks:
        if raw.get("platform") != platform:
            continue
        model = raw.get("model") or {}
        lean  = model.get("lean") or "OVER"
        po    = float(model.get("p_over") or 0.0)
        pu    = float(model.get("p_under") or 0.0)
        side, p_win = (
            ("UNDER", pu if pu > 0 else 1.0 - po)
            if lean == "UNDER"
            else ("OVER", po if po > 0 else 1.0 - pu)
        )
        if p_win <= 0 or p_win >= 1:
            continue
        player = raw.get("player")
        market = raw.get("market")
        line   = raw.get("dfs_line")
        team   = raw.get("team_abbr")
        if player is None or market is None or line is None or team is None:
            continue
        opp   = raw.get("opponent_abbr")
        t_s   = str(team).strip().upper()
        opp_s = str(opp).strip().upper() if opp is not None else ""
        game_key = tuple(sorted([t_s, opp_s])) if t_s and opp_s else None
        legs.append({
            "pick":          raw,
            "player":        str(player),
            "market":        str(market),
            "line":          float(line),
            "team":          t_s,
            "opponent_abbr": opp,
            "game_total":    _ud_game_ctx(raw, "game_total"),
            "game_key":      game_key,
            "side":          side,
            "p_win":         p_win,
            "prop_key":      (str(player), str(market)),
        })
    return legs


def _legs_for_nleg_combos(legs: list[dict], n_leg: int) -> list[dict]:
    """Use full leg pool for 2/3-leg; top pool by p_win for 5/6-leg combos."""
    if n_leg < 5 or len(legs) <= _NLEG_COMBO_POOL_MAX:
        return legs
    ranked = sorted(legs, key=lambda x: x["p_win"], reverse=True)
    return ranked[:_NLEG_COMBO_POOL_MAX]


def _net_mult_for(platform: str, n_leg: int) -> float:
    return _DFS_PLATFORM_PAYOUTS.get(platform, _DEFAULT_PAYOUTS).get(n_leg, _DEFAULT_PAYOUTS[n_leg])


# ─────────────────────────────────────────────────────────────────────────────
# SLATE ROW BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_row_nleg_dfs(
    legs_chunk: list[dict],
    *,
    net_mult: float,
    stake: float,
    kelly_fraction: float,
    row_extras: dict | None = None,
) -> dict:
    """Build a slate row using a fixed platform net-profit multiplier."""
    ps       = [x["p_win"] for x in legs_chunk]
    parlay_p = float(np.prod(ps))
    b        = net_mult
    ev_dol   = parlay_p * stake * b - (1.0 - parlay_p) * stake
    ev_pct   = (ev_dol / stake) * 100.0 if stake else 0.0
    k_full   = (parlay_p * b - (1.0 - parlay_p)) / b if b > 1e-12 else 0.0
    kelly    = max(0.0, k_full * kelly_fraction) * 100.0

    row: dict = {
        "PARLAY_PROB":     round(parlay_p, 5),
        "EV":              round(ev_pct, 3),
        "EV_DOLLARS":      round(ev_dol, 4),
        "KELLY":           round(kelly, 4),
        "STAKE_DOLLARS":   stake,
        "NET_PAYOUT_MULT": net_mult,
        "N_LEGS":          len(legs_chunk),
    }
    legs_out = []
    for leg in legs_chunk:
        pk = dict(leg["pick"])
        pk["side"]  = leg["side"]
        pk["p_win"] = round(leg["p_win"], 4)
        legs_out.append(pk)
    row["LEGS"] = legs_out
    if row_extras:
        row.update(row_extras)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_candidates_dfs_nleg(
    legs: list[dict],
    n_leg: int,
    stake: float,
    kelly_fraction: float,
    hi_total: float,
    net_mult: float,
) -> list[dict]:
    pool = _legs_for_nleg_combos(legs, n_leg)
    if len(pool) < n_leg:
        return []

    records = []
    for idxs in combinations(range(len(pool)), n_leg):
        chunk = [pool[i] for i in idxs]
        if not _valid_n_leg(chunk, n_leg):
            continue
        tier, profile, anchor_p, anchor_name = _strategy_tier_profile_nleg(chunk, hi_total, n_leg)
        if tier <= 0:
            continue
        ordered = _order_n_leg_row(chunk, n_leg)
        extras: dict = {
            "STRATEGY_TIER": tier,
            "COMBO_PROFILE": profile,
            "ANCHOR_WIN_PROB": round(anchor_p, 4),
        }
        if anchor_name is not None:
            extras["ANCHOR_NAME"] = anchor_name
        records.append(
            _build_row_nleg_dfs(
                ordered,
                net_mult=net_mult,
                stake=stake,
                kelly_fraction=kelly_fraction,
                row_extras=extras,
            )
        )

    sort_key = (
        (lambda r: (r.get("STRATEGY_TIER", 0), r.get("ANCHOR_WIN_PROB", 0.0), r["EV_DOLLARS"]))
        if n_leg >= 3
        else (lambda r: (r.get("STRATEGY_TIER", 0), r["EV_DOLLARS"]))
    )
    records.sort(key=sort_key, reverse=True)
    return records


def _write_slate_json(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_ud_json_ready(records), indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: BUILD SLATES
# ─────────────────────────────────────────────────────────────────────────────

def build_dfs_slate_records(
    all_picks: list[dict],
    platform: str,
    *,
    stake_dollars: float = 10.0,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
    verbose: bool = True,
) -> dict[int, list[dict]]:
    """
    Build greedy 2-/3-/5-/6-leg parlays in memory for one DFS platform.

    ``all_picks`` is the ``picks`` list from dfs_sharp_aligned JSON (or the
    equivalent list of enriched dicts from ``enrich_dfs_picks``).

    Returns ``{leg_count: [parlay_row, ...]}`` (JSON-ready dicts). Missing or
    empty leg counts map to ``[]``.
    """
    result: dict[int, list[dict]] = {n: [] for n in SLATE_LEG_COUNTS}

    legs = _prep_legs_dfs(all_picks, platform)
    if len(legs) < 2:
        if verbose:
            print(f"  [{platform}] ≥2 legs needed after filtering (have {len(legs)}) — skipping")
        return result

    hi_total = _high_total_threshold(legs)

    for n_leg in SLATE_LEG_COUNTS:
        if len(legs) < n_leg:
            if verbose:
                print(f"  [{platform}] {n_leg}-leg: need ≥{n_leg} legs (have {len(legs)}) — skipping")
            continue

        net_mult = _net_mult_for(platform, n_leg)
        candidates = _generate_candidates_dfs_nleg(
            legs, n_leg, stake_dollars, kelly_fraction, hi_total, net_mult
        )
        slate = _greedy_slates(candidates, top_n)
        if not slate:
            if verbose:
                print(f"  [{platform}] {n_leg}-leg: no slates after filtering")
            continue

        result[n_leg] = _ud_json_ready(slate)
        if verbose:
            print(f"  {platform} {n_leg}-leg: {len(slate)} parlays")

    return result


def build_dfs_slates_from_aligned(
    aligned_path: str | Path,
    platform: str,
    *,
    out_paths: dict[int, str | Path | None] | None = None,
    out_2leg: str | Path | None = None,
    out_3leg: str | Path | None = None,
    out_5leg: str | Path | None = None,
    out_6leg: str | Path | None = None,
    stake_dollars: float = 10.0,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
    verbose: bool = True,
) -> dict[int, str | None]:
    """
    Build 2-, 3-, 5-, and 6-leg slates (skip 4) for one DFS platform from the
    dfs_sharp_aligned JSON written by enrich_dfs_picks().

    Pass ``out_paths={2: path, 3: path, 5: path, 6: path}`` or individual
    ``out_2leg`` / ``out_3leg`` / ``out_5leg`` / ``out_6leg`` kwargs.

    Returns ``{leg_count: output_path_or_None}`` for each count in SLATE_LEG_COUNTS.
    """
    paths: dict[int, str | Path | None] = dict(out_paths or {})
    for n_leg, kw in ((2, out_2leg), (3, out_3leg), (5, out_5leg), (6, out_6leg)):
        if kw is not None:
            paths[n_leg] = kw

    result: dict[int, str | None] = {n: None for n in SLATE_LEG_COUNTS}

    path = Path(aligned_path).expanduser().resolve()
    if not path.is_file():
        if verbose:
            print(f"  [{platform}] aligned JSON not found: {path}")
        return result

    _, all_picks = load_sharp_aligned(path)
    records = build_dfs_slate_records(
        all_picks,
        platform,
        stake_dollars=stake_dollars,
        top_n=top_n,
        kelly_fraction=kelly_fraction,
        verbose=verbose,
    )

    for n_leg in SLATE_LEG_COUNTS:
        out = paths.get(n_leg)
        slate = records.get(n_leg) or []
        if out is None or not slate:
            continue
        out_path = Path(out)
        _write_slate_json(slate, out_path)
        result[n_leg] = str(out_path)
        if verbose:
            print(f"  {platform} {n_leg}-leg: {len(slate)} slates → {out_path}")

    return result
