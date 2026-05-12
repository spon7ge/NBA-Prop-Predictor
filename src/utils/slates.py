"""
slates.py — leg preparation and slate building for DFS platforms.

Public API:
  line_probs_for_market        – model predictions + book lines → P_OVER/P_UNDER per player
  build_dfs_slates_from_aligned – 2- and 3-leg slates from dfs_sharp_aligned JSON

Internal helpers:
  _line_lookup_from_lines_df
  _json_ready
  _prep_legs_dfs
  _build_row_nleg_dfs
  _generate_candidates_dfs_2leg
  _generate_candidates_dfs_3leg
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.underdog_slates import (
    _valid_two_leg,
    _valid_three_leg,
    _strategy_tier_profile_2leg,
    _strategy_tier_profile_3leg,
    _order_three_leg_row,
    _high_total_threshold,
    _greedy_slates,
    load_sharp_aligned,
    _json_ready as _ud_json_ready,
    _game_ctx as _ud_game_ctx,
)

# Net-profit multipliers per platform: (2-leg, 3-leg)
_DFS_PLATFORM_PAYOUTS: dict[str, tuple[float, float]] = {
    "PrizePicks":       (2.0, 4.5),
    "Underdog":         (2.0, 4.5),
    "Betr DFS":         (2.0, 4.5),
    "DraftKings Pick6": (2.0, 4.0),
}


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
    for i, leg in enumerate(legs_chunk, start=1):
        pk     = leg["pick"]
        gc     = pk.get("game_context") or {}
        form   = pk.get("form") or {}
        vs_opp = pk.get("vs_opp") or {}
        model  = pk.get("model") or {}
        row[f"NAME {i}"]                = leg["player"]
        row[f"TEAM {i}"]                = leg["team"]
        row[f"MARKET {i}"]              = leg["market"]
        row[f"PROP_KEY_{i}"]            = leg["prop_key"]
        row[f"LINE {i}"]                = leg["line"]
        row[f"SIDE {i}"]                = leg["side"]
        row[f"PREDICTION {i}"]          = model.get("stat_q50")
        row[f"MODEL_PROB {i}"]          = round(leg["p_win"], 4)
        row[f"OPPONENT {i}"]            = leg.get("opponent_abbr")
        row[f"SPREAD {i}"]              = gc.get("spread")
        row[f"GAME_TOTAL {i}"]          = gc.get("game_total")
        row[f"TOTAL {i}"]               = gc.get("game_total")
        row[f"OPP_DEF_RATING_RANK {i}"] = gc.get("opp_def_rating_rank")
        row[f"OPP_PACE_RANK {i}"]       = gc.get("opp_pace_rank")
        row[f"OVER_RATE_L5 {i}"]        = form.get("over_l5")
        row[f"OVER_RATE_L10 {i}"]       = form.get("over_l10")
        row[f"OVER_RATE_L15 {i}"]       = form.get("over_l15")
        row[f"AVG_STAT_VS_MATCHUP {i}"] = vs_opp.get("avg_stat")
        row[f"MATCHUP_GAMES {i}"]       = vs_opp.get("n_games")
    if row_extras:
        row.update(row_extras)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_candidates_dfs_2leg(legs, stake, kelly_fraction, hi_total, net_mult):
    records = []
    for idxs in combinations(range(len(legs)), 2):
        chunk = [legs[i] for i in idxs]
        if not _valid_two_leg(chunk):
            continue
        a, b = chunk
        tier, profile = _strategy_tier_profile_2leg(a, b, hi_total)
        extras = {"STRATEGY_TIER": tier, "COMBO_PROFILE": profile, "ANCHOR_WIN_PROB": 0.0}
        records.append(_build_row_nleg_dfs(chunk, net_mult=net_mult, stake=stake, kelly_fraction=kelly_fraction, row_extras=extras))
    records.sort(key=lambda r: (r.get("STRATEGY_TIER", 0), r["EV_DOLLARS"]), reverse=True)
    return records


def _generate_candidates_dfs_3leg(legs, stake, kelly_fraction, hi_total, net_mult):
    records = []
    for idxs in combinations(range(len(legs)), 3):
        chunk = [legs[i] for i in idxs]
        if not _valid_three_leg(chunk):
            continue
        tier, profile, anchor_p, anchor_name = _strategy_tier_profile_3leg(chunk, hi_total)
        if tier <= 0:
            continue
        ordered = _order_three_leg_row(chunk)
        extras: dict = {"STRATEGY_TIER": tier, "COMBO_PROFILE": profile, "ANCHOR_WIN_PROB": round(anchor_p, 4)}
        if anchor_name is not None:
            extras["ANCHOR_NAME"] = anchor_name
        records.append(_build_row_nleg_dfs(ordered, net_mult=net_mult, stake=stake, kelly_fraction=kelly_fraction, row_extras=extras))
    records.sort(key=lambda r: (r.get("STRATEGY_TIER", 0), r.get("ANCHOR_WIN_PROB", 0.0), r["EV_DOLLARS"]), reverse=True)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: BUILD SLATES
# ─────────────────────────────────────────────────────────────────────────────

def build_dfs_slates_from_aligned(
    aligned_path: str | Path,
    platform: str,
    *,
    out_2leg: str | Path | None = None,
    out_3leg: str | Path | None = None,
    stake_dollars: float = 10.0,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
    verbose: bool = True,
) -> tuple[str | None, str | None]:
    """
    Build 2- and 3-leg slates for one DFS platform from the dfs_sharp_aligned JSON
    written by enrich_dfs_picks(). Uses the same strategy-tiering logic as
    underdog_slates.py: same-game preference, high-total preference, 2+1 structure.

    Returns (path_2leg, path_3leg) — None for each if not produced.
    """
    path = Path(aligned_path).expanduser().resolve()
    if not path.is_file():
        if verbose:
            print(f"  [{platform}] aligned JSON not found: {path}")
        return None, None

    _, all_picks = load_sharp_aligned(path)
    legs = _prep_legs_dfs(all_picks, platform)
    if len(legs) < 2:
        if verbose:
            print(f"  [{platform}] ≥2 legs needed after filtering (have {len(legs)}) — skipping")
        return None, None

    net_mult_2, net_mult_3 = _DFS_PLATFORM_PAYOUTS.get(platform, (2.0, 4.5))
    hi_total = _high_total_threshold(legs)
    path_2: str | None = None
    path_3: str | None = None

    cand2  = _generate_candidates_dfs_2leg(legs, stake_dollars, kelly_fraction, hi_total, net_mult_2)
    slate2 = _greedy_slates(cand2, top_n)
    if slate2 and out_2leg:
        out2 = Path(out_2leg)
        out2.parent.mkdir(parents=True, exist_ok=True)
        out2.write_text(json.dumps(_ud_json_ready(slate2), indent=2), encoding="utf-8")
        path_2 = str(out2)
        if verbose:
            print(f"  {platform} 2-leg: {len(slate2)} slates → {out2}")

    if len(legs) >= 3 and out_3leg:
        cand3  = _generate_candidates_dfs_3leg(legs, stake_dollars, kelly_fraction, hi_total, net_mult_3)
        slate3 = _greedy_slates(cand3, top_n)
        if slate3:
            out3 = Path(out_3leg)
            out3.parent.mkdir(parents=True, exist_ok=True)
            out3.write_text(json.dumps(_ud_json_ready(slate3), indent=2), encoding="utf-8")
            path_3 = str(out3)
            if verbose:
                print(f"  {platform} 3-leg: {len(slate3)} slates → {out3}")

    return path_2, path_3
