import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from itertools import combinations

from src.pipeline.props_pipeline.min_pipeline import min_pipeline
from src.pipeline.props_pipeline.ppm_pipeline import ppm_pipeline
from src.pipeline.props_pipeline.apm_pipeline import apm_pipeline
from src.pipeline.props_pipeline.rpm_pipeline import rpm_pipeline
from src.utils.team_info import *
from src.utils.scrap_starters import NBADailyLineups

# Populated at startup by main(); empty default prevents import-time side-effects
outPlayers: dict = {}

def predict_min_times_rate(
    names,
    min_stats_df,
    prop_stats_df,
    current_date,
    *,
    rate_pipeline,
    rate_quantile_models,
    min_quantile_models,
    stat_prefix,  # "PTS", "AST", "REB", or "PTS+AST" (display market label)
    verbose=True,
):
    """
    For each name: min quantiles × rate quantiles → implied stat (same quantile index).
    """
    q10, q50, q90 = "q_0.10", "q_0.50", "q_0.90"
    records = []
    # MARKET label (stat_prefix) vs per-minute column in prop_stats_df
    rate_col_by_market = {
        "PTS": "PTS_PER_MIN",
        "AST": "AST_PER_MIN",
        "REB": "REB_PER_MIN",
    }

    for raw_name in names:
        name = raw_name
        try:
            min_feats = min_pipeline(min_stats_df, name, current_date)
            if min_feats is None:
                raise ValueError("min_pipeline returned None (need >= 10 games)")

            rate_feats = rate_pipeline(prop_stats_df, name, current_date)
            min_arr = np.asarray(min_feats, dtype=float).reshape(1, -1)
            rate_arr = np.asarray(rate_feats, dtype=float).reshape(1, -1)

            m10 = float(min_quantile_models[q10].predict(min_arr)[0])
            m50 = float(min_quantile_models[q50].predict(min_arr)[0])
            m90 = float(min_quantile_models[q90].predict(min_arr)[0])

            r10 = float(rate_quantile_models[q10].predict(rate_arr)[0])
            r50 = float(rate_quantile_models[q50].predict(rate_arr)[0])
            r90 = float(rate_quantile_models[q90].predict(rate_arr)[0])

            pdf = prop_stats_df[prop_stats_df["PLAYER_NAME"] == name].sort_values("GAME_DATE")
            rate_col = rate_col_by_market.get(stat_prefix, f"{stat_prefix}_PER_MIN")
            if rate_col not in pdf.columns:
                raise KeyError(rate_col)
            rate_history = pdf[rate_col].dropna().tail(15).tolist()

            s10, s50, s90 = m10 * r10, m50 * r50, m90 * r90
            row = {
                "PLAYER_NAME": name,
                "MARKET": stat_prefix,
                "MIN_Q10": round(m10, 2),
                "MIN_Q50": round(m50, 2),
                "MIN_Q90": round(m90, 2),
                "RATE_Q10": round(r10, 4),
                "RATE_Q50": round(r50, 4),
                "RATE_Q90": round(r90, 4),
                f"STAT_Q10": round(s10, 2),
                f"STAT_Q50": round(s50, 2),
                f"STAT_Q90": round(s90, 2),
                f"RATE_HISTORY": rate_history,
            }
        except Exception as e:
            if verbose:
                print(f"[SKIP] {name}: {e}")
            continue
        records.append(row)

    return pd.DataFrame(records)

# ---------------------------------------------------------
# 1. THE CORE SIMULATION ENGINE (POISSON-BASED)
# ---------------------------------------------------------
def triangular_clip(
    row,
    q10,
    q50,
    q90,
    lo_scale=0.80,
    hi_scale=1.20,
    low=0.0,
    high=np.inf,
    n_sims=10000,
):
    """
    Triangular distribution using model quantiles.
    Designed for minutes simulation.
    """

    q10_val = float(row[q10])
    q50_val = float(row[q50])
    q90_val = float(row[q90])

    # Fallback if bad values
    if not (np.isfinite(q10_val) and np.isfinite(q50_val) and np.isfinite(q90_val)):
        base = q50_val if np.isfinite(q50_val) else 0.0
        return np.full(n_sims, base)

    # --- 1. Base bounds ---
    left = q10_val * lo_scale
    mode = q50_val
    right = q90_val * hi_scale

    # --- 2. Enforce valid ordering ---
    left = min(left, mode)
    right = max(right, mode)

    # --- 3. Optional skew (minutes tend to downside risk) ---
    # small right shift to reflect foul trouble / blowouts
    mode = mode * 1.03

    # ensure still valid after shift
    mode = min(max(mode, left), right)

    # --- 4. Sample ---
    samples = np.random.triangular(left, mode, right, size=n_sims)

    # --- 5. Clip to realistic bounds ---
    samples = np.clip(samples, low, high)

    return samples

def triangular_clip_with_u(row, q10, q50, q90, U, lo_scale, hi_scale, low, high, mode_skew=1.03):
    """
    Inverse transform sampling with degenerate case protection.
    """
    q10_val, q50_val, q90_val = float(row[q10]), float(row[q50]), float(row[q90])
    
    left = q10_val * lo_scale
    mode = q50_val * mode_skew
    right = q90_val * hi_scale
    
    left, right = min(left, mode), max(right, mode)
    
    # Degenerate guard: if variance is zero, return the mode
    if np.isclose(left, right):
        return np.full_like(U, np.clip(mode, low, high))
    
    prob_mode = (mode - left) / (right - left)
    
    samples = np.where(
        U < prob_mode,
        left + np.sqrt(U * (mode - left) * (right - left)),
        right - np.sqrt((1 - U) * (right - mode) * (right - left))
    )
    return np.clip(samples, low, high)

def run_pts_simulation(row, n_sims=10_000, anchor_weight=0.3, decay=0.85, min_history=5):
    n_anchor = int(n_sims * anchor_weight)
    n_empirical = n_sims - n_anchor
    
    U = np.random.uniform(0, 1, n_sims)
    np.random.shuffle(U)
    
    # Anchor: Fully Coupled (Min/Rate tied to same U slice)
    U_anchor = U[:n_anchor]
    sim_min_anchor = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U_anchor, 0.75, 1.25, 0, 48)
    sim_ppm_anchor = triangular_clip_with_u(row, "RATE_Q10", "RATE_Q50", "RATE_Q90", U_anchor, 0.85, 1.15, 0, None)
    
    # Empirical: Independent Assumption
    # Explicitly acknowledging independence assumption here.
    if row.get("RATE_HISTORY") is not None and len(row["RATE_HISTORY"]) >= min_history:
        h = np.array(row["RATE_HISTORY"], dtype=float)
        weights = np.array([decay ** i for i in range(len(h) - 1, -1, -1)])
        weights /= weights.sum()
        
        empirical_rates = np.random.choice(h, size=n_empirical, p=weights)
        # Robust jitter: capped to prevent outlier-driven variance expansion
        jitter_std = np.clip(np.std(h) * 0.05, 0.01, 0.5)
        empirical_rates += np.random.normal(0, jitter_std, size=n_empirical)
        
        sim_min_empirical = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U[n_anchor:], 0.75, 1.25, 0, 48)
        
        # Result arrays
        sim_min = np.concatenate([sim_min_empirical, sim_min_anchor])
        sim_ppm = np.concatenate([np.clip(empirical_rates, 0, None), sim_ppm_anchor])
    else:
        # Fallback
        sim_min = sim_min_anchor
        sim_ppm = sim_ppm_anchor
        
    return sim_min * sim_ppm
# ---------------------------------------------------------
# 2. LINE LOOKUP & MAPPING
# ---------------------------------------------------------

def player_scenarios(df: pd.DataFrame, player_name: str, stat_name: str) -> dict:
    """
    Historical splits for a player covering the context signals
    that the base model does not capture:
        1. Active stars count  (roster context)
        2. Opponent pace       (game-speed context, proxied by Vegas total)
        3. Spread              (game-script / blowout risk)
        4. Home / Away         (venue context)

    Each split uses Bayesian shrinkage toward the player's overall median:
        shrunk = (n * split_median + k * overall_median) / (n + k)
    """
    pdf = df[df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")
    overall_median = pdf[stat_name].median()
    total_n = len(pdf)

    def split_stats(subset, k=10):
        n = len(subset)
        if n == 0:
            return {"median": None, "shrunk_median": None, "delta": 0.0, "hit_rate_vs_overall": None, "n": 0}
        split_median = subset[stat_name].median()
        shrunk = (n * split_median + k * overall_median) / (n + k)
        hit_rate = (subset[stat_name] >= overall_median).mean()
        return {
            "median":              round(split_median, 4),
            "shrunk_median":       round(shrunk, 4),
            "delta":               round(shrunk - overall_median, 4),
            "hit_rate_vs_overall": round(hit_rate, 4),
            "n":                   n,
        }

    K_ACTIVE_STARS = 10
    K_PACE         = 10
    K_SPREAD       = 10
    K_HOME_AWAY    = 5

    active_stars = {
        i: split_stats(pdf[pdf["ACTIVE_STARS_COUNT"] == i], k=K_ACTIVE_STARS)
        for i in [0, 1, 2, 3]
    }
    opp_pace = {
        "high_pace":   split_stats(pdf[pdf["GAME_TOTAL"] > 234.5],                                              k=K_PACE),
        "middle_pace": split_stats(pdf[(pdf["GAME_TOTAL"] >= 225.0) & (pdf["GAME_TOTAL"] <= 234.5)], k=K_PACE),
        "low_pace":    split_stats(pdf[pdf["GAME_TOTAL"] < 225.0],                                              k=K_PACE),
    }
    spread = {
        "favorite": split_stats(pdf[pdf["TEAM_SPREAD"] < 0], k=K_SPREAD),
        "underdog": split_stats(pdf[pdf["TEAM_SPREAD"] > 0], k=K_SPREAD),
    }
    home_away = {
        "home": split_stats(pdf[pdf["IS_HOME"] == 1], k=K_HOME_AWAY),
        "away": split_stats(pdf[pdf["IS_HOME"] == 0], k=K_HOME_AWAY),
    }
    overall_iqr = (
        pdf[stat_name].quantile(0.75) - pdf[stat_name].quantile(0.25)
        if total_n > 0 else None
    )
    return {
        "player":         player_name,
        "stat":           stat_name,
        "overall_median": round(overall_median, 4) if pd.notna(overall_median) else None,
        "overall_iqr":    round(overall_iqr, 4) if overall_iqr is not None and pd.notna(overall_iqr) else None,
        "total_games":    total_n,
        "active_stars":   active_stars,
        "opp_pace":       opp_pace,
        "spread":         spread,
        "home_away":      home_away,
    }


def adjust_predictions(
    preds_df: pd.DataFrame,
    base_df: pd.DataFrame,
    game_contexts: dict,
    *,
    min_adjust_weight: float = 0.5,
    rate_adjust_weight: float = 0.5,
    delta_cap_min: float = 4.0,
    delta_cap_rate: float = 0.04,
    pace_high_cutoff: float = 234.5,
    pace_low_cutoff: float = 225.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adjusts MIN_Q50 and RATE_Q50 from predict_min_times_rate using
    player_scenarios + game_context, then recalculates all STAT quantiles.

    Args:
        preds_df:           Output of predict_min_times_rate.
        base_df:            Game-log DataFrame used to compute player_scenarios.
        game_contexts:      {player_name: dict} where each dict is the output of
                            get_game_context (keys: active_stars, spread, total, is_home).
        min_adjust_weight:  Blend fraction for minutes adjustment  (0 = off, 1 = full delta).
        rate_adjust_weight: Blend fraction for rate adjustment     (0 = off, 1 = full delta).
        delta_cap_min:      Hard cap on total minutes delta (minutes).
        delta_cap_rate:     Hard cap on total rate delta (per-minute units).
        pace_high_cutoff:   Vegas total above this → high-pace bucket.
        pace_low_cutoff:    Vegas total below this → low-pace bucket.
        verbose:            Print per-player adjustment summary.

    Returns:
        Adjusted copy of preds_df, ready for line_probs_for_market().
    """
    rate_col_by_market = {
        "PTS": "PTS_PER_MIN",
        "AST": "AST_PER_MIN",
        "REB": "REB_PER_MIN",
    }

    def _delta(sc: dict, *key_path) -> float:
        node = sc
        for k in key_path:
            if not isinstance(node, dict):
                return 0.0
            node = node.get(k)
        if not isinstance(node, dict):
            return 0.0
        d = node.get("delta", 0.0)
        return float(d) if d is not None else 0.0

    out_rows = []
    for _, row in preds_df.iterrows():
        row = row.copy()
        name    = row["PLAYER_NAME"]
        market  = row["MARKET"]
        rate_col = rate_col_by_market.get(market, f"{market}_PER_MIN")

        ctx = game_contexts.get(name)
        if ctx is None or isinstance(ctx, str):
            if verbose:
                print(f"[NO CONTEXT] {name} — using raw model prediction")
            out_rows.append(row)
            continue

        n_stars  = int(ctx.get("active_stars", 1))
        spread   = ctx.get("spread")
        total    = ctx.get("total")
        is_home  = ctx.get("is_home", True)

        min_sc  = player_scenarios(base_df, name, "MIN")
        rate_sc = player_scenarios(base_df, name, rate_col)

        # ── minutes deltas ──────────────────────────────────────────────
        min_d_stars  = _delta(min_sc, "active_stars", min(n_stars, 3))
        min_d_spread = (
            _delta(min_sc, "spread", "favorite") if spread is not None and spread < 0
            else _delta(min_sc, "spread", "underdog") if spread is not None and spread > 0
            else 0.0
        )
        min_d_pace = (
            _delta(min_sc, "opp_pace", "high_pace")   if total is not None and total > pace_high_cutoff
            else _delta(min_sc, "opp_pace", "low_pace") if total is not None and total < pace_low_cutoff
            else _delta(min_sc, "opp_pace", "middle_pace")
        )
        min_d_home  = _delta(min_sc, "home_away", "home" if is_home else "away")
        min_delta   = float(np.clip(min_d_stars + min_d_spread + min_d_pace + min_d_home,
                                    -delta_cap_min, delta_cap_min))

        # ── rate deltas ─────────────────────────────────────────────────
        rate_d_stars  = _delta(rate_sc, "active_stars", min(n_stars, 3))
        rate_d_spread = (
            _delta(rate_sc, "spread", "favorite") if spread is not None and spread < 0
            else _delta(rate_sc, "spread", "underdog") if spread is not None and spread > 0
            else 0.0
        )
        rate_d_pace = (
            _delta(rate_sc, "opp_pace", "high_pace")   if total is not None and total > pace_high_cutoff
            else _delta(rate_sc, "opp_pace", "low_pace") if total is not None and total < pace_low_cutoff
            else _delta(rate_sc, "opp_pace", "middle_pace")
        )
        rate_d_home  = _delta(rate_sc, "home_away", "home" if is_home else "away")
        rate_delta   = float(np.clip(rate_d_stars + rate_d_spread + rate_d_pace + rate_d_home,
                                     -delta_cap_rate, delta_cap_rate))

        # ── apply adjustments ───────────────────────────────────────────
        orig_min_q50  = float(row["MIN_Q50"])
        orig_rate_q50 = float(row["RATE_Q50"])

        new_min_q50  = max(orig_min_q50  + min_adjust_weight  * min_delta,  0.0)
        new_rate_q50 = max(orig_rate_q50 + rate_adjust_weight * rate_delta, 0.0)

        # Scale Q10/Q90 proportionally to preserve distribution shape
        if orig_min_q50 > 0:
            min_scale = new_min_q50 / orig_min_q50
            row["MIN_Q10"] = round(float(row["MIN_Q10"]) * min_scale, 2)
            row["MIN_Q90"] = round(float(row["MIN_Q90"]) * min_scale, 2)
        if orig_rate_q50 > 0:
            rate_scale = new_rate_q50 / orig_rate_q50
            row["RATE_Q10"] = round(float(row["RATE_Q10"]) * rate_scale, 4)
            row["RATE_Q90"] = round(float(row["RATE_Q90"]) * rate_scale, 4)

        row["MIN_Q50"]  = round(new_min_q50,  2)
        row["RATE_Q50"] = round(new_rate_q50, 4)

        # Scale RATE_HISTORY so the empirical branch in run_pts_simulation
        # reflects the same context shift as the quantiles.
        if orig_rate_q50 > 0 and row.get("RATE_HISTORY") is not None:
            rate_scale = new_rate_q50 / orig_rate_q50
            row["RATE_HISTORY"] = [round(r * rate_scale, 6) for r in row["RATE_HISTORY"]]

        row["STAT_Q10"] = round(float(row["MIN_Q10"]) * float(row["RATE_Q10"]), 2)
        row["STAT_Q50"] = round(new_min_q50 * new_rate_q50, 2)
        row["STAT_Q90"] = round(float(row["MIN_Q90"]) * float(row["RATE_Q90"]), 2)

        if verbose:
            print(
                f"{name} [{market}]  "
                f"MIN: {orig_min_q50:.1f}→{new_min_q50:.1f} (Δ{min_delta:+.2f})  "
                f"RATE: {orig_rate_q50:.4f}→{new_rate_q50:.4f} (Δ{rate_delta:+.4f})"
            )

        out_rows.append(row)

    return pd.DataFrame(out_rows).reset_index(drop=True)

# Reverse map: full name → abbreviation (mirrors team_name_map from get_game_spread)
TEAM_NAME_TO_ABBREV = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

def get_game_context(base_df, player_name, team_odds, bookmaker_name='DraftKings', out_players=None):
    pdf = base_df[base_df['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
    if pdf.empty:
        return f"Player '{player_name}' not found in base_df"

    team_name = pdf['TEAM_NAME'].iloc[-1]

    games = team_odds.to_dict('records') if isinstance(team_odds, pd.DataFrame) else team_odds

    game_data = next(
        (g for g in games if g.get('home_team') == team_name or g.get('away_team') == team_name),
        None
    )
    if game_data is None:
        return f"No game found for team '{team_name}'"

    bk = next(
        (b for b in game_data.get('bookmakers', []) if b.get('bookmaker') == bookmaker_name),
        None
    )
    if bk is None:
        return f"Bookmaker '{bookmaker_name}' not found for this game"

    # --- active_stars ---
    team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name)
    stars = team3StarsPerTeam.get(team_abbrev, [])
    _out = out_players if out_players is not None else outPlayers
    out = _out.get(team_abbrev, [])
    active_stars = [s for s in stars if s not in out]

    result = {
        'player': player_name,
        'team': team_name,
        'opponent': game_data['away_team'] if game_data['home_team'] == team_name else game_data['home_team'],
        'is_home': game_data['home_team'] == team_name,
        'commence_time': game_data['commence_time'],
        'bookmaker': bookmaker_name,
        'active_stars': len(active_stars),   # count: 0, 1, 2, or 3
        'active_star_names': active_stars,   # optional: the names for debugging
        'spread': None,
        'spread_price': None,
        'total': None,
        'total_over_price': None,
        'total_under_price': None,
    }

    for market in bk.get('markets', []):
        if market['market_key'] == 'spreads':
            for outcome in market['outcomes']:
                if outcome['name'] == team_name:
                    result['spread'] = outcome['point']
                    result['spread_price'] = outcome['price']
        elif market['market_key'] == 'totals':
            for outcome in market['outcomes']:
                if outcome['name'] == 'Over':
                    result['total'] = outcome['point']
                    result['total_over_price'] = outcome['price']
                elif outcome['name'] == 'Under':
                    result['total_under_price'] = outcome['price']

    return result
# ---------------------------------------------------------
# 3. LINE LOOKUP & MAPPING
# ---------------------------------------------------------

def _line_lookup_from_lines_df(ldf: pd.DataFrame) -> dict:
    """Build a direct name → line lookup from the lines DataFrame."""
    d = {}
    for _, r in ldf.iterrows():
        book_name = str(r["NAME"]).strip()
        d[book_name] = r["LINE"]
    return d

# ---------------------------------------------------------
# 4. EXECUTION LOOP
# ---------------------------------------------------------

def line_probs_for_market(preds_df, lines_df, sim_fn, n_sims=10_000):
    line_by_name = _line_lookup_from_lines_df(lines_df)
    rows = []
    for _, row in preds_df.iterrows():
        name = row["PLAYER_NAME"]
        market = row["MARKET"]
        line = line_by_name.get(name)
        if line is None or np.isnan(row.get("MIN_Q50", np.nan)) or np.isnan(row.get("RATE_Q50", np.nan)):
            rows.append({
                "PLAYER_NAME": name, "MARKET": market, "LINE": np.nan,
                "MIN_Q50": np.nan, "STAT_Q50": np.nan,
                "P_OVER": np.nan, "P_UNDER": np.nan,
            })
            continue
        sims = sim_fn(row, n_sims=n_sims)
        line_f = float(line)
        rows.append({
            "PLAYER_NAME": name, "MARKET": market, "LINE": line_f,
            "MIN_Q10": round(row["MIN_Q10"], 2),
            "MIN_Q50": round(row["MIN_Q50"], 2),
            "MIN_Q90": round(row["MIN_Q90"], 2),
            "STAT_Q10": round(row["STAT_Q10"], 2),
            "STAT_Q50": round(row["STAT_Q50"], 2),
            "STAT_Q90": round(row["STAT_Q90"], 2),
            "P_OVER": round(float(np.mean(sims > line_f)), 3),
            "P_UNDER": round(float(np.mean(sims < line_f)), 3),
        })
    return pd.DataFrame(rows)


def _json_ready(obj):
    """Recursively convert slate rows for json.dump (tuples, numpy scalars, NaN)."""
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
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


def build_greedy_slate(
    prob_df: pd.DataFrame,
    min_df: pd.DataFrame,
    min_ev: float = 0.0,
    min_kelly: float = 0.0,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
    *,
    json_path: str | Path = "greedy_slate.json",
) -> str:
    """
    Build a greedy 2-leg PrizePicks slate from model probabilities.

    Args:
        prob_df:        Props data — PLAYER_NAME, MARKET, LINE, STAT_Q50, P_OVER, P_UNDER
        min_df:         NBA game log — PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE
        min_ev:         Minimum EV% to include a pair (default 0.0)
        min_kelly:      Minimum Kelly fraction to include a pair (default 0.0)
        top_n:          Max pairs to return in final slate
        kelly_fraction: Scale full Kelly (0.5 = half Kelly, recommended)
        json_path:      File path to write the slate as JSON (indent=2).

    Returns:
        Absolute path to the written JSON file.
    """

    # --- Latest team per player ---
    latest_team = (
        min_df.sort_values("GAME_DATE")
              .groupby("PLAYER_NAME", sort=False)["TEAM_ABBREVIATION"]
              .last()
    )

    # --- Prepare legs ---
    legs = prob_df.dropna(subset=["LINE", "P_OVER", "P_UNDER"]).copy()
    legs["LINE"] = legs["LINE"].astype(float)
    legs["TEAM"] = legs["PLAYER_NAME"].map(latest_team)
    legs = legs.dropna(subset=["TEAM"]).reset_index(drop=True)
    legs["PROP_KEY"] = list(zip(legs["PLAYER_NAME"], legs["MARKET"]))

    # --- Build all valid pairs ---
    records = []
    for i, j in combinations(legs.index, 2):
        r1, r2 = legs.loc[i], legs.loc[j]

        if r1["PLAYER_NAME"] == r2["PLAYER_NAME"]:
            continue
        if r1["TEAM"] == r2["TEAM"]:
            continue

        side1, p1 = _pick_side_and_prob(r1)
        side2, p2 = _pick_side_and_prob(r2)

        parlay_p = p1 * p2
        ev       = (parlay_p * 2) - ((1 - parlay_p) * 1)
        kelly    = max((2 * parlay_p - (1 - parlay_p)) / 2, 0) * kelly_fraction

        if ev < min_ev or kelly < min_kelly:
            continue

        records.append({
            # --- Core bet info ---
            "NAME 1":       r1["PLAYER_NAME"],
            "NAME 2":       r2["PLAYER_NAME"],
            "TEAM 1":       r1["TEAM"],
            "TEAM 2":       r2["TEAM"],
            "MARKET 1":     r1["MARKET"],
            "MARKET 2":     r2["MARKET"],
            "PROP_KEY_1":   r1["PROP_KEY"],
            "PROP_KEY_2":   r2["PROP_KEY"],
            "LINE 1":       r1["LINE"],
            "LINE 2":       r2["LINE"],
            "SIDE 1":       side1,
            "SIDE 2":       side2,
            "PREDICTION 1": round(float(r1["STAT_Q50"]), 1),
            "PREDICTION 2": round(float(r2["STAT_Q50"]), 1),
            'MIN PREDICTION 1': round(float(r1["MIN_Q50"]), 1),
            'MIN PREDICTION 2': round(float(r2["MIN_Q50"]), 1),
            "MODEL_PROB 1": round(p1, 3),
            "MODEL_PROB 2": round(p2, 3),
            "PARLAY_PROB":  round(parlay_p, 3),
            "EV":           round(ev * 100, 2),
            "KELLY":        round(kelly * 100, 2),

            # --- Game context ---
            "OPPONENT 1":       r1["OPPONENT"],
            "OPPONENT 2":       r2["OPPONENT"],
            "SPREAD 1":         r1["TEAM_SPREAD"],
            "SPREAD 2":         r2["TEAM_SPREAD"],
            "TOTAL 1":          r1["GAME_TOTAL"],
            "TOTAL 2":          r2["GAME_TOTAL"],
            "OPP_DEF_RATING 1": r1["OPP_DEF_RATING"],
            "OPP_DEF_RATING 2": r2["OPP_DEF_RATING"],
            "OPP_DEF_RANK 1":   r1["OPP_RANK_DEF_RATING"],
            "OPP_DEF_RANK 2":   r2["OPP_RANK_DEF_RATING"],
            "OPP_PACE 1":       r1["OPP_PACE"],
            "OPP_PACE 2":       r2["OPP_PACE"],
            "OPP_PACE_RANK 1":  r1["OPP_PACE_RANK"],
            "OPP_PACE_RANK 2":  r2["OPP_PACE_RANK"],

            # --- Market odds & implied prob ---
            "ODDS_OVER 1":      r1["ODDS_OVER"],
            "ODDS_OVER 2":      r2["ODDS_OVER"],
            "ODDS_UNDER 1":     r1["ODDS_UNDER"],
            "ODDS_UNDER 2":     r2["ODDS_UNDER"],
            "IMP_PROB_OVER 1":  round(float(r1["IMP_PROB_OVER"]), 3),
            "IMP_PROB_OVER 2":  round(float(r2["IMP_PROB_OVER"]), 3),
            "IMP_PROB_UNDER 1": round(float(r1["IMP_PROB_UNDER"]), 3),
            "IMP_PROB_UNDER 2": round(float(r2["IMP_PROB_UNDER"]), 3),

            # --- Model edge ---
            "EDGE 1":           round(float(r1["EDGE"]), 3),
            "EDGE 2":           round(float(r2["EDGE"]), 3),
            "MED_EDGE 1":       round(float(r1["MED_EDGE"]), 3),
            "MED_EDGE 2":       round(float(r2["MED_EDGE"]), 3),
            "Z_SCORE 1":        round(float(r1["Z_SCORE"]), 3),
            "Z_SCORE 2":        round(float(r2["Z_SCORE"]), 3),
            "EV_OVER 1":        round(float(r1["EV_OVER"]), 3),
            "EV_OVER 2":        round(float(r2["EV_OVER"]), 3),
            "EV_UNDER 1":       round(float(r1["EV_UNDER"]), 3),
            "EV_UNDER 2":       round(float(r2["EV_UNDER"]), 3),

            # --- Recent form ---
            "AVG_STAT_L10 1":   round(float(r1["AVG_STAT_L10"]), 1),
            "AVG_STAT_L10 2":   round(float(r2["AVG_STAT_L10"]), 1),
            "MED_STAT_L10 1":   round(float(r1["MED_STAT_L10"]), 1),
            "MED_STAT_L10 2":   round(float(r2["MED_STAT_L10"]), 1),
            "STD_STAT_L10 1":   round(float(r1["STD_STAT_L10"]), 1),
            "STD_STAT_L10 2":   round(float(r2["STD_STAT_L10"]), 1),
            "OVER_RATE_L5 1":   round(float(r1["OVER_RATE_L5"]), 3),
            "OVER_RATE_L5 2":   round(float(r2["OVER_RATE_L5"]), 3),
            "OVER_RATE_L10 1":  round(float(r1["OVER_RATE_L10"]), 3),
            "OVER_RATE_L10 2":  round(float(r2["OVER_RATE_L10"]), 3),
            "OVER_RATE_L15 1":  round(float(r1["OVER_RATE_L15"]), 3),
            "OVER_RATE_L15 2":  round(float(r2["OVER_RATE_L15"]), 3),
            "OVER_RATE_SEASON 1": round(float(r1["OVER_RATE_SEASON"]), 3),
            "OVER_RATE_SEASON 2": round(float(r2["OVER_RATE_SEASON"]), 3),

            # --- Minutes & usage ---
            "AVG_MIN_L10 1":    round(float(r1["AVG_MIN_L10"]), 1),
            "AVG_MIN_L10 2":    round(float(r2["AVG_MIN_L10"]), 1),
            "STD_MIN_L10 1":    round(float(r1["STD_MIN_L10"]), 1),
            "STD_MIN_L10 2":    round(float(r2["STD_MIN_L10"]), 1),
            "AVG_USG_L10 1":    round(float(r1["AVG_USG_L10"]), 3),
            "AVG_USG_L10 2":    round(float(r2["AVG_USG_L10"]), 3),
            "STD_USG_L10 1":    round(float(r1["STD_USG_L10"]), 3),
            "STD_USG_L10 2":    round(float(r2["STD_USG_L10"]), 3),

            # --- Matchup history ---
            "AVG_STAT_VS_MATCHUP 1": round(float(r1["AVG_STAT_VS_MATCHUP"]), 1),
            "AVG_STAT_VS_MATCHUP 2": round(float(r2["AVG_STAT_VS_MATCHUP"]), 1),
            "MATCHUP_GAMES 1":       int(r1["MATCHUP_GAMES"]),
            "MATCHUP_GAMES 2":       int(r2["MATCHUP_GAMES"]),
        })

    pair_sorted = sorted(records, key=lambda r: r["PARLAY_PROB"], reverse=True)

    # --- Greedy: each (player, market) used at most once ---
    used_props, slate_rows = set(), []
    for row in pair_sorted:
        k1, k2 = row["PROP_KEY_1"], row["PROP_KEY_2"]
        if k1 in used_props or k2 in used_props:
            continue
        used_props.add(k1)
        used_props.add(k2)
        slate_rows.append(row)

    slate_list = sorted(slate_rows, key=lambda r: r["EV"], reverse=True)[:top_n]

    out_path = Path(json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(slate_list)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"Legs: {len(legs)}  |  Pairs: {len(records)}  |  Slate: {len(slate_list)}  |  JSON: {out_path}"
    )
    return str(out_path)


def _pick_side_and_prob(row) -> tuple[str, float]:
    """Return (side, prob) for a leg based on model prediction vs line."""
    line = float(row["LINE"])
    q50  = float(row["STAT_Q50"])
    po, pu = float(row["P_OVER"]), float(row["P_UNDER"])
    if q50 > line:
        return "over", po
    if q50 < line:
        return "under", pu
    return ("over", po) if po >= pu else ("under", pu)

def build_greedy_slate_3leg(
    prob_df: pd.DataFrame,
    min_df: pd.DataFrame,
    min_ev: float = 0.0,
    min_kelly: float = 0.0,
    top_n: int = 10,
    kelly_fraction: float = 0.5,
    *,
    win_profit_units: float = 5.0,
    json_path: str | Path = "greedy_slate_3leg.json",
) -> str:
    """
    Greedy 3-leg slate from model probabilities (same schema extension as 2-leg).

    win_profit_units: net profit if the 3-leg wins, per 1 unit staked (book-specific).
        Your 2-leg builder uses 2.0 here implicitly. Set this to match your 3-leg payout.
    """
    latest_team = (
        min_df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME", sort=False)["TEAM_ABBREVIATION"]
        .last()
    )

    legs = prob_df.dropna(subset=["LINE", "P_OVER", "P_UNDER"]).copy()
    legs["LINE"] = legs["LINE"].astype(float)
    legs["TEAM"] = legs["PLAYER_NAME"].map(latest_team)
    legs = legs.dropna(subset=["TEAM"]).reset_index(drop=True)
    legs["PROP_KEY"] = list(zip(legs["PLAYER_NAME"], legs["MARKET"]))

    b = float(win_profit_units)
    records = []
    for i, j, k in combinations(legs.index, 3):
        r1, r2, r3 = legs.loc[i], legs.loc[j], legs.loc[k]

        names = {r1["PLAYER_NAME"], r2["PLAYER_NAME"], r3["PLAYER_NAME"]}
        if len(names) < 3:
            continue
        teams = {r1["TEAM"], r2["TEAM"], r3["TEAM"]}
        if len(teams) < 3:
            continue

        side1, p1 = _pick_side_and_prob(r1)
        side2, p2 = _pick_side_and_prob(r2)
        side3, p3 = _pick_side_and_prob(r3)

        parlay_p = p1 * p2 * p3
        ev = (parlay_p * b) - ((1 - parlay_p) * 1)
        kelly = max((b * parlay_p - (1 - parlay_p)) / b, 0) * kelly_fraction

        if ev < min_ev or kelly < min_kelly:
            continue

        def leg_fields(r, side, p, n: int):
            sn = str(n)
            return {
                f"NAME {n}": r["PLAYER_NAME"],
                f"TEAM {n}": r["TEAM"],
                f"MARKET {n}": r["MARKET"],
                f"PROP_KEY_{n}": r["PROP_KEY"],
                f"LINE {n}": r["LINE"],
                f"SIDE {n}": side,
                f"PREDICTION {n}": round(float(r["STAT_Q50"]), 1),
                f"MIN PREDICTION {n}": round(float(r["MIN_Q50"]), 1),
                f"MODEL_PROB {n}": round(p, 3),
                f"OPPONENT {n}": r["OPPONENT"],
                f"SPREAD {n}": r["TEAM_SPREAD"],
                f"TOTAL {n}": r["GAME_TOTAL"],
                f"OPP_DEF_RATING {n}": r["OPP_DEF_RATING"],
                f"OPP_DEF_RANK {n}": r["OPP_RANK_DEF_RATING"],
                f"OPP_PACE {n}": r["OPP_PACE"],
                f"OPP_PACE_RANK {n}": r["OPP_PACE_RANK"],
                f"ODDS_OVER {n}": r["ODDS_OVER"],
                f"ODDS_OVER {n}": r["ODDS_OVER"],
                f"ODDS_UNDER {n}": r["ODDS_UNDER"],
                f"IMP_PROB_OVER {n}": round(float(r["IMP_PROB_OVER"]), 3),
                f"IMP_PROB_UNDER {n}": round(float(r["IMP_PROB_UNDER"]), 3),
                f"EDGE {n}": round(float(r["EDGE"]), 3),
                f"MED_EDGE {n}": round(float(r["MED_EDGE"]), 3),
                f"Z_SCORE {n}": round(float(r["Z_SCORE"]), 3),
                f"EV_OVER {n}": round(float(r["EV_OVER"]), 3),
                f"EV_UNDER {n}": round(float(r["EV_UNDER"]), 3),
                f"AVG_STAT_L10 {n}": round(float(r["AVG_STAT_L10"]), 1),
                f"MED_STAT_L10 {n}": round(float(r["MED_STAT_L10"]), 1),
                f"STD_STAT_L10 {n}": round(float(r["STD_STAT_L10"]), 1),
                f"OVER_RATE_L5 {n}": round(float(r["OVER_RATE_L5"]), 3),
                f"OVER_RATE_L10 {n}": round(float(r["OVER_RATE_L10"]), 3),
                f"OVER_RATE_L15 {n}": round(float(r["OVER_RATE_L15"]), 3),
                f"OVER_RATE_SEASON {n}": round(float(r["OVER_RATE_SEASON"]), 3),
                f"AVG_MIN_L10 {n}": round(float(r["AVG_MIN_L10"]), 1),
                f"STD_MIN_L10 {n}": round(float(r["STD_MIN_L10"]), 1),
                f"AVG_USG_L10 {n}": round(float(r["AVG_USG_L10"]), 3),
                f"STD_USG_L10 {n}": round(float(r["STD_USG_L10"]), 3),
                f"AVG_STAT_VS_MATCHUP {n}": round(float(r["AVG_STAT_VS_MATCHUP"]), 1),
                f"MATCHUP_GAMES {n}": int(r["MATCHUP_GAMES"]),
            }

        row = {
            **leg_fields(r1, side1, p1, 1),
            **leg_fields(r2, side2, p2, 2),
            **leg_fields(r3, side3, p3, 3),
            "PARLAY_PROB": round(parlay_p, 3),
            "EV": round(ev * 100, 2),
            "KELLY": round(kelly * 100, 2),
        }
        records.append(row)

    triple_sorted = sorted(records, key=lambda r: r["PARLAY_PROB"], reverse=True)

    used_props: set = set()
    slate_rows = []
    for row in triple_sorted:
        k1, k2, k3 = row["PROP_KEY_1"], row["PROP_KEY_2"], row["PROP_KEY_3"]
        if k1 in used_props or k2 in used_props or k3 in used_props:
            continue
        used_props.add(k1)
        used_props.add(k2)
        used_props.add(k3)
        slate_rows.append(row)

    slate_list = sorted(slate_rows, key=lambda r: r["EV"], reverse=True)[:top_n]

    out_path = Path(json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_json_ready(slate_list), indent=2),
        encoding="utf-8",
    )
    print(
        f"Legs: {len(legs)}  |  Triples: {len(records)}  |  Slate: {len(slate_list)}  |  JSON: {out_path}"
    )
    return str(out_path)

# Helper function for leg_fields
def leg_fields(r, side, p, n: int):
    return {
        f"NAME {n}": r["PLAYER_NAME"],
        f"TEAM {n}": r["TEAM"],
        f"MARKET {n}": r["MARKET"],
        f"PROP_KEY_{n}": r["PROP_KEY"],
        f"LINE {n}": r["LINE"],
        f"SIDE {n}": side,
        f"PREDICTION {n}": round(float(r["STAT_Q50"]), 1),
        f"MIN PREDICTION {n}": round(float(r["MIN_Q50"]), 1),
        f"MODEL_PROB {n}": round(p, 3),
        f"OPPONENT {n}": r["OPPONENT"],
        f"SPREAD {n}": r["TEAM_SPREAD"],
        f"TOTAL {n}": r["GAME_TOTAL"],
        f"OPP_DEF_RATING {n}": r["OPP_DEF_RATING"],
        f"OPP_DEF_RANK {n}": r["OPP_RANK_DEF_RATING"],
        f"OPP_PACE {n}": r["OPP_PACE"],
        f"OPP_PACE_RANK {n}": r["OPP_PACE_RANK"],
        f"ODDS_OVER {n}": r["ODDS_OVER"],
        f"ODDS_UNDER {n}": r["ODDS_UNDER"],
        f"IMP_PROB_OVER {n}": round(float(r["IMP_PROB_OVER"]), 3),
        f"IMP_PROB_UNDER {n}": round(float(r["IMP_PROB_UNDER"]), 3),
        f"EDGE {n}": round(float(r["EDGE"]), 3),
        f"MED_EDGE {n}": round(float(r["MED_EDGE"]), 3),
        f"Z_SCORE {n}": round(float(r["Z_SCORE"]), 3),
        f"EV_OVER {n}": round(float(r["EV_OVER"]), 3),
        f"EV_UNDER {n}": round(float(r["EV_UNDER"]), 3),
        f"AVG_STAT_L10 {n}": round(float(r["AVG_STAT_L10"]), 1),
        f"MED_STAT_L10 {n}": round(float(r["MED_STAT_L10"]), 1),
        f"STD_STAT_L10 {n}": round(float(r["STD_STAT_L10"]), 1),
        f"OVER_RATE_L5 {n}": round(float(r["OVER_RATE_L5"]), 3),
        f"OVER_RATE_L10 {n}": round(float(r["OVER_RATE_L10"]), 3),
        f"OVER_RATE_L15 {n}": round(float(r["OVER_RATE_L15"]), 3),
        f"OVER_RATE_SEASON {n}": round(float(r["OVER_RATE_SEASON"]), 3),
        f"AVG_MIN_L10 {n}": round(float(r["AVG_MIN_L10"]), 1),
        f"STD_MIN_L10 {n}": round(float(r["STD_MIN_L10"]), 1),
        f"AVG_USG_L10 {n}": round(float(r["AVG_USG_L10"]), 3),
        f"STD_USG_L10 {n}": round(float(r["STD_USG_L10"]), 3),
        f"AVG_STAT_VS_MATCHUP {n}": round(float(r["AVG_STAT_VS_MATCHUP"]), 1),
        f"MATCHUP_GAMES {n}": int(r["MATCHUP_GAMES"]),
    }


# ---------------------------------------------------------
# DAILY PIPELINE
# ---------------------------------------------------------

def main(
    *,
    min_ev: float = 0.5,
    min_kelly: float = 0.10,
    kelly_fraction: float = 0.5,
    top_n: int = 10,
    n_sims: int = 10_000,
    bookmakers: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Run the full daily prop-prediction pipeline end-to-end:
      1. Scrape lineups & injury updates
      2. Load game logs, team odds, player lines, and models
      3. Predict MIN × RATE quantiles for PTS / AST / REB
      4. Adjust predictions with player_scenarios + game_context
      5. Simulate line probabilities
      6. Enrich with bookmaker contextual stats
      7. Save combined all_line_probs.json
      8. Build 2-leg and 3-leg greedy slates per bookmaker
    """
    from src.utils.helpers import (
        load_base_df, load_team_odds, load_player_lines,
        load_models, merge_with_bookmaker,
        BOOKMAKER_SLATE_PATHS, BOOKMAKER_3LEG_PATHS,
    )

    if bookmakers is None:
        bookmakers = list(BOOKMAKER_SLATE_PATHS.keys())

    today_str    = datetime.today().strftime('%Y%m%d')
    current_date = datetime.today().strftime('%Y-%m-%d')

    # ── 1. Lineups & injury scrape ───────────────────────────────────────────
    global outPlayers
    print("\n── Scraping lineups ──")
    scraper = NBADailyLineups("https://www.rotowire.com/basketball/nba-lineups.php")
    scraper.getDict()
    outPlayers = scraper.getOutPlayers()
    scraper.updateTeamInfo()
    if verbose:
        print(f"  Out players: {outPlayers}")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    print("\n── Loading data ──")
    base_df   = load_base_df()
    team_odds = load_team_odds()
    lines_dfs, lines_us = load_player_lines(today_str)

    lines_dfs_pts = lines_dfs[lines_dfs['CATEGORY'] == 'player_points']
    lines_dfs_ast = lines_dfs[lines_dfs['CATEGORY'] == 'player_assists']
    lines_dfs_reb = lines_dfs[lines_dfs['CATEGORY'] == 'player_rebounds']
    lines_us_pts  = lines_us[lines_us['CATEGORY']  == 'player_points']
    lines_us_ast  = lines_us[lines_us['CATEGORY']  == 'player_assists']
    lines_us_reb  = lines_us[lines_us['CATEGORY']  == 'player_rebounds']

    pts_names = lines_dfs_pts['NAME'].unique()
    ast_names = lines_dfs_ast['NAME'].unique()
    reb_names = lines_dfs_reb['NAME'].unique()

    # ── 3. Load models ───────────────────────────────────────────────────────
    print("\n── Loading models ──")
    models = load_models()
    min_q_models = models["min"]["quantile_models"]
    ppm_q_models = models["ppm"]["quantile_models"]
    apm_q_models = models["apm"]["quantile_models"]
    rpm_q_models = models["rpm"]["quantile_models"]

    # ── 4. Predict ───────────────────────────────────────────────────────────
    print("\n── Running predictions ──")
    pts_preds = predict_min_times_rate(
        pts_names, base_df, base_df, current_date,
        rate_pipeline=ppm_pipeline,
        rate_quantile_models=ppm_q_models,
        min_quantile_models=min_q_models,
        stat_prefix="PTS",
        verbose=verbose,
    )
    ast_preds = predict_min_times_rate(
        ast_names, base_df, base_df, current_date,
        rate_pipeline=apm_pipeline,
        rate_quantile_models=apm_q_models,
        min_quantile_models=min_q_models,
        stat_prefix="AST",
        verbose=verbose,
    )
    reb_preds = predict_min_times_rate(
        reb_names, base_df, base_df, current_date,
        rate_pipeline=rpm_pipeline,
        rate_quantile_models=rpm_q_models,
        min_quantile_models=min_q_models,
        stat_prefix="REB",
        verbose=verbose,
    )

    # ── 5. Context adjustments ───────────────────────────────────────────────
    print("\n── Adjusting predictions ──")
    all_names = set(pts_preds["PLAYER_NAME"]) | set(ast_preds["PLAYER_NAME"]) | set(reb_preds["PLAYER_NAME"])
    game_contexts = {
        name: get_game_context(base_df, name, team_odds)
        for name in all_names
    }
    pts_preds = adjust_predictions(pts_preds, base_df, game_contexts, verbose=verbose)
    ast_preds = adjust_predictions(ast_preds, base_df, game_contexts, verbose=verbose)
    reb_preds = adjust_predictions(reb_preds, base_df, game_contexts, verbose=verbose)

    # ── 6. Simulate line probabilities ───────────────────────────────────────
    print("\n── Simulating line probabilities ──")
    raw_probs = pd.concat([
        line_probs_for_market(pts_preds, lines_dfs_pts, run_pts_simulation, n_sims=n_sims),
        line_probs_for_market(ast_preds, lines_dfs_ast, run_pts_simulation, n_sims=n_sims),
        line_probs_for_market(reb_preds, lines_dfs_reb, run_pts_simulation, n_sims=n_sims),
    ], ignore_index=True)
    print(f"  {len(raw_probs)} raw prop lines")

    # ── 7. Enrich per bookmaker ───────────────────────────────────────────────
    print("\n── Enriching with bookmaker context ──")
    pra_lines_dfs = pd.concat([lines_dfs_pts, lines_dfs_ast, lines_dfs_reb], ignore_index=True)
    pra_lines_us  = pd.concat([lines_us_pts,  lines_us_ast,  lines_us_reb],  ignore_index=True)

    bookmaker_dfs: dict[str, pd.DataFrame] = {}
    for bk in bookmakers:
        enriched = merge_with_bookmaker(
            raw_probs, pra_lines_dfs, pra_lines_us, base_df, team_odds, bk
        )
        bookmaker_dfs[bk] = enriched

    # ── 8. Save combined JSON ─────────────────────────────────────────────────
    non_empty = [df for df in bookmaker_dfs.values() if not df.empty]
    if non_empty:
        all_line_probs = pd.concat(non_empty, ignore_index=True)
        out_path = Path('data/props/ev_analysis/all_line_probs.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_line_probs.to_json(out_path, orient='records', lines=True)
        print(f"\nSaved {len(all_line_probs)} rows → {out_path}")
    else:
        print("\nNo enriched lines — all_line_probs.json not written")
        return

    # ── 9. Build slates ───────────────────────────────────────────────────────
    print("\n── Building slates ──")
    min_df = base_df  # game-log used for team lookup in slates
    for bk, enriched in bookmaker_dfs.items():
        if enriched.empty:
            continue
        slate_2leg = BOOKMAKER_SLATE_PATHS.get(bk)
        slate_3leg = BOOKMAKER_3LEG_PATHS.get(bk)
        if slate_2leg:
            build_greedy_slate(
                prob_df=enriched, min_df=min_df,
                min_ev=min_ev, min_kelly=min_kelly,
                kelly_fraction=kelly_fraction, top_n=top_n,
                json_path=slate_2leg,
            )
        if slate_3leg:
            build_greedy_slate_3leg(
                prob_df=enriched, min_df=min_df,
                min_ev=min_ev, min_kelly=min_kelly,
                kelly_fraction=kelly_fraction, top_n=top_n,
                json_path=slate_3leg,
            )

    # ── 10. Log to ledger ─────────────────────────────────────────────────────
    print("\n── Logging predictions & slates ──")
    from src.utils import log as _log

    slate_paths = {
        bk: {
            "2leg": BOOKMAKER_SLATE_PATHS.get(bk, ""),
            "3leg": BOOKMAKER_3LEG_PATHS.get(bk, ""),
        }
        for bk in bookmakers
    }
    _log.snapshot(all_line_probs, slate_paths, date=current_date)

    print("\n── Done ──")


if __name__ == "__main__":
    main()