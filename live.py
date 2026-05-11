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
from src.utils.underdog_slates import (
    _valid_two_leg, _valid_three_leg,
    _strategy_tier_profile_2leg, _strategy_tier_profile_3leg,
    _order_three_leg_row, _high_total_threshold, _greedy_slates,
    load_sharp_aligned,
    _json_ready as _ud_json_ready,
    _game_ctx as _ud_game_ctx,
)

# Populated at startup by main(); empty default prevents import-time side-effects
outPlayers: dict = {}


def coerce_nonneg_monotone_quantiles(q10: float, q50: float, q90: float) -> tuple[float, float, float]:
    """
    Betting-facing cleanup for stacked quantile regressors.

    Separate XGBoost quantile heads minimize pinball loss on unbounded reals, so tails
    can cross (q50 < q10) or go slightly negative — invalid for minutes, per-minute rates,
    and triangular sampling in ``run_pts_simulation``. We impose:

    - support in [0, ∞) for minutes and rates (points/assists/rebounds can't be negative);
    - ordering q10 ≤ q50 ≤ q90 so triangular parameterization stays well-defined.

    A final monotone pass on implied STAT quantiles avoids nonsensical under/over tails
    in edge cases where min×rate crosses between deciles due to independence of the two
    model stacks.

    Minor inconsistency can remain between STAT_Q50 and MIN_Q50×RATE_Q50 after the STAT
    pass; MC paths still draw from sanitized MIN_/RATE_ quantiles.
    """
    a = max(float(q10), 0.0)
    b = max(float(q50), 0.0)
    c = max(float(q90), 0.0)
    b = max(b, a)
    c = max(c, b)
    return a, b, c


def _sanitize_prediction_row_quantiles(row) -> None:
    """Mutate a prediction row Series/dict in place (MIN_/RATE_/STAT_ columns)."""
    m10 = float(row["MIN_Q10"])
    m50 = float(row["MIN_Q50"])
    m90 = float(row["MIN_Q90"])
    r10 = float(row["RATE_Q10"])
    r50 = float(row["RATE_Q50"])
    r90 = float(row["RATE_Q90"])

    m10, m50, m90 = coerce_nonneg_monotone_quantiles(m10, m50, m90)
    r10, r50, r90 = coerce_nonneg_monotone_quantiles(r10, r50, r90)

    row["MIN_Q10"] = round(m10, 2)
    row["MIN_Q50"] = round(m50, 2)
    row["MIN_Q90"] = round(m90, 2)
    row["RATE_Q10"] = round(r10, 4)
    row["RATE_Q50"] = round(r50, 4)
    row["RATE_Q90"] = round(r90, 4)

    s10, s50, s90 = m10 * r10, m50 * r50, m90 * r90
    s10, s50, s90 = coerce_nonneg_monotone_quantiles(s10, s50, s90)
    row["STAT_Q10"] = round(s10, 2)
    row["STAT_Q50"] = round(s50, 2)
    row["STAT_Q90"] = round(s90, 2)


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

    Raw XGB predictions are coerced to nonnegative, ordered q10≤q50≤q90 anchors for
    ``run_pts_simulation`` and sensible betting tails (see ``coerce_nonneg_monotone_quantiles``).
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
            rate_history = pdf[rate_col].dropna().tail(10).tolist()

            m10, m50, m90 = coerce_nonneg_monotone_quantiles(m10, m50, m90)
            r10, r50, r90 = coerce_nonneg_monotone_quantiles(r10, r50, r90)

            row = {
                "PLAYER_NAME": name,
                "MARKET": stat_prefix,
                "MIN_Q10": round(m10, 2),
                "MIN_Q50": round(m50, 2),
                "MIN_Q90": round(m90, 2),
                "RATE_Q10": round(r10, 4),
                "RATE_Q50": round(r50, 4),
                "RATE_Q90": round(r90, 4),
                "RATE_HISTORY": rate_history,
            }
            s10, s50, s90 = m10 * r10, m50 * r50, m90 * r90
            s10, s50, s90 = coerce_nonneg_monotone_quantiles(s10, s50, s90)
            row["STAT_Q10"] = round(s10, 2)
            row["STAT_Q50"] = round(s50, 2)
            row["STAT_Q90"] = round(s90, 2)
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
        jitter_std = np.clip(np.std(h) * 0.3, 0.05, 1.0)
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

def player_scenarios(
    df: pd.DataFrame,
    player_name: str,
    stat_name: str,
    min_minutes: int = 15,
    last_n: int = 5,
    min_playoff_games: int = 5,
) -> dict:
    """
    Historical splits for a player covering the context signals
    that the base model does not capture:
        1. Active stars count  (roster context)        — dynamic, no silent gaps
        2. Opponent pace       (game-speed context)    — player-relative percentile buckets
        3. Spread              (game-script risk)      — includes pick'em
        4. Home / Away         (venue context)
        5. Last-N recency      (role / form drift)
        6. Interaction splits  (home×pace, away×spread)
        7. Starting vs bench   (``STARTING`` / ``START_POSITION`` prior-game flags)
        8. Playoff vs regular season, close/blowout playoff splits, uplift

    Preprocessing:
        - Drops games below `min_minutes` (garbage time / early DNP returns)
        - `SPREAD_ABS` is derived from `TEAM_SPREAD` when absent (for playoff close/blowout splits).
        - If ``STARTING`` is missing but ``START_POSITION`` exists, derives ``STARTING`` for splits.

    Bayesian shrinkage toward the player's overall median:
        shrunk = (n * split_median + k * overall_median) / (n + k)
        k is adaptive: max(5, total_n // 10) — trusts splits more for large samples.

    If neither ``STARTING`` nor ``START_POSITION`` exists, ``starting`` in the returned
    dict is empty and downstream adjustments omit the role term.
    """

    # ── 0. filter + sort ────────────────────────────────────────────────────
    pdf = (
        df[df["PLAYER_NAME"] == player_name]
        .copy()
        .sort_values("GAME_DATE")
    )
    if "STARTING" not in pdf.columns and "START_POSITION" in pdf.columns:
        pdf["STARTING"] = pdf["START_POSITION"].notna().astype(int)
    if "MIN" in pdf.columns:
        pdf = pdf[pdf["MIN"] >= min_minutes]

    if "SPREAD_ABS" not in pdf.columns and "TEAM_SPREAD" in pdf.columns:
        pdf["SPREAD_ABS"] = pdf["TEAM_SPREAD"].abs()
    elif "SPREAD_ABS" not in pdf.columns:
        pdf["SPREAD_ABS"] = np.nan

    total_n = len(pdf)
    overall_median = pdf[stat_name].median() if total_n > 0 else None

    _empty_split = {
        "median": None,
        "shrunk_median": None,
        "delta": 0.0,
        "hit_rate_vs_overall": None,
        "n": 0,
    }

    if total_n == 0 or pd.isna(overall_median):
        return {
            "player": player_name,
            "stat": stat_name,
            "overall_median": None,
            "overall_iqr": None,
            "total_games": 0,
            "min_minutes_filter": min_minutes,
            "active_stars": {},
            "opp_pace": {},
            "spread": {},
            "home_away": {},
            "recency": {},
            "interactions": {},
            "starting": {},
            "playoff": {
                "regular_season": dict(_empty_split),
                "playoffs": dict(_empty_split),
                "n_playoff_games": 0,
                "has_playoff_history": False,
                "last_playoff_run": dict(_empty_split),
                "playoff_uplift": None,
            },
        }

    # ── 1. adaptive k ───────────────────────────────────────────────────────
    K_BASE = max(5, total_n // 10)
    K_HOME_AWAY = max(3, total_n // 20)   # venue is a weaker prior — shrink less

    # ── 2. split_stats helper ───────────────────────────────────────────────
    def split_stats(subset, k=K_BASE) -> dict:
        n = len(subset)
        if n == 0:
            return {
                "median": None,
                "shrunk_median": None,
                "delta": 0.0,
                "hit_rate_vs_overall": None,
                "n": 0,
            }
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

    # ── 3. active stars — dynamic, no silent gaps ────────────────────────────
    star_counts = sorted(pdf["ACTIVE_STARS_COUNT"].dropna().unique().astype(int))
    active_stars = {i: split_stats(pdf[pdf["ACTIVE_STARS_COUNT"] == i]) for i in star_counts}

    # ── 4. opponent pace — player-relative percentile buckets ───────────────
    p33 = pdf["GAME_TOTAL"].quantile(0.33)
    p67 = pdf["GAME_TOTAL"].quantile(0.67)
    opp_pace = {
        "high_pace":   split_stats(pdf[pdf["GAME_TOTAL"] >  p67]),
        "middle_pace": split_stats(pdf[(pdf["GAME_TOTAL"] >= p33) & (pdf["GAME_TOTAL"] <= p67)]),
        "low_pace":    split_stats(pdf[pdf["GAME_TOTAL"] <  p33]),
        "_pace_thresholds": {"p33": round(p33, 2), "p67": round(p67, 2)},  # auditable
    }

    # ── 5. spread — pick'em included ────────────────────────────────────────
    spread = {
        "favorite": split_stats(pdf[pdf["TEAM_SPREAD"] <  0]),
        "pick_em":  split_stats(pdf[pdf["TEAM_SPREAD"] == 0]),
        "underdog": split_stats(pdf[pdf["TEAM_SPREAD"] >  0]),
    }

    # ── 6. home / away ──────────────────────────────────────────────────────
    home_away = {
        "home": split_stats(pdf[pdf["IS_HOME"] == 1], k=K_HOME_AWAY),
        "away": split_stats(pdf[pdf["IS_HOME"] == 0], k=K_HOME_AWAY),
    }

    # ── 7. recency ──────────────────────────────────────────────────────────
    recency = {
        f"last_{last_n}": split_stats(pdf.tail(last_n)),
        f"prior":         split_stats(pdf.iloc[:-last_n] if total_n > last_n else pdf.iloc[0:0]),
    }

    # ── 8. interaction splits ────────────────────────────────────────────────
    is_home  = pdf["IS_HOME"] == 1
    is_away  = pdf["IS_HOME"] == 0
    hi_pace  = pdf["GAME_TOTAL"] > p67
    lo_pace  = pdf["GAME_TOTAL"] < p33
    favorite = pdf["TEAM_SPREAD"] < 0
    underdog = pdf["TEAM_SPREAD"] > 0

    interactions = {
        "home_high_pace":   split_stats(pdf[is_home  & hi_pace]),
        "home_low_pace":    split_stats(pdf[is_home  & lo_pace]),
        "away_high_pace":   split_stats(pdf[is_away  & hi_pace]),
        "away_low_pace":    split_stats(pdf[is_away  & lo_pace]),
        "away_underdog":    split_stats(pdf[is_away  & underdog]),
        "home_favorite":    split_stats(pdf[is_home  & favorite]),
    }

    # ── 8.5 starting vs bench (same stat_name as other splits) ───────────────
    if "STARTING" in pdf.columns:
        st = pdf["STARTING"].fillna(0).astype(float).astype(int).clip(0, 1)
        starting = {
            "started": split_stats(pdf[st.eq(1)]),
            "bench":   split_stats(pdf[st.eq(0)]),
        }
    else:
        starting = {}

    # ── 9. playoff splits ───────────────────────────────────────────────────
    if "IS_PLAYOFF" in pdf.columns:
        is_po = pdf["IS_PLAYOFF"].fillna(0).astype(int).eq(1)
        playoff_pdf = pdf.loc[is_po]
        reg_pdf = pdf.loc[~is_po]
    else:
        playoff_pdf = pdf.iloc[0:0]
        reg_pdf = pdf

    n_playoff = len(playoff_pdf)
    n_reg = len(reg_pdf)

    playoff: dict = {
        "regular_season": split_stats(reg_pdf),
        "playoffs": split_stats(playoff_pdf),
        "n_playoff_games": n_playoff,
        "has_playoff_history": n_playoff >= min_playoff_games,
    }

    last_playoff_run = (
        playoff_pdf.sort_values("GAME_DATE").tail(last_n)
        if n_playoff else playoff_pdf
    )
    playoff["last_playoff_run"] = split_stats(last_playoff_run)

    if n_playoff >= min_playoff_games:
        sab = playoff_pdf["SPREAD_ABS"]
        close_mask = sab.notna() & (sab <= 4)
        blow_mask = sab.notna() & (sab > 4)
        playoff["playoff_close"] = split_stats(playoff_pdf.loc[close_mask])
        playoff["playoff_blowout"] = split_stats(playoff_pdf.loc[blow_mask])

    if n_playoff >= min_playoff_games and pd.notna(overall_median):
        playoff_median = playoff_pdf[stat_name].median()
        reg_median = reg_pdf[stat_name].median()
        playoff["playoff_uplift"] = (
            round(float(playoff_median - reg_median), 4)
            if pd.notna(playoff_median) and pd.notna(reg_median)
            else None
        )
    else:
        playoff["playoff_uplift"] = None

    # ── 10. overall IQR ─────────────────────────────────────────────────────
    overall_iqr = pdf[stat_name].quantile(0.75) - pdf[stat_name].quantile(0.25)

    return {
        "player":            player_name,
        "stat":              stat_name,
        "overall_median":    round(overall_median, 4),
        "overall_iqr":       round(overall_iqr, 4) if pd.notna(overall_iqr) else None,
        "total_games":       total_n,
        "min_minutes_filter": min_minutes,
        "adaptive_k":        {"base": K_BASE, "home_away": K_HOME_AWAY},
        "active_stars":      active_stars,
        "opp_pace":          opp_pace,
        "spread":            spread,
        "home_away":         home_away,
        "recency":           recency,
        "interactions":      interactions,
        "starting":          starting,
        "playoff":           playoff,
    }


def adjust_predictions(
    preds_df: pd.DataFrame,
    base_df: pd.DataFrame,
    game_contexts: dict,
    *,
    min_adjust_weight: float = 1.0,
    rate_adjust_weight: float = 1.0,
    delta_cap_min: float = 5.0,
    delta_cap_rate: float = 0.5,
    use_interactions: bool = True,
    use_recency: bool = True,
    recency_weight: float = 0.5,
    min_interaction_n: int = 15,
    min_playoff_games: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adjusts MIN_Q50 and RATE_Q50 from predict_min_times_rate using
    player_scenarios + game_context, then recalculates all STAT quantiles.

    Args:
        preds_df:            Output of predict_min_times_rate.
        base_df:             Game-log DataFrame used to compute player_scenarios.
        game_contexts:       {player_name: dict} output of get_game_context
                             (keys: active_stars, spread, total, is_home, is_playoff,
                             expected_starting = last logged 1/0 starter flag when available).
        min_adjust_weight:   Blend fraction for minutes adjustment  (0=off, 1=full delta).
        rate_adjust_weight:  Blend fraction for rate adjustment     (0=off, 1=full delta).
        delta_cap_min:       Hard cap on total minutes delta (minutes).
        delta_cap_rate:      Hard cap on total rate delta (per-minute units).
        use_interactions:    Pull away_underdog / home_favorite / pace×venue splits
                             when n >= min_interaction_n, replacing two additive deltas.
        use_recency:         Blend last_N delta into minutes and rate adjustments.
        recency_weight:      Weight on the recency delta (0=ignore, 1=full recency signal).
                             Blended as: final_delta = (1-recency_weight)*base + recency_weight*recent
        min_interaction_n:   Minimum n in an interaction split to trust it over additive.
        min_playoff_games:   Minimum playoff games to treat playoff splits as trustworthy
                             (see player_scenarios "playoff" block).
        verbose:             Print per-player adjustment summary.
    """
    rate_col_by_market = {
        "PTS": "PTS_PER_MIN",
        "AST": "AST_PER_MIN",
        "REB": "REB_PER_MIN",
    }

    # ── 1. pre-compute all scenario caches ──────────────────────────────────
    unique_names = preds_df["PLAYER_NAME"].unique()
    unique_rate_cols = {
        rate_col_by_market.get(m, f"{m}_PER_MIN")
        for m in preds_df["MARKET"].unique()
    }

    scenario_cache: dict[tuple, dict] = {}
    for name in unique_names:
        for col in ["MIN", *unique_rate_cols]:
            scenario_cache[(name, col)] = player_scenarios(
                base_df, name, col, min_playoff_games=min_playoff_games
            )

    # ── helpers ─────────────────────────────────────────────────────────────
    def _delta(sc: dict, *key_path) -> tuple[float, int]:
        """Return (delta, n) — n=0 means the split was missing or empty."""
        node = sc
        for k in key_path:
            if not isinstance(node, dict):
                return 0.0, 0
            node = node.get(k)
        if not isinstance(node, dict):
            return 0.0, 0
        d = node.get("delta", 0.0)
        n = node.get("n", 0)
        return (float(d) if d is not None else 0.0), int(n)

    def _pace_key(sc: dict, total: float | None) -> str:
        thresholds = sc.get("opp_pace", {}).get("_pace_thresholds", {})
        p33 = thresholds.get("p33")
        p67 = thresholds.get("p67")
        if total is None or p33 is None or p67 is None:
            return "middle_pace"
        if total > p67:
            return "high_pace"
        if total < p33:
            return "low_pace"
        return "middle_pace"

    def _spread_key(spread: float | None) -> str | None:
        if spread is None:
            return None
        if spread < 0:
            return "favorite"
        if spread > 0:
            return "underdog"
        return "pick_em"

    def _interaction_key(is_home: bool, spread_key: str | None, pace_key: str) -> str | None:
        venue = "home" if is_home else "away"
        if spread_key == "underdog" and not is_home:
            return "away_underdog"
        if spread_key == "favorite" and is_home:
            return "home_favorite"
        if pace_key == "high_pace" and is_home:
            return "home_high_pace"
        if pace_key == "low_pace" and is_home:
            return "home_low_pace"
        if pace_key == "high_pace" and not is_home:
            return "away_high_pace"
        if pace_key == "low_pace" and not is_home:
            return "away_low_pace"
        return None

    def _starting_role_key(expected_starting) -> str | None:
        """
        Maps ctx['expected_starting'] (typically last game's 0/1) to scenarios keys.
        """
        if expected_starting is None or (
            isinstance(expected_starting, float) and np.isnan(expected_starting)
        ):
            return None
        try:
            return "started" if int(expected_starting) == 1 else "bench"
        except (TypeError, ValueError):
            return None

    def _playoff_delta(sc: dict, spread_val: float | None) -> tuple[float, int, str | None]:
        """
        Three tiers: no playoff games → 0; small sample → heavy shrink toward overall;
        enough games → _delta from playoff splits, preferring close/blowout when spread known.
        """
        po = sc.get("playoff")
        if not isinstance(po, dict):
            return 0.0, 0, None

        n_po = int(po.get("n_playoff_games", 0))
        om = sc.get("overall_median")

        if n_po == 0:
            return 0.0, n_po, "none"

        spread_abs = abs(float(spread_val)) if spread_val is not None else None

        if n_po < min_playoff_games:
            pl_stats = po.get("playoffs") or {}
            pm = pl_stats.get("median")
            if pm is None or om is None or pd.isna(om):
                return 0.0, n_po, "small_no_median"
            k_playoff = 30.0
            d = (n_po * float(pm) + k_playoff * float(om)) / (n_po + k_playoff) - float(om)
            return float(d), n_po, "shrunk"

        if spread_abs is not None and spread_abs <= 4:
            d, n_sub = _delta(sc, "playoff", "playoff_close")
            tier = "playoff_close"
        elif spread_abs is not None and spread_abs > 4:
            d, n_sub = _delta(sc, "playoff", "playoff_blowout")
            tier = "playoff_blowout"
        else:
            d, n_sub = _delta(sc, "playoff", "playoffs")
            tier = "playoffs"

        if n_sub == 0 and spread_abs is not None:
            d, n_sub = _delta(sc, "playoff", "playoffs")
            tier = "playoffs_fallback"

        return d, n_po, tier

    def _build_delta(
        sc: dict,
        n_stars: int,
        spread_key: str | None,
        pace_key: str,
        is_home: bool,
        *,
        is_playoff_ctx: bool,
        ctx_spread: float | None,
        starting_role_key: str | None,
    ) -> tuple[float, dict]:
        """
        Returns (total_delta, component_log).
        Replaces spread+venue additive pair with interaction split when
        use_interactions=True and n >= min_interaction_n.
        """
        star_key  = min(n_stars, max(sc.get("active_stars", {}).keys(), default=3))
        d_stars, n_stars_n = _delta(sc, "active_stars", star_key)
        d_pace,  n_pace    = _delta(sc, "opp_pace", pace_key)

        interaction_key = _interaction_key(is_home, spread_key, pace_key) if use_interactions else None
        d_interaction, n_interaction = (
            _delta(sc, "interactions", interaction_key)
            if interaction_key else (0.0, 0)
        )

        use_ix = use_interactions and interaction_key and n_interaction >= min_interaction_n
        if use_ix:
            d_spread = 0.0;  n_spread = 0
            d_home   = 0.0;  n_home   = 0
            d_ix     = d_interaction
        else:
            d_spread, n_spread = (
                _delta(sc, "spread", spread_key) if spread_key and spread_key != "pick_em"
                else (0.0, 0)
            )
            d_home, n_home = _delta(sc, "home_away", "home" if is_home else "away")
            d_ix = 0.0

        base_delta = d_stars + d_pace + (d_ix if use_ix else d_spread + d_home)

        d_start, n_start = (0.0, 0)
        if starting_role_key is not None:
            ds, ns = _delta(sc, "starting", starting_role_key)
            if ns >= min_interaction_n:
                d_start, n_start = ds, ns
        base_delta = base_delta + d_start

        d_playoff, n_playoff_g, po_tier = (
            _playoff_delta(sc, ctx_spread) if is_playoff_ctx else (0.0, 0, None)
        )
        base_delta = base_delta + d_playoff

        recency_delta = 0.0
        if use_recency:
            last_key = next(
                (k for k in sc.get("recency", {}) if k.startswith("last_")), None
            )
            if last_key:
                d_rec, n_rec = _delta(sc, "recency", last_key)
                recency_delta = d_rec if n_rec >= min_interaction_n else 0.0

        total = (
            (1 - recency_weight) * base_delta + recency_weight * recency_delta
            if use_recency and recency_delta != 0.0
            else base_delta
        )

        log = {
            "stars":       (round(d_stars, 4),  n_stars_n),
            "pace":        (round(d_pace,  4),  n_pace),
            "interaction": (round(d_ix,    4),  n_interaction) if use_ix else None,
            "spread":      (round(d_spread, 4), n_spread)      if not use_ix else None,
            "home_away":   (round(d_home,  4),  n_home)        if not use_ix else None,
            "starting":    (round(d_start, 4), n_start),
            "recency":     (round(recency_delta, 4), None)      if use_recency else None,
            "playoff":     (round(d_playoff, 4), n_playoff_g, po_tier)
            if is_playoff_ctx
            else None,
            "used_interaction": use_ix,
            "interaction_key":  interaction_key if use_ix else None,
            "starting_role_key": starting_role_key,
        }
        return total, log

    def _set_adj_missing(row, err=None):
        for col in [
            "ADJ_CONTEXT_OK", "ADJ_CONTEXT_ERR", "ADJ_ACTIVE_STARS",
            "ADJ_STARS_MISSING", "ADJ_SPREAD_ROLE", "ADJ_CTX_SPREAD",
            "ADJ_CTX_TOTAL", "ADJ_MIN_DELTA", "ADJ_RATE_DELTA",
            "ADJ_MIN_SHIFT", "ADJ_RATE_SHIFT", "ADJ_USED_INTERACTION",
            "ADJ_MIN_LOG", "ADJ_RATE_LOG",
        ]:
            row[col] = None
        row["ADJ_CONTEXT_OK"]  = False
        row["ADJ_CONTEXT_ERR"] = err

    # ── 2. main loop ─────────────────────────────────────────────────────────
    out_rows = []
    for _, row in preds_df.iterrows():
        row      = row.copy()
        name     = row["PLAYER_NAME"]
        market   = row["MARKET"]
        rate_col = rate_col_by_market.get(market, f"{market}_PER_MIN")

        ctx = game_contexts.get(name)
        if ctx is None or not ctx.get("ok"):
            err = ctx.get("error") if isinstance(ctx, dict) else "missing_context"
            if verbose:
                print(f"[NO CONTEXT] {name} — raw prediction used")
            _set_adj_missing(row, err=err)
            _sanitize_prediction_row_quantiles(row)
            out_rows.append(row)
            continue

        n_stars   = int(ctx.get("active_stars", 1))
        spread    = ctx.get("spread")
        total     = ctx.get("total")
        is_home   = bool(ctx.get("is_home", True))
        is_playoff_ctx = bool(ctx.get("is_playoff", False))

        min_sc   = scenario_cache[(name, "MIN")]
        rate_sc  = scenario_cache[(name, rate_col)]

        spread_key = _spread_key(spread)
        min_pace   = _pace_key(min_sc,  total)
        rate_pace  = _pace_key(rate_sc, total)
        role_k     = _starting_role_key(ctx.get("expected_starting"))

        raw_min_delta,  min_log  = _build_delta(
            min_sc,  n_stars, spread_key, min_pace,  is_home,
            is_playoff_ctx=is_playoff_ctx,
            ctx_spread=float(spread) if spread is not None else None,
            starting_role_key=role_k,
        )
        raw_rate_delta, rate_log = _build_delta(
            rate_sc, n_stars, spread_key, rate_pace, is_home,
            is_playoff_ctx=is_playoff_ctx,
            ctx_spread=float(spread) if spread is not None else None,
            starting_role_key=role_k,
        )

        if verbose and is_playoff_ctx and min_log.get("playoff"):
            _dp, n_pg, tier = min_log["playoff"]
            if n_pg == 0 and tier == "none":
                print(f"[PLAYOFF] {name} — no playoff history, using raw prediction")

        min_delta  = float(np.clip(raw_min_delta,  -delta_cap_min,  delta_cap_min))
        rate_delta = float(np.clip(raw_rate_delta, -delta_cap_rate, delta_cap_rate))

        # ── apply ──────────────────────────────────────────────────────
        orig_min_q50  = float(row["MIN_Q50"])
        orig_rate_q50 = float(row["RATE_Q50"])

        new_min_q50  = max(orig_min_q50  + min_adjust_weight  * min_delta,  0.0)
        new_rate_q50 = max(orig_rate_q50 + rate_adjust_weight * rate_delta, 0.0)

        # shift Q10/Q90 absolutely — preserves IQR width
        row["MIN_Q10"]  = round(max(float(row["MIN_Q10"])  + min_delta,  0.0), 2)
        row["MIN_Q90"]  = round(max(float(row["MIN_Q90"])  + min_delta,  0.0), 2)
        row["RATE_Q10"] = round(max(float(row["RATE_Q10"]) + rate_delta, 0.0), 4)
        row["RATE_Q90"] = round(max(float(row["RATE_Q90"]) + rate_delta, 0.0), 4)
        row["MIN_Q50"]  = round(new_min_q50,  2)
        row["RATE_Q50"] = round(new_rate_q50, 4)

        # RATE_HISTORY: adjust only if not already scaling quantiles, pick one
        if orig_rate_q50 > 0 and row.get("RATE_HISTORY") is not None:
            row["RATE_HISTORY"] = [
                round(r + rate_adjust_weight * rate_delta, 6)
                for r in row["RATE_HISTORY"]
            ]

        row["STAT_Q10"] = round(float(row["MIN_Q10"]) * float(row["RATE_Q10"]), 2)
        row["STAT_Q50"] = round(new_min_q50 * new_rate_q50, 2)
        row["STAT_Q90"] = round(float(row["MIN_Q90"]) * float(row["RATE_Q90"]), 2)

        _sanitize_prediction_row_quantiles(row)

        # ── metadata ───────────────────────────────────────────────────
        row["ADJ_CONTEXT_OK"]      = True
        row["ADJ_CONTEXT_ERR"]     = None
        row["ADJ_ACTIVE_STARS"]    = n_stars
        row["ADJ_STARS_MISSING"]   = max(0, 3 - min(n_stars, 3))
        row["ADJ_SPREAD_ROLE"]     = spread_key
        row["ADJ_CTX_SPREAD"]      = float(spread) if spread is not None else None
        row["ADJ_CTX_TOTAL"]       = float(total)  if total  is not None else None
        row["ADJ_MIN_DELTA"]       = round(min_delta,  4)
        row["ADJ_RATE_DELTA"]      = round(rate_delta, 6)
        row["ADJ_MIN_SHIFT"]       = round(new_min_q50  - orig_min_q50,  2)
        row["ADJ_RATE_SHIFT"]      = round(new_rate_q50 - orig_rate_q50, 4)
        row["ADJ_USED_INTERACTION"]= min_log.get("used_interaction", False)
        row["ADJ_MIN_LOG"]         = min_log
        row["ADJ_RATE_LOG"]        = rate_log

        if verbose:
            ix_tag = f" [ix: {min_log.get('interaction_key')}]" if min_log.get("used_interaction") else ""
            print(
                f"{name} [{market}]{ix_tag}  "
                f"pace_bucket={min_pace}  "
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

def get_game_context(
    base_df: pd.DataFrame,
    player_name: str,
    team_odds: pd.DataFrame | list[dict],
    *,
    bookmaker_priority: list[str] = ("DraftKings", "FanDuel", "BetMGM"),
    out_players: dict | None = None,
    stars_by_team: dict | None = None,
    team_abbrev_map: dict | None = None,
    is_playoff: bool = False,
) -> dict:
    """
    Builds the game-context dict consumed by adjust_predictions().

    Returns a dict always. On failure, "ok" is False and "error" describes why.
    On success, "ok" is True and all expected keys are present,
    including ``expected_starting`` when derivable from the latest game row
    (``STARTING`` or ``START_POSITION`` on ``base_df``).

    Args:
        base_df:           Game-log DataFrame (used to resolve player → team).
        player_name:       Exact PLAYER_NAME string as it appears in base_df.
        team_odds:         Odds feed — either a DataFrame or list of game dicts.
        bookmaker_priority: Bookmakers tried in order; first hit wins.
        out_players:       {team_abbrev: [player, ...]} of inactive players.
                           Falls back to global outPlayers if None.
        stars_by_team:     {team_abbrev: [player, ...]} of star designations.
                           Falls back to global team3StarsPerTeam if None.
        team_abbrev_map:   {full_team_name: abbrev} lookup.
                           Falls back to global TEAM_NAME_TO_ABBREV if None.
        is_playoff:        When True, adjust_predictions applies tiered playoff deltas.
    """
    # ── resolve injected vs global dependencies ──────────────────────────────
    _out_players   = out_players   if out_players   is not None else outPlayers
    _stars_by_team = stars_by_team if stars_by_team is not None else team3StarsPerTeam
    _abbrev_map    = team_abbrev_map if team_abbrev_map is not None else TEAM_NAME_TO_ABBREV

    def _fail(reason: str) -> dict:
        return {
            "ok": False, "error": reason,
            "player": player_name,
            "team": None, "opponent": None, "is_home": None,
            "active_stars": None, "active_star_names": None,
            "spread": None, "spread_price": None,
            "total": None, "total_over_price": None, "total_under_price": None,
            "bookmaker": None, "commence_time": None,
            "is_playoff": None,
            "expected_starting": None,
        }

    # ── 1. resolve player → team ─────────────────────────────────────────────
    pdf = base_df[base_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")
    if pdf.empty:
        return _fail(f"player '{player_name}' not found in base_df")

    team_name   = pdf["TEAM_NAME"].iloc[-1]
    team_abbrev = _abbrev_map.get(team_name)

    # ── 2. find the game — normalized match ──────────────────────────────────
    games = team_odds.to_dict("records") if isinstance(team_odds, pd.DataFrame) else team_odds

    def _team_matches(feed_name: str) -> bool:
        """Tolerates case, punctuation, and common abbreviation mismatches."""
        norm = lambda s: s.lower().replace(".", "").replace("-", " ").strip()
        if norm(feed_name) == norm(team_name):
            return True
        if team_abbrev and norm(feed_name) == norm(team_abbrev):
            return True
        return False

    game_data = next(
        (g for g in games
         if _team_matches(g.get("home_team", ""))
         or _team_matches(g.get("away_team", ""))),
        None,
    )
    if game_data is None:
        return _fail(f"no game found for team '{team_name}' (abbrev: {team_abbrev})")

    is_home  = _team_matches(game_data.get("home_team", ""))
    opponent = game_data["away_team"] if is_home else game_data["home_team"]

    # ── 3. bookmaker — first available from priority list ────────────────────
    available_books = {b.get("bookmaker"): b for b in game_data.get("bookmakers", [])}
    bk = None
    used_bookmaker = None
    for book in bookmaker_priority:
        if book in available_books:
            bk = available_books[book]
            used_bookmaker = book
            break

    if bk is None:
        return _fail(
            f"none of {list(bookmaker_priority)} found; "
            f"available: {list(available_books.keys())}"
        )

    # ── 4. active stars ──────────────────────────────────────────────────────
    stars  = _stars_by_team.get(team_abbrev, [])
    out    = _out_players.get(team_abbrev, [])
    active = [s for s in stars if s not in out]

    # ── 5. parse markets ─────────────────────────────────────────────────────
    spread = spread_price = total = over_price = under_price = None

    for market in bk.get("markets", []):
        key = market.get("market_key")
        if key == "spreads":
            for outcome in market.get("outcomes", []):
                if _team_matches(outcome.get("name", "")):
                    spread       = outcome.get("point")
                    spread_price = outcome.get("price")
        elif key == "totals":
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == "Over":
                    total      = outcome.get("point")
                    over_price = outcome.get("price")
                elif outcome.get("name") == "Under":
                    under_price = outcome.get("price")

    # ── 6. infer likely role tonight from most recent logged game ─────────────
    last_row = pdf.iloc[-1]
    expected_starting = None
    if "STARTING" in pdf.columns:
        v = last_row["STARTING"]
        if pd.notna(v):
            try:
                expected_starting = int(np.clip(round(float(v)), 0, 1))
            except (TypeError, ValueError):
                expected_starting = None
    elif "START_POSITION" in pdf.columns:
        expected_starting = int(bool(pd.notna(last_row["START_POSITION"])))

    # ── 7. warn on missing lines ─────────────────────────────────────────────
    missing = [k for k, v in [("spread", spread), ("total", total)] if v is None]
    if missing:
        import warnings
        warnings.warn(
            f"{player_name} ({team_name}): missing lines {missing} "
            f"from {used_bookmaker} — context signals will be partial",
            stacklevel=2,
        )

    return {
        "ok":                True,
        "error":             None,
        "player":            player_name,
        "team":              team_name,
        "team_abbrev":       team_abbrev,
        "opponent":          opponent,
        "is_home":           is_home,
        "commence_time":     game_data.get("commence_time"),
        "bookmaker":         used_bookmaker,
        "active_stars":      len(active),
        "active_star_names": active,
        "spread":            spread,
        "spread_price":      spread_price,
        "total":             total,
        "total_over_price":  over_price,
        "total_under_price": under_price,
        "is_playoff":        is_playoff,
        "expected_starting": expected_starting,
    }
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
    if isinstance(obj, np.bool_):
        return bool(obj)
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


def classify_bet(row: dict) -> tuple[str, int]:
    """
    Score a 2-leg PrizePicks pair across 6 dimensions and return
    a (classification, score) tuple.

    Scoring:
        13–15  → STRONG
        8–12   → MARGINAL
        0–7    → SKIP
    """
    score = 0

    # ── 1. EV (0–3 pts) ──────────────────────────────────────────────────────
    ev = row["EV"]
    if ev >= 15:
        score += 3
    elif ev >= 10:
        score += 2
    elif ev >= 5:
        score += 1

    # ── 2. Parlay probability (0–2 pts) ──────────────────────────────────────
    pp = row["PARLAY_PROB"]
    if pp >= 0.65:
        score += 2
    elif pp >= 0.58:
        score += 1

    # ── 3. Individual leg strength (0–2 pts) ─────────────────────────────────
    mp1, mp2 = row["MODEL_PROB 1"], row["MODEL_PROB 2"]
    if mp1 >= 0.65 and mp2 >= 0.65:
        score += 2
    elif mp1 >= 0.60 and mp2 >= 0.60:
        score += 1

    # ── 4. Model edge over implied prob (0–2 pts) ────────────────────────────
    e1, e2 = row["EDGE 1"], row["EDGE 2"]
    if e1 >= 0.07 and e2 >= 0.07:
        score += 2
    elif e1 >= 0.04 and e2 >= 0.04:
        score += 1

    # ── 5. Recent form confirms side (0–2 pts) ───────────────────────────────
    #   For OVER picks we want OVER_RATE_L10 ≥ 0.60.
    #   For UNDER picks we want OVER_RATE_L10 ≤ 0.40 (i.e. under rate ≥ 0.60).
    def form_ok(side: str, over_rate: float) -> bool:
        if side == "OVER":
            return over_rate >= 0.60
        return over_rate <= 0.40

    f1 = form_ok(row["SIDE 1"], row["OVER_RATE_L10 1"])
    f2 = form_ok(row["SIDE 2"], row["OVER_RATE_L10 2"])
    if f1 and f2:
        score += 2
    elif f1 or f2:
        score += 1

    # ── 6. Player consistency — low minutes & stat variance (0–2 pts) ────────
    std1, std2 = row["STD_STAT_L10 1"], row["STD_STAT_L10 2"]
    sm1, sm2   = row["STD_MIN_L10 1"],  row["STD_MIN_L10 2"]
    consistent = std1 < 4.0 and std2 < 4.0
    stable_min = sm1  < 3.0 and sm2  < 3.0
    if consistent and stable_min:
        score += 2
    elif consistent or stable_min:
        score += 1

    # ── 7. Kelly confirmation (0–1 pt) ───────────────────────────────────────
    if row["KELLY"] >= 5.0:
        score += 1

    # ── 8. Matchup history depth (0–1 pt) ────────────────────────────────────
    if row["MATCHUP_GAMES 1"] >= 3 and row["MATCHUP_GAMES 2"] >= 3:
        score += 1

    # ── 9. Median edge sanity check (0–1 pt) ─────────────────────────────────
    #   MED_EDGE should be positive on chosen side; negative is a red flag.
    if row["MED_EDGE 1"] > 0 and row["MED_EDGE 2"] > 0:
        score += 1

    # ── Classification ────────────────────────────────────────────────────────
    if score >= 13:
        label = "STRONG"
    elif score >= 8:
        label = "MARGINAL"
    else:
        label = "SKIP"

    return label, score


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

    # ── Latest team per player ────────────────────────────────────────────────
    latest_team = (
        min_df.sort_values("GAME_DATE")
              .groupby("PLAYER_NAME", sort=False)["TEAM_ABBREVIATION"]
              .last()
    )

    # ── Prepare legs ──────────────────────────────────────────────────────────
    legs = prob_df.dropna(subset=["LINE", "P_OVER", "P_UNDER"]).copy()
    legs["LINE"]  = legs["LINE"].astype(float)
    legs["TEAM"]  = legs["PLAYER_NAME"].map(latest_team)
    legs = legs.dropna(subset=["TEAM"]).reset_index(drop=True)
    legs["PROP_KEY"] = list(zip(legs["PLAYER_NAME"], legs["MARKET"]))

    # ── Build all valid pairs ─────────────────────────────────────────────────
    records = []
    for i, j in combinations(legs.index, 2):
        r1, r2 = legs.loc[i], legs.loc[j]

        # No same player, no same team
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
            # ── Core bet info ──────────────────────────────────────────────────
            "NAME 1":         r1["PLAYER_NAME"],
            "NAME 2":         r2["PLAYER_NAME"],
            "TEAM 1":         r1["TEAM"],
            "TEAM 2":         r2["TEAM"],
            "MARKET 1":       r1["MARKET"],
            "MARKET 2":       r2["MARKET"],
            "PROP_KEY_1":     r1["PROP_KEY"],
            "PROP_KEY_2":     r2["PROP_KEY"],
            "LINE 1":         r1["LINE"],
            "LINE 2":         r2["LINE"],
            "SIDE 1":         side1,
            "SIDE 2":         side2,
            "PREDICTION 1":   round(float(r1["STAT_Q50"]), 1),
            "PREDICTION 2":   round(float(r2["STAT_Q50"]), 1),
            "MIN PREDICTION 1": round(float(r1["MIN_Q50"]), 1),
            "MIN PREDICTION 2": round(float(r2["MIN_Q50"]), 1),
            "MODEL_PROB 1":   round(p1, 3),
            "MODEL_PROB 2":   round(p2, 3),
            "PARLAY_PROB":    round(parlay_p, 3),
            "EV":             round(ev * 100, 2),
            "KELLY":          round(kelly * 100, 2),

            # ── Game context ───────────────────────────────────────────────────
            "OPPONENT 1":         r1["OPPONENT"],
            "OPPONENT 2":         r2["OPPONENT"],
            "SPREAD 1":           r1["TEAM_SPREAD"],
            "SPREAD 2":           r2["TEAM_SPREAD"],
            "TOTAL 1":            r1["GAME_TOTAL"],
            "TOTAL 2":            r2["GAME_TOTAL"],
            "OPP_DEF_RATING 1":   r1["OPP_DEF_RATING"],
            "OPP_DEF_RATING 2":   r2["OPP_DEF_RATING"],
            "OPP_DEF_RANK 1":     r1["OPP_RANK_DEF_RATING"],
            "OPP_DEF_RANK 2":     r2["OPP_RANK_DEF_RATING"],
            "OPP_PACE 1":         r1["OPP_PACE"],
            "OPP_PACE 2":         r2["OPP_PACE"],
            "OPP_PACE_RANK 1":    r1["OPP_PACE_RANK"],
            "OPP_PACE_RANK 2":    r2["OPP_PACE_RANK"],

            # ── Market odds & implied prob ─────────────────────────────────────
            "ODDS_OVER 1":        r1["ODDS_OVER"],
            "ODDS_OVER 2":        r2["ODDS_OVER"],
            "ODDS_UNDER 1":       r1["ODDS_UNDER"],
            "ODDS_UNDER 2":       r2["ODDS_UNDER"],
            "IMP_PROB_OVER 1":    round(float(r1["IMP_PROB_OVER"]), 3),
            "IMP_PROB_OVER 2":    round(float(r2["IMP_PROB_OVER"]), 3),
            "IMP_PROB_UNDER 1":   round(float(r1["IMP_PROB_UNDER"]), 3),
            "IMP_PROB_UNDER 2":   round(float(r2["IMP_PROB_UNDER"]), 3),

            # ── Model edge ────────────────────────────────────────────────────
            "EDGE 1":             round(float(r1["EDGE"]), 3),
            "EDGE 2":             round(float(r2["EDGE"]), 3),
            "MED_EDGE 1":         round(float(r1["MED_EDGE"]), 3),
            "MED_EDGE 2":         round(float(r2["MED_EDGE"]), 3),
            "Z_SCORE 1":          round(float(r1["Z_SCORE"]), 3),
            "Z_SCORE 2":          round(float(r2["Z_SCORE"]), 3),
            "EV_OVER 1":          round(float(r1["EV_OVER"]), 3),
            "EV_OVER 2":          round(float(r2["EV_OVER"]), 3),
            "EV_UNDER 1":         round(float(r1["EV_UNDER"]), 3),
            "EV_UNDER 2":         round(float(r2["EV_UNDER"]), 3),

            # ── Recent form ───────────────────────────────────────────────────
            "AVG_STAT_L10 1":     round(float(r1["AVG_STAT_L10"]), 1),
            "AVG_STAT_L10 2":     round(float(r2["AVG_STAT_L10"]), 1),
            "MED_STAT_L10 1":     round(float(r1["MED_STAT_L10"]), 1),
            "MED_STAT_L10 2":     round(float(r2["MED_STAT_L10"]), 1),
            "STD_STAT_L10 1":     round(float(r1["STD_STAT_L10"]), 1),
            "STD_STAT_L10 2":     round(float(r2["STD_STAT_L10"]), 1),
            "OVER_RATE_L5 1":     round(float(r1["OVER_RATE_L5"]), 3),
            "OVER_RATE_L5 2":     round(float(r2["OVER_RATE_L5"]), 3),
            "OVER_RATE_L10 1":    round(float(r1["OVER_RATE_L10"]), 3),
            "OVER_RATE_L10 2":    round(float(r2["OVER_RATE_L10"]), 3),
            "OVER_RATE_L15 1":    round(float(r1["OVER_RATE_L15"]), 3),
            "OVER_RATE_L15 2":    round(float(r2["OVER_RATE_L15"]), 3),
            "OVER_RATE_SEASON 1": round(float(r1["OVER_RATE_SEASON"]), 3),
            "OVER_RATE_SEASON 2": round(float(r2["OVER_RATE_SEASON"]), 3),

            # ── Minutes & usage ───────────────────────────────────────────────
            "AVG_MIN_L10 1":      round(float(r1["AVG_MIN_L10"]), 1),
            "AVG_MIN_L10 2":      round(float(r2["AVG_MIN_L10"]), 1),
            "STD_MIN_L10 1":      round(float(r1["STD_MIN_L10"]), 1),
            "STD_MIN_L10 2":      round(float(r2["STD_MIN_L10"]), 1),
            "AVG_USG_L10 1":      round(float(r1["AVG_USG_L10"]), 3),
            "AVG_USG_L10 2":      round(float(r2["AVG_USG_L10"]), 3),
            "STD_USG_L10 1":      round(float(r1["STD_USG_L10"]), 3),
            "STD_USG_L10 2":      round(float(r2["STD_USG_L10"]), 3),

            # ── Matchup history ───────────────────────────────────────────────
            "AVG_STAT_VS_MATCHUP 1": round(float(r1["AVG_STAT_VS_MATCHUP"]), 1),
            "AVG_STAT_VS_MATCHUP 2": round(float(r2["AVG_STAT_VS_MATCHUP"]), 1),
            "MATCHUP_GAMES 1":       int(r1["MATCHUP_GAMES"]),
            "MATCHUP_GAMES 2":       int(r2["MATCHUP_GAMES"]),
        })

    pair_sorted = sorted(records, key=lambda r: r["PARLAY_PROB"], reverse=True)

    # ── Greedy: each (player, market) used at most once ───────────────────────
    used_props, slate_rows = set(), []
    for row in pair_sorted:
        k1, k2 = row["PROP_KEY_1"], row["PROP_KEY_2"]
        if k1 in used_props or k2 in used_props:
            continue
        used_props.add(k1)
        used_props.add(k2)
        slate_rows.append(row)

    # ── Tag each row with classification BEFORE final sort/truncate ───────────
    for row in slate_rows:
        label, score = classify_bet(row)
        row["BET_CLASS"] = label
        row["BET_SCORE"] = score

    # ── Final sort: STRONG first, then by EV desc, keep top_n ────────────────
    CLASS_ORDER = {"STRONG": 0, "MARGINAL": 1, "SKIP": 2}
    slate_list = sorted(
        slate_rows,
        key=lambda r: (CLASS_ORDER[r["BET_CLASS"]], -r["EV"]),
    )[:top_n]

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = Path(json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(slate_list)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── Summary print ─────────────────────────────────────────────────────────
    class_counts = {"STRONG": 0, "MARGINAL": 0, "SKIP": 0}
    for row in slate_list:
        class_counts[row["BET_CLASS"]] += 1

    print(
        f"Legs: {len(legs)}  |  Pairs: {len(records)}  |  Slate: {len(slate_list)}  |  "
        f"STRONG: {class_counts['STRONG']}  |  "
        f"MARGINAL: {class_counts['MARGINAL']}  |  "
        f"SKIP: {class_counts['SKIP']}  |  "
        f"JSON: {out_path}"
    )
    return str(out_path)

def classify_bet_3leg(row: dict) -> tuple[str, int]:
    """
    Score a 3-leg PrizePicks triple across 6 dimensions.
    All 3 legs must clear thresholds (weakest leg sets the floor).

    Scoring:
        13–15  → STRONG
        8–12   → MARGINAL
        0–7    → SKIP
    """
    score = 0

    # ── 1. EV (0–3 pts) ── 3-leg needs higher EV bar to justify variance ─────
    ev = row["EV"]
    if ev >= 25:
        score += 3
    elif ev >= 15:
        score += 2
    elif ev >= 8:
        score += 1

    # ── 2. Parlay probability (0–2 pts) ──────────────────────────────────────
    pp = row["PARLAY_PROB"]
    if pp >= 0.50:
        score += 2
    elif pp >= 0.42:
        score += 1

    # ── 3. All individual legs strong — weakest leg governs (0–2 pts) ─────────
    probs = [row[f"MODEL_PROB {n}"] for n in (1, 2, 3)]
    min_prob = min(probs)
    if min_prob >= 0.65:
        score += 2
    elif min_prob >= 0.60:
        score += 1

    # ── 4. Model edge over implied prob — all 3 legs (0–2 pts) ───────────────
    edges = [row[f"EDGE {n}"] for n in (1, 2, 3)]
    min_edge = min(edges)
    if min_edge >= 0.07:
        score += 2
    elif min_edge >= 0.04:
        score += 1

    # ── 5. Recent form confirms side — all 3 legs (0–2 pts) ──────────────────
    def form_ok(side: str, over_rate: float) -> bool:
        return over_rate >= 0.60 if side == "OVER" else over_rate <= 0.40

    form_results = [
        form_ok(row[f"SIDE {n}"], row[f"OVER_RATE_L10 {n}"])
        for n in (1, 2, 3)
    ]
    if all(form_results):
        score += 2
    elif sum(form_results) >= 2:
        score += 1

    # ── 6. Consistency — low stat & minutes variance across all legs (0–2 pts) 
    stds  = [row[f"STD_STAT_L10 {n}"] for n in (1, 2, 3)]
    smins = [row[f"STD_MIN_L10 {n}"]  for n in (1, 2, 3)]
    consistent = all(s < 4.0 for s in stds)
    stable_min = all(s < 3.0 for s in smins)
    if consistent and stable_min:
        score += 2
    elif consistent or stable_min:
        score += 1

    # ── 7. Kelly confirmation (0–1 pt) ───────────────────────────────────────
    if row["KELLY"] >= 3.0:   # lower bar: 3-leg kelly is naturally smaller
        score += 1

    # ── 8. Matchup history depth — all 3 legs (0–1 pt) ───────────────────────
    if all(row[f"MATCHUP_GAMES {n}"] >= 3 for n in (1, 2, 3)):
        score += 1

    # ── 9. Median edge sanity check — all 3 legs positive (0–1 pt) ───────────
    if all(row[f"MED_EDGE {n}"] > 0 for n in (1, 2, 3)):
        score += 1

    # ── Classification ────────────────────────────────────────────────────────
    if score >= 13:
        label = "STRONG"
    elif score >= 8:
        label = "MARGINAL"
    else:
        label = "SKIP"

    return label, score


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
    legs["LINE"]     = legs["LINE"].astype(float)
    legs["TEAM"]     = legs["PLAYER_NAME"].map(latest_team)
    legs             = legs.dropna(subset=["TEAM"]).reset_index(drop=True)
    legs["PROP_KEY"] = list(zip(legs["PLAYER_NAME"], legs["MARKET"]))

    b = float(win_profit_units)
    records = []
    for i, j, k in combinations(legs.index, 3):
        r1, r2, r3 = legs.loc[i], legs.loc[j], legs.loc[k]

        if len({r1["PLAYER_NAME"], r2["PLAYER_NAME"], r3["PLAYER_NAME"]}) < 3:
            continue
        if len({r1["TEAM"], r2["TEAM"], r3["TEAM"]}) < 3:
            continue

        side1, p1 = _pick_side_and_prob(r1)
        side2, p2 = _pick_side_and_prob(r2)
        side3, p3 = _pick_side_and_prob(r3)

        parlay_p = p1 * p2 * p3
        ev       = (parlay_p * b) - ((1 - parlay_p) * 1)
        kelly    = max((b * parlay_p - (1 - parlay_p)) / b, 0) * kelly_fraction

        if ev < min_ev or kelly < min_kelly:
            continue

        def leg_fields(r, side, p, n: int) -> dict:
            return {
                f"NAME {n}":               r["PLAYER_NAME"],
                f"TEAM {n}":               r["TEAM"],
                f"MARKET {n}":             r["MARKET"],
                f"PROP_KEY_{n}":           r["PROP_KEY"],
                f"LINE {n}":               r["LINE"],
                f"SIDE {n}":               side,
                f"PREDICTION {n}":         round(float(r["STAT_Q50"]), 1),
                f"MIN PREDICTION {n}":     round(float(r["MIN_Q50"]), 1),
                f"MODEL_PROB {n}":         round(p, 3),
                f"OPPONENT {n}":           r["OPPONENT"],
                f"SPREAD {n}":             r["TEAM_SPREAD"],
                f"TOTAL {n}":              r["GAME_TOTAL"],
                f"OPP_DEF_RATING {n}":     r["OPP_DEF_RATING"],
                f"OPP_DEF_RANK {n}":       r["OPP_RANK_DEF_RATING"],
                f"OPP_PACE {n}":           r["OPP_PACE"],
                f"OPP_PACE_RANK {n}":      r["OPP_PACE_RANK"],
                f"ODDS_OVER {n}":          r["ODDS_OVER"],
                f"ODDS_UNDER {n}":         r["ODDS_UNDER"],
                f"IMP_PROB_OVER {n}":      round(float(r["IMP_PROB_OVER"]), 3),
                f"IMP_PROB_UNDER {n}":     round(float(r["IMP_PROB_UNDER"]), 3),
                f"EDGE {n}":               round(float(r["EDGE"]), 3),
                f"MED_EDGE {n}":           round(float(r["MED_EDGE"]), 3),
                f"Z_SCORE {n}":            round(float(r["Z_SCORE"]), 3),
                f"EV_OVER {n}":            round(float(r["EV_OVER"]), 3),
                f"EV_UNDER {n}":           round(float(r["EV_UNDER"]), 3),
                f"AVG_STAT_L10 {n}":       round(float(r["AVG_STAT_L10"]), 1),
                f"MED_STAT_L10 {n}":       round(float(r["MED_STAT_L10"]), 1),
                f"STD_STAT_L10 {n}":       round(float(r["STD_STAT_L10"]), 1),
                f"OVER_RATE_L5 {n}":       round(float(r["OVER_RATE_L5"]), 3),
                f"OVER_RATE_L10 {n}":      round(float(r["OVER_RATE_L10"]), 3),
                f"OVER_RATE_L15 {n}":      round(float(r["OVER_RATE_L15"]), 3),
                f"OVER_RATE_SEASON {n}":   round(float(r["OVER_RATE_SEASON"]), 3),
                f"AVG_MIN_L10 {n}":        round(float(r["AVG_MIN_L10"]), 1),
                f"STD_MIN_L10 {n}":        round(float(r["STD_MIN_L10"]), 1),
                f"AVG_USG_L10 {n}":        round(float(r["AVG_USG_L10"]), 3),
                f"STD_USG_L10 {n}":        round(float(r["STD_USG_L10"]), 3),
                f"AVG_STAT_VS_MATCHUP {n}": round(float(r["AVG_STAT_VS_MATCHUP"]), 1),
                f"MATCHUP_GAMES {n}":       int(r["MATCHUP_GAMES"]),
            }

        records.append({
            **leg_fields(r1, side1, p1, 1),
            **leg_fields(r2, side2, p2, 2),
            **leg_fields(r3, side3, p3, 3),
            "PARLAY_PROB": round(parlay_p, 3),
            "EV":          round(ev * 100, 2),
            "KELLY":       round(kelly * 100, 2),
        })

    triple_sorted = sorted(records, key=lambda r: r["PARLAY_PROB"], reverse=True)

    # ── Greedy: each (player, market) used at most once ───────────────────────
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

    # ── Tag each row with classification BEFORE final sort/truncate ───────────
    for row in slate_rows:
        label, score = classify_bet_3leg(row)
        row["BET_CLASS"] = label
        row["BET_SCORE"] = score

    # ── Final sort: STRONG first, then by EV desc, keep top_n ────────────────
    CLASS_ORDER = {"STRONG": 0, "MARGINAL": 1, "SKIP": 2}
    slate_list = sorted(
        slate_rows,
        key=lambda r: (CLASS_ORDER[r["BET_CLASS"]], -r["EV"]),
    )[:top_n]

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = Path(json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_json_ready(slate_list), indent=2),
        encoding="utf-8",
    )

    # ── Summary print ─────────────────────────────────────────────────────────
    class_counts = {"STRONG": 0, "MARGINAL": 0, "SKIP": 0}
    for row in slate_list:
        class_counts[row["BET_CLASS"]] += 1

    print(
        f"Legs: {len(legs)}  |  Triples: {len(records)}  |  Slate: {len(slate_list)}  |  "
        f"STRONG: {class_counts['STRONG']}  |  "
        f"MARGINAL: {class_counts['MARGINAL']}  |  "
        f"SKIP: {class_counts['SKIP']}  |  "
        f"JSON: {out_path}"
    )
    return str(out_path)


# Net profit multiplier (profit / stake) for winning the full parlay per platform.
# 2-leg PrizePicks pays 3x total → net = 2.0; 3-leg pays ~5.5x → net = 4.5, etc.
_DFS_PLATFORM_PAYOUTS: dict[str, tuple[float, float]] = {
    "PrizePicks":       (2.0, 4.5),
    "Underdog":         (2.0, 4.5),
    "Betr DFS":         (2.0, 4.5),
    "DraftKings Pick6": (2.0, 4.0),
}


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
        side, p_win = ("UNDER", pu if pu > 0 else 1.0 - po) if lean == "UNDER" else ("OVER", po if po > 0 else 1.0 - pu)
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
        "PARLAY_PROB":    round(parlay_p, 5),
        "EV":             round(ev_pct, 3),
        "EV_DOLLARS":     round(ev_dol, 4),
        "KELLY":          round(kelly, 4),
        "STAKE_DOLLARS":  stake,
        "NET_PAYOUT_MULT": net_mult,
        "N_LEGS":         len(legs_chunk),
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


def _generate_candidates_dfs_2leg(legs, stake, kelly_fraction, hi_total, net_mult):
    records = []
    for idxs in combinations(range(len(legs)), 2):
        chunk = [legs[i] for i in idxs]
        if not _valid_two_leg(chunk):
            continue
        a, b  = chunk
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
      6/7. Enrich DFS picks via ``enrich_dfs_picks`` (sharp context + model probs) → all_line_probs
      8. Build greedy 2-/3-leg slates per bookmaker from dfs_sharp_aligned picks
      9. Append predictions & slates to the ledger (`log.snapshot`)
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

    print(f"Lines for date: {current_date}")

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
        name: get_game_context(base_df, name, team_odds, is_playoff=True)
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

    # ── 6/7. Enrich DFS picks with sharp context & model probs ──────────────
    print("\n── Enriching DFS picks ──")
    from src.utils.generalized_best_bets_v2 import enrich_dfs_picks as _enrich_dfs_picks

    _circa_files = list(Path("data/props/circa+betonline_team_lines").glob("circa+betonline_*.json"))
    _team_odds_src = str(max(_circa_files, key=lambda f: f.stat().st_mtime)) if _circa_files else None

    enriched_path, aligned_path, all_line_probs = _enrich_dfs_picks(
        dfs_df=lines_dfs,
        us_df=lines_us,
        base_df=base_df,
        all_line_probs=raw_probs,
        team_odds_source=_team_odds_src,
        dfs_platforms=None,
        current_date=current_date,
        verbose=verbose,
    )

    if all_line_probs.empty:
        print("\nNo enriched lines — skipping snapshot")
        return

    # ── 8. Build slates from dfs_sharp_aligned picks ─────────────────────────
    print("\n── Building slates ──")
    for bk in bookmakers:
        build_dfs_slates_from_aligned(
            aligned_path,
            platform=bk,
            out_2leg=BOOKMAKER_SLATE_PATHS.get(bk),
            out_3leg=BOOKMAKER_3LEG_PATHS.get(bk),
            stake_dollars=10.0,
            top_n=top_n,
            kelly_fraction=kelly_fraction,
            verbose=verbose,
        )

    # ── 9. Log to ledger ─────────────────────────────────────────────────────
    print("\n── Logging predictions & slates ──")
    from src.utils import log as _log

    # Rename key columns so log.snapshot can find its dedup/side fields,
    # then expand the `model` nested dict to flat columns alongside all
    # other enriched context (sharp, consensus, game_context, etc. stay as-is).
    _log_df = all_line_probs.rename(columns={
        "player":   "PLAYER_NAME",
        "market":   "MARKET",
        "dfs_line": "LINE",
        "platform": "LINE_BOOKMAKER",
    }).copy()
    if "model" in _log_df.columns:
        _model_flat = _log_df.pop("model").apply(pd.Series).rename(columns={
            "p_over":   "P_OVER",
            "p_under":  "P_UNDER",
            "min_q10":  "MIN_Q10",
            "min_q50":  "MIN_Q50",
            "min_q90":  "MIN_Q90",
            "stat_q10": "STAT_Q10",
            "stat_q50": "STAT_Q50",
            "stat_q90": "STAT_Q90",
        })
        _log_df = pd.concat([_log_df.reset_index(drop=True), _model_flat.reset_index(drop=True)], axis=1)

    slate_paths = {
        bk: {
            "2leg": BOOKMAKER_SLATE_PATHS.get(bk, ""),
            "3leg": BOOKMAKER_3LEG_PATHS.get(bk, ""),
        }
        for bk in bookmakers
    }
    _log.snapshot(_log_df, slate_paths, date=current_date)

    print("\n── Done ──")


if __name__ == "__main__":
    main()