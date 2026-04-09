import json
from pathlib import Path

import pandas as pd
import numpy as np
from itertools import combinations
from src.pipeline.props_pipeline.min_pipeline import min_pipeline

def predict_min_times_rate(
    names,
    min_stats_df,
    prop_stats_df,
    current_date,
    *,
    name_dict,
    rate_pipeline,
    rate_quantile_models,
    min_quantile_models,
    stat_prefix,  # "PTS", "AST", or "REB"
    verbose=True,
):
    """
    For each name: min quantiles × rate quantiles → implied stat (same quantile index).
    """
    q10, q50, q90 = "q_0.10", "q_0.50", "q_0.90"
    records = []

    for raw_name in names:
        name = name_dict.get(raw_name, raw_name) if raw_name in name_dict else raw_name
        if raw_name in name_dict:
            name = name_dict[raw_name]

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
            rate_col = f"{stat_prefix}_PER_MIN"  # "PTS_PER_MIN", "AST_PER_MIN", "REB_PER_MIN"
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

def run_stat_simulation(
    row,
    n_sims: int = 10_000,
    anchor_weight: float = 0.3,
    decay: float = 0.85,
    min_history: int = 5,
) -> np.ndarray:
    if not 0.0 <= anchor_weight <= 1.0:
        raise ValueError(f"anchor_weight must be in [0, 1], got {anchor_weight}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")

    # --- Minutes simulation ---
    sim_min = triangular_clip(
        row,
        "MIN_Q10", "MIN_Q50", "MIN_Q90",
        lo_scale=0.75, hi_scale=1.25,
        low=0, high=48,
        n_sims=n_sims,
    )

    # --- Rate simulation ---
    rate_history = row.get("RATE_HISTORY") if isinstance(row, (dict, pd.Series)) else None

    if rate_history is not None and len(rate_history) >= min_history:
        h = np.array(rate_history, dtype=float)

        # Recency weights: most recent game gets highest weight
        weights = np.array([decay ** i for i in range(len(h) - 1, -1, -1)])
        weights /= weights.sum()

        n_anchor = int(n_sims * anchor_weight)
        n_empirical = n_sims - n_anchor

        # Empirical draws with mild jitter to smooth inter-game variance
        empirical_samples = np.random.choice(h, size=n_empirical, p=weights)
        jitter_std = np.std(h) * 0.05
        empirical_samples = np.clip(
            empirical_samples + np.random.normal(0, jitter_std, size=n_empirical),
            0, None,
        )

        # Anchor draws: triangular distribution centred on Q50 rather than a point mass
        anchor_samples = triangular_clip(
            row,
            "RATE_Q10", "RATE_Q50", "RATE_Q90",
            lo_scale=0.85, hi_scale=1.15,
            low=0, high=None,
            n_sims=n_anchor,
        ) if n_anchor > 0 else np.empty(0)

        sim_rate = np.concatenate([empirical_samples, anchor_samples])

    else:
        sim_rate = triangular_clip(
            row,
            "RATE_Q10", "RATE_Q50", "RATE_Q90",
            lo_scale=0.70, hi_scale=1.30,
            low=0, high=None,
            n_sims=n_sims,
        )
    # --- Poisson draw ---
    # Clip lambda to (0, inf) — np.random.poisson raises on negative lam
    expected_val = np.clip(sim_min * sim_rate, 0, None)
    return np.random.poisson(lam=expected_val)

def run_pts_simulation(
    row,
    n_sims: int = 10_000,
    anchor_weight: float = 0.3,
    decay: float = 0.85,
    min_history: int = 5,
) -> np.ndarray:
    if not 0.0 <= anchor_weight <= 1.0:
        raise ValueError(f"anchor_weight must be in [0, 1], got {anchor_weight}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")

    # --- Minutes simulation (unchanged) ---
    sim_min = triangular_clip(
        row,
        "MIN_Q10", "MIN_Q50", "MIN_Q90",
        lo_scale=0.75, hi_scale=1.25,
        low=0, high=48,
        n_sims=n_sims,
    )

    # --- PPM simulation ---
    rate_history = row.get("RATE_HISTORY") if isinstance(row, (dict, pd.Series)) else None

    if rate_history is not None and len(rate_history) >= min_history:
        h = np.array(rate_history, dtype=float)

        # Recency weights: most recent game gets highest weight
        weights = np.array([decay ** i for i in range(len(h) - 1, -1, -1)])
        weights /= weights.sum()

        n_anchor = int(n_sims * anchor_weight)
        n_empirical = n_sims - n_anchor

        # Empirical draws with mild jitter to smooth inter-game variance
        empirical_samples = np.random.choice(h, size=n_empirical, p=weights)
        jitter_std = np.std(h) * 0.05
        empirical_samples = np.clip(empirical_samples + np.random.normal(0, jitter_std, size=n_empirical), 0, None)

        # Anchor draws: triangular distribution centred on Q50 rather than a point mass
        anchor_samples = triangular_clip(
            row,
            "RATE_Q10", "RATE_Q50", "RATE_Q90",
            lo_scale=0.85, hi_scale=1.15,
            low=0, high=None,
            n_sims=n_anchor,
        ) if n_anchor > 0 else np.empty(0)

        sim_ppm = np.concatenate([empirical_samples, anchor_samples])

    else:
        sim_ppm = triangular_clip(
            row,
            "RATE_Q10", "RATE_Q50", "RATE_Q90",
            lo_scale=0.70, hi_scale=1.30,
            low=0, high=None,
            n_sims=n_sims,
        )

    # Paired multiply — no shuffle needed; both arrays are already independently random
    return sim_min * sim_ppm

# ---------------------------------------------------------
# 2. LINE LOOKUP & MAPPING
# ---------------------------------------------------------

def _line_lookup_from_lines_df(ldf: pd.DataFrame, name_dict: dict) -> dict:
    """
    Maps sportsbook names to canonical dataset names.
    Handles 'Jokic' vs 'Jokić' logic.
    """
    d = {}
    for _, r in ldf.iterrows():
        book_name = str(r["NAME"]).strip()
        line = r["LINE"]
        d[book_name] = line
        
        # Add the canonical version if it exists in your nameDict
        canon = name_dict.get(book_name)
        if canon:
            d[canon] = line
            
    # Reverse check: ensure variants point to the line if canon was found
    for variant, canon in name_dict.items():
        if canon in d and variant not in d:
            d[variant] = d[canon]
    return d

# ---------------------------------------------------------
# 3. EXECUTION LOOP
# ---------------------------------------------------------

def line_probs_for_market(preds_df, lines_df, name_dict, sim_fn, n_sims=10_000):
    line_by_name = _line_lookup_from_lines_df(lines_df, name_dict)
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
            "MIN_Q50": round(row["MIN_Q50"], 2),
            "STAT_Q50": round(row["STAT_Q50"], 2),
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