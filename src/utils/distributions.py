"""
distributions.py — simulation primitives for NBA prop modelling.

Current (production):
  triangular_clip            – standalone minutes sampler (n_sims path)
  triangular_clip_with_u     – inverse-CDF triangular (shared-U copula path)
  run_pts_simulation         – continuous points (min × ppm)
  run_count_simulation       – Poisson counts (AST/REB)

Experimental (swap-in replacements, same call signature):
  sample_gamma_rate          – Gamma anchor for PPM/rate (replaces triangular anchor)
  run_pts_simulation_gamma   – points using Gamma rate anchor
  run_count_simulation_nbinom – Negative-Binomial counts (overdispersion-aware)
  run_count_simulation_zip   – Zero-Inflated Poisson (BLK/STL excess zeros)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _decay_weights(n: int, decay: float) -> np.ndarray:
    w = np.array([decay ** i for i in range(n - 1, -1, -1)])
    return w / w.sum()


def _empirical_rates(history: np.ndarray, n: int, decay: float, jitter_lo: float, jitter_hi: float) -> np.ndarray:
    weights = _decay_weights(len(history), decay)
    samples = np.random.choice(history, size=n, p=weights)
    jitter_std = np.clip(np.std(history) * 0.3, jitter_lo, jitter_hi)
    return samples + np.random.normal(0, jitter_std, size=n)


def _opp_rate_multiplier(row: dict, *, use_pace: bool = False, clip_lo: float = 0.88, clip_hi: float = 1.12) -> float:
    """
    Opponent-adjustment multiplier applied to the rate component of simulations.

    DEF_RATING: higher = worse defense → multiplier > 1 (boost player rate).
    PACE: higher = more possessions → multiplier > 1 (use for count stats only).
    Returns 1.0 when any required key is missing or non-finite (backward-compatible).
    Clipped to [clip_lo, clip_hi] to prevent extreme swings from small samples.
    """
    mult = 1.0
    opp_def = row.get("OPP_DEF_RATING")
    avg_def = row.get("LEAGUE_AVG_DEF_RATING")
    if opp_def is not None and avg_def is not None and np.isfinite(opp_def) and np.isfinite(avg_def) and avg_def > 0:
        mult *= opp_def / avg_def
    if use_pace:
        opp_pace = row.get("OPP_PACE")
        avg_pace = row.get("LEAGUE_AVG_PACE")
        if opp_pace is not None and avg_pace is not None and np.isfinite(opp_pace) and np.isfinite(avg_pace) and avg_pace > 0:
            mult *= opp_pace / avg_pace
    return float(np.clip(mult, clip_lo, clip_hi))


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION: TRIANGULAR SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────

def triangular_clip(
    row,
    q10,
    q50,
    q90,
    lo_scale=0.80,
    hi_scale=1.20,
    low=0.0,
    high=np.inf,
    n_sims=10_000,
):
    """Triangular sampler — standalone n_sims path (used for minutes)."""
    q10_val = float(row[q10])
    q50_val = float(row[q50])
    q90_val = float(row[q90])

    if not (np.isfinite(q10_val) and np.isfinite(q50_val) and np.isfinite(q90_val)):
        base = q50_val if np.isfinite(q50_val) else 0.0
        return np.full(n_sims, base)

    left  = q10_val * lo_scale
    mode  = q50_val
    right = q90_val * hi_scale

    left  = min(left, mode)
    right = max(right, mode)

    # downside skew: foul trouble / blowouts
    mode = min(max(mode * 0.97, left), right)

    return np.clip(np.random.triangular(left, mode, right, size=n_sims), low, high)


def triangular_clip_with_u(row, q10, q50, q90, U, lo_scale, hi_scale, low, high, mode_skew=1.03):
    """Inverse-CDF triangular for shared-U copula paths."""
    q10_val, q50_val, q90_val = float(row[q10]), float(row[q50]), float(row[q90])

    left  = q10_val * lo_scale
    mode  = q50_val * mode_skew
    right = q90_val * hi_scale

    left, right = min(left, mode), max(right, mode)

    if np.isclose(left, right):
        return np.full_like(U, np.clip(mode, low, high))

    prob_mode = (mode - left) / (right - left)

    samples = np.where(
        U < prob_mode,
        left  + np.sqrt(U       * (mode  - left)  * (right - left)),
        right - np.sqrt((1 - U) * (right - mode) * (right - left)),
    )
    return np.clip(samples, low, high)


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION: SIMULATION RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_pts_simulation(row, n_sims=50_000, anchor_weight=0.3, decay=0.85, min_history=10):
    n_anchor    = int(n_sims * anchor_weight)
    n_empirical = n_sims - n_anchor

    U        = np.random.uniform(0, 1, n_sims)
    U_anchor = U[:n_anchor]

    sim_min_anchor = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U_anchor, 0.75, 1.25, 0, 48)
    sim_ppm_anchor = triangular_clip_with_u(row, "RATE_Q10", "RATE_Q50", "RATE_Q90", U_anchor, 0.85, 1.15, 0, None)

    h = row.get("RATE_HISTORY")
    m = row.get("MIN_HISTORY")

    use_paired = (
        h is not None and m is not None
        and len(h) >= min_history and len(m) == len(h)
    )
    use_rate_only = not use_paired and h is not None and len(h) >= min_history

    if use_paired:
        h = np.array(h, dtype=float)
        m = np.array(m, dtype=float)

        # Decay weights over game index (most recent = highest weight)
        weights = _decay_weights(len(h), decay)

        # Resample GAME INDICES — minutes and rate come from the same game
        idx = np.random.choice(len(h), size=n_empirical, p=weights)
        emp_min   = np.clip(m[idx] + np.random.normal(0, np.std(m) * 0.15, n_empirical), 0, 48)
        emp_rates = np.clip(h[idx] + np.random.normal(0, np.std(h) * 0.15, n_empirical), 0, None)

        sim_min = np.concatenate([sim_min_anchor, emp_min])
        sim_ppm = np.concatenate([sim_ppm_anchor, emp_rates])

    elif use_rate_only:
        # Fallback: no MIN_HISTORY, original triangular minutes path
        h         = np.array(h, dtype=float)
        emp_rates = _empirical_rates(h, n_empirical, decay, 0.05, 1.0)
        sim_min_emp = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U[n_anchor:], 0.75, 1.25, 0, 48)

        sim_min = np.concatenate([sim_min_anchor, sim_min_emp])
        sim_ppm = np.concatenate([sim_ppm_anchor, emp_rates])

    else:
        sim_min = sim_min_anchor
        sim_ppm = sim_ppm_anchor

    return sim_min * sim_ppm * _opp_rate_multiplier(row)


def run_count_simulation(row, n_sims=50_000, anchor_weight=0.3, decay=0.85, min_history=10):
    """
    Poisson count simulation (AST / REB).
    λ per path = sim_min × sim_rate.  Final draw: Poisson(λ).

    Empirical path (70%): paired MIN_HISTORY + RATE_HISTORY resampling by game
    index preserves natural minutes/rate correlation.  Falls back to rate-only
    triangular minutes if MIN_HISTORY is absent.
    """
    n_anchor    = int(n_sims * anchor_weight)
    n_empirical = n_sims - n_anchor

    U        = np.random.uniform(0, 1, n_sims)
    U_anchor = U[:n_anchor]

    sim_min_anchor  = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90",  U_anchor, 0.75, 1.25, 0, 48)
    sim_rate_anchor = triangular_clip_with_u(row, "RATE_Q10", "RATE_Q50", "RATE_Q90", U_anchor, 0.85, 1.15, 0, None)

    h = row.get("RATE_HISTORY")
    m = row.get("MIN_HISTORY")

    use_paired    = h is not None and m is not None and len(h) >= min_history and len(m) == len(h)
    use_rate_only = not use_paired and h is not None and len(h) >= min_history

    if use_paired:
        h = np.array(h, dtype=float)
        m = np.array(m, dtype=float)

        weights = _decay_weights(len(h), decay)
        idx     = np.random.choice(len(h), size=n_empirical, p=weights)

        emp_min  = np.clip(m[idx] + np.random.normal(0, np.std(m) * 0.15, n_empirical), 0, 48)
        emp_rate = np.clip(h[idx] + np.random.normal(0, np.std(h) * 0.15, n_empirical), 0, None)

        sim_min  = np.concatenate([sim_min_anchor,  emp_min])
        sim_rate = np.concatenate([sim_rate_anchor, emp_rate])

    elif use_rate_only:
        h           = np.array(h, dtype=float)
        emp_rate    = _empirical_rates(h, n_empirical, decay, 0.02, 0.5)
        sim_min_emp = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U[n_anchor:], 0.75, 1.25, 0, 48)

        sim_min  = np.concatenate([sim_min_anchor,  sim_min_emp])
        sim_rate = np.concatenate([sim_rate_anchor, emp_rate])

    else:
        sim_min  = sim_min_anchor
        sim_rate = sim_rate_anchor

    lam = np.clip(sim_min * sim_rate * _opp_rate_multiplier(row, use_pace=True), 1e-6, None)
    return np.random.poisson(lam).astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL: GAMMA RATE ANCHOR
# ─────────────────────────────────────────────────────────────────────────────

def _fit_gamma_from_quantiles(q10_val: float, q50_val: float, q90_val: float):
    """
    Fit Gamma(k, θ) from quantiles using moment matching.
    IQR-range → σ estimate; median → μ estimate.
    Returns (k, theta) or None if degenerate.
    """
    mu    = q50_val
    sigma = (q90_val - q10_val) / 2.56  # ≈ 2 std deviations span

    if mu <= 0 or sigma <= 0:
        return None

    k     = (mu / sigma) ** 2
    theta = sigma ** 2 / mu
    return k, theta


def sample_gamma_rate(row, q10_key, q50_key, q90_key, U, lo_scale=0.85, hi_scale=1.15):
    """
    Gamma anchor sampler — drop-in for triangular_clip_with_u on the rate axis.
    Uses inverse-CDF (scipy.stats.gamma.ppf) so it accepts a shared U vector
    and preserves copula coupling with the minutes draw.

    Falls back to triangular if Gamma fit is degenerate.
    """
    from scipy.stats import gamma as gamma_dist

    q10 = float(row[q10_key]) * lo_scale
    q50 = float(row[q50_key])
    q90 = float(row[q90_key]) * hi_scale

    fit = _fit_gamma_from_quantiles(q10, q50, q90)
    if fit is None:
        return triangular_clip_with_u(row, q10_key, q50_key, q90_key, U, lo_scale, hi_scale, 0, None)

    k, theta = fit
    # Clip U away from exact 0/1 to avoid ppf overflow
    U_safe = np.clip(U, 1e-6, 1 - 1e-6)
    return gamma_dist.ppf(U_safe, a=k, scale=theta)


def run_pts_simulation_gamma(row, n_sims=10_000, anchor_weight=0.3, decay=0.85, min_history=5):
    """
    Points simulation — Gamma anchor for the rate component.
    Same empirical slice as run_pts_simulation; only the anchor rate sampler changes.
    """
    n_anchor    = int(n_sims * anchor_weight)
    n_empirical = n_sims - n_anchor

    U = np.random.uniform(0, 1, n_sims)
    np.random.shuffle(U)

    U_anchor       = U[:n_anchor]
    sim_min_anchor = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U_anchor, 0.75, 1.25, 0, 48)
    # ← Gamma instead of triangular for the rate
    sim_ppm_anchor = sample_gamma_rate(row, "RATE_Q10", "RATE_Q50", "RATE_Q90", U_anchor)
    sim_ppm_anchor = np.clip(sim_ppm_anchor, 0, None)

    h = row.get("RATE_HISTORY")
    if h is not None and len(h) >= min_history:
        h = np.array(h, dtype=float)
        emp_rates   = _empirical_rates(h, n_empirical, decay, 0.05, 1.0)
        sim_min_emp = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U[n_anchor:], 0.75, 1.25, 0, 48)
        sim_min     = np.concatenate([sim_min_emp, sim_min_anchor])
        sim_ppm     = np.concatenate([np.clip(emp_rates, 0, None), sim_ppm_anchor])
    else:
        sim_min = sim_min_anchor
        sim_ppm = sim_ppm_anchor

    return sim_min * sim_ppm


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL: NEGATIVE BINOMIAL (overdispersed counts — AST / REB / 3PM)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_nbinom(lam: np.ndarray):
    """
    Fit Negative-Binomial dispersion parameter r from a λ array.
    r is estimated per-path via moment matching on cross-path variance.
    Returns scalar r.  Falls back to r=5 if variance ≤ mean (not overdispersed).
    """
    mu  = np.mean(lam)
    var = np.var(lam)
    if var > mu and mu > 0:
        r = max(mu ** 2 / (var - mu), 0.5)
    else:
        r = 5.0  # mildly overdispersed default
    return r


def run_count_simulation_nbinom(row, n_sims=10_000, anchor_weight=0.3, decay=0.85, min_history=5):
    """
    Negative-Binomial count simulation — overdispersion-aware replacement for
    run_count_simulation.  Use for AST, REB, 3PM.

    Same min × rate path as Poisson version; final draw is NegBinom(r, p)
    where r is fit from the cross-path λ distribution.
    """
    n_anchor    = int(n_sims * anchor_weight)
    n_empirical = n_sims - n_anchor

    U = np.random.uniform(0, 1, n_sims)
    np.random.shuffle(U)

    U_anchor         = U[:n_anchor]
    sim_min_anchor   = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U_anchor, 0.75, 1.25, 0, 48)
    sim_rate_anchor  = triangular_clip_with_u(row, "RATE_Q10", "RATE_Q50", "RATE_Q90", U_anchor, 0.85, 1.15, 0, None)

    h = row.get("RATE_HISTORY")
    if h is not None and len(h) >= min_history:
        h = np.array(h, dtype=float)
        emp_rates   = _empirical_rates(h, n_empirical, decay, 0.02, 0.5)
        sim_min_emp = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U[n_anchor:], 0.75, 1.25, 0, 48)
        sim_min     = np.concatenate([sim_min_emp, sim_min_anchor])
        sim_rate    = np.concatenate([np.clip(emp_rates, 0, None), sim_rate_anchor])
    else:
        sim_min  = sim_min_anchor
        sim_rate = sim_rate_anchor

    lam = np.clip(sim_min * sim_rate, 1e-6, None)

    r = _fit_nbinom(lam)
    p = r / (r + lam)  # per-path success probability
    # NegBinom parameterised as: number of failures before r successes
    return np.random.negative_binomial(r, np.clip(p, 1e-6, 1 - 1e-6)).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL: ZERO-INFLATED POISSON (excess zeros — BLK / STL)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_pi_zero(history: np.ndarray) -> float:
    """
    Estimate the excess zero probability beyond what Poisson predicts.
    pi_excess = max(0, observed_zero_rate − Poisson_zero_rate)
    """
    if len(history) == 0:
        return 0.0
    obs_zero   = np.mean(history == 0)
    mu         = np.mean(history)
    pois_zero  = np.exp(-mu) if mu > 0 else 1.0
    return float(np.clip(obs_zero - pois_zero, 0, 0.8))


def run_count_simulation_zip(row, n_sims=10_000, anchor_weight=0.3, decay=0.85, min_history=5):
    """
    Zero-Inflated Poisson count simulation.  Use for BLK and STL where
    the observed zero rate exceeds what Poisson(λ) predicts.

    pi_zero is estimated from RATE_HISTORY (converted to integer counts via
    history × median_minutes).  Falls back to plain Poisson when pi_zero ≈ 0.
    """
    n_anchor    = int(n_sims * anchor_weight)
    n_empirical = n_sims - n_anchor

    U = np.random.uniform(0, 1, n_sims)
    np.random.shuffle(U)

    U_anchor         = U[:n_anchor]
    sim_min_anchor   = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U_anchor, 0.75, 1.25, 0, 48)
    sim_rate_anchor  = triangular_clip_with_u(row, "RATE_Q10", "RATE_Q50", "RATE_Q90", U_anchor, 0.85, 1.15, 0, None)

    pi_zero = 0.0
    h = row.get("RATE_HISTORY")
    if h is not None and len(h) >= min_history:
        h = np.array(h, dtype=float)
        # Approximate game counts from per-minute rates × median minutes
        med_min   = float(row.get("MIN_Q50", 30))
        approx_counts = np.round(h * med_min).astype(int)
        pi_zero   = _estimate_pi_zero(approx_counts.astype(float))

        emp_rates   = _empirical_rates(h, n_empirical, decay, 0.02, 0.5)
        sim_min_emp = triangular_clip_with_u(row, "MIN_Q10", "MIN_Q50", "MIN_Q90", U[n_anchor:], 0.75, 1.25, 0, 48)
        sim_min     = np.concatenate([sim_min_emp, sim_min_anchor])
        sim_rate    = np.concatenate([np.clip(emp_rates, 0, None), sim_rate_anchor])
    else:
        sim_min  = sim_min_anchor
        sim_rate = sim_rate_anchor

    lam    = np.clip(sim_min * sim_rate, 1e-6, None)
    counts = np.random.poisson(lam).astype(float)

    if pi_zero > 0.01:
        zero_mask         = np.random.uniform(size=n_sims) < pi_zero
        counts[zero_mask] = 0.0

    return counts
