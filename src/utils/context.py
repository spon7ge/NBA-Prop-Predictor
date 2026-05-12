import numpy as np
import pandas as pd

from src.utils.team_info import team3StarsPerTeam

outPlayers: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# QUANTILE COERCION
# ─────────────────────────────────────────────────────────────────────────────

def coerce_nonneg_monotone_quantiles(q10: float, q50: float, q90: float) -> tuple[float, float, float]:
    """Force q10 ≤ q50 ≤ q90 and clip to [0, ∞) — required before triangular sampling."""
    a = max(float(q10), 0.0)
    b = max(float(q50), 0.0)
    c = max(float(q90), 0.0)
    b = max(b, a)
    c = max(c, b)
    return a, b, c


def _sanitize_prediction_row_quantiles(row) -> None:
    """Mutate a prediction row in place — enforces valid MIN/RATE/STAT quantiles."""
    m10, m50, m90 = coerce_nonneg_monotone_quantiles(row["MIN_Q10"],  row["MIN_Q50"],  row["MIN_Q90"])
    r10, r50, r90 = coerce_nonneg_monotone_quantiles(row["RATE_Q10"], row["RATE_Q50"], row["RATE_Q90"])
    row["MIN_Q10"]  = round(m10, 2);  row["MIN_Q50"]  = round(m50, 2);  row["MIN_Q90"]  = round(m90, 2)
    row["RATE_Q10"] = round(r10, 4);  row["RATE_Q50"] = round(r50, 4);  row["RATE_Q90"] = round(r90, 4)
    s10, s50, s90 = coerce_nonneg_monotone_quantiles(m10 * r10, m50 * r50, m90 * r90)
    row["STAT_Q10"] = round(s10, 2);  row["STAT_Q50"] = round(s50, 2);  row["STAT_Q90"] = round(s90, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

def player_scenarios(
    df: pd.DataFrame,
    player_name: str,
    stat_name: str,
    min_minutes: int = 15,
) -> dict:
    """
    Historical splits across 4 context signals:
      1. active_stars  — teammate injury effects      (ACTIVE_STARS_COUNT)
      2. opp_pace      — game-speed via GAME_TOTAL percentile buckets
      3. spread        — game-script risk              (TEAM_SPREAD)
      4. opp_def       — matchup difficulty            (OPP_DEF_RATING percentile buckets)

    Each split is Bayesian-shrunk toward the player's overall median:
        shrunk = (n * split_median + k * overall_median) / (n + k)
        k = max(5, total_n // 10)
    """
    pdf = (
        df[df["PLAYER_NAME"] == player_name]
        .copy()
        .sort_values("GAME_DATE")
    )
    if "MIN" in pdf.columns:
        pdf = pdf[pdf["MIN"] >= min_minutes]

    total_n        = len(pdf)
    overall_median = pdf[stat_name].median() if total_n > 0 else None
    _empty         = {"median": None, "shrunk_median": None, "delta": 0.0, "hit_rate_vs_overall": None, "n": 0}

    if total_n == 0 or pd.isna(overall_median):
        return {
            "player": player_name, "stat": stat_name,
            "overall_median": None, "overall_iqr": None, "total_games": 0,
            "active_stars": {}, "opp_pace": {}, "spread": {}, "opp_def": {},
        }

    K = max(5, total_n // 10)

    def split_stats(subset) -> dict:
        n = len(subset)
        if n == 0:
            return dict(_empty)
        med    = subset[stat_name].median()
        shrunk = (n * med + K * overall_median) / (n + K)
        return {
            "median":              round(med, 4),
            "shrunk_median":       round(shrunk, 4),
            "delta":               round(shrunk - overall_median, 4),
            "hit_rate_vs_overall": round((subset[stat_name] >= overall_median).mean(), 4),
            "n":                   n,
        }

    # ── 1. Active stars ──────────────────────────────────────────────────────
    star_counts  = sorted(pdf["ACTIVE_STARS_COUNT"].dropna().unique().astype(int))
    active_stars = {i: split_stats(pdf[pdf["ACTIVE_STARS_COUNT"] == i]) for i in star_counts}

    # ── 2. Opponent pace (game total) ────────────────────────────────────────
    p33, p67 = pdf["GAME_TOTAL"].quantile(0.33), pdf["GAME_TOTAL"].quantile(0.67)
    opp_pace = {
        "high_pace":   split_stats(pdf[pdf["GAME_TOTAL"] >  p67]),
        "middle_pace": split_stats(pdf[(pdf["GAME_TOTAL"] >= p33) & (pdf["GAME_TOTAL"] <= p67)]),
        "low_pace":    split_stats(pdf[pdf["GAME_TOTAL"] <  p33]),
        "_thresholds": {"p33": round(p33, 2), "p67": round(p67, 2)},
    }

    # ── 3. Spread ────────────────────────────────────────────────────────────
    spread = {
        "favorite": split_stats(pdf[pdf["TEAM_SPREAD"] <  0]),
        "pick_em":  split_stats(pdf[pdf["TEAM_SPREAD"] == 0]),
        "underdog": split_stats(pdf[pdf["TEAM_SPREAD"] >  0]),
    }

    # ── 4. Opponent defensive rating ─────────────────────────────────────────
    opp_def: dict = {}
    if "OPP_DEF_RATING" in pdf.columns:
        d33, d67 = pdf["OPP_DEF_RATING"].quantile(0.33), pdf["OPP_DEF_RATING"].quantile(0.67)
        opp_def = {
            "strong_def":  split_stats(pdf[pdf["OPP_DEF_RATING"] <  d33]),  # lower = better defense
            "mid_def":     split_stats(pdf[(pdf["OPP_DEF_RATING"] >= d33) & (pdf["OPP_DEF_RATING"] <= d67)]),
            "weak_def":    split_stats(pdf[pdf["OPP_DEF_RATING"] >  d67]),
            "_thresholds": {"p33": round(d33, 2), "p67": round(d67, 2)},
        }

    overall_iqr = pdf[stat_name].quantile(0.75) - pdf[stat_name].quantile(0.25)
    return {
        "player":         player_name,
        "stat":           stat_name,
        "overall_median": round(overall_median, 4),
        "overall_iqr":    round(overall_iqr, 4) if pd.notna(overall_iqr) else None,
        "total_games":    total_n,
        "active_stars":   active_stars,
        "opp_pace":       opp_pace,
        "spread":         spread,
        "opp_def":        opp_def,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def adjust_predictions(
    preds_df: pd.DataFrame,
    base_df: pd.DataFrame,
    game_contexts: dict,
    *,
    min_adjust_weight: float = 1.0,
    rate_adjust_weight: float = 1.0,
    delta_cap_min: float = 5.0,
    delta_cap_rate: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Shift MIN_Q50 and RATE_Q50 using player_scenarios deltas across 4 signals:
    active stars, opponent pace, spread, opponent defensive rating.
    Q10/Q90 shift by the same absolute amount to preserve IQR width.
    """
    rate_col_by_market = {"PTS": "PTS_PER_MIN", "AST": "AST_PER_MIN", "REB": "REB_PER_MIN"}

    unique_names     = preds_df["PLAYER_NAME"].unique()
    unique_rate_cols = {rate_col_by_market.get(m, f"{m}_PER_MIN") for m in preds_df["MARKET"].unique()}

    scenario_cache: dict[tuple, dict] = {}
    for name in unique_names:
        for col in ["MIN", *unique_rate_cols]:
            scenario_cache[(name, col)] = player_scenarios(base_df, name, col)

    # ── key helpers ───────────────────────────────────────────────────────────
    def _delta(sc, *key_path) -> tuple[float, int]:
        node = sc
        for k in key_path:
            if not isinstance(node, dict):
                return 0.0, 0
            node = node.get(k)
        if not isinstance(node, dict):
            return 0.0, 0
        d = node.get("delta", 0.0)
        return (float(d) if d is not None else 0.0), int(node.get("n", 0))

    def _pace_key(sc, total) -> str:
        t = sc.get("opp_pace", {}).get("_thresholds", {})
        if total is None or not t:
            return "middle_pace"
        if total > t.get("p67", float("inf")):
            return "high_pace"
        if total < t.get("p33", float("-inf")):
            return "low_pace"
        return "middle_pace"

    def _spread_key(spread) -> str | None:
        if spread is None: return None
        if spread < 0:     return "favorite"
        if spread > 0:     return "underdog"
        return "pick_em"

    def _def_key(sc, opp_def_rating) -> str | None:
        if opp_def_rating is None:
            return None
        t = sc.get("opp_def", {}).get("_thresholds", {})
        if not t:
            return None
        if opp_def_rating < t.get("p33", float("inf")):
            return "strong_def"
        if opp_def_rating > t.get("p67", float("-inf")):
            return "weak_def"
        return "mid_def"

    def _build_delta(sc, n_stars, s_key, pace_key, def_key) -> tuple[float, dict]:
        star_key         = min(n_stars, max(sc.get("active_stars", {}).keys(), default=3))
        d_stars, n_st    = _delta(sc, "active_stars", star_key)
        d_pace,  _       = _delta(sc, "opp_pace", pace_key)
        d_spread, _      = _delta(sc, "spread", s_key) if s_key and s_key != "pick_em" else (0.0, 0)
        d_def,   _       = _delta(sc, "opp_def", def_key) if def_key else (0.0, 0)
        total = d_stars + d_pace + d_spread + d_def
        log   = {
            "stars":   (round(d_stars,  4), n_st),
            "pace":    (round(d_pace,   4), pace_key),
            "spread":  (round(d_spread, 4), s_key),
            "opp_def": (round(d_def,    4), def_key),
        }
        return total, log

    def _set_adj_missing(row, err=None):
        for col in ["ADJ_CONTEXT_OK", "ADJ_CONTEXT_ERR", "ADJ_ACTIVE_STARS",
                    "ADJ_CTX_SPREAD", "ADJ_CTX_TOTAL", "ADJ_CTX_OPP_DEF",
                    "ADJ_MIN_DELTA", "ADJ_RATE_DELTA", "ADJ_MIN_LOG", "ADJ_RATE_LOG"]:
            row[col] = None
        row["ADJ_CONTEXT_OK"]  = False
        row["ADJ_CONTEXT_ERR"] = err

    # ── main loop ─────────────────────────────────────────────────────────────
    out_rows = []
    for _, row in preds_df.iterrows():
        row    = row.copy()
        name   = row["PLAYER_NAME"]
        market = row["MARKET"]
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

        n_stars        = int(ctx.get("active_stars", 1))
        spread         = ctx.get("spread")
        total          = ctx.get("total")
        opp_def_rating = ctx.get("opp_def_rating")

        min_sc  = scenario_cache[(name, "MIN")]
        rate_sc = scenario_cache[(name, rate_col)]

        s_key     = _spread_key(spread)
        min_pace  = _pace_key(min_sc,  total)
        rate_pace = _pace_key(rate_sc, total)
        min_def   = _def_key(min_sc,  opp_def_rating)
        rate_def  = _def_key(rate_sc, opp_def_rating)

        raw_min_delta,  min_log  = _build_delta(min_sc,  n_stars, s_key, min_pace,  min_def)
        raw_rate_delta, rate_log = _build_delta(rate_sc, n_stars, s_key, rate_pace, rate_def)

        min_delta  = float(np.clip(raw_min_delta,  -delta_cap_min,  delta_cap_min))
        rate_delta = float(np.clip(raw_rate_delta, -delta_cap_rate, delta_cap_rate))

        orig_min_q50  = float(row["MIN_Q50"])
        orig_rate_q50 = float(row["RATE_Q50"])
        new_min_q50   = max(orig_min_q50  + min_adjust_weight  * min_delta,  0.0)
        new_rate_q50  = max(orig_rate_q50 + rate_adjust_weight * rate_delta, 0.0)

        row["MIN_Q10"]  = round(max(float(row["MIN_Q10"])  + min_delta,  0.0), 2)
        row["MIN_Q90"]  = round(max(float(row["MIN_Q90"])  + min_delta,  0.0), 2)
        row["RATE_Q10"] = round(max(float(row["RATE_Q10"]) + rate_delta, 0.0), 4)
        row["RATE_Q90"] = round(max(float(row["RATE_Q90"]) + rate_delta, 0.0), 4)
        row["MIN_Q50"]  = round(new_min_q50,  2)
        row["RATE_Q50"] = round(new_rate_q50, 4)

        if orig_rate_q50 > 0 and row.get("RATE_HISTORY") is not None:
            row["RATE_HISTORY"] = [round(r + rate_adjust_weight * rate_delta, 6) for r in row["RATE_HISTORY"]]

        row["STAT_Q10"] = round(float(row["MIN_Q10"]) * float(row["RATE_Q10"]), 2)
        row["STAT_Q50"] = round(new_min_q50 * new_rate_q50, 2)
        row["STAT_Q90"] = round(float(row["MIN_Q90"]) * float(row["RATE_Q90"]), 2)

        _sanitize_prediction_row_quantiles(row)

        row["ADJ_CONTEXT_OK"]  = True
        row["ADJ_CONTEXT_ERR"] = None
        row["ADJ_ACTIVE_STARS"]= n_stars
        row["ADJ_CTX_SPREAD"]  = float(spread)         if spread         is not None else None
        row["ADJ_CTX_TOTAL"]   = float(total)          if total          is not None else None
        row["ADJ_CTX_OPP_DEF"] = float(opp_def_rating) if opp_def_rating is not None else None
        row["ADJ_MIN_DELTA"]   = round(min_delta,  4)
        row["ADJ_RATE_DELTA"]  = round(rate_delta, 6)
        row["ADJ_MIN_LOG"]     = min_log
        row["ADJ_RATE_LOG"]    = rate_log

        if verbose:
            print(
                f"{name} [{market}]  "
                f"pace={min_pace}  def={min_def}  spread={s_key}  stars={n_stars}  "
                f"MIN: {orig_min_q50:.1f}→{new_min_q50:.1f} (Δ{min_delta:+.2f})  "
                f"RATE: {orig_rate_q50:.4f}→{new_rate_q50:.4f} (Δ{rate_delta:+.4f})"
            )

        out_rows.append(row)

    return pd.DataFrame(out_rows).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# GAME CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

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
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
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
    Build the game-context dict consumed by adjust_predictions().
    Derives opp_def_rating from recent matchups in base_df.
    Returns ok=True on success, ok=False with error on failure.
    """
    _out_players   = out_players     if out_players     is not None else outPlayers
    _stars_by_team = stars_by_team   if stars_by_team   is not None else team3StarsPerTeam
    _abbrev_map    = team_abbrev_map if team_abbrev_map is not None else TEAM_NAME_TO_ABBREV

    def _fail(reason):
        return {"ok": False, "error": reason, "player": player_name,
                "team": None, "opponent": None, "is_home": None,
                "active_stars": None, "spread": None, "total": None,
                "opp_def_rating": None, "is_playoff": None}

    # ── 1. Player → team ─────────────────────────────────────────────────────
    pdf = base_df[base_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")
    if pdf.empty:
        return _fail(f"player '{player_name}' not found in base_df")

    team_name   = pdf["TEAM_NAME"].iloc[-1]
    team_abbrev = _abbrev_map.get(team_name)

    # ── 2. Find game ─────────────────────────────────────────────────────────
    games = team_odds.to_dict("records") if isinstance(team_odds, pd.DataFrame) else team_odds

    def _matches(feed_name: str) -> bool:
        norm = lambda s: s.lower().replace(".", "").replace("-", " ").strip()
        return norm(feed_name) == norm(team_name) or (team_abbrev and norm(feed_name) == norm(team_abbrev))

    game_data = next(
        (g for g in games if _matches(g.get("home_team", "")) or _matches(g.get("away_team", ""))),
        None,
    )
    if game_data is None:
        return _fail(f"no game found for '{team_name}' (abbrev: {team_abbrev})")

    is_home  = _matches(game_data.get("home_team", ""))
    opponent = game_data["away_team"] if is_home else game_data["home_team"]

    # ── 3. Bookmaker ─────────────────────────────────────────────────────────
    available = {b.get("bookmaker"): b for b in game_data.get("bookmakers", [])}
    bk, used_bookmaker = None, None
    for book in bookmaker_priority:
        if book in available:
            bk, used_bookmaker = available[book], book
            break
    if bk is None:
        return _fail(f"none of {list(bookmaker_priority)} found; available: {list(available.keys())}")

    # ── 4. Active stars ──────────────────────────────────────────────────────
    stars  = _stars_by_team.get(team_abbrev, [])
    active = [s for s in stars if s not in _out_players.get(team_abbrev, [])]

    # ── 5. Lines ─────────────────────────────────────────────────────────────
    spread = total = None
    for market in bk.get("markets", []):
        key = market.get("market_key")
        if key == "spreads":
            for o in market.get("outcomes", []):
                if _matches(o.get("name", "")):
                    spread = o.get("point")
        elif key == "totals":
            for o in market.get("outcomes", []):
                if o.get("name") == "Over":
                    total = o.get("point")

    # ── 6. Opponent defensive rating ─────────────────────────────────────────
    opp_def_rating = None
    opp_abbrev = _abbrev_map.get(opponent)
    if opp_abbrev and "OPP_DEF_RATING" in base_df.columns and "MATCHUP" in base_df.columns:
        recent_vs = (
            base_df[base_df["MATCHUP"].str.contains(opp_abbrev, na=False)]
            .sort_values("GAME_DATE")
            .tail(30)
        )
        if not recent_vs.empty:
            opp_def_rating = round(float(recent_vs["OPP_DEF_RATING"].mean()), 2)

    missing = [k for k, v in [("spread", spread), ("total", total)] if v is None]
    if missing:
        import warnings
        warnings.warn(
            f"{player_name} ({team_name}): missing lines {missing} from {used_bookmaker}",
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
        "total":             total,
        "opp_def_rating":    opp_def_rating,
        "is_playoff":        is_playoff,
    }
