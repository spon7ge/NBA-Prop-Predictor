"""Helpers for NBA points scoring discovery notebooks."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

PRIOR_TOKENS = ("_ewm_", "_season_avg", "_lag1", "_lag", "_roll", "_trend", "_rank", "_std", "_var")
ID_META = {
    "game_id", "player_id", "team_id", "opp_team_id", "player_name", "matchup",
    "game_date", "season_year", "season", "pos", "starting",
}
CONTEXT_COLS = {
    "days_rest", "is_back_to_back", "games_played", "team_games_played",
    "games_played_last_7_days", "games_played_last_14_days", "min_sum_last_7_days",
    "starter_roll10_pct",
}
MARKET_TOKENS = ("over_under", "ou_", "implied", "spread", "line", "total")

# concept → exact or substring matchers for available / proxy columns
WISHLIST_ITEMS: list[dict] = [
    {"category": "Player traditional", "concept": "Minutes", "column_matchers": ["minutes", "base_min"], "proxy_matchers": ["track_minutes"]},
    {"category": "Player traditional", "concept": "FGA", "column_matchers": ["fga_per_min", "base_fga"], "proxy_matchers": []},
    {"category": "Player traditional", "concept": "3PA", "column_matchers": ["fg3a_per_min", "base_fg3a"], "proxy_matchers": []},
    {"category": "Player traditional", "concept": "FTA", "column_matchers": ["fta_per_min", "base_fta"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "Usage %", "column_matchers": ["usg_pct", "adv_usg"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "TS%", "column_matchers": ["ts_pct", "adv_ts"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "eFG%", "column_matchers": ["efg_pct", "adv_efg"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "Offensive Rating", "column_matchers": ["off_rating", "adv_off"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "Assist %", "column_matchers": ["ast_pct", "adv_ast_pct"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "Turnover % / TOV", "column_matchers": ["tov_pct", "tov_per_min", "adv_tov"], "proxy_matchers": []},
    {"category": "Player advanced", "concept": "Pace / possessions", "column_matchers": ["pace", "poss", "adv_pace", "adv_poss"], "proxy_matchers": []},
    {"category": "Tracking", "concept": "Touches", "column_matchers": ["tchs", "track_tchs"], "proxy_matchers": []},
    {"category": "Tracking", "concept": "Time of possession", "column_matchers": [], "proxy_matchers": ["tchs", "pass_per_min"]},
    {"category": "Tracking", "concept": "Contested / uncontested FGA", "column_matchers": ["cfga", "ufga", "dfga"], "proxy_matchers": []},
    {"category": "Tracking", "concept": "Catch-and-shoot / pull-up", "column_matchers": [], "proxy_matchers": []},
    {"category": "Tracking", "concept": "Drive frequency", "column_matchers": [], "proxy_matchers": []},
    {"category": "Tracking", "concept": "Shot quality / expected eFG", "column_matchers": [], "proxy_matchers": []},
    {"category": "Team context", "concept": "Team pace", "column_matchers": ["team_pace"], "proxy_matchers": []},
    {"category": "Team context", "concept": "Team offensive rating / efficiency", "column_matchers": ["team_off", "team_ts"], "proxy_matchers": ["team_pts"]},
    {"category": "Opponent defense", "concept": "Opponent defensive rating", "column_matchers": ["opp_def_rating"], "proxy_matchers": []},
    {"category": "Opponent defense", "concept": "Rim protection / scheme rates", "column_matchers": [], "proxy_matchers": []},
    {"category": "Opponent defense", "concept": "Allowed shot profile (corner/paint/transition)", "column_matchers": [], "proxy_matchers": ["opp_pts", "opp_fg"]},
    {"category": "Individual matchups", "concept": "Primary defender", "column_matchers": [], "proxy_matchers": []},
    {"category": "Individual matchups", "concept": "Height / wingspan mismatch", "column_matchers": [], "proxy_matchers": []},
    {"category": "Vegas", "concept": "Game total / team implied total / spread", "column_matchers": ["over_under", "implied", "spread"], "proxy_matchers": ["ou_"]},
    {"category": "Vegas", "concept": "Line movement / steam / consensus", "column_matchers": [], "proxy_matchers": []},
    {"category": "Injury", "concept": "Usage redistribution when teammate OUT", "column_matchers": [], "proxy_matchers": []},
    {"category": "Rest", "concept": "Back-to-back / days rest", "column_matchers": ["days_rest", "is_back_to_back"], "proxy_matchers": []},
    {"category": "Rest", "concept": "Travel / time zones / trip length", "column_matchers": [], "proxy_matchers": []},
    {"category": "Coaching", "concept": "Rotation stability / minutes volatility", "column_matchers": ["starter_roll", "min_"], "proxy_matchers": ["track_minutes_"]},
    {"category": "Coaching", "concept": "Blowout substitution tendencies", "column_matchers": [], "proxy_matchers": ["plus_minus"]},
]


def _matches(columns: set[str], matchers: Sequence[str]) -> list[str]:
    hits: list[str] = []
    for m in matchers:
        m_low = m.lower()
        for c in columns:
            if m_low in c.lower() and c not in hits:
                hits.append(c)
    return hits


def build_coverage_map(columns: Iterable[str]) -> pd.DataFrame:
    colset = set(columns)
    rows = []
    for item in WISHLIST_ITEMS:
        exact = _matches(colset, item["column_matchers"])
        proxy = _matches(colset, item["proxy_matchers"]) if not exact else []
        if exact:
            status = "available"
            matched = exact
        elif proxy:
            status = "partial"
            matched = proxy
        else:
            status = "missing"
            matched = []
        rows.append({
            "category": item["category"],
            "concept": item["concept"],
            "status": status,
            "matched_columns": ", ".join(matched[:8]),
        })
    return pd.DataFrame(rows)


def derive_pts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pts" in out.columns:
        return out
    if "pts_per_min" not in out.columns or "minutes" not in out.columns:
        raise ValueError("derive_pts requires pts_per_min and minutes")
    out["pts"] = out["pts_per_min"].astype(float) * out["minutes"].astype(float)
    return out


def lineage_for(col: str) -> str:
    c = col.lower()
    if col in ID_META or c in ID_META:
        return "excluded"
    if any(tok in c for tok in MARKET_TOKENS):
        return "market"
    if col in CONTEXT_COLS or c in CONTEXT_COLS or c.startswith("starter_"):
        return "context"
    if c.startswith("opp_"):
        return "opponent"
    if c.startswith("team_"):
        return "team"
    if any(tok in c for tok in PRIOR_TOKENS) or c.startswith(("base_", "adv_", "track_")):
        return "prior_player"
    return "same_game"


def split_feature_pools(
    columns: Iterable[str],
    *,
    targets: Sequence[str],
) -> dict[str, list[str]]:
    target_set = set(targets)
    same_game: list[str] = []
    predictive: list[str] = []
    excluded: list[str] = []
    for col in columns:
        if col in target_set or col in ID_META:
            excluded.append(col)
            continue
        lin = lineage_for(col)
        if lin == "excluded":
            excluded.append(col)
            continue
        if lin == "same_game":
            same_game.append(col)
        elif lin in {"prior_player", "team", "opponent", "context", "market"}:
            # only keep prior-safe engineered / context / market cols
            if lin == "prior_player" and not (
                any(tok in col for tok in PRIOR_TOKENS) or col.startswith(("base_", "adv_", "track_"))
            ):
                same_game.append(col)
            else:
                predictive.append(col)
        else:
            excluded.append(col)
    return {
        "same_game": sorted(set(same_game)),
        "predictive": sorted(set(predictive)),
        "excluded": sorted(set(excluded)),
    }


def rank_univariate(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    use = [f for f in features if f in df.columns and f != target]
    if not use:
        return pd.DataFrame(columns=["feature", "spearman", "mi", "lineage"])
    y = df[target].astype(float)
    mask = y.notna()
    rows = []
    X = df.loc[mask, use].apply(pd.to_numeric, errors="coerce")
    y = y.loc[mask]
    # drop all-null features
    use = [c for c in use if X[c].notna().any()]
    X = X[use]
    med = X.median(numeric_only=True)
    X_filled = X.fillna(med)
    mi = mutual_info_regression(X_filled, y, random_state=random_state)
    mi_map = dict(zip(use, mi))
    for feat in use:
        spear = X[feat].corr(y, method="spearman")
        rows.append({
            "feature": feat,
            "spearman": float(spear) if pd.notna(spear) else 0.0,
            "mi": float(mi_map[feat]),
            "lineage": lineage_for(feat),
        })
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    out = out.sort_values(["mi", "abs_spearman"], ascending=False).drop(columns=["abs_spearman"])
    return out.reset_index(drop=True)


def season_rank_stability(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    season_col: str = "season_year",
    *,
    top_n: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    seasons = [s for s in df[season_col].dropna().unique().tolist()]
    rank_maps: list[pd.Series] = []
    for season in seasons:
        sub = df.loc[df[season_col] == season]
        if len(sub) < 50:
            continue
        ranks = rank_univariate(sub, features, target, random_state=random_state)
        if ranks.empty:
            continue
        s = ranks.set_index("feature")["mi"].rank(ascending=False)
        rank_maps.append(s)
    if len(rank_maps) < 2:
        base = rank_univariate(df, features, target, random_state=random_state)
        base["stability"] = np.nan
        return base.head(top_n)
    mat = pd.concat(rank_maps, axis=1)
    # pairwise Spearman of rank vectors across seasons, then mean per feature via leave-one?
    # Use feature-wise std of ranks as inverse stability; also overall pairwise.
    pair_cors = []
    cols = list(mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair_cors.append(mat.iloc[:, i].corr(mat.iloc[:, j], method="spearman"))
    global_stab = float(np.nanmean(pair_cors)) if pair_cors else np.nan
    # per-feature: lower rank variance → higher stability
    rank_std = mat.std(axis=1)
    stab = 1.0 / (1.0 + rank_std)
    overall = rank_univariate(df, features, target, random_state=random_state).set_index("feature")
    out = overall.copy()
    out["stability"] = stab.reindex(out.index)
    out["season_rank_corr_mean"] = global_stab
    out = out.reset_index()
    return out.sort_values(["stability", "mi"], ascending=False).head(top_n).reset_index(drop=True)


def merge_driver_shortlist(
    univariate_df: pd.DataFrame,
    shap_df: pd.DataFrame | None = None,
    perm_df: pd.DataFrame | None = None,
    *,
    top_k: int = 25,
) -> pd.DataFrame:
    uni = univariate_df.copy()
    uni["uni_rank"] = uni["mi"].rank(ascending=False, method="first")
    out = uni[["feature", "spearman", "mi", "lineage", "uni_rank"]]
    if shap_df is not None and not shap_df.empty:
        s = shap_df.copy()
        s["shap_rank"] = s["mean_abs_shap"].rank(ascending=False)
        out = out.merge(s[["feature", "mean_abs_shap", "shap_rank"]], on="feature", how="left")
    if perm_df is not None and not perm_df.empty:
        p = perm_df.copy()
        p["perm_rank"] = p["importance_mean"].rank(ascending=False)
        out = out.merge(
            p[["feature", "importance_mean", "perm_rank"]], on="feature", how="left",
        )
    rank_cols = [c for c in ("uni_rank", "shap_rank", "perm_rank") if c in out.columns]
    out["consensus_rank"] = out[rank_cols].mean(axis=1)
    return out.sort_values("consensus_rank").head(top_k).reset_index(drop=True)
