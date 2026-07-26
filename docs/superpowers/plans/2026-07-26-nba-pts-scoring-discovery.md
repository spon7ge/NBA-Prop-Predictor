# NBA Points Scoring Discovery Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `models/nba/pts_scoring_discovery.ipynb` that maps wishlist coverage, ranks same-game and prior-only drivers of `pts` / `pts_per_min` / `minutes`, probes extra `nba_api` endpoints, and emits a PPM shortlist plus ingest priorities.

**Architecture:** Put pure ranking/coverage helpers in `models/shared/scoring_discovery.py` (unit-tested). The notebook loads season parquets the same way MIN does, enriches with `ContextFeatureEngineer`, then calls those helpers for anatomy, predictive ranks, stability, and deliverables. Optional live endpoint probes stay notebook-local with a toggle and local cache.

**Tech Stack:** Jupyter (nbformat 4), pandas, numpy, scikit-learn (mutual_info_regression, permutation_importance), XGBoost, shap (optional soft-import), nba_api, pytest.

**Spec:** `docs/superpowers/specs/2026-07-26-nba-pts-scoring-discovery-design.md`

## Global Constraints

- NBA only; seasons `["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]`
- Load: `data/processed/{season}_Regular_Season_training_data.parquet` then `ContextFeatureEngineer(league="nba").enrich`
- Skip missing season files with a warning; do not rebuild from raw in this task
- Filter: `(minutes >= 5) | (starting == 1)`
- Holdout: `HOLDOUT_SEASON = "2025-26"` — fit model-based ranks on pre-holdout only
- Derive `pts = pts_per_min * minutes` (raw `pts` is not on the training parquet)
- Same-game features never enter predictive rankings
- No production artifact save; no `fetch.py` / silver / gold changes; no Supabase upserts from probes
- `RUN_ENDPOINT_PROBES` defaults to `False` so offline runs still produce sections 1–8 and 10
- Fixed `RANDOM_SEED = 42`
- Clear all stored cell outputs when writing the notebook

---

## File map

| File | Responsibility |
|------|----------------|
| `models/shared/scoring_discovery.py` | Pure helpers: wishlist status, pool split, univariate ranks, season stability, shortlist merge |
| `models/shared/test_scoring_discovery.py` | Unit tests for helpers |
| `models/nba/pts_scoring_discovery.ipynb` | End-to-end discovery notebook |
| `models/README.md` | One-line pointer to the new discovery notebook |
| `data/raw/cache/discovery/` | Optional probe cache (gitignored if not already; do not commit large dumps) |

---

### Task 1: Scoring discovery helpers + tests

**Files:**
- Create: `models/shared/scoring_discovery.py`
- Create: `models/shared/test_scoring_discovery.py`

**Interfaces:**
- Consumes: pandas DataFrames / column lists
- Produces:
  - `WISHLIST_ITEMS: list[dict]` — static checklist entries with `category`, `concept`, `column_matchers`, `proxy_matchers`
  - `build_coverage_map(columns: Iterable[str]) -> pd.DataFrame` — columns `category, concept, status, matched_columns`
  - `derive_pts(df: pd.DataFrame) -> pd.DataFrame` — adds `pts` from `pts_per_min * minutes`
  - `split_feature_pools(columns: Iterable[str], *, targets: Sequence[str]) -> dict[str, list[str]]` — keys `same_game`, `predictive`, `excluded`
  - `lineage_for(col: str) -> str` — one of `same_game|prior_player|team|opponent|context|market|excluded`
  - `rank_univariate(df, features, target, *, random_state=42) -> pd.DataFrame` — `feature, spearman, mi, lineage`
  - `season_rank_stability(df, features, target, season_col="season_year", *, top_n=30, random_state=42) -> pd.DataFrame` — mean Spearman rank correlation of feature ranks across seasons
  - `merge_driver_shortlist(univariate_df, shap_df=None, perm_df=None, *, top_k=25) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for models.shared.scoring_discovery helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.shared import scoring_discovery as sd


def test_derive_pts_multiplies_rate_by_minutes():
    df = pd.DataFrame({"pts_per_min": [0.5, 1.0], "minutes": [20.0, 30.0]})
    out = sd.derive_pts(df)
    assert out["pts"].tolist() == pytest.approx([10.0, 30.0])


def test_build_coverage_map_statuses():
    cols = [
        "fga_per_min",
        "base_fga_per_min_ewm_hl10",
        "adv_usg_pct_season_avg",
        "track_tchs_per_min_ewm_hl10",
        "opp_def_rating_ewm_hl10",
        "days_rest",
        "is_back_to_back",
    ]
    cov = sd.build_coverage_map(cols)
    assert set(cov["status"]) <= {"available", "partial", "missing"}
    usg = cov.loc[cov["concept"].str.contains("Usage", case=False)].iloc[0]
    assert usg["status"] in {"available", "partial"}
    matchup = cov.loc[cov["concept"].str.contains("Primary defender", case=False)]
    assert not matchup.empty
    assert matchup.iloc[0]["status"] == "missing"


def test_split_feature_pools_keeps_prior_out_of_same_game():
    cols = [
        "fga_per_min",
        "tchs_per_min",
        "base_fga_per_min_ewm_hl10",
        "adv_usg_pct_season_avg",
        "opp_pace_ewm_hl10",
        "days_rest",
        "player_id",
        "pts_per_min",
        "pts",
        "minutes",
    ]
    pools = sd.split_feature_pools(cols, targets=["pts", "pts_per_min", "minutes"])
    assert "fga_per_min" in pools["same_game"]
    assert "base_fga_per_min_ewm_hl10" in pools["predictive"]
    assert "fga_per_min" not in pools["predictive"]
    assert "pts" in pools["excluded"]
    assert "player_id" in pools["excluded"]


def test_lineage_for_prefixes():
    assert sd.lineage_for("base_pts_per_min_ewm_hl5") == "prior_player"
    assert sd.lineage_for("opp_def_rating_ewm_hl10") == "opponent"
    assert sd.lineage_for("team_pace_ewm_hl10") == "team"
    assert sd.lineage_for("days_rest") == "context"
    assert sd.lineage_for("fga_per_min") == "same_game"


def test_rank_univariate_orders_strong_signal_first():
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    df = pd.DataFrame({
        "signal": signal,
        "noise": noise,
        "y": signal * 2 + rng.normal(scale=0.1, size=n),
    })
    ranks = sd.rank_univariate(df, ["signal", "noise"], "y", random_state=42)
    assert ranks.iloc[0]["feature"] == "signal"
    assert abs(ranks.loc[ranks["feature"] == "signal", "spearman"].iloc[0]) > 0.8


def test_season_rank_stability_returns_score():
    rng = np.random.default_rng(1)
    rows = []
    for season in ("2023-24", "2024-25"):
        for _ in range(200):
            s = rng.normal()
            rows.append({
                "season_year": season,
                "good": s,
                "noise": rng.normal(),
                "y": s + rng.normal(scale=0.2),
            })
    df = pd.DataFrame(rows)
    stab = sd.season_rank_stability(
        df, ["good", "noise"], "y", season_col="season_year", top_n=2, random_state=42,
    )
    assert "stability" in stab.columns
    assert stab.loc[stab["feature"] == "good", "stability"].iloc[0] >= \
        stab.loc[stab["feature"] == "noise", "stability"].iloc[0]


def test_merge_driver_shortlist_intersects_signals():
    uni = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "spearman": [0.9, 0.2, 0.5],
        "mi": [0.8, 0.1, 0.4],
        "lineage": ["prior_player"] * 3,
    })
    shap = pd.DataFrame({"feature": ["a", "c", "b"], "mean_abs_shap": [1.0, 0.5, 0.1]})
    out = sd.merge_driver_shortlist(uni, shap_df=shap, top_k=2)
    assert list(out["feature"])[:2] == ["a", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest models/shared/test_scoring_discovery.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `models.shared.scoring_discovery`.

- [ ] **Step 3: Implement `models/shared/scoring_discovery.py`**

```python
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
    uni["uni_rank"] = np.arange(1, len(uni) + 1)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest models/shared/test_scoring_discovery.py -v`

Expected: all PASS. If `test_build_coverage_map_statuses` fails on Usage status, adjust matchers so `adv_usg` counts as available/partial. If Primary defender is not missing, ensure that wishlist row has empty matchers.

- [ ] **Step 5: Commit**

```bash
git add models/shared/scoring_discovery.py models/shared/test_scoring_discovery.py
git commit -m "$(cat <<'EOF'
Add scoring-discovery helpers for points driver analysis.

EOF
)"
```

---

### Task 2: Notebook scaffold — imports, config, load, sanity

**Files:**
- Create: `models/nba/pts_scoring_discovery.ipynb`

**Interfaces:**
- Consumes: season parquets + `ContextFeatureEngineer` + `derive_pts`
- Produces: notebook variable `df` (filtered, with `pts`), `SEASONS`, `TARGETS`, `HOLDOUT_SEASON`, `RANDOM_SEED`, `RUN_ENDPOINT_PROBES`

- [ ] **Step 1: Create the notebook with these cells (empty outputs)**

**Markdown — title**

```markdown
# NBA points scoring discovery

Explore what drives `pts`, `pts_per_min`, and `minutes` using existing training parquets + optional `nba_api` probes.

Spec: `docs/superpowers/specs/2026-07-26-nba-pts-scoring-discovery-design.md`
```

**Code — imports / root** (mirror MIN pattern):

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve repo root (directory that contains data/)
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    for cand in [ROOT.parent, *ROOT.parents]:
        if (cand / "data").exists():
            ROOT = cand
            break
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
print("cwd:", Path.cwd())
```

**Code — config**

```python
from src.pipeline.features.context_features import ContextFeatureEngineer
from models.shared.scoring_discovery import (
    build_coverage_map,
    derive_pts,
    lineage_for,
    merge_driver_shortlist,
    rank_univariate,
    season_rank_stability,
    split_feature_pools,
)

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
HOLDOUT_SEASON = "2025-26"
TARGETS = ["pts", "pts_per_min", "minutes"]
RANDOM_SEED = 42
RUN_ENDPOINT_PROBES = False  # set True only when live nba_api pulls are desired
np.random.seed(RANDOM_SEED)
```

**Code — load**

```python
frames = []
missing = []
for yr in SEASONS:
    path = Path(f"data/processed/{yr}_Regular_Season_training_data.parquet")
    if not path.exists():
        missing.append(yr)
        print(f"⚠ missing parquet for {yr}: {path}")
        continue
    season_df = pd.read_parquet(path)
    season_df = ContextFeatureEngineer(league="nba", season=yr).enrich(season_df)
    frames.append(season_df)
    print(f"✓ {yr}: {len(season_df):,} rows")

if not frames:
    raise FileNotFoundError("No season parquets found under data/processed/")

df = pd.concat(frames, ignore_index=True)
df = derive_pts(df)
df = df[(df["minutes"] >= 5) | (df["starting"] == 1)].copy()
print(f"Combined: {len(df):,} rows × {df.shape[1]} cols | missing seasons: {missing or 'none'}")
df[["season_year", "pts", "pts_per_min", "minutes"]].describe()
```

**Code — sanity**

```python
dupes = df.duplicated(subset=["game_id", "player_id"]).sum()
print(f"Duplicate game_id+player_id: {dupes}")
print(df.groupby("season_year").size())
print("Target nulls:", {t: int(df[t].isna().sum()) for t in TARGETS})
```

- [ ] **Step 2: Smoke-check load cell logic without full enrich (optional quick check)**

Run from repo root:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from models.shared.scoring_discovery import derive_pts
p = Path('data/processed/2024-25_Regular_Season_training_data.parquet')
assert p.exists()
df = derive_pts(pd.read_parquet(p))
assert 'pts' in df.columns
print(df[['pts','pts_per_min','minutes']].head(2))
PY
```

Expected: prints two rows; no exception.

- [ ] **Step 3: Commit**

```bash
git add models/nba/pts_scoring_discovery.ipynb
git commit -m "$(cat <<'EOF'
Scaffold NBA points scoring discovery notebook load path.

EOF
)"
```

---

### Task 3: Coverage map + same-game scoring anatomy

**Files:**
- Modify: `models/nba/pts_scoring_discovery.ipynb`

**Interfaces:**
- Consumes: `df`, `build_coverage_map`, `split_feature_pools`, `rank_univariate`
- Produces: `coverage_df`, `pools`, `anatomy_ranks` dict keyed by target

- [ ] **Step 1: Add markdown warning + coverage + anatomy cells**

**Markdown**

```markdown
## Wishlist coverage

Status is based on columns present **after** load+enrich — not aspirational names.
`partial` = proxy only (e.g. contested FGA ≈ contest rate).
```

**Code**

```python
coverage_df = build_coverage_map(df.columns)
display(coverage_df)
print(coverage_df["status"].value_counts())
```

**Markdown**

```markdown
## Scoring anatomy (same-game)

**NOT model features.** These are contemporaneous associations that explain how points are produced in the same game. Using them pre-tip is leakage.
```

**Code**

```python
pools = split_feature_pools(df.columns, targets=TARGETS)
print({k: len(v) for k, v in pools.items()})
# leakage audit
overlap = set(pools["same_game"]) & set(pools["predictive"])
print("same_game ∩ predictive:", overlap or "∅ (ok)")

anatomy_ranks = {}
for target in TARGETS:
    ranks = rank_univariate(df, pools["same_game"], target, random_state=RANDOM_SEED)
    anatomy_ranks[target] = ranks
    print(f"\n=== same-game drivers of {target} ===")
    display(ranks.head(20))
```

**Code — volume × efficiency scatter (keep simple)**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
pairs = [
    ("fga_per_min", "pts"),
    ("tchs_per_min", "pts_per_min"),
    ("minutes", "pts"),
]
sample = df.sample(min(8000, len(df)), random_state=RANDOM_SEED)
for ax, (x, y) in zip(axes, pairs):
    if x not in sample.columns or y not in sample.columns:
        ax.set_title(f"missing {x}/{y}")
        continue
    ax.scatter(sample[x], sample[y], s=4, alpha=0.15)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Mentally verify / quick assert in a scratch cell (can delete after)**

```python
assert "fga_per_min" in pools["same_game"]
assert "base_fga_per_min_ewm_hl10" in pools["predictive"]
assert coverage_df.loc[coverage_df["concept"] == "Primary defender", "status"].iloc[0] == "missing"
```

- [ ] **Step 3: Commit**

```bash
git add models/nba/pts_scoring_discovery.ipynb
git commit -m "$(cat <<'EOF'
Add coverage map and same-game scoring anatomy cells.

EOF
)"
```

---

### Task 4: Predictive univariate ranks + season stability + model-based ranks

**Files:**
- Modify: `models/nba/pts_scoring_discovery.ipynb`

**Interfaces:**
- Consumes: `pools["predictive"]`, pre-holdout slice, `models.shared.analysis` optionally
- Produces: `pred_ranks`, `stability`, `shap_tables` / `perm_tables`, `shortlists` per target

- [ ] **Step 1: Add predictive univariate + stability cells**

**Markdown**

```markdown
## Predictive discovery (prior-only, leak-safe)

Candidates are EWM / season_avg / lag / roll / team / opp / context / market columns only.
Model-based ranks fit on seasons **before** `HOLDOUT_SEASON`; holdout MAE is diagnostic.
```

**Code**

```python
pred_cols = pools["predictive"]
# Drop any accidental target leakage in feature names
pred_cols = [c for c in pred_cols if c not in TARGETS]

pred_ranks = {}
stability = {}
for target in TARGETS:
    r = rank_univariate(df, pred_cols, target, random_state=RANDOM_SEED)
    pred_ranks[target] = r
    stability[target] = season_rank_stability(
        df, pred_cols, target, season_col="season_year", top_n=40, random_state=RANDOM_SEED,
    )
    print(f"\n=== prior drivers of {target} ===")
    display(r.head(25))
    print(f"--- stability ({target}) ---")
    display(stability[target].head(15))
```

- [ ] **Step 2: Add lightweight XGB + SHAP + permutation for each target**

Use a **capped** feature set for speed: top 40 by MI from univariate ranks on the train pool.

```python
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    print("shap not available — skipping SHAP tables")

train_mask = df["season_year"] != HOLDOUT_SEASON
hold_mask = df["season_year"] == HOLDOUT_SEASON
train_df = df.loc[train_mask]
hold_df = df.loc[hold_mask]

XGB_DISC = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=4,
)

shap_tables = {}
perm_tables = {}
holdout_diag = {}

for target in TARGETS:
    top_feats = [
        f for f in pred_ranks[target]["feature"].head(40).tolist()
        if f in train_df.columns
    ]
    X_tr = train_df[top_feats].apply(pd.to_numeric, errors="coerce")
    y_tr = train_df[target].astype(float)
    med = X_tr.median()
    X_tr = X_tr.fillna(med)
    model = XGBRegressor(**XGB_DISC)
    model.fit(X_tr, y_tr)

    if len(hold_df):
        X_ho = hold_df[top_feats].apply(pd.to_numeric, errors="coerce").fillna(med)
        y_ho = hold_df[target].astype(float)
        pred = model.predict(X_ho)
        if target == "pts_per_min" and "base_pts_per_min_season_avg" in hold_df.columns:
            holdout_diag[target] = {
                "model_mae": float(mean_absolute_error(y_ho, pred)),
                "naive_mae": float(mean_absolute_error(
                    y_ho, hold_df["base_pts_per_min_season_avg"].astype(float),
                )),
                "naive": "base_pts_per_min_season_avg",
            }
        elif target == "minutes" and "base_min_season_avg" in hold_df.columns:
            holdout_diag[target] = {
                "model_mae": float(mean_absolute_error(y_ho, pred)),
                "naive_mae": float(mean_absolute_error(
                    y_ho, hold_df["base_min_season_avg"].astype(float),
                )),
                "naive": "base_min_season_avg",
            }
        elif (
            target == "pts"
            and "base_pts_per_min_season_avg" in hold_df.columns
            and "track_minutes_season_avg" in hold_df.columns
        ):
            naive = (
                hold_df["base_pts_per_min_season_avg"].astype(float)
                * hold_df["track_minutes_season_avg"].astype(float)
            )
            holdout_diag[target] = {
                "model_mae": float(mean_absolute_error(y_ho, pred)),
                "naive_mae": float(mean_absolute_error(
                    y_ho, naive.fillna(y_ho.median()),
                )),
                "naive": "ppm_season_avg * track_minutes_season_avg",
            }
        else:
            holdout_diag[target] = {
                "model_mae": float(mean_absolute_error(y_ho, pred)),
                "naive_mae": None,
                "naive": None,
            }

    sample_idx = X_tr.sample(min(5000, len(X_tr)), random_state=RANDOM_SEED).index
    perm = permutation_importance(
        model,
        X_tr.loc[sample_idx],
        y_tr.loc[sample_idx],
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=2,
    )
    perm_tables[target] = pd.DataFrame({
        "feature": top_feats,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_tr.loc[sample_idx])
        shap_tables[target] = pd.DataFrame({
            "feature": top_feats,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
    else:
        shap_tables[target] = pd.DataFrame(columns=["feature", "mean_abs_shap"])

    print(f"\n=== model ranks {target} ===")
    display(perm_tables[target].head(15))
    if HAS_SHAP:
        display(shap_tables[target].head(15))

print("Holdout diagnostic:", holdout_diag)
```

Optional collinearity on shortlist candidates:

```python
from models.shared.analysis import analyze_correlations

for target in TARGETS:
    feats = pred_ranks[target]["feature"].head(20).tolist()
    analyze_correlations(train_df, feats, threshold=0.95, title=f"Predictive corr ({target})")
```

- [ ] **Step 3: Commit**

```bash
git add models/nba/pts_scoring_discovery.ipynb
git commit -m "$(cat <<'EOF'
Add predictive univariate, stability, and model-based driver ranks.

EOF
)"
```

---

### Task 5: Optional endpoint probes

**Files:**
- Modify: `models/nba/pts_scoring_discovery.ipynb`

**Interfaces:**
- Consumes: `RUN_ENDPOINT_PROBES`, nba_api
- Produces: `probe_inventory` DataFrame; optional cache under `data/raw/cache/discovery/`

- [ ] **Step 1: Add probe cells**

**Markdown**

```markdown
## Endpoint probes (optional)

Inventory-only pulls for surfaces not in `GameLogs`. No DB upserts.
Default `RUN_ENDPOINT_PROBES = False`.
```

**Code**

```python
from datetime import datetime
import time
import json

PROBE_CACHE = Path("data/raw/cache/discovery")
PROBE_CACHE.mkdir(parents=True, exist_ok=True)

probe_inventory = pd.DataFrame(columns=["endpoint", "ok", "n_rows", "n_cols", "columns_head", "error"])

def _probe(name: str, fn):
    global probe_inventory
    cache_path = PROBE_CACHE / f"{name}.parquet"
    try:
        time.sleep(0.6)
        df_p = fn()
        if df_p is None or df_p.empty:
            raise RuntimeError("empty frame")
        df_p.to_parquet(cache_path, index=False)
        row = {
            "endpoint": name,
            "ok": True,
            "n_rows": len(df_p),
            "n_cols": df_p.shape[1],
            "columns_head": ", ".join(map(str, df_p.columns[:20])),
            "error": "",
        }
    except Exception as exc:
        row = {
            "endpoint": name,
            "ok": False,
            "n_rows": 0,
            "n_cols": 0,
            "columns_head": "",
            "error": str(exc)[:200],
        }
    probe_inventory = pd.concat([probe_inventory, pd.DataFrame([row])], ignore_index=True)
    print(row)

if RUN_ENDPOINT_PROBES:
    from nba_api.stats.endpoints import (
        leaguehustlestatsplayer,
        leaguedashplayerptshot,
        leaguedashptdefend,
        boxscorematchupsv3,
        playergamelogs,
    )

    season = "2024-25"

    _probe(
        "LeagueHustleStatsPlayer",
        lambda: leaguehustlestatsplayer.LeagueHustleStatsPlayer(
            season=season, per_mode_time="PerGame",
        ).get_data_frames()[0],
    )
    _probe(
        "LeagueDashPlayerPtShot",
        lambda: leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season=season, per_mode_simple="PerGame",
        ).get_data_frames()[0],
    )
    _probe(
        "LeagueDashPtDefend",
        lambda: leaguedashptdefend.LeagueDashPtDefend(
            season=season, defense_category="Overall",
        ).get_data_frames()[0],
    )
    # Extra measure types on PlayerGameLogs (sample)
    for measure in ("Misc", "Scoring", "Usage"):
        _probe(
            f"PlayerGameLogs_{measure}",
            lambda m=measure: playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
                measure_type_player_game_logs_nullable=m,
            ).get_data_frames()[0].head(500),
        )
    # Matchups: one sample game_id from df if available
    sample_gid = str(df["game_id"].dropna().astype(str).iloc[0]).zfill(10)
    _probe(
        "BoxScoreMatchupsV3",
        lambda: boxscorematchupsv3.BoxScoreMatchupsV3(game_id=sample_gid).get_data_frames()[0],
    )
else:
    print("Skipping endpoint probes (RUN_ENDPOINT_PROBES=False)")

display(probe_inventory)
```

- [ ] **Step 2: Offline sanity** — with flag False, cell should print skip message and empty/prior inventory without errors.

- [ ] **Step 3: Commit**

```bash
git add models/nba/pts_scoring_discovery.ipynb
git commit -m "$(cat <<'EOF'
Add optional nba_api endpoint inventory probes to discovery notebook.

EOF
)"
```

---

### Task 6: Deliverables cell + README pointer

**Files:**
- Modify: `models/nba/pts_scoring_discovery.ipynb`
- Modify: `models/README.md`

**Interfaces:**
- Consumes: all ranking tables + `coverage_df` + `probe_inventory`
- Produces: printed PPM shortlist + worth-adding gaps

- [ ] **Step 1: Add deliverables cell**

```python
print("=" * 72)
print("DELIVERABLES")
print("=" * 72)

shortlists = {}
for target in TARGETS:
    shortlists[target] = merge_driver_shortlist(
        pred_ranks[target],
        shap_df=shap_tables.get(target),
        perm_df=perm_tables.get(target),
        top_k=20,
    )
    print(f"\n### Shortlist → {target}")
    display(shortlists[target])

print("\n### Coverage gaps (missing)")
display(coverage_df.loc[coverage_df["status"] == "missing"])

print("\n### Coverage partial (proxies only)")
display(coverage_df.loc[coverage_df["status"] == "partial"])

print("\n### PPM candidate suggestions (prior_player/team/opp/context intersecting pts_per_min shortlist)")
ppm_suggest = shortlists["pts_per_min"].copy()
# Compare to current PPM_FEATURES for awareness
CURRENT_PPM = {
    "base_pts_per_min_ewm_hl5",
    "base_pts_per_min_season_avg",
    "base_fga_per_min_ewm_hl10",
    "base_fg3a_per_min_ewm_hl10",
    "base_fta_per_min_ewm_hl10",
    "ts_pct_x_usg_pct",
    "adv_poss_ewm_hl5",
    "track_tchs_per_min_ewm_hl10",
    "track_cfga_per_min_ewm_hl10",
    "track_ufga_per_min_ewm_hl10",
    "opp_def_rating_ewm_hl10",
    "opp_pace_ewm_hl10",
    "team_pace_ewm_hl10",
}
ppm_suggest["already_in_ppm"] = ppm_suggest["feature"].isin(CURRENT_PPM)
display(ppm_suggest)

print("\n### Worth adding (from missing wishlist + failed/successful probes)")
worth = coverage_df.loc[coverage_df["status"] == "missing", ["category", "concept"]].copy()
display(worth)
if len(probe_inventory):
    display(probe_inventory.sort_values(["ok", "endpoint"], ascending=[False, True]))

print("\nHoldout diagnostic MAE:", holdout_diag)
print("\nDone. Anatomy ≠ predictive. Do not feed same-game cols into PPM.")
```

- [ ] **Step 2: Update `models/README.md` layout table**

Under the Layout / notebooks table (or Status section), add one line:

```markdown
| Discovery | scoring drivers (`pts`, `pts_per_min`, `minutes`) | `nba/pts_scoring_discovery.ipynb` |
```

Keep the edit minimal — do not rewrite the README.

- [ ] **Step 3: Commit**

```bash
git add models/nba/pts_scoring_discovery.ipynb models/README.md
git commit -m "$(cat <<'EOF'
Add discovery deliverables and README pointer for scoring notebook.

EOF
)"
```

---

### Task 7: Verification

**Files:**
- Read-only checks on notebook + helpers

- [ ] **Step 1: Unit tests still pass**

Run: `python3 -m pytest models/shared/test_scoring_discovery.py -v`  
Expected: PASS

- [ ] **Step 2: Notebook JSON parses**

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('models/nba/pts_scoring_discovery.ipynb')
nb = json.loads(p.read_text())
assert nb.get('nbformat') == 4
sources = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for needle in [
    "pts_scoring_discovery",
    "derive_pts",
    "build_coverage_map",
    "split_feature_pools",
    "RUN_ENDPOINT_PROBES",
    "NOT model features",
    "HOLDOUT_SEASON",
    "merge_driver_shortlist",
]:
    assert needle in sources, needle
# no outputs required; ensure cells cleared preferred
print("ok", len(nb["cells"]), "cells")
PY
```

Expected: `ok N cells`

- [ ] **Step 3: Leakage audit snippet present**

Confirm notebook source contains `same_game ∩ predictive` (or equivalent) and `RUN_ENDPOINT_PROBES = False`.

- [ ] **Step 4: Final commit only if verification fixed anything**; otherwise done.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Wishlist coverage map | Task 3 (+ helpers Task 1) |
| Same-game anatomy for 3 targets | Task 3 |
| Prior-only predictive ranks + per-season stability | Task 4 |
| Model-based SHAP/perm (+ optional corr) | Task 4 |
| Holdout diagnostic only | Task 4 |
| Optional nba_api probes, no upsert | Task 5 |
| Deliverables: shortlist + gaps | Task 6 |
| Parquet + ContextFeatureEngineer load | Task 2 |
| Derive `pts` | Task 1–2 |
| No fetch/silver changes / no artifact save | Global + Tasks 2–6 |
| Offline mode works | Task 5 default False |
| README pointer | Task 6 |

## Self-review notes

- Fixed ambiguity: training parquet has no raw `pts` — plan always derives it.
- Same-game advanced rates (`usg_pct` etc.) are generally **not** on the parquet; anatomy uses available same-game rate cols; coverage marks advanced concepts via prior `adv_*` as available/partial.
- Permutation sampling uses explicit `sample_idx` to avoid misaligned y.
- Probe list locked to concrete endpoints discovered in this repo’s `nba_api` install.
