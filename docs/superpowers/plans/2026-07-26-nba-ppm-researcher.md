# NBA PPM Researcher Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `models/nba/points/researcher.ipynb` as a classical quantitative investigation of `pts_per_min` that produces quality profiles, target analysis, prior-only driver ranks, segmentation/temporal notes, and modeling-readiness conclusions for `model.ipynb`.

**Architecture:** Single self-contained Jupyter notebook with notebook-local helper functions (no new `models/shared/` modules). Load season parquets + `ContextFeatureEngineer`, split feature pools into predictive / same_game / excluded, then run the 11 research sections with tables and plots. Clear all stored outputs when writing the file.

**Tech Stack:** Jupyter (nbformat 4), pandas, numpy, scipy, matplotlib, seaborn, scikit-learn (`mutual_info_regression` only).

**Spec:** `docs/superpowers/specs/2026-07-26-nba-ppm-researcher-design.md`

## Global Constraints

- NBA only; target `pts_per_min` only
- Seasons `["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]`
- Load: `data/processed/{season}_Regular_Season_training_data.parquet` then `ContextFeatureEngineer(league="nba").enrich`
- Skip missing season files with a warning
- Filter: `(minutes >= 5) | (starting == 1)`
- Holdout: `HOLDOUT_SEASON = "2025-26"` — descriptive drift only; do not use for shortlist decisions
- Predictive ranks use prior-only features; same-game is anatomy only
- No XGB / SHAP / Optuna; no production artifact save; no pipeline/ingest changes
- Independent of scoring discovery
- `RANDOM_SEED = 42`
- Clear all stored cell outputs when writing the notebook
- Project-root resolution must work from `models/nba/points/` (climb until `data/` exists)

---

## File map

| File | Responsibility |
|------|----------------|
| `models/nba/points/researcher.ipynb` | Full 11-section PPM research notebook |
| `models/README.md` | Optional one-line pointer under Layout / Prop table if easy; skip if disruptive |

---

### Task 1: Preamble + Problem Definition + load

**Files:**
- Create: `models/nba/points/researcher.ipynb`

**Interfaces:**
- Produces: `df` filtered analysis frame; config constants; pool-split helpers defined for later cells

- [ ] **Step 1: Create notebook with imports + robust project root**

```python
import warnings
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

project_root = Path.cwd().resolve()
while not (project_root / "data").exists() and project_root != project_root.parent:
    project_root = project_root.parent
assert (project_root / "data").exists(), f"Could not find repo root from {Path.cwd()}"
sys.path.insert(0, str(project_root))
os.chdir(project_root)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", context="notebook")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

- [ ] **Step 2: Add Section 1 markdown** stating target, observation/prediction unit, horizon, research vs modeling metrics, assumptions, bias, leakage (copy locked wording from spec).

- [ ] **Step 3: Config + load + filter cell**

```python
from src.pipeline.features.context_features import ContextFeatureEngineer

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
HOLDOUT_SEASON = "2025-26"
TARGET = "pts_per_min"
ID_COLS = ["game_id", "player_id", "team_id", "player_name", "game_date", "season_year", "matchup"]

frames = []
for season in SEASONS:
    path = Path(f"data/processed/{season}_Regular_Season_training_data.parquet")
    if not path.exists():
        print(f"WARNING: missing {path} — skipping")
        continue
    season_df = pd.read_parquet(path)
    season_df = ContextFeatureEngineer(league="nba").enrich(season_df)
    if "pos" in season_df.columns and "position_encoded" not in season_df.columns:
        season_df["position_encoded"] = season_df["pos"].map(
            {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
        )
    frames.append(season_df)

df_raw = pd.concat(frames, ignore_index=True)
n_before = len(df_raw)
df = df_raw[(df_raw["minutes"] >= 5) | (df_raw["starting"] == 1)].copy()
df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
print(f"Loaded {len(frames)} seasons | rows {n_before:,} → {len(df):,} after filter")
print(f"Duplicate (game_id, player_id): {df.duplicated(subset=['game_id','player_id']).sum()}")
```

- [ ] **Step 4: Define notebook-local pool helpers**

```python
PRIOR_TOKENS = ("_ewm_", "_season_avg", "_lag1", "_roll")
CONTEXT_EXACT = {
    "days_rest", "is_back_to_back", "starting", "position_encoded",
    "games_played",  # pre-tip cumulative within season if present
}
MARKET_TOKENS = ("implied", "total", "spread", "line", "odds")
SAME_GAME_EXACT_PREFIXES = ()  # classified by exclusion of prior tokens

def lineage_for(col: str) -> str:
    c = col.lower()
    if col == TARGET or col in {"minutes", "pts"} or col in ID_COLS:
        return "excluded"
    if col in CONTEXT_EXACT or c.startswith("is_") or "rest" in c or "b2b" in c:
        return "context"
    if any(t in c for t in MARKET_TOKENS):
        return "market"
    if c.startswith("opp_"):
        return "opponent"
    if c.startswith("team_"):
        return "team"
    if any(t in c for t in PRIOR_TOKENS) or c.startswith(("base_", "adv_", "track_")) and any(t in c for t in PRIOR_TOKENS):
        return "prior_player"
    if any(t in c for t in PRIOR_TOKENS):
        return "prior_player"
    # same-game rates / raw box without prior suffix
    if c.endswith("_per_min") or c in {
        "fga_per_min", "fg3a_per_min", "fta_per_min", "usg_pct", "ts_pct", "efg_pct"
    }:
        return "same_game"
    if c.startswith(("base_", "adv_", "track_")) and not any(t in c for t in PRIOR_TOKENS):
        return "same_game"
    if pd.api.types.is_numeric_dtype(df[col]) if col in df.columns else False:
        # numeric leftovers: treat cautiously
        return "same_game" if not any(t in c for t in PRIOR_TOKENS) else "prior_player"
    return "excluded"

def split_feature_pools(columns):
    pools = {"predictive": [], "same_game": [], "excluded": []}
    for col in columns:
        lin = lineage_for(col)
        if lin == "excluded" or col == TARGET:
            pools["excluded"].append(col)
        elif lin == "same_game":
            pools["same_game"].append(col)
        elif lin in {"prior_player", "team", "opponent", "context", "market"}:
            pools["predictive"].append(col)
        else:
            pools["excluded"].append(col)
    return pools
```

Fix lineage logic carefully in implementation so `base_*_ewm_*` → prior_player and `fga_per_min` → same_game. Verify with a printed audit: intersection empty.

- [ ] **Step 5: Commit**

```bash
git add models/nba/points/researcher.ipynb
git commit -m "scaffold PPM researcher notebook with load and problem definition"
```

---

### Task 2: Dataset overview + data quality (Sections 2–3)

**Files:**
- Modify: `models/nba/points/researcher.ipynb`

- [ ] **Step 1: Section 2 — overview cells**  
  Print shape, season value counts, dtype counts, column-family counts (`base_`, `adv_`, `track_`, `opp_`, `team_`, context), `df.head()`.

- [ ] **Step 2: Section 3 — `profile_columns(df)` helper + table**  
  For every column compute: dtype, missing_pct, nunique, cardinality_band (`const`/`low`/`med`/`high`), is_constant, is_near_constant (>99% mode), outlier_pct (IQR on numerics), invalid_pct heuristics, memory_bytes.  
  Also total frame memory.  
  Flag suspicious rows into a `flags` column / separate table.

- [ ] **Step 3: Missingness viz**  
  Bar chart of missing % for columns with missing_pct > 0 (top 40). Optional family-level mean missing bar.

- [ ] **Step 4: Commit**

```bash
git add models/nba/points/researcher.ipynb
git commit -m "add dataset overview and column quality profile to PPM researcher"
```

---

### Task 3: Target analysis (Section 4)

**Files:**
- Modify: `models/nba/points/researcher.ipynb`

- [ ] **Step 1: Moments + outlier rate** on `df[TARGET]` (drop NA for stats).

- [ ] **Step 2: Plots** — histogram, boxplot, QQ (`stats.probplot`).

- [ ] **Step 3: Seasonality / trend** — mean/median by `season_year`; by month from `game_date`.

- [ ] **Step 4: Autocorrelation** — within-player lag-1 corr of TARGET (group by player_id, shift). Report mean of per-player corr for players with ≥10 games.

- [ ] **Step 5: Stationarity note (markdown)** — panel framing; optional plot of league-daily mean TARGET over time.

- [ ] **Step 6: Modeling implications markdown** (skew → quantiles/MAE; tails; autocorrelation → prior features; season shifts → holdout).

- [ ] **Step 7: Commit**

```bash
git add models/nba/points/researcher.ipynb
git commit -m "add pts_per_min target analysis to PPM researcher"
```

---

### Task 4: Feature exploration + relationships (Sections 5–6)

**Files:**
- Modify: `models/nba/points/researcher.ipynb`

- [ ] **Step 1: Section 5** — build pools; print counts by lineage; sample descriptive stats for top prior families; markdown banner that same_game is NOT for modeling.

- [ ] **Step 2: Optional same-game anatomy table** — Spearman of same_game numerics vs TARGET (labeled LEAKAGE IF USED PRE-TIP), top 15 only.

- [ ] **Step 3: Section 6 predictive ranks** on pre-holdout rows only for shortlist decisions:

```python
df_rank = df[df["season_year"] != HOLDOUT_SEASON].copy()
# median-fill predictors for MI; document the rule
```

Compute Spearman and MI for each predictive numeric; output sorted table with lineage. Cap MI subsample if needed (e.g. min(n, 20000) with RANDOM_SEED).

- [ ] **Step 4: Collinearity** among top 25 by |Spearman| — corr heatmap or `analyze` table of pairs > 0.9.

- [ ] **Step 5: Scatter/hex** for top 6 drivers vs TARGET.

- [ ] **Step 6: Commit**

```bash
git add models/nba/points/researcher.ipynb
git commit -m "add feature exploration and prior-only relationship ranks"
```

---

### Task 5: Segmentation + temporal (Sections 7–8)

**Files:**
- Modify: `models/nba/points/researcher.ipynb`

- [ ] **Step 1: Segmentation** — group TARGET mean/std/count by `starting`; minutes tiers (`<15`,`15-25`,`25-35`,`35+`); PPM tiers on target; `pos` or `position_encoded` if present.

- [ ] **Step 2: Temporal** — TARGET by season; feature rank stability: for each season in train seasons, rank top features by |Spearman|, compute pairwise Spearman rank-correlation of those ranks across seasons; summarize mean stability.

- [ ] **Step 3: Holdout drift (descriptive)** — compare TARGET mean/std and top-feature means train vs holdout; print clearly as diagnostic only.

- [ ] **Step 4: Commit**

```bash
git add models/nba/points/researcher.ipynb
git commit -m "add segmentation and temporal stability analysis"
```

---

### Task 6: Engineering ideas + readiness + conclusions (Sections 9–11)

**Files:**
- Modify: `models/nba/points/researcher.ipynb`
- Modify: `models/README.md` (optional one-line pointer)

- [ ] **Step 1: Section 9 markdown** — ranked FE ideas grounded in findings (e.g. keep/extend `ts_pct_x_usg_pct`, rest interactions, opponent defensive splits). Hypotheses only.

- [ ] **Step 2: Section 10** — leakage audit print (`predictive ∩ same_game`); recommended shortlist (top stable prior features, exclude near-constants / high-missing); avoid list; remind MAE + holdout rules for `model.ipynb`.

- [ ] **Step 3: Section 11 conclusions** — bullet answers to success criteria; concrete `PPM_FEATURES` suggestions vs current list in `model.ipynb`.

- [ ] **Step 4: Validate notebook JSON**

```bash
python -c "import nbformat; nbformat.read('models/nba/points/researcher.ipynb', as_version=4); print('ok')"
```

- [ ] **Step 5: Smoke-run critical helpers** (optional script or `jupyter execute` if env allows). At minimum: load parquet path exists and pool split unit checks via a tiny inline assert cell.

- [ ] **Step 6: Commit**

```bash
git add models/nba/points/researcher.ipynb models/README.md
git commit -m "finish PPM researcher with readiness checklist and conclusions"
```

---

## Self-review checklist (author)

1. Spec coverage: all 11 sections, quality fields, target plots, prior-only ranks, holdout-not-for-shortlist, non-goals respected.  
2. No TBD/TODO placeholders in plan steps.  
3. Root resolution handles `models/nba/points/` depth.  
4. Outputs cleared in written notebook.
