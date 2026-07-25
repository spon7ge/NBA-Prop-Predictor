# NBA APM/RPM/PPM Notebook Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `apm_nba_model.ipynb`, `rpm_nba_model.ipynb`, and `ppm_nba_model.ipynb` as structural twins of `min_nba_model.ipynb`, using `models.shared.*` helpers while preserving each prop’s features, filters, XGB params, and rate tiers.

**Architecture:** Treat MIN as the template. Add a small structure-check script that asserts shared-helper usage and forbids inlined training loops. Rewrite each rate notebook cell-by-cell from that template, inserting only the prop-specific ad-hoc feature + config + filter cells. Clear all outputs.

**Tech Stack:** Jupyter notebooks (nbformat 4), pandas, XGBoost quantile models via `models.shared`, pytest for the structure checker.

**Spec:** `docs/superpowers/specs/2026-07-24-nba-apm-rpm-ppm-notebook-cleanup-design.md`

## Global Constraints

- Seasons loaded: `["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]`
- `ContextFeatureEngineer(league="nba")` on every season load
- Holdout: `HOLDOUT_SEASON = "2025-26"`
- Naive: primary `base_{stat}_season_avg`, secondary `base_{stat}_lag1`
- Keep `ppm_df` / `ppm_holdout` aliases from `prepare_splits` (MIN compatibility)
- Do not change feature membership, XGB hyperparams, or minute filters vs current notebooks
- Do not update `hypothesis.txt` freeze/decision dates in this plan
- Do not touch WNBA notebooks
- Clear all stored cell outputs on rewrite
- No inlined `fit_quantile_models`, `score_fold`, walk-forward loops, Optuna study bodies, or raw `joblib.dump`

---

## File map

| File | Responsibility |
|---|---|
| `models/nba/min_nba_model.ipynb` | Read-only template |
| `models/nba/apm_nba_model.ipynb` | Rewrite — assists/min |
| `models/nba/rpm_nba_model.ipynb` | Rewrite — rebounds/min |
| `models/nba/ppm_nba_model.ipynb` | Rewrite — points/min |
| `models/shared/test_notebook_structure.py` | Structure assertions for cleaned NBA notebooks |
| `models.shared.{splits,train,baselines,analysis,artifacts,metrics}` | Unchanged — consumed by notebooks |

---

### Task 1: Structure checker (failing tests first)

**Files:**
- Create: `models/shared/test_notebook_structure.py`
- Test: `models/shared/test_notebook_structure.py`

**Interfaces:**
- Consumes: notebook JSON under `models/nba/*_nba_model.ipynb`
- Produces: pytest cases that rate notebooks must pass after rewrite; MIN must already pass the shared-helper checks

- [ ] **Step 1: Write the failing structure tests**

```python
"""Structural checks for cleaned NBA prop notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NBA_DIR = Path(__file__).resolve().parents[1] / "nba"
MIN_NB = NBA_DIR / "min_nba_model.ipynb"
RATE_NBS = [
    NBA_DIR / "apm_nba_model.ipynb",
    NBA_DIR / "rpm_nba_model.ipynb",
    NBA_DIR / "ppm_nba_model.ipynb",
]

REQUIRED_IMPORT_SNIPPETS = [
    "from models.shared.splits import prepare_splits",
    "from models.shared.train import run_timeseries_cv, run_walk_forward, evaluate_holdout",
    "from models.shared.baselines import (",
    "run_naive_comparison",
    "evaluate_holdout_vs_naive",
    "run_quantile_linear_baseline",
    "from models.shared.analysis import run_feature_ablation, analyze_correlations",
    "from models.shared.artifacts import save_model_bundle, load_model_bundle, predict_quantiles",
    "from models.shared.metrics import pinball_50",
]

FORBIDDEN_SNIPPETS = [
    "def fit_quantile_models",
    "def score_fold",
    "joblib.dump",
    "optuna.create_study",
    "TimeSeriesSplit(",
]

REQUIRED_SEASON = 'seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]'
REQUIRED_ENGINEER = 'ContextFeatureEngineer(league="nba")'


def _sources(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def _assert_shared_pipeline(path: Path) -> None:
    src = _sources(path)
    assert REQUIRED_SEASON in src, f"{path.name}: missing full season list"
    assert REQUIRED_ENGINEER in src, f"{path.name}: missing league='nba' engineer"
    for snippet in REQUIRED_IMPORT_SNIPPETS:
        assert snippet in src, f"{path.name}: missing `{snippet}`"
    for snippet in FORBIDDEN_SNIPPETS:
        assert snippet not in src, f"{path.name}: still contains `{snippet}`"
    assert "save_model_bundle(" in src
    assert "run_quantile_linear_baseline(" in src


def _assert_cleared_outputs(path: Path) -> None:
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == [], f"{path.name} cell {i} has outputs"
            assert cell.get("execution_count") is None, f"{path.name} cell {i} has execution_count"


def test_min_notebook_is_cleaned_template():
    # MIN may retain run outputs; only require shared-pipeline structure.
    _assert_shared_pipeline(MIN_NB)


@pytest.mark.parametrize("path", RATE_NBS, ids=lambda p: p.name)
def test_rate_notebook_matches_min_structure(path: Path):
    _assert_shared_pipeline(path)
    _assert_cleared_outputs(path)


def test_apm_preserves_features_and_naive():
    src = _sources(NBA_DIR / "apm_nba_model.ipynb")
    for feat in [
        "base_ast_per_min_ewm_hl10",
        "track_pass_per_min_ewm_hl10",
        "adv_ast_pct_ewm_hl10",
        "position_encoded",
        "adv_poss_ewm_hl10",
    ]:
        assert feat in src
    assert 'TARGET_COL = "ast_per_min"' in src
    assert 'NAIVE_PRIMARY = "base_ast_per_min_season_avg"' in src
    assert 'NAIVE_SECONDARY = "base_ast_per_min_lag1"' in src
    assert 'ARTIFACT_STEM = "apm_nba_model"' in src
    assert "1553" in src  # preserved n_estimators
    assert "minutes" in src and ">= 10" in src


def test_rpm_preserves_features_and_naive():
    src = _sources(NBA_DIR / "rpm_nba_model.ipynb")
    for feat in [
        "base_reb_per_min_season_avg",
        "adv_reb_pct_season_avg",
        "track_rbc_per_min_ewm_hl10",
        "position_encoded",
        "reb_per_min_roll10",
    ]:
        assert feat in src
    assert 'TARGET_COL = "reb_per_min"' in src
    assert 'NAIVE_PRIMARY = "base_reb_per_min_season_avg"' in src
    assert "ARTIFACT_STEM = \"rpm_nba_model\"" in src
    assert "reb_per_min_roll10_vs_season_z" not in src  # unused z-score dropped


def test_ppm_preserves_features_and_naive():
    src = _sources(NBA_DIR / "ppm_nba_model.ipynb")
    assert "ts_pct_x_usg_pct" in src
    assert 'TARGET_COL = "pts_per_min"' in src
    assert 'NAIVE_PRIMARY = "base_pts_per_min_season_avg"' in src
    assert "ARTIFACT_STEM = \"ppm_nba_model\"" in src
    assert "FeatureEngineer" not in src  # dead rebuild scaffolding gone
    assert "list(set(" not in src
```

- [ ] **Step 2: Run tests — expect MIN pass, rate notebooks fail**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
python -m pytest models/shared/test_notebook_structure.py -v
```

Expected: `test_min_notebook_is_cleaned_template` PASS; rate-notebook tests FAIL (missing shared imports / still contain forbidden snippets).

- [ ] **Step 3: Commit checker**

```bash
git add models/shared/test_notebook_structure.py
git commit -m "$(cat <<'EOF'
Add structure tests for cleaned NBA prop notebooks.

EOF
)"
```

---

### Task 2: Rewrite APM notebook

**Files:**
- Modify: `models/nba/apm_nba_model.ipynb` (full rewrite)
- Test: `models/shared/test_notebook_structure.py`

**Interfaces:**
- Consumes: MIN cell flow; APM config from spec
- Produces: cleaned `apm_nba_model.ipynb` with empty outputs

- [ ] **Step 1: Build notebook JSON from MIN skeleton**

Use Python (do not hand-edit giant JSON). Read MIN, clear outputs, then replace / insert prop-specific cells.

```python
import json
import copy
from pathlib import Path

ROOT = Path("/Users/alexgonzalez/Documents/NBA-Prop-Predictor")
min_nb = json.loads((ROOT / "models/nba/min_nba_model.ipynb").read_text())


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")[:-1]]
        + ([source.split("\n")[-1] + "\n"] if source.split("\n")[-1] != "" else [])
        if source.endswith("\n") or True
        else [source],
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source if source.endswith("\n") else source + "\n"]}


def clear_nb(nb: dict) -> dict:
    out = copy.deepcopy(nb)
    for c in out["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return out
```

Prefer writing sources as lists of lines ending in `\n` (nbformat style), e.g.:

```python
def src_lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)
```

- [ ] **Step 2: Write APM-specific cells (exact content)**

**Cell 0 — imports** (copy MIN cell 0 verbatim).

**Cell 1 — load** (copy MIN cell 1 verbatim — already has `league="nba"` and full seasons).

**Cell 2 — sanity** (copy MIN cell 2 verbatim).

**Cell 3 — ad-hoc `position_encoded`:**

```python
df["position_encoded"] = df["pos"].map({"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5})
print(f"position_encoded filled: {df['position_encoded'].notna().sum():,} / {len(df):,}")
```

**Cell 4 — prop config:**

```python
# ── Prop-specific config (NBA APM) ────────────────────────────────────────────
APM_FEATURES = [
    "base_ast_per_min_ewm_hl10",
    "track_pass_per_min_ewm_hl10",
    "adv_ast_pct_ewm_hl10",
    "position_encoded",
    "adv_poss_ewm_hl10",
]

HOLDOUT_SEASON = "2025-26"
ID_COLS = ["game_id", "player_id", "season_year", "player_name", "game_date"]
TARGET_COL = "ast_per_min"
ROLE_COL = "starting"

NAIVE_PRIMARY = "base_ast_per_min_season_avg"
NAIVE_SECONDARY = "base_ast_per_min_lag1"
NAIVE_COLS = [NAIVE_PRIMARY, NAIVE_SECONDARY]
ALPHA = 0.05

QUANTILES = [0.10, 0.50, 0.90]

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    n_estimators=1553,
    max_depth=4,
    learning_rate=0.016209908567987385,
    subsample=0.6707418178831205,
    colsample_bytree=0.5074990705838847,
    reg_alpha=0.33354531117582426,
    reg_lambda=0.06707791013177704,
    min_child_weight=6,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50,
)

APM_TIERS = {
    "<0.10 ast/min": lambda a: a < 0.10,
    "0.10-0.20 ast/min": lambda a: (a >= 0.10) & (a < 0.20),
    "0.20-0.35 ast/min": lambda a: (a >= 0.20) & (a < 0.35),
    "0.35+ ast/min": lambda a: a >= 0.35,
}

ARTIFACT_STEM = "apm_nba_model"

from models.shared.splits import prepare_splits
from models.shared.train import run_timeseries_cv, run_walk_forward, evaluate_holdout
from models.shared.baselines import (
    run_naive_comparison,
    evaluate_holdout_vs_naive,
    run_quantile_linear_baseline,
)
from models.shared.analysis import run_feature_ablation, analyze_correlations
from models.shared.artifacts import save_model_bundle, load_model_bundle, predict_quantiles
from models.shared.metrics import pinball_50

print(f"Number of features: {len(APM_FEATURES)}")
APM_FEATURES
```

**Cell 5 — filter:**

```python
df = df[(df["minutes"] >= 10) | (df["starting"] == 1)]
print(f"After filtering minutes: {df[APM_FEATURES].shape[0]} rows")
```

**Cell 6 — splits** (MIN cell 5, rename `MIN_FEATURES` → `APM_FEATURES`):

```python
splits = prepare_splits(
    df,
    holdout_season=HOLDOUT_SEASON,
    features=APM_FEATURES,
    target_col=TARGET_COL,
    naive_primary=NAIVE_PRIMARY,
    naive_secondary=NAIVE_SECONDARY,
    id_cols=ID_COLS,
    role_col=ROLE_COL,
    extra_keep=["minutes"],
)
ppm_df = splits["ppm_df"]
ppm_holdout = splits["ppm_holdout"]
X = splits["X"]
y = splits["y"]
```

**Cell 7 — tune stub** (copy MIN cell 6 verbatim).

**Cell 8 — CV + WF** (MIN cell 7 with `APM_FEATURES`/`APM_TIERS`):

```python
tscv_results = run_timeseries_cv(
    X, y, ppm_df,
    xgb_params=XGB_PARAMS,
    role_col=ROLE_COL,
    tiers=APM_TIERS,
    quantiles=QUANTILES,
)
wf = run_walk_forward(
    X, y, ppm_df,
    xgb_params=XGB_PARAMS,
    role_col=ROLE_COL,
    tiers=APM_TIERS,
    quantiles=QUANTILES,
)

wf_results = wf["wf_results"]
models_last = wf["models_last"]
preds_last = wf["preds_last"]
X_val_last = wf["X_val_last"]
y_val_last = wf["y_val_last"]
starting_last = wf["starting_last"]
last_fold = wf["last_fold"]
```

**Cell 9 — naive markdown:**

```markdown
### Naive baseline — APM hypothesis test

**H0:** The assists/min model does not predict significantly better than a naive baseline.  
**H1:** The assists/min model predicts significantly better than naive.

**Frozen primary naive:** `base_ast_per_min_season_avg` (season-to-date expanding mean, shift-then-expand — no same-game leakage).  
**Secondary (reported only):** `base_ast_per_min_lag1`.

**Decision rule:** on the same player-games, reject H0 if model MAE < naive MAE **and** one-sided Wilcoxon signed-rank on `|y − ŷ|` has p < 0.05 (model errors stochastically smaller).
```

**Cells 10–22:** Copy MIN cells 9–22 with these substitutions everywhere:
- `MIN_FEATURES` → `APM_FEATURES`
- `MIN_TIERS` → `APM_TIERS`
- residual plot unit `"min"` → `"AST/min"`
- axis labels `Actual MIN` / `Predicted MIN` → `Actual AST/min` / `Predicted AST/min`
- correlation title `NBA MIN` → `NBA APM`
- markdown “minutes model” / “MIN hypothesis” already replaced in cell 9; complexity ladder markdown should say APM / `APM_FEATURES`

Critical residual-plot label edits (from MIN residual cell):

```python
ax.set_title(f"{name}  |  R²={r2:.3f}  MAE={mae:.2f} AST/min")
ax.set_xlabel("Actual AST/min")
ax.set_ylabel("Predicted AST/min")
# ...
print(f"  {name}: R²={m['R2']:+.3f}  MAE={m['MAE']:.3f} AST/min")
# interval width prints: use "AST/min" instead of "min"
# residual summary: use "AST/min"
# Residuals by minutes tier → "Residuals by rate tier:"
```

SHAP mean bar uses `index=APM_FEATURES`. Feature audit uses `APM_FEATURES`. Holdout / linear / save cells use `APM_FEATURES` and `ARTIFACT_STEM`.

- [ ] **Step 3: Write notebook file with empty outputs + MIN metadata**

```python
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": min_nb["metadata"],
    "cells": [/* ordered cells above */],
}
(ROOT / "models/nba/apm_nba_model.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
```

- [ ] **Step 4: Run APM-focused tests**

```bash
python -m pytest models/shared/test_notebook_structure.py::test_apm_preserves_features_and_naive models/shared/test_notebook_structure.py::test_rate_notebook_matches_min_structure -k apm -v
```

Expected: APM structural + preserve tests PASS. (RPM/PPM may still fail.)

Also sanity-check no forbidden strings:

```bash
rg -n "fit_quantile_models|score_fold|joblib\.dump|optuna\.create_study" models/nba/apm_nba_model.ipynb
```

Expected: no matches.

- [ ] **Step 5: Commit APM**

```bash
git add models/nba/apm_nba_model.ipynb
git commit -m "$(cat <<'EOF'
Rewrite NBA APM notebook to match cleaned MIN shared pipeline.

EOF
)"
```

---

### Task 3: Rewrite RPM notebook

**Files:**
- Modify: `models/nba/rpm_nba_model.ipynb` (full rewrite)
- Test: `models/shared/test_notebook_structure.py`

**Interfaces:**
- Consumes: same MIN skeleton as Task 2
- Produces: cleaned `rpm_nba_model.ipynb`

- [ ] **Step 1: Clone Task 2 pattern with RPM cells**

Cells 0–2: same as APM/MIN.

**Cell 3 — `position_encoded`:** same as APM.

**Cell 4 — `reb_per_min_roll10` only (no z-score):**

```python
# Prior-games rolling mean of reb_per_min (player × season), leakage-safe via shift(1)
STAT, WINDOW = "reb_per_min", 10
season_col = "season_year" if "season_year" in df.columns else "season"

df = df.sort_values(["player_id", season_col, "game_date"]).reset_index(drop=True)
shifted = df.groupby(["player_id", season_col], sort=False)[STAT].shift(1)
g = shifted.groupby([df["player_id"], df[season_col]], sort=False)

df[f"{STAT}_roll{WINDOW}"] = (
    g.rolling(WINDOW, min_periods=max(2, WINDOW // 2))
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

print(
    f"{STAT}_roll{WINDOW} defined on "
    f"{df[f'{STAT}_roll{WINDOW}'].notna().sum():,} / {len(df):,} rows"
)
```

**Cell 5 — config:**

```python
# ── Prop-specific config (NBA RPM) ────────────────────────────────────────────
RPM_FEATURES = [
    "base_reb_per_min_season_avg",
    "adv_reb_pct_season_avg",
    "track_rbc_per_min_ewm_hl10",
    "position_encoded",
    "adv_reb_pct_ewm_hl10",
    "base_reb_per_min_ewm_hl10",
    "track_orbc_per_min_ewm_hl10",
    "opp_pace_ewm_hl10",
    "track_drbc_per_min_ewm_hl10",
    "reb_per_min_roll10",
]

HOLDOUT_SEASON = "2025-26"
ID_COLS = ["game_id", "player_id", "season_year", "player_name", "game_date"]
TARGET_COL = "reb_per_min"
ROLE_COL = "starting"

NAIVE_PRIMARY = "base_reb_per_min_season_avg"
NAIVE_SECONDARY = "base_reb_per_min_lag1"
NAIVE_COLS = [NAIVE_PRIMARY, NAIVE_SECONDARY]
ALPHA = 0.05

QUANTILES = [0.10, 0.50, 0.90]

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    n_estimators=1655,
    max_depth=4,
    learning_rate=0.03703121107469649,
    subsample=0.548428652232707,
    colsample_bytree=0.8325873147129733,
    reg_alpha=0.0059881368676087034,
    reg_lambda=0.050446181208082655,
    min_child_weight=4,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50,
)

RPM_TIERS = {
    "<0.15 reb/min": lambda a: a < 0.15,
    "0.15-0.25 reb/min": lambda a: (a >= 0.15) & (a < 0.25),
    "0.25-0.40 reb/min": lambda a: (a >= 0.25) & (a < 0.40),
    "0.40+ reb/min": lambda a: a >= 0.40,
}

ARTIFACT_STEM = "rpm_nba_model"

from models.shared.splits import prepare_splits
from models.shared.train import run_timeseries_cv, run_walk_forward, evaluate_holdout
from models.shared.baselines import (
    run_naive_comparison,
    evaluate_holdout_vs_naive,
    run_quantile_linear_baseline,
)
from models.shared.analysis import run_feature_ablation, analyze_correlations
from models.shared.artifacts import save_model_bundle, load_model_bundle, predict_quantiles
from models.shared.metrics import pinball_50

print(f"Number of features: {len(RPM_FEATURES)}")
RPM_FEATURES
```

**Cell 6 — filter:**

```python
df = df[(df["minutes"] >= 10) | (df["starting"] == 1)]
print(f"After filtering minutes: {df[RPM_FEATURES].shape[0]} rows")
```

Remaining cells: same as APM/MIN with `RPM_FEATURES`, `RPM_TIERS`, unit `REB/min`, title `NBA RPM`, hypothesis markdown for rebounds/min.

- [ ] **Step 2: Write file, clear outputs**

Same JSON write pattern as Task 2 → `models/nba/rpm_nba_model.ipynb`.

- [ ] **Step 3: Run RPM tests**

```bash
python -m pytest models/shared/test_notebook_structure.py -k rpm -v
rg -n "fit_quantile_models|score_fold|joblib\.dump|reb_per_min_roll10_vs_season_z" models/nba/rpm_nba_model.ipynb
```

Expected: RPM tests PASS; no forbidden / z-score matches.

- [ ] **Step 4: Commit RPM**

```bash
git add models/nba/rpm_nba_model.ipynb
git commit -m "$(cat <<'EOF'
Rewrite NBA RPM notebook to match cleaned MIN shared pipeline.

EOF
)"
```

---

### Task 4: Rewrite PPM notebook

**Files:**
- Modify: `models/nba/ppm_nba_model.ipynb` (full rewrite)
- Test: `models/shared/test_notebook_structure.py`

**Interfaces:**
- Consumes: MIN skeleton; PPM config from spec
- Produces: cleaned `ppm_nba_model.ipynb`

- [ ] **Step 1: Cells 0–2 same as MIN**

**Cell 3 — interaction feature:**

```python
df["ts_pct_x_usg_pct"] = df["adv_ts_pct_season_avg"] * df["adv_usg_pct_season_avg"]
```

**No** `position_encoded` (not in `PPM_FEATURES`). **No** FeatureEngineer rebuild cell/markdown.

**Cell 4 — config** (ordered list, no `list(set(...))`):

```python
# ── Prop-specific config (NBA PPM) ────────────────────────────────────────────
PPM_FEATURES = [
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
]

HOLDOUT_SEASON = "2025-26"
ID_COLS = ["game_id", "player_id", "season_year", "player_name", "game_date"]
TARGET_COL = "pts_per_min"
ROLE_COL = "starting"

NAIVE_PRIMARY = "base_pts_per_min_season_avg"
NAIVE_SECONDARY = "base_pts_per_min_lag1"
NAIVE_COLS = [NAIVE_PRIMARY, NAIVE_SECONDARY]
ALPHA = 0.05

QUANTILES = [0.10, 0.50, 0.90]

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    n_estimators=1300,
    max_depth=4,
    learning_rate=0.014326702971361573,
    subsample=0.6865962871949852,
    colsample_bytree=0.5990331287465477,
    reg_alpha=0.11598736995372527,
    reg_lambda=0.48362325914167376,
    min_child_weight=6,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=50,
)

PPM_TIERS = {
    "<0.3 pts/min": lambda a: a < 0.3,
    "0.3-0.5 pts/min": lambda a: (a >= 0.3) & (a < 0.5),
    "0.5-0.7 pts/min": lambda a: (a >= 0.5) & (a < 0.7),
    "0.7+ pts/min": lambda a: a >= 0.7,
}

ARTIFACT_STEM = "ppm_nba_model"

from models.shared.splits import prepare_splits
from models.shared.train import run_timeseries_cv, run_walk_forward, evaluate_holdout
from models.shared.baselines import (
    run_naive_comparison,
    evaluate_holdout_vs_naive,
    run_quantile_linear_baseline,
)
from models.shared.analysis import run_feature_ablation, analyze_correlations
from models.shared.artifacts import save_model_bundle, load_model_bundle, predict_quantiles
from models.shared.metrics import pinball_50

print(f"Number of features: {len(PPM_FEATURES)}")
PPM_FEATURES
```

**Cell 5 — filter (`minutes >= 5`):**

```python
df = df[(df["minutes"] >= 5) | (df["starting"] == 1)]
print(f"After filtering minutes: {df[PPM_FEATURES].shape[0]} rows")
```

Remaining cells: MIN twins with `PPM_FEATURES`, `PPM_TIERS`, unit `PTS/min`, title `NBA PPM`.

- [ ] **Step 2: Write file + run tests**

```bash
python -m pytest models/shared/test_notebook_structure.py -k ppm -v
rg -n "FeatureEngineer|fit_quantile_models|joblib\.dump|list\(set\(" models/nba/ppm_nba_model.ipynb
```

Expected: PPM tests PASS; no matches for dead scaffolding / forbidden helpers.

- [ ] **Step 3: Commit PPM**

```bash
git add models/nba/ppm_nba_model.ipynb
git commit -m "$(cat <<'EOF'
Rewrite NBA PPM notebook to match cleaned MIN shared pipeline.

EOF
)"
```

---

### Task 5: Full verification + smoke import

**Files:**
- Verify only (no product changes unless a test fails)

- [ ] **Step 1: Run full structure suite**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
python -m pytest models/shared/test_notebook_structure.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Shared-module smoke import**

```bash
python - <<'PY'
from models.shared.splits import prepare_splits
from models.shared.train import run_timeseries_cv, run_walk_forward, evaluate_holdout
from models.shared.baselines import (
    run_naive_comparison,
    evaluate_holdout_vs_naive,
    run_quantile_linear_baseline,
)
from models.shared.analysis import run_feature_ablation, analyze_correlations
from models.shared.artifacts import save_model_bundle, load_model_bundle, predict_quantiles
from models.shared.metrics import pinball_50
print("shared imports ok")
PY
```

Expected: `shared imports ok`

- [ ] **Step 3: Diff-style sanity — shared markers present in all three**

```bash
for f in apm rpm ppm; do
  echo "== $f =="
  rg -n "prepare_splits|run_timeseries_cv|run_naive_comparison|run_quantile_linear_baseline|save_model_bundle" \
    "models/nba/${f}_nba_model.ipynb" | head
done
```

Expected: each file shows those markers.

- [ ] **Step 4: Final commit only if Step 1–3 required tiny fixes; otherwise done**

If fixes were needed:

```bash
git add models/nba/*_nba_model.ipynb models/shared/test_notebook_structure.py
git commit -m "$(cat <<'EOF'
Fix leftover structure gaps in cleaned NBA rate notebooks.

EOF
)"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Full MIN twin structure + shared helpers | 2–4 |
| Seasons 2020-21…2025-26 + `league="nba"` | 1 (assert), 2–4 |
| Keep features / filters / XGB / tiers | 1 (assert), 2–4 configs |
| Ad-hoc only if needed; drop dead scaffolding | 2–4 |
| Naive season_avg + lag1 | 2–4 |
| Hypothesis + ladder + artifacts | 2–4 |
| Clear outputs | 1 assert + 2–4 write |
| No WNBA / no hypothesis freeze update | Global constraints |
| Verification without full train | Task 5 |

No placeholders remain. Feature/XGB/tier values copied from current notebooks and the approved spec.
