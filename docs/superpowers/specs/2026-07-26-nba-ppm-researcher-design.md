# NBA PPM researcher notebook

**Date:** 2026-07-26  
**Status:** Approved for planning  
**Notebook:** `models/nba/points/researcher.ipynb`  
**League:** NBA only  
**Approach:** Classical quantitative research (Approach 1) — no model fitting beyond optional descriptive baselines

## Goal

Conduct a complete quantitative investigation of `pts_per_min` before further PPM model development. Understand what drives the target, identify predictive features, quantify relationships, evaluate feature stability, and leave reusable research that makes edits to `models/nba/points/model.ipynb` straightforward.

This notebook is **independent** of `pts_scoring_discovery`. It may overlap findings but must not depend on discovery outputs.

## Decisions (locked)

| Choice | Value |
|--------|-------|
| Primary target | `pts_per_min` only |
| Relation to discovery | Independent full EDA |
| Feature universe | Full-frame quality audit; predictive ranks on **prior-only** features; same-game stats as anatomy only |
| Population | `(minutes >= 5) \| (starting == 1)` |
| Seasons | `2020-21` … `2025-26` |
| Holdout | `2025-26` — temporal/stability diagnostics only; **not** for shortlist decisions |
| Stack | pandas, numpy, scipy, matplotlib, seaborn (no XGB/SHAP/Optuna) |
| Shared modules | None new — notebook-local helpers only |

## Problem definition (Section 1 content)

**Prediction problem:** Pre-tip prediction of a player’s points per minute in an upcoming NBA regular-season game, for the PPM → points prop path.

| Element | Definition |
|---------|------------|
| Target variable | `pts_per_min` (continuous) |
| Observation unit | Player-game row after the minutes/starter filter |
| Prediction unit | One forecast per player per upcoming game (pre-tip) |
| Prediction horizon | Next game only |
| Research success metrics | Association strength + stability (Spearman, mutual information, season rank stability) |
| Modeling success metric (downstream) | MAE on `pts_per_min` (aligned with `models/README.md`); quantile coverage is a counter-metric in the trainer, not this notebook |

### Assumptions (must be stated in-notebook)

1. Regular-season training parquets are the analysis surface; playoffs are out of scope.
2. Row filter matches `model.ipynb`: `(minutes >= 5) | (starting == 1)`.
3. Predictive features are as-of prior games only (EWM / season_avg / lag / roll / context / market when present).
4. “Driver” means association or descriptive contribution, not proven causality.
5. `ContextFeatureEngineer(league="nba")` enrich is part of the load path and matches research/live intent for context fields.
6. Raw box-score `pts` may equal `pts_per_min * minutes` when both exist; the research target remains `pts_per_min`.

### Bias sources (must be flagged)

- Survivor / playing-time bias from the minutes filter (low-minute benches underrepresented unless starting).
- Starter enrichment if `starting == 1` admits short outings.
- Season and rule-environment shifts (e.g. pace, three-point era effects across 2020–26).
- Position / role mix changes within players over time.
- Missingness correlated with role (tracking gaps, late-season call-ups).

### Leakage sources (must be flagged)

- Same-game counting and rate stats (FGA, TS%, usage, touches, etc.) if used as pre-tip features.
- Target-derived or contemporaneous efficiency that embeds the night’s scoring.
- Any post-tip / post-game market closes if present on the frame.
- Using holdout (`2025-26`) to choose the feature shortlist.
- Player or game IDs treated as numeric predictors.

## Data sources

```text
for season in SEASONS:
    read data/processed/{season}_Regular_Season_training_data.parquet
    ContextFeatureEngineer(league="nba").enrich(season_df)
concat → filter → analyze
```

If a season parquet is missing, skip with a clear warning; do not fail the whole notebook. Do not rebuild features from raw in this task.

## Notebook structure (11 sections)

Path: `models/nba/points/researcher.ipynb`

1. **Problem Definition** — markdown: target, units, horizon, metrics, assumptions, bias, leakage  
2. **Dataset Overview** — shape, seasons, dtypes summary, column families, sample rows  
3. **Data Quality Assessment** — per-column profile; missingness viz; suspicious flags  
4. **Target Analysis** — distribution, moments, skew/kurtosis, outliers, hist/box/QQ, seasonality/trend, player-level lag-1 autocorrelation, panel stationarity notes, modeling implications  
5. **Feature Exploration** — prior-only pool by lineage; key family distributions; same-game anatomy called out separately  
6. **Relationship Analysis** — Spearman + MI vs `pts_per_min` (prior-only); collinearity among top candidates; scatter/hex for top drivers  
7. **Segmentation** — starter vs bench; minutes tiers; PPM tiers; team/position if columns exist  
8. **Temporal Analysis** — by season/month; rank stability across seasons; descriptive holdout drift (no shortlist fitting on holdout)  
9. **Feature Engineering Ideas** — ranked hypotheses only (not implemented)  
10. **Modeling Readiness** — leakage audit, recommended shortlist vs avoid list, split/metric reminders for `model.ipynb`  
11. **Conclusions** — executive findings + concrete next steps for `PPM_FEATURES` / trainer

### Cell preamble (before Section 1)

- Imports + project-root resolution matching other `models/nba/*` notebooks  
- Config: `SEASONS`, `HOLDOUT_SEASON = "2025-26"`, filter, `RANDOM_SEED = 42`, `TARGET = "pts_per_min"`  
- Load + enrich + concat + filter + duplicate check on `(game_id, player_id)`

## Methods

### Data quality (every column)

Profile table columns:

- dtype  
- missing %  
- nunique / cardinality band  
- constant / near-constant (e.g. >99% single value)  
- numeric outlier rate (IQR rule)  
- invalid-value heuristics (negatives where impossible; percentages outside plausible [0,1] or [0,100])  
- memory usage  

Also report:

- duplicate player-game keys  
- missingness visualization (bar by column family; optional heatmap on a column sample)  
- suspicious-column flags: high-missing, leakage-risk, ID-like, constant, near-constant, broken-scale  

### Target analysis

- mean, median, variance, std, skew, kurtosis, outlier rate  
- histogram, boxplot, QQ plot  
- mean/median by season and by calendar month  
- player-level lag-1 autocorrelation of `pts_per_min` (sort by player, date)  
- stationarity framed as panel/league-mean shifts across seasons (optional league-daily mean diagnostic); do not treat the stacked frame as one AR series without caveat  
- explicit modeling implications paragraph  

### Feature pools (notebook-local)

| Pool | Rule |
|------|------|
| `predictive` | Prior-only patterns: `ewm_`, `season_avg`, `lag1`, `roll`, plus context (`days_rest`, B2B, starter flags as known pre-tip) and market fields if present |
| `same_game` | Contemporaneous box / tracking / advanced rates without prior suffixes — anatomy only |
| `excluded` | Target(s), IDs, meta, dates-as-raw if unsafe, target-derived |

Every ranked feature gets a lineage label: `prior_player` | `team` | `opponent` | `context` | `market` | `same_game` | `excluded`.

Same-game features never enter Sections 6–8 predictive rankings.

### Relationship / segmentation / temporal

- Spearman + mutual information on prior-only numerics vs `pts_per_min`  
- Pairwise correlation among top shortlist candidates  
- Segments: starter vs bench; minutes tiers; PPM tiers; position/team if available  
- Season rank stability (correlation of feature ranks across seasons)  
- Holdout drift: compare feature/target moments train vs `2025-26` descriptively only  

### Missing values for MI

Document in-notebook (e.g. median fill for numeric predictors after row filter) and apply consistently before mutual information.

## Deliverables

1. Written problem definition (assumptions, bias, leakage)  
2. Full column quality profile + suspicious flags  
3. Target characterization + modeling implications  
4. Prior-only driver ranking (Spearman + MI) with lineage  
5. Segmentation + temporal stability notes (incl. descriptive holdout drift)  
6. Feature-engineering idea list (hypotheses)  
7. Modeling-readiness checklist: shortlist, avoid list, leakage audit  
8. Conclusions with concrete next steps for `model.ipynb`  

## Non-goals

- Training or saving PPM artifacts  
- XGB / SHAP / Optuna  
- Changes to `src/pipeline/*`, silver/gold, or new endpoint ingest  
- Depending on or replacing scoring discovery  
- WNBA  
- Implementing new engineered features in this notebook  

## Verification

- Notebook is valid nbformat JSON and opens in Jupyter  
- Runs with project-root resolution matching other model notebooks  
- Predictive pool ∩ same-game pool = ∅; targets/IDs excluded from predictive ranks  
- Holdout season not used to choose the shortlist  
- Offline-only (parquet + local enrich); no DB writes  
- Stored cell outputs cleared when writing the notebook file  

## Success criteria

A reader can answer:

1. Is the data clean enough to model?  
2. What does `pts_per_min` look like, and what does that imply for loss/metrics?  
3. Which prior-only features stably associate with the target?  
4. Where do segments and seasons disagree?  
5. What should `model.ipynb` / `PPM_FEATURES` try or drop next?
