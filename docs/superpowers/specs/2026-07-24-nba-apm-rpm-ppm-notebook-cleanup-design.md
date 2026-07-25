# NBA APM / RPM / PPM notebook cleanup

**Date:** 2026-07-24  
**Status:** Approved for planning  
**Template:** `models/nba/min_nba_model.ipynb`  
**Targets:** `models/nba/apm_nba_model.ipynb`, `rpm_nba_model.ipynb`, `ppm_nba_model.ipynb`

## Goal

Make APM, RPM, and PPM NBA notebooks structural twins of the cleaned MIN notebook: shared train/eval helpers, naive hypothesis test, complexity ladder, and artifact save — while preserving each prop’s features, filters, XGB params, and rate tiers.

## Non-goals

- WNBA notebooks
- Changing feature sets, hyperparams, or filters (except structural wiring)
- Re-running training / updating experiment log results in this task
- Moving ad-hoc feature engineering into the pipeline (keep notebook-local when required)

## Approach

Rewrite each notebook as a clone of MIN’s cell flow. Swap prop-specific config only. Delete inlined CV/helpers/joblib code in favor of `models.shared.*`.

## Shared structure (matches MIN)

1. **Imports / project root** — lean imports; resolve repo root via `Path` + `data/` existence check; `sys.path` + `chdir`.
2. **Load** — seasons `["2020-21", …, "2025-26"]`; `ContextFeatureEngineer(league="nba").enrich` per season; concat.
3. **Sanity** — duplicate check + describe core rate columns.
4. **Ad-hoc features** (only if final feature list needs them):
   - APM / RPM: `position_encoded` from `pos`
   - RPM: `reb_per_min_roll10` (prior-games rolling mean; drop unused z-score scaffolding)
   - PPM: `ts_pct_x_usg_pct = adv_ts_pct_season_avg * adv_usg_pct_season_avg`
   - Drop: PPM dead FeatureEngineer rebuild cell + markdown; RPM unused `*_vs_season_z` columns
5. **Prop config cell** — features, holdout, IDs, target, role, naive primary/secondary, alpha, quantiles, XGB params, tiers, `ARTIFACT_STEM`, imports from `models.shared.{splits,train,baselines,analysis,artifacts,metrics}`.
6. **Filter** — existing minute/starter gate.
7. **`prepare_splits(...)`** — keep MIN’s `ppm_df` / `ppm_holdout` aliases for drop-in compatibility with shared helpers.
8. **Optional Optuna** — commented `tune_xgb_quantile` reload stub (MIN style), not inline Optuna.
9. **`run_timeseries_cv` + `run_walk_forward`**
10. **Naive hypothesis markdown + `run_naive_comparison`**
11. **Residual visuals** — same plots; axis labels/units use prop target (e.g. AST/min), not “MIN”.
12. **SHAP beeswarm + mean |SHAP| bar**
13. **Permutation importance** via `pinball_50` scorer
14. **Feature ablation** via `run_feature_ablation`
15. **Feature audit table** (SHAP + perm + ablation consensus)
16. **`analyze_correlations`**
17. **`evaluate_holdout`**
18. **`evaluate_holdout_vs_naive`**
19. **Complexity ladder markdown + `run_quantile_linear_baseline`**
20. **`save_model_bundle` / `load_model_bundle` / `predict_quantiles`**

Clear all stored cell outputs on rewrite.

## Prop-specific config (preserve)

### APM (`apm_nba_model`)

| Field | Value |
|---|---|
| Feature list name | `APM_FEATURES` |
| Features | `base_ast_per_min_ewm_hl10`, `track_pass_per_min_ewm_hl10`, `adv_ast_pct_ewm_hl10`, `position_encoded`, `adv_poss_ewm_hl10` |
| Target | `ast_per_min` |
| Filter | `(minutes >= 10) \| (starting == 1)` |
| Naive primary / secondary | `base_ast_per_min_season_avg` / `base_ast_per_min_lag1` |
| Tiers | `<0.10`, `0.10–0.20`, `0.20–0.35`, `0.35+` ast/min |
| XGB | existing APM params (`n_estimators=1553`, …) |
| Artifact stem | `apm_nba_model` |

### RPM (`rpm_nba_model`)

| Field | Value |
|---|---|
| Feature list name | `RPM_FEATURES` |
| Features | `base_reb_per_min_season_avg`, `adv_reb_pct_season_avg`, `track_rbc_per_min_ewm_hl10`, `position_encoded`, `adv_reb_pct_ewm_hl10`, `base_reb_per_min_ewm_hl10`, `track_orbc_per_min_ewm_hl10`, `opp_pace_ewm_hl10`, `track_drbc_per_min_ewm_hl10`, `reb_per_min_roll10` |
| Target | `reb_per_min` |
| Filter | `(minutes >= 10) \| (starting == 1)` |
| Naive primary / secondary | `base_reb_per_min_season_avg` / `base_reb_per_min_lag1` |
| Tiers | `<0.15`, `0.15–0.25`, `0.25–0.40`, `0.40+` reb/min |
| XGB | existing RPM params (`n_estimators=1655`, …) |
| Artifact stem | `rpm_nba_model` |

### PPM (`ppm_nba_model`)

| Field | Value |
|---|---|
| Feature list name | `PPM_FEATURES` (ordered list; drop `list(set(...))`) |
| Features | current 13-feature set including `ts_pct_x_usg_pct` |
| Target | `pts_per_min` |
| Filter | `(minutes >= 5) \| (starting == 1)` |
| Naive primary / secondary | `base_pts_per_min_season_avg` / `base_pts_per_min_lag1` |
| Tiers | `<0.3`, `0.3–0.5`, `0.5–0.7`, `0.7+` pts/min |
| XGB | existing PPM params (`n_estimators=1300`, …) |
| Artifact stem | `ppm_nba_model` |

Common: `HOLDOUT_SEASON = "2025-26"`, `ID_COLS` / `ROLE_COL` / `QUANTILES` / `ALPHA = 0.05` same as MIN.

## Naming / plot labels

- Rename misuse of `MIN_FEATURES` / `MIN_TIERS` in APM/RPM to `APM_*` / `RPM_*` (or `*_TIERS`).
- Residual plot titles/labels/units use the prop rate (AST/min, REB/min, PTS/min), not minutes.

## Hypothesis wiring

Wire the same decision rule as MIN / `models/docs/hypothesis.txt`:

- Primary naive = season avg; secondary = lag1 (report only).
- Walk-forward last fold diagnostic; holdout authoritative.
- Do **not** freeze dates or mark H0 decided in docs as part of this cleanup (structure only).

Optional follow-up (out of scope): update `hypothesis.txt` PPM primary from TBD → `base_pts_per_min_season_avg` once frozen.

## Verification

- Notebooks parse as valid JSON / open in Jupyter.
- Cell order and shared imports match MIN (diff structure, not outputs).
- No remaining inlined `fit_quantile_models` / `score_fold` / walk-forward loops / raw `joblib.dump`.
- Feature lists and XGB params unchanged vs pre-cleanup sources.
- Smoke: import shared modules from a kernel with cwd at repo root (no full train required for this task).

## Success criteria

Someone who knows MIN can open APM/RPM/PPM and navigate the same cells with only the config/ad-hoc blocks differing by prop.
