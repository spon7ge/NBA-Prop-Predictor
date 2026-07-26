# NBA points scoring discovery notebook

**Date:** 2026-07-26  
**Status:** Approved for planning  
**Notebook:** `models/nba/pts_scoring_discovery.ipynb`  
**League:** NBA only (WNBA clone later if useful)

## Goal

Identify every available (and missing) source of information that helps explain why a player scores points, using existing pipeline endpoints plus lightweight probes of additional `nba_api` surfaces.

Deliverables from one notebook run:

1. Wishlist coverage map (`available` / `partial` / `missing`)
2. Same-game scoring anatomy rankings for `pts`, `pts_per_min`, and `minutes`
3. Leakage-safe predictive driver rankings (overall + per-season stability)
4. Candidate feature shortlist for PPM
5. Prioritized list of endpoints / data sources worth adding

## Decisions (locked)

| Choice | Value |
|--------|-------|
| Scope | Current engineered data **plus** live `nba_api` probes of missing surfaces |
| Targets | Dual: `pts` and `pts_per_min`; `minutes` analyzed as its own driver |
| History | Seasons `2020-21` … `2025-26` with per-season breakdown |
| Analysis layers | Both: same-game anatomy **and** prior-only predictive discovery |
| Approach | Layered discovery (coverage + anatomy + predictive + probes + shortlist) |

## Non-goals

- Training or saving a production PPM / points model artifact
- Wiring new endpoints into `src/pipeline/fetch.py`, silver, or gold
- Full injury usage-redistribution modeling
- Live steam / opening–closing line movement pipelines
- Primary-defender matchup feature engineering beyond inventory + stubs
- WNBA

## Data sources

### Already in pipeline (primary analysis surface)

Load via the same path as the MIN notebook:

```text
for season in SEASONS:
    read data/processed/{season}_Regular_Season_training_data.parquet
    ContextFeatureEngineer(league="nba").enrich(season_df)
concat → filter → analyze
```

If a season parquet is missing, skip it with a clear warning (do not fail the whole notebook). Do not rebuild features from raw in this discovery task unless parquet load fails for all seasons and a fallback is explicitly needed.

Underlying raw tables / feature families:

| Family | Source | Examples relevant to scoring |
|--------|--------|------------------------------|
| Player base | `PlayerGameLogs` Base | `min`, `pts`, `fga`, `fg3a`, `fta`, shooting % |
| Player advanced | `PlayerGameLogs` Advanced | `usg_pct`, `ts_pct`, `efg_pct`, `off_rating`, `ast_pct`, `tov`-related, `pace`, `poss` |
| Tracking | `BoxScorePlayerTrackV3` | touches, passes, contested/uncontested/defended FGA, rebound chances, speed/distance |
| Team / opp | Team Base + Advanced | pace, ratings, shooting volume, team efficiency |
| Context | `ContextFeatureEngineer` | `days_rest`, `is_back_to_back`, starter flags, trends/ranks where present |
| Market (when present on frame) | Rotowire / odds enrichment | game total, team implied total / line fields if joined |

### Wishlist coverage map

The notebook maintains an explicit checklist grouped by the discovery brief:

- Player traditional / advanced / pace-adjusted
- Tracking / shot creation proxies
- Team context
- Opponent defense (beyond raw DEF_RATING)
- Individual matchups
- Vegas information
- Injury / usage redistribution
- Rest / schedule
- Coaching / rotation

Each item is labeled `available`, `partial`, or `missing` based on **actual columns present after load**, not aspirational names. Partial means a proxy exists (e.g. contested FGA ≈ contest proxy) but the exact concept is not ingested.

### Endpoint probes (discovery only)

Optional cells pull **small samples** from candidate `nba_api` endpoints not yet in `GameLogs` (examples to try; exact set chosen at implementation by what `nba_api` exposes and what returns data):

- Hustle / defensive dashboards
- Shot type / shooting dashboards (pull-up, catch-and-shoot proxies)
- Additional measure types on player/team game logs if available beyond Base/Advanced

Rules for probes:

- Sample season or sample game/player only
- Column inventory + row counts; no upsert to Supabase
- Rate-limit + retry; failures degrade to a note, notebook continues
- Toggleable flag (e.g. `RUN_ENDPOINT_PROBES = False`) for offline runs
- Cache locally under something like `data/raw/cache/discovery/` when successful

## Notebook structure

Path: `models/nba/pts_scoring_discovery.ipynb`

Cell flow:

1. **Imports / project root** — MIN-style `Path` + `data/` check; `sys.path` + `chdir`
2. **Config** — `SEASONS`, targets, filter `(minutes >= 5) | (starting == 1)`, `HOLDOUT_SEASON = "2025-26"`, `RUN_ENDPOINT_PROBES`, random seed
3. **Load + sanity** — read season parquets, enrich, concat; duplicate check; missingness by season; warn/skip missing files
4. **Wishlist coverage map** — table from column presence
5. **Scoring anatomy (same-game)** — Spearman + MI vs `pts` / `pts_per_min` / `minutes`; a few volume×efficiency plots; markdown banner: **NOT model features / leakage if used pre-tip**
6. **Predictive candidate pool** — auto-select prior-only columns (suffix/pattern: `ewm_`, `season_avg`, `lag1`, `roll`, `opp_`, `team_`, rest/B2B, market fields if present); drop same-game counting stats and ID/meta columns; assign lineage labels
7. **Univariate ranks** — Spearman + MI overall; per-season ranks; stability = rank correlation across seasons
8. **Model-based ranks** — fit lightweight XGB on pre-holdout seasons only; mean \|SHAP\| + permutation importance; optional leave-one-out via `models.shared.analysis.run_feature_ablation`; holdout MAE vs season-avg naive is diagnostic only
9. **Endpoint probes** — optional live inventory
10. **Deliverables** — top drivers, PPM shortlist, worth-adding gaps, notes for missing wishlist items

## Analysis rules

- “Driver” means association or predictive contribution, not proven causality.
- Same-game features are never mixed into pre-tip rankings.
- Every candidate gets a lineage label: `same_game` | `prior_player` | `team` | `opponent` | `context` | `market`.
- Missing columns are reported and skipped; they must not crash the notebook.
- High-cardinality IDs, post-game outcomes, and target-derived columns are excluded from predictive features.
- Missing values handled consistently before MI / XGBoost (document the rule in-notebook, e.g. median fill for numeric predictors after filtering).
- Fixed random seed for reproducibility.
- Holdout (`2025-26`) is reserved for predictive ranks / diagnostic MAE only — not for shortlist fitting decisions.
- No production artifact save from this notebook.

## Reuse

| Piece | Use |
|-------|-----|
| `ContextFeatureEngineer` | Primary load path |
| `models.shared.analysis.analyze_correlations` | Pairwise collinearity among shortlist candidates |
| `models.shared.analysis.run_feature_ablation` | Optional leave-one-out on shortlist |
| MIN notebook import / root pattern | Consistency with other `models/nba/*` notebooks |

Notebook-local helpers are fine for coverage map, MI ranks, season-stability, and probe inventory. Do **not** expand `fetch.py` in this task.

## Verification

- Notebook is valid JSON and opens in Jupyter.
- Runs sequentially with cwd at repo root (or notebook root resolution matching MIN).
- Leakage audit printout: predictive pool contains only prior-safe patterns; same-game set is disjoint.
- Coverage-map statuses match loaded columns.
- No DB writes; probes write only optional local cache.
- Offline mode (`RUN_ENDPOINT_PROBES=False`) still produces sections 1–8 and 10.
- Final deliverables cell prints rankings for all three targets, season stability summary, PPM shortlist, and missing-data priorities.

## Success criteria

Someone reading the notebook can answer:

1. Of the scoring-driver wishlist, what do we already have?
2. Same-game: what composes points vs points-per-minute vs minutes?
3. Pre-tip: which features stably predict each target across seasons?
4. What should we try next in PPM?
5. Which new endpoints are highest priority to ingest?
