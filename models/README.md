# Prop models — methodology

Quantile ML for NBA / WNBA player props (MIN, PTS via PPM, REB via RPM, AST via APM).
This folder is the **research + training** surface. Live inference stays in
`src/live_pipeline/` and loads versioned artifacts from `models/saved_models/`.

> **Rule:** document the hypothesis and freeze the naive baseline **before**
> treating a holdout number as a decision. Training math changes belong in
> experiment-log rows, not silent notebook edits.

---

## Layout

```
models/
├── README.md                 ← you are here
├── docs/
│   ├── hypothesis.md         # per-model H0 / H1 + frozen naives
│   ├── leakage_and_splits.md # as-of rules, date-safe WF, holdout
│   ├── experiment_log.txt    # E0…En decisions
│   └── postmortem_leakage.md # bugs caught mid-project
├── shared/                   # reusable helpers — no prop-specific constants
│   ├── baselines.py          # naive H0 tests
│   ├── splits.py             # holdout + date walk-forward
│   ├── metrics.py            # quantile fold scoring
│   ├── train.py              # fit / CV / holdout orchestration
│   ├── analysis.py           # ablation + correlation
│   └── artifacts.py          # save / load / predict
├── nba/                      # league notebooks (prop config in notebook)
├── wnba/
├── saved_models/             # joblib bundles + metrics sidecars
└── reports/                  # holdout tables / plots for the log
```

| Prop | Target | Notebooks |
|------|--------|-----------|
| MIN | `minutes` | `nba/min_nba_model.ipynb`, `wnba/min_wnba_model.ipynb` |
| PPM | `pts_per_min` | `nba/ppm_*`, `wnba/ppm_*` |
| APM | `ast_per_min` | `nba/apm_*`, `wnba/apm_*` |
| RPM | `reb_per_min` | `nba/rpm_*`, `wnba/rpm_*` |
| Discovery | scoring drivers (`pts`, `pts_per_min`, `minutes`) | `nba/pts_scoring_discovery.ipynb` |

---

## Problem

Predict pre-tip player outcomes as calibrated distributions (p10 / p50 / p90),
then convert to Over/Under lean vs the sportsbook line for live EV.

Each prop model is tested **independently** against a frozen naive baseline.
See [docs/hypothesis.md](docs/hypothesis.md).

---

## Metrics

| Role | Metric | Notes |
|------|--------|-------|
| **Primary** | MAE on the prop target | Optimization / H0 test |
| **Counter** | [p10, p90] coverage (~80%) + interval width | Stops useless wide bands |
| **Counter** | Line-hit / Brier vs market (when lines available) | MAE↑ can still hurt betting |
| **Slices** | Starters vs bench; minutes tiers | Overall MAE can hide star failure |

Hypothesis decision uses **MAE + paired Wilcoxon** on absolute errors
(α = 0.05, one-sided: model errors smaller than naive). Coverage and Brier
are reported; they do not redefine H0.

---

## Splits (summary)

```
[ older seasons ── date-safe walk-forward folds ── | LOCKED HOLDOUT season ]
```

- Entire `game_date` on one side of every fold (no intra-game leakage).
- Holdout never used for early stopping, feature selection, or hyperparams.
- Feature builder for research should match the live path
  (`src/pipeline/features/` → `src/live_pipeline/`).

Details: [docs/leakage_and_splits.md](docs/leakage_and_splits.md).

---

## Baseline bar

| Layer | What | Role |
|-------|------|------|
| **Naive (H0)** | Frozen per prop (e.g. season-to-date MIN) | Hypothesis test |
| **Strong** | Current `saved_models/*.joblib` on the same holdout | “Did we regress?” |
| **Market** | Consensus Over/Under (when available) | Directional usefulness |

---

## Cost / complexity kill switch

Pre-registered: after naive tests + feature prune, a heavier model (e.g. small
MLP on GPU) ships **only if** holdout MAE improves by ≥ 2% **and** line Brier
does not worsen. Otherwise keep XGB quantile — latency and ops win for Airflow.

Log the rejection as an experiment row; do not delete the failed run.

---

## Status (scaffold)

| Item | Status |
|------|--------|
| Folder + docs scaffold | ✅ |
| `shared/` helpers | ✅ |
| WNBA MIN notebook uses `shared/` + local config | ✅ |
| NBA MIN / other props migrated to `shared/` | ⬜ |
| PPM / APM / RPM naive H0 | ⬜ |
| Live parity audit | ⬜ |

---

## Reproduce

1. Build / refresh training parquets via the feature pipeline.
2. Open the league×prop notebook under `nba/` or `wnba/`.
3. Run split → walk-forward → holdout → naive test cells **in order**.
4. Save joblib under `saved_models/`; record the row in
   [docs/experiment_log.txt](docs/experiment_log.txt).

Root product README: [../README.md](../README.md).
