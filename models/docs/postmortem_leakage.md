# Postmortem — leakage and wrong assumptions

Honest mid-project failures. Fill numbers when you re-run clean comparisons.

---

## P1 — Intra-game / same-date leakage in validation

**Wrong assumption:** Row-wise or index-based time splits were “temporal enough.”

**What went wrong:** Players from the same `game_date` could land in both train
and val, so teammate/opponent structure leaked across the fold.

**How caught:** Walk-forward MAE looked optimistic vs a locked season holdout;
re-splitting by unique `game_date` closed part of the gap.

**Fix:** Date-based walk-forward masks (entire tip on one side). Documented in
[leakage_and_splits.md](leakage_and_splits.md).

**Before / after (fill in):**

| Setup | WF MAE | Holdout MAE | Gap |
|-------|--------|-------------|-----|
| Row-wise / leaky | | | |
| Date-safe WF | | | |

---

## P2 — Holdout used as early-stopping eval_set

**Wrong assumption:** Using the holdout (or a slice that included it) as
`eval_set` only “chooses n_estimators,” not the model.

**What went wrong:** `best_iteration` tuned to the season you claimed was blind.

**How caught:** Policy review + comparing “holdout-in-eval_set” vs early stop on
train-pool tail only.

**Fix:** Early stop on last ~10% of **train pool** dates/rows; predict holdout
once.

---

## P3 — Flattened quantile heads

**Wrong assumption:** Three quantile models with the same features would
automatically spread.

**What went wrong:** p10 / p50 / p90 predictions collapsed toward the same value;
empirical coverage drifted from nominal 10/50/90.

**How caught:** Per-quantile calibration table on validation / holdout.

**Fix:** Monitor coverage per quantile; investigate objective / fit when
nominal coverage fails (see notebook calibration cells).

**Before / after (fill in):**

| | Q10 cov | Q50 cov | Q90 cov |
|--|---------|---------|---------|
| Flattened | | | |
| After fix | | | |

---

## Template for new entries

**Wrong assumption:**  
**What went wrong:**  
**How caught:**  
**Fix:**  
**Evidence (table or link to `reports/`):**
